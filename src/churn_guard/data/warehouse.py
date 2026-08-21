"""Load the raw dataset into Postgres and build the analytics layer.

Run it::

    docker compose up -d postgres          # start the database
    uv run python -m churn_guard.data.warehouse

What this gives the project
---------------------------
A SQL layer between the source file and everything that consumes it. Three
reasons it exists, none of them cosmetic:

**Scale.** 7,000 rows fit in pandas. 50 million do not. Pushing filtering and
aggregation into the database is how analytics works once data outgrows memory.

**One version of the truth.** The churn definition, the ``TotalCharges`` fix and
the category collapsing live in one view. An analyst querying Postgres and a
dashboard reading the same view cannot disagree about the numbers.

**BI tools speak SQL.** Power BI and Tableau cannot import a pandas script. They
connect to a database, which is what makes the analytics marts in
``sql/03_analytics.sql`` directly consumable.

What deliberately stays out
---------------------------
Model feature engineering. Serving receives a single JSON record with no
database to query, so features must be computed by the same fitted pipeline in
both training and inference. Reimplementing them in SQL would create two copies
of the logic that drift apart — the train/serve skew the pipeline exists to
prevent. SQL owns *what is true about customers*; the pipeline owns *what the
model eats*.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from churn_guard.config import PROJECT_ROOT, ConfigSection, load_config
from churn_guard.data.ingest import load_raw_data
from churn_guard.exception import DataIngestionError
from churn_guard.logger import get_logger

logger = get_logger(__name__)


def warehouse_url(cfg: ConfigSection, hide_password: bool = False) -> str:
    """Build the connection URL from environment variables.

    Credentials come from the environment, never from the repository. The
    defaults in ``config.yaml`` match ``docker-compose.yml`` so local
    development works with no setup, but a real deployment overrides every one
    of them and nothing secret is ever committed.
    """
    settings = cfg.warehouse
    defaults = settings.defaults

    host = os.getenv(settings.host_env, defaults.host)
    port = os.getenv(settings.port_env, str(defaults.port))
    user = os.getenv(settings.user_env, defaults.user)
    password = os.getenv(settings.password_env, defaults.password)
    database = os.getenv(settings.database_env, defaults.database)

    shown = "***" if hide_password else password
    return f"postgresql+psycopg://{user}:{shown}@{host}:{port}/{database}"


def get_engine(cfg: ConfigSection | None = None) -> Engine:
    """Create a SQLAlchemy engine for the warehouse.

    ``pool_pre_ping`` sends a cheap liveness check before handing out a pooled
    connection. Without it, a connection that died while idle — a database
    restart, a network blip — surfaces as a confusing error on the next query.
    """
    cfg = cfg or load_config()
    return create_engine(warehouse_url(cfg), pool_pre_ping=True, future=True)


def wait_for_database(engine: Engine, timeout_seconds: int = 60) -> None:
    """Block until the database accepts connections.

    A freshly started Postgres container reports "running" before it is ready to
    serve queries. Retrying beats a sleep: it returns as soon as the database is
    actually up, instead of always waiting for the worst case.
    """
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Database ready (attempt %d).", attempt)
            return
        except OperationalError as err:
            if time.monotonic() >= deadline:
                raise DataIngestionError(
                    f"Database not reachable after {timeout_seconds}s. Is it "
                    f"running? Start it with: docker compose up -d postgres"
                ) from err
            time.sleep(2)


def run_sql_scripts(engine: Engine, cfg: ConfigSection) -> list[str]:
    """Execute every .sql file in order.

    Filenames are numbered because order matters: schemas before tables, tables
    before the views built on them. Running them as plain files — rather than
    embedding SQL in Python strings — keeps the SQL readable, diffable in
    review, and openable in any database tool.
    """
    sql_dir: Path = PROJECT_ROOT / str(cfg.warehouse.sql_dir)
    scripts = sorted(sql_dir.glob("*.sql"))
    if not scripts:
        raise DataIngestionError(f"No .sql files found in {sql_dir}")

    executed: list[str] = []
    for script in scripts:
        statements = script.read_text(encoding="utf-8")
        # autocommit: DDL should land as it runs, so a later failure does not
        # roll back the schema that already succeeded.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(statements))
        logger.info("Ran %s", script.name)
        executed.append(script.name)
    return executed


def load_raw(engine: Engine, frame: pd.DataFrame, cfg: ConfigSection) -> int:
    """Insert the source data into ``raw.customers``.

    Appends into the table created by the DDL rather than letting pandas invent
    the schema. That keeps the typed columns and CHECK constraints, so a source
    file with an unexpected category fails the load instead of silently adding
    a new one downstream.
    """
    rows = frame.to_sql(
        name=str(cfg.warehouse.raw_table),
        schema=str(cfg.warehouse.raw_schema),
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    logger.info("Loaded %d rows into raw.%s", len(frame), cfg.warehouse.raw_table)
    return rows if rows is not None else len(frame)


def read_sql(query: str, engine: Engine | None = None) -> pd.DataFrame:
    """Run a query and return a dataframe.

    The bridge between the two layers: SQL does the heavy aggregation, pandas
    receives a small result to plot or model.
    """
    engine = engine or get_engine()
    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 82}\n  {title}\n{'=' * 82}")


def main() -> None:
    """Build the warehouse end to end, then show what it contains."""
    cfg = load_config()
    engine = get_engine(cfg)

    logger.info("Connecting to %s", warehouse_url(cfg, hide_password=True))
    wait_for_database(engine)

    frame = load_raw_data(cfg)

    logger.info("Building schemas, tables and views...")
    run_sql_scripts(engine, cfg)
    load_raw(engine, frame, cfg)

    _print_section("1. HEADLINE KPIs  (analytics.v_kpi_summary)")
    kpis = read_sql("SELECT * FROM analytics.v_kpi_summary", engine).iloc[0]
    print(f"\n  customers                  {int(kpis.total_customers):>12,}")
    print(f"  churned                    {int(kpis.churned_customers):>12,}")
    print(f"  churn rate (customers)     {kpis.churn_rate:>12.2%}")
    print(f"  churn rate (revenue)       {kpis.revenue_churn_rate:>12.2%}   <-- higher!")
    print(f"  monthly revenue            ${kpis.total_monthly_revenue:>11,.0f}")
    print(f"  monthly revenue lost       ${kpis.monthly_revenue_lost:>11,.0f}")
    print(f"  avg tenure (all / churned) {kpis.avg_tenure_months:>7.1f} / {kpis.avg_tenure_churned:.1f} months")

    _print_section("2. CHURN BY SEGMENT, WITH ROLLUP SUBTOTALS")
    segments = read_sql(
        """
        SELECT contract_type, internet_service, customers, churn_rate,
               annual_revenue_at_risk, churn_index
        FROM analytics.v_churn_by_segment
        WHERE aggregation_level = 0
        ORDER BY churn_rate DESC LIMIT 6
        """,
        engine,
    )
    print(f"\n  {'contract':<16}{'internet':<14}{'n':>7}{'churn':>9}{'index':>8}{'$ at risk/yr':>15}")
    print("  " + "-" * 69)
    for _, row in segments.iterrows():
        print(f"  {row.contract_type:<16}{row.internet_service:<14}{int(row.customers):>7,}"
              f"{row.churn_rate:>9.1%}{row.churn_index:>8.2f}{row.annual_revenue_at_risk:>15,.0f}")

    _print_section("3. RETENTION CURVE  (LAG shows the change between bands)")
    curve = read_sql("SELECT * FROM analytics.v_retention_curve", engine)
    print(f"\n  {'band':<10}{'customers':>11}{'churn':>9}{'change':>9}{'% of base':>11}")
    print("  " + "-" * 50)
    for _, row in curve.iterrows():
        change = "     —" if pd.isna(row.churn_rate_change) else f"{row.churn_rate_change:>+9.1%}"
        print(f"  {row.tenure_band:<10}{int(row.customers):>11,}{row.churn_rate:>9.1%}{change}{row.pct_of_base:>10.1f}%")

    _print_section("4. WHERE THE MONEY IS  (ranked, with running share of loss)")
    risk = read_sql("SELECT * FROM analytics.v_revenue_at_risk LIMIT 5", engine)
    print(f"\n  {'#':>3}  {'contract':<16}{'payment':<26}{'churn':>8}{'$ lost/yr':>13}{'cum %':>8}")
    print("  " + "-" * 76)
    for _, row in risk.iterrows():
        print(f"  {int(row.risk_rank):>3}  {row.contract_type:<16}{row.payment_method:<26}"
              f"{row.churn_rate:>8.1%}{row.annual_revenue_lost:>13,.0f}{row.cumulative_pct_of_loss:>7.0f}%")

    _print_section("5. SIMPSON'S PARADOX, AS A QUERY")
    paradox = read_sql("SELECT * FROM analytics.v_addon_paradox", engine)
    print(f"\n  {'add-ons':>8}{'all customers':>16}{'with internet':>16}{'no internet':>14}")
    print("  " + "-" * 54)
    for _, row in paradox.iterrows():
        with_net = f"{row.churn_rate_with_internet:.1%}" if pd.notna(row.churn_rate_with_internet) else "—"
        no_net = f"{row.churn_rate_no_internet:.1%}" if pd.notna(row.churn_rate_no_internet) else "—"
        print(f"  {int(row.addon_service_count):>8}{row.churn_rate_all:>15.1%}{with_net:>16}{no_net:>14}")
    print("\n  The 'all customers' column reverses at 0 add-ons. Conditioning on")
    print("  internet access removes the confounder and the trend is monotonic.")

    _print_section("6. RULES-BASED RISK DECILES  (NTILE) — the model's SQL rival")
    deciles = read_sql(
        """
        SELECT risk_decile,
               COUNT(*)                                            AS customers,
               ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END), 4) AS actual_churn_rate,
               ROUND(SUM(annual_revenue), 0)                       AS annual_revenue
        FROM analytics.v_customer_risk
        GROUP BY risk_decile ORDER BY risk_decile
        """,
        engine,
    )
    print(f"\n  {'decile':>7}{'customers':>11}{'actual churn':>15}{'annual revenue':>17}")
    print("  " + "-" * 51)
    for _, row in deciles.iterrows():
        bar = "#" * int(row.actual_churn_rate * 30)
        print(f"  {int(row.risk_decile):>7}{int(row.customers):>11,}{row.actual_churn_rate:>14.1%}"
              f"{row.annual_revenue:>16,.0f}  {bar}")

    print(f"\n{'=' * 82}")
    logger.info("Warehouse ready. Connect Power BI / DBeaver to localhost:5432 (db: churnguard).")


if __name__ == "__main__":
    main()
