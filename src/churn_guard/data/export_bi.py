"""Export the analytics marts to CSV for a BI tool.

Run it::

    docker compose up -d postgres
    uv run python -m churn_guard.data.warehouse    # build the views first
    uv run python -m churn_guard.data.export_bi

Why an export step exists
-------------------------
Not every consumer can reach the database. Free BI tiers read files rather than
connections, analysts want a spreadsheet, and a colleague wants something they
can open without credentials. The SQL still does the real work — joins,
aggregation, window functions — and the export is the handoff.

This is a normal pattern rather than a workaround: extract-based reporting is how
most analytics runs at scale, because a dashboard hitting the warehouse live on
every filter click is both slow and expensive.

What gets exported
------------------
One wide detail table plus the pre-aggregated marts.

The detail table matters most: given ``customers.csv`` a BI tool can build almost
any chart itself, so the dashboard is not limited to the aggregations we happened
to think of. The marts cover the cases SQL does better than a BI tool — ROLLUP
subtotals, LAG deltas, running shares.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_guard.config import PROJECT_ROOT, ConfigSection, load_config
from churn_guard.data.warehouse import get_engine, read_sql
from churn_guard.logger import get_logger

logger = get_logger(__name__)

EXPORT_DIRNAME = "exports"

# The detail table. v_customers holds the cleaned attributes and v_customer_risk
# the rules-based score; joining them gives one row per customer with everything
# a dashboard might group by.
DETAIL_QUERY = """
SELECT
    c.customer_id,
    c.gender,
    c.is_senior,
    c.has_partner,
    c.has_dependents,
    c.tenure_months,
    c.tenure_band,
    c.contract_type,
    c.internet_service,
    c.payment_method,
    c.is_automatic_payment,
    c.paperless_billing,
    c.phone_service,
    c.multiple_lines,
    c.online_security,
    c.online_backup,
    c.device_protection,
    c.tech_support,
    c.streaming_tv,
    c.streaming_movies,
    c.addon_service_count,
    c.has_internet,
    c.monthly_charges,
    c.total_charges,
    c.annual_revenue,
    r.risk_score,
    r.risk_decile,
    r.risk_rank,
    r.risk_tier,
    -- Exported as text rather than boolean: BI tools handle 'Yes'/'No' more
    -- predictably in legends and filters than true/false.
    CASE WHEN c.has_churned THEN 'Yes' ELSE 'No' END AS churned,
    -- Pre-computed so the dashboard can sum revenue at risk without writing
    -- a calculated field, which keeps the numbers consistent across sheets.
    CASE WHEN c.has_churned THEN c.annual_revenue ELSE 0 END AS annual_revenue_lost
FROM analytics.v_customers c
JOIN analytics.v_customer_risk r USING (customer_id)
"""

MARTS = {
    "kpi_summary": "SELECT * FROM analytics.v_kpi_summary",
    "churn_by_segment": "SELECT * FROM analytics.v_churn_by_segment",
    "retention_curve": "SELECT * FROM analytics.v_retention_curve",
    "revenue_at_risk": "SELECT * FROM analytics.v_revenue_at_risk",
    "addon_paradox": "SELECT * FROM analytics.v_addon_paradox",
}


def export_all(cfg: ConfigSection | None = None) -> dict[str, Path]:
    """Write the detail table and every mart to ``data/exports/``.

    Returns:
        Mapping of export name to the file written.
    """
    cfg = cfg or load_config()
    engine = get_engine(cfg)

    out_dir: Path = PROJECT_ROOT / "data" / EXPORT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    queries = {"customers": DETAIL_QUERY, **MARTS}

    for name, query in queries.items():
        frame: pd.DataFrame = read_sql(query, engine)
        destination = out_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        written[name] = destination
        logger.info(
            "%-18s %6d rows x %2d cols -> %s",
            name, len(frame), frame.shape[1], destination.name,
        )

    return written


def main() -> None:
    cfg = load_config()
    written = export_all(cfg)

    print(f"\n{'=' * 74}")
    print("  ANALYTICS EXTRACTS")
    print(f"{'=' * 74}\n")
    for name, path in written.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name:<20} {size_kb:>8.1f} KB   {path.relative_to(PROJECT_ROOT)}")

    print("\n  Primary data source: customers.csv  (one row per customer)")
    print("  Open Tableau Public -> Connect -> Text file -> select customers.csv")
    print(f"\n{'=' * 74}\n")


if __name__ == "__main__":
    main()
