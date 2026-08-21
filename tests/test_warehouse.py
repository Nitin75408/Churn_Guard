"""SQL warehouse layer.

These tests skip when Postgres is unreachable, so the suite still passes on a
fresh clone with nothing running. Start the database with::

    docker compose up -d postgres
    uv run python -m churn_guard.data.warehouse

What is worth testing here is not that SQL executes — it is that the *business
rules encoded in SQL agree with the ones encoded in Python*. Two definitions of
churn that disagree is the classic warehouse failure, and it is silent.
"""

from __future__ import annotations

import pytest

from churn_guard.config import load_config


def _warehouse_available() -> bool:
    try:
        from sqlalchemy import text

        from churn_guard.data.warehouse import get_engine

        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1 FROM raw.customers LIMIT 1"))
        return True
    except Exception:
        return False


needs_warehouse = pytest.mark.skipif(
    not _warehouse_available(),
    reason="Postgres not reachable — run 'docker compose up -d postgres' and load it",
)


@pytest.fixture(scope="module")
def query():
    from churn_guard.data.warehouse import get_engine, read_sql

    engine = get_engine()
    return lambda sql: read_sql(sql, engine)


@needs_warehouse
class TestRawLayer:
    def test_row_count_matches_the_source(self, query):
        assert int(query("SELECT COUNT(*) AS n FROM raw.customers").n[0]) == 7043

    def test_total_charges_is_stored_as_text(self, query):
        """Raw mirrors the source faithfully, blanks included.

        Storing it as NUMERIC would either fail the load or silently coerce the
        blanks, hiding a real data-quality fact from anyone reading the table.
        """
        dtype = query(
            """
            SELECT data_type FROM information_schema.columns
            WHERE table_schema = 'raw' AND table_name = 'customers'
              AND column_name = 'TotalCharges'
            """
        ).data_type[0]
        assert dtype == "text"

    def test_customer_ids_are_unique(self, query):
        result = query(
            "SELECT COUNT(*) AS total, COUNT(DISTINCT \"customerID\") AS distinct_ids FROM raw.customers"
        )
        assert int(result.total[0]) == int(result.distinct_ids[0])


@needs_warehouse
class TestCleanView:
    def test_blank_total_charges_became_zero(self, query):
        """The 11 blanks are all tenure = 0 — never billed, so 0 is correct."""
        result = query(
            """
            SELECT COUNT(*) AS zero_charge,
                   COUNT(*) FILTER (WHERE tenure_months = 0) AS also_zero_tenure
            FROM analytics.v_customers WHERE total_charges = 0
            """
        )
        assert int(result.zero_charge[0]) == 11
        assert int(result.also_zero_tenure[0]) == 11

    def test_not_applicable_categories_were_collapsed(self, query):
        leftovers = query(
            """
            SELECT COUNT(*) AS n FROM analytics.v_customers
            WHERE 'No internet service' IN (online_security, online_backup,
                  device_protection, tech_support, streaming_tv, streaming_movies)
               OR multiple_lines = 'No phone service'
            """
        )
        assert int(leftovers.n[0]) == 0

    def test_churn_definition_matches_the_raw_table(self, query):
        """The SQL churn rate must equal the source's.

        A warehouse whose definition drifts from the source is worse than no
        warehouse: every dashboard built on it is confidently wrong.
        """
        result = query(
            """
            SELECT (SELECT AVG(CASE WHEN "Churn" = 'Yes' THEN 1.0 ELSE 0 END) FROM raw.customers)     AS raw_rate,
                   (SELECT AVG(CASE WHEN has_churned    THEN 1.0 ELSE 0 END) FROM analytics.v_customers) AS clean_rate
            """
        )
        assert float(result.raw_rate[0]) == pytest.approx(float(result.clean_rate[0]))

    def test_sql_churn_rate_matches_python(self, query):
        """Cross-language check: SQL and pandas must agree on the same fact."""
        from churn_guard.data.ingest import load_raw_data

        cfg = load_config()
        python_rate = (
            load_raw_data(cfg, validate=False)[cfg.data.target_column]
            == cfg.data.positive_label
        ).mean()
        sql_rate = float(
            query(
                "SELECT AVG(CASE WHEN has_churned THEN 1.0 ELSE 0 END) AS r FROM analytics.v_customers"
            ).r[0]
        )
        assert sql_rate == pytest.approx(python_rate, abs=1e-6)

    def test_addon_count_never_exceeds_six(self, query):
        result = query(
            "SELECT MIN(addon_service_count) AS lo, MAX(addon_service_count) AS hi FROM analytics.v_customers"
        )
        assert int(result.lo[0]) == 0
        assert int(result.hi[0]) == 6


@needs_warehouse
class TestAnalyticsMarts:
    def test_revenue_churn_exceeds_customer_churn(self, query):
        """Churners spend more than average, so revenue lost outpaces headcount.

        If this ever inverted it would change the business case, so it is worth
        pinning rather than leaving as a remembered observation.
        """
        kpis = query("SELECT * FROM analytics.v_kpi_summary").iloc[0]
        assert float(kpis.revenue_churn_rate) > float(kpis.churn_rate)

    def test_rollup_grand_total_matches_the_customer_count(self, query):
        """GROUPING(...) = 3 is the ROLLUP grand-total row."""
        total = query(
            "SELECT customers FROM analytics.v_churn_by_segment WHERE aggregation_level = 3"
        )
        assert int(total.customers[0]) == 7043

    def test_retention_improves_with_tenure(self, query):
        curve = query("SELECT * FROM analytics.v_retention_curve")
        rates = [float(r) for r in curve.churn_rate]
        assert rates == sorted(rates, reverse=True), "churn should fall as tenure grows"

    def test_addon_relationship_is_monotonic_once_conditioned(self, query):
        """Simpson's paradox: monotonic among internet customers, not overall."""
        paradox = query(
            "SELECT * FROM analytics.v_addon_paradox WHERE customers_with_internet > 0"
        )
        conditioned = [float(r) for r in paradox.churn_rate_with_internet]
        assert conditioned == sorted(conditioned, reverse=True)

        overall = [float(r) for r in paradox.churn_rate_all]
        assert overall != sorted(overall, reverse=True), (
            "the unconditioned series should NOT be monotonic — that is the paradox"
        )

    def test_risk_deciles_separate_churners(self, query):
        """The rules-based score must actually rank, or it is not a baseline."""
        deciles = query(
            """
            SELECT risk_decile, AVG(CASE WHEN has_churned THEN 1.0 ELSE 0 END) AS rate
            FROM analytics.v_customer_risk GROUP BY risk_decile ORDER BY risk_decile
            """
        )
        top = float(deciles.rate.iloc[0])
        bottom = float(deciles.rate.iloc[-1])
        assert top > 0.5, f"top decile churn {top:.1%} is too low to be useful"
        assert bottom < 0.1, f"bottom decile churn {bottom:.1%} is too high"
        assert top > bottom * 5

    def test_every_customer_gets_a_risk_tier(self, query):
        result = query(
            """
            SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE risk_tier IS NULL) AS missing
            FROM analytics.v_customer_risk
            """
        )
        assert int(result.n[0]) == 7043
        assert int(result.missing[0]) == 0
