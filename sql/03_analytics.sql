-- =============================================================================
-- 03 — Analytics marts (gold layer)
-- =============================================================================
--
-- Aggregated views a BI tool or an analyst reads directly. Each one answers a
-- specific business question, and each is built on analytics.v_customers so the
-- churn definition can never drift between them.

-- -----------------------------------------------------------------------------
-- Headline KPIs. One row, designed for dashboard cards.
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_kpi_summary CASCADE;

CREATE VIEW analytics.v_kpi_summary AS
SELECT
    COUNT(*)                                                   AS total_customers,
    -- FILTER is the readable form of a conditional aggregate, and clearer than
    -- SUM(CASE WHEN ... THEN 1 ELSE 0 END) once you have several of them.
    COUNT(*) FILTER (WHERE has_churned)                        AS churned_customers,
    ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END), 4)     AS churn_rate,
    ROUND(SUM(monthly_charges), 2)                             AS total_monthly_revenue,
    ROUND(SUM(monthly_charges) FILTER (WHERE has_churned), 2)  AS monthly_revenue_lost,
    -- Share of revenue lost, which is the number leadership actually reacts to.
    -- It exceeds the customer churn rate whenever churners are above-average
    -- spenders, which is exactly the case here.
    ROUND(
        SUM(monthly_charges) FILTER (WHERE has_churned)
      / NULLIF(SUM(monthly_charges), 0), 4)                    AS revenue_churn_rate,
    ROUND(AVG(tenure_months), 1)                               AS avg_tenure_months,
    ROUND(AVG(tenure_months) FILTER (WHERE has_churned), 1)    AS avg_tenure_churned,
    ROUND(AVG(monthly_charges), 2)                             AS avg_monthly_charges
FROM analytics.v_customers;

-- -----------------------------------------------------------------------------
-- Churn by contract and internet service, with subtotals.
-- ROLLUP produces the per-combination rows *and* the per-contract subtotals
-- *and* the grand total in a single pass — three queries collapsed into one.
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_churn_by_segment CASCADE;

CREATE VIEW analytics.v_churn_by_segment AS
SELECT
    COALESCE(contract_type, 'ALL CONTRACTS')                  AS contract_type,
    COALESCE(internet_service, 'ALL SERVICES')                AS internet_service,
    GROUPING(contract_type, internet_service)                 AS aggregation_level,
    COUNT(*)                                                  AS customers,
    COUNT(*) FILTER (WHERE has_churned)                       AS churned,
    ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END), 4)    AS churn_rate,
    ROUND(SUM(monthly_charges) FILTER (WHERE has_churned) * 12, 2)
                                                              AS annual_revenue_at_risk,
    -- Lift against the company-wide rate. A window function over the whole
    -- result set gives each row access to the global average without a
    -- self-join or a subquery.
    ROUND(
        AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)
      / NULLIF((SELECT AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)
                FROM analytics.v_customers), 0), 2)           AS churn_index
FROM analytics.v_customers
GROUP BY ROLLUP (contract_type, internet_service)
ORDER BY aggregation_level, churn_rate DESC;

-- -----------------------------------------------------------------------------
-- Retention curve by tenure band, with the drop between consecutive bands.
-- LAG reaches into the previous row, which is how you compute period-over-period
-- change without joining a table to itself.
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_retention_curve CASCADE;

CREATE VIEW analytics.v_retention_curve AS
WITH bands AS (
    SELECT
        tenure_band,
        MIN(tenure_months)                                     AS band_start,
        COUNT(*)                                               AS customers,
        AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)           AS churn_rate,
        SUM(monthly_charges)                                   AS monthly_revenue
    FROM analytics.v_customers
    GROUP BY tenure_band
)
SELECT
    tenure_band,
    customers,
    ROUND(churn_rate, 4)                                       AS churn_rate,
    ROUND(1 - churn_rate, 4)                                   AS retention_rate,
    ROUND(monthly_revenue, 2)                                  AS monthly_revenue,
    -- Change versus the previous band. Negative means churn is falling as
    -- customers mature, which is the expected shape.
    ROUND(churn_rate - LAG(churn_rate) OVER (ORDER BY band_start), 4)
                                                               AS churn_rate_change,
    ROUND(100.0 * customers / SUM(customers) OVER (), 1)       AS pct_of_base
FROM bands
ORDER BY band_start;

-- -----------------------------------------------------------------------------
-- Revenue at risk, ranked. Answers "where should retention spend go first?"
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_revenue_at_risk CASCADE;

CREATE VIEW analytics.v_revenue_at_risk AS
WITH segment AS (
    SELECT
        contract_type,
        payment_method,
        COUNT(*)                                               AS customers,
        COUNT(*) FILTER (WHERE has_churned)                    AS churned,
        AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)           AS churn_rate,
        SUM(monthly_charges * 12) FILTER (WHERE has_churned)   AS annual_revenue_lost
    FROM analytics.v_customers
    GROUP BY contract_type, payment_method
)
SELECT
    contract_type,
    payment_method,
    customers,
    churned,
    ROUND(churn_rate, 4)                                       AS churn_rate,
    ROUND(annual_revenue_lost, 2)                              AS annual_revenue_lost,
    RANK() OVER (ORDER BY annual_revenue_lost DESC)            AS risk_rank,
    -- Running share of total loss. Usually shows that a handful of segments
    -- account for most of the damage, which is what makes targeting worthwhile.
    ROUND(100.0 * SUM(annual_revenue_lost) OVER (ORDER BY annual_revenue_lost DESC)
        / SUM(annual_revenue_lost) OVER (), 1)                 AS cumulative_pct_of_loss
FROM segment
WHERE annual_revenue_lost > 0
ORDER BY annual_revenue_lost DESC;

-- -----------------------------------------------------------------------------
-- The Simpson's paradox finding, as a queryable view.
-- Aggregated over everyone, churn appears to RISE from 0 to 1 add-on. Split by
-- whether the customer has internet at all, the relationship is monotonic.
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_addon_paradox CASCADE;

CREATE VIEW analytics.v_addon_paradox AS
SELECT
    addon_service_count,
    -- The misleading aggregate.
    COUNT(*)                                                   AS customers_all,
    ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END), 4)     AS churn_rate_all,
    -- The honest figure, conditioned on the confounder.
    COUNT(*) FILTER (WHERE has_internet)                       AS customers_with_internet,
    ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)
          FILTER (WHERE has_internet), 4)                      AS churn_rate_with_internet,
    COUNT(*) FILTER (WHERE NOT has_internet)                   AS customers_no_internet,
    ROUND(AVG(CASE WHEN has_churned THEN 1 ELSE 0 END)
          FILTER (WHERE NOT has_internet), 4)                  AS churn_rate_no_internet
FROM analytics.v_customers
GROUP BY addon_service_count
ORDER BY addon_service_count;

-- -----------------------------------------------------------------------------
-- Per-customer rules-based risk score, entirely in SQL.
--
-- Deliberately NOT the model — it is the transparent heuristic the model must
-- beat, and it gives the dashboard a risk ranking without a Python round trip.
-- NTILE splits customers into deciles, which is how retention teams usually
-- want the list: "work the top decile."
-- -----------------------------------------------------------------------------
DROP VIEW IF EXISTS analytics.v_customer_risk CASCADE;

CREATE VIEW analytics.v_customer_risk AS
WITH scored AS (
    SELECT
        customer_id,
        contract_type,
        internet_service,
        payment_method,
        tenure_months,
        tenure_band,
        monthly_charges,
        annual_revenue,
        addon_service_count,
        has_churned,
        -- Weights come from the EDA spreads: contract type separates churners
        -- most, then internet service, then payment method.
          CASE contract_type WHEN 'Month-to-month' THEN 40
                             WHEN 'One year'       THEN 10
                             ELSE 0 END
        + CASE WHEN tenure_months <= 6  THEN 25
               WHEN tenure_months <= 12 THEN 15
               WHEN tenure_months <= 24 THEN 8
               ELSE 0 END
        + CASE internet_service WHEN 'Fiber optic' THEN 20
                                WHEN 'DSL'         THEN 5
                                ELSE 0 END
        + CASE WHEN payment_method = 'Electronic check' THEN 15 ELSE 0 END
        + CASE WHEN addon_service_count = 0 AND has_internet THEN 10 ELSE 0 END
        - CASE WHEN addon_service_count >= 4 THEN 10 ELSE 0 END
                                                               AS risk_score
    FROM analytics.v_customers
)
SELECT
    customer_id,
    contract_type,
    internet_service,
    payment_method,
    tenure_months,
    tenure_band,
    monthly_charges,
    annual_revenue,
    addon_service_count,
    risk_score,
    NTILE(10) OVER (ORDER BY risk_score DESC)                  AS risk_decile,
    RANK()    OVER (ORDER BY risk_score DESC)                  AS risk_rank,
    CASE WHEN NTILE(10) OVER (ORDER BY risk_score DESC) = 1  THEN 'high'
         WHEN NTILE(10) OVER (ORDER BY risk_score DESC) <= 3  THEN 'medium'
         ELSE 'low' END                                        AS risk_tier,
    has_churned
FROM scored;

COMMENT ON VIEW analytics.v_customer_risk IS
  'Transparent rules-based risk score. The baseline the ML model must beat, and the ranking the dashboard uses.';
