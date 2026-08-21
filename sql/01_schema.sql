-- =============================================================================
-- 01 — Schemas and the raw landing table (bronze layer)
-- =============================================================================
--
-- The raw table is a FAITHFUL MIRROR of the source file. Column names keep the
-- source's casing and total_charges stays TEXT even though it holds numbers,
-- because 11 rows contain a blank string.
--
-- That is deliberate. A raw layer that "helpfully" cleans as it loads destroys
-- the evidence: if a cleaning rule turns out to be wrong six months later, you
-- need the original to re-derive from. Raw is append-only and never edited.
-- Every correction happens downstream in a view, where it is visible, version
-- controlled, and reversible.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

COMMENT ON SCHEMA raw IS
  'Landing zone. Faithful copy of source data. Never edited in place.';
COMMENT ON SCHEMA analytics IS
  'Cleaned views and aggregated marts. Everything here is derived from raw.';

DROP TABLE IF EXISTS raw.customers CASCADE;

CREATE TABLE raw.customers (
    "customerID"       TEXT PRIMARY KEY,
    "gender"           TEXT     NOT NULL,
    "SeniorCitizen"    SMALLINT NOT NULL,
    "Partner"          TEXT     NOT NULL,
    "Dependents"       TEXT     NOT NULL,
    "tenure"           INTEGER  NOT NULL,
    "PhoneService"     TEXT     NOT NULL,
    "MultipleLines"    TEXT     NOT NULL,
    "InternetService"  TEXT     NOT NULL,
    "OnlineSecurity"   TEXT     NOT NULL,
    "OnlineBackup"     TEXT     NOT NULL,
    "DeviceProtection" TEXT     NOT NULL,
    "TechSupport"      TEXT     NOT NULL,
    "StreamingTV"      TEXT     NOT NULL,
    "StreamingMovies"  TEXT     NOT NULL,
    "Contract"         TEXT     NOT NULL,
    "PaperlessBilling" TEXT     NOT NULL,
    "PaymentMethod"    TEXT     NOT NULL,
    "MonthlyCharges"   NUMERIC(10, 2) NOT NULL,
    -- TEXT, not NUMERIC: the source stores a blank string for customers with
    -- tenure 0. Declaring it NUMERIC would fail the load, and coercing during
    -- load would hide a real data-quality fact from anyone reading the table.
    "TotalCharges"     TEXT,
    "Churn"            TEXT     NOT NULL,

    -- Constraints as documentation AND enforcement. If the upstream file ever
    -- ships an unexpected category, the load fails loudly here rather than
    -- silently producing a new one-hot column three steps later.
    CONSTRAINT chk_senior   CHECK ("SeniorCitizen" IN (0, 1)),
    CONSTRAINT chk_tenure   CHECK ("tenure" >= 0),
    CONSTRAINT chk_churn    CHECK ("Churn" IN ('Yes', 'No')),
    CONSTRAINT chk_contract CHECK ("Contract" IN ('Month-to-month', 'One year', 'Two year')),
    CONSTRAINT chk_internet CHECK ("InternetService" IN ('DSL', 'Fiber optic', 'No'))
);

COMMENT ON TABLE raw.customers IS
  'Telco churn source data, loaded verbatim. TotalCharges is TEXT because the source contains blank strings.';

-- Indexes on the columns the analytics views group by. Unnecessary at 7k rows,
-- included because at 50M rows their absence is the difference between a
-- dashboard that loads and one that times out.
CREATE INDEX idx_customers_contract ON raw.customers ("Contract");
CREATE INDEX idx_customers_internet ON raw.customers ("InternetService");
CREATE INDEX idx_customers_churn    ON raw.customers ("Churn");
