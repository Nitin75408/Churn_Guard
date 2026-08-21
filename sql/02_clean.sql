-- =============================================================================
-- 02 — Cleaned customer view (silver layer)
-- =============================================================================
--
-- One view, one job: make raw data trustworthy and consistently named. Every
-- downstream query and every dashboard reads from here, so a fix applied once
-- propagates everywhere. That is the point of a shared warehouse layer — the
-- analyst and the data scientist cannot disagree about what "churned" means,
-- because there is only one definition and it lives here.
--
-- Note what this view does NOT do: it does not build model features. Feature
-- engineering stays in the scikit-learn pipeline, because serving receives one
-- JSON record with no database to query. Two implementations of the same
-- feature logic would drift apart, which is exactly the train/serve skew the
-- pipeline exists to prevent.

DROP VIEW IF EXISTS analytics.v_customers CASCADE;

CREATE VIEW analytics.v_customers AS
SELECT
    "customerID"                                        AS customer_id,
    "gender"                                            AS gender,
    ("SeniorCitizen" = 1)                               AS is_senior,
    ("Partner"    = 'Yes')                              AS has_partner,
    ("Dependents" = 'Yes')                              AS has_dependents,
    "tenure"                                            AS tenure_months,

    -- The blank-string fix, applied in exactly one place.
    -- NULLIF turns '' into NULL, then COALESCE fills 0. These rows are all
    -- tenure = 0: brand-new customers who have never been billed, so the true
    -- value is 0. Median imputation would claim they had already spent ~$1,400
    -- and would invert the strongest signal in the dataset.
    COALESCE(NULLIF(TRIM("TotalCharges"), '')::NUMERIC, 0)::NUMERIC(10, 2)
                                                        AS total_charges,
    "MonthlyCharges"                                    AS monthly_charges,

    "PhoneService"                                      AS phone_service,
    -- 'No phone service' and 'No internet service' are not customer choices —
    -- they are consequences of not having the parent service, and that fact is
    -- already recorded in phone_service / internet_service. Collapsing them to
    -- 'No' removes six redundant categories without losing any information.
    CASE WHEN "MultipleLines" = 'No phone service' THEN 'No'
         ELSE "MultipleLines" END                       AS multiple_lines,
    "InternetService"                                   AS internet_service,
    CASE WHEN "OnlineSecurity"   = 'No internet service' THEN 'No' ELSE "OnlineSecurity"   END AS online_security,
    CASE WHEN "OnlineBackup"     = 'No internet service' THEN 'No' ELSE "OnlineBackup"     END AS online_backup,
    CASE WHEN "DeviceProtection" = 'No internet service' THEN 'No' ELSE "DeviceProtection" END AS device_protection,
    CASE WHEN "TechSupport"      = 'No internet service' THEN 'No' ELSE "TechSupport"      END AS tech_support,
    CASE WHEN "StreamingTV"      = 'No internet service' THEN 'No' ELSE "StreamingTV"      END AS streaming_tv,
    CASE WHEN "StreamingMovies"  = 'No internet service' THEN 'No' ELSE "StreamingMovies"  END AS streaming_movies,

    "Contract"                                          AS contract_type,
    ("PaperlessBilling" = 'Yes')                        AS paperless_billing,
    "PaymentMethod"                                     AS payment_method,
    -- The two automatic methods churn at ~15%, the two manual ones at 19-46%.
    -- Removing a recurring decision point is what matters, so the flag is
    -- worth naming explicitly rather than leaving implicit in four categories.
    ("PaymentMethod" LIKE '%automatic%')                AS is_automatic_payment,

    -- The business definition of churn, written down once.
    ("Churn" = 'Yes')                                   AS has_churned,

    -- Tenure bands from the EDA: churn falls steeply then flattens, so equal
    -- width buckets would hide the shape. The final band is open-ended because
    -- the source tops out at 72 months but a real customer can exceed it.
    CASE
        WHEN "tenure" <= 6  THEN '0-6mo'
        WHEN "tenure" <= 12 THEN '7-12mo'
        WHEN "tenure" <= 24 THEN '13-24mo'
        WHEN "tenure" <= 48 THEN '25-48mo'
        ELSE                     '49mo+'
    END                                                 AS tenure_band,

    -- Count of paid add-ons. FILTER is the readable way to do a conditional
    -- count; the alternative is a stack of CASE WHEN ... THEN 1 ELSE 0 sums.
    (CASE WHEN "OnlineSecurity"   = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN "OnlineBackup"     = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN "DeviceProtection" = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN "TechSupport"      = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN "StreamingTV"      = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN "StreamingMovies"  = 'Yes' THEN 1 ELSE 0 END)
                                                        AS addon_service_count,
    ("InternetService" <> 'No')                         AS has_internet,

    -- Annualised revenue, the unit the business actually plans in.
    ("MonthlyCharges" * 12)::NUMERIC(10, 2)             AS annual_revenue
FROM raw.customers;

COMMENT ON VIEW analytics.v_customers IS
  'Cleaned customer view. Single source of truth for churn definition, category collapsing and the TotalCharges fix.';
