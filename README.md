# ChurnGuard

**Customer churn prediction that optimises money, not accuracy.**

[![Live demo](https://img.shields.io/badge/demo-live-brightgreen?style=flat&logo=streamlit&logoColor=white)](https://churnguard-jluuzdkc7cndyetu9vpqvs.streamlit.app/)
[![CI](https://github.com/Nitin75408/Churn_Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Nitin75408/Churn_Guard/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

### ▶ **[Try the live demo](https://churnguard-jluuzdkc7cndyetu9vpqvs.streamlit.app/)**

Move the tenure slider to 1 month on a month-to-month fiber plan and watch the
risk score, then switch to a two-year contract. The **Retention worklist** tab
shows the ranked call list and its hit rate.

An end-to-end machine learning system: **SQL analytics warehouse** → reproducible
data pipeline → leakage-free training → cost-based decisioning → explainable REST
API → demo dashboard → containerised deployment → CI.

---

## The headline

> Of the **75 highest-risk customers** the model identifies — the retention
> team's actual monthly capacity — **84% genuinely churned**, against a 26.5%
> base rate. **A 3.2× lift.**

| | |
|---|---|
| PR-AUC (sealed test set) | **0.662** |
| ROC-AUC | **0.853** |
| Recall at the operating threshold | **0.796** |
| Precision@75 | **0.840** |
| Net retention value | **$18,006 per 1,000 customers scored** |
| Lift over the existing business rule | **+46%** ($18,006 vs $12,324) |

Every figure measured **once**, on a test set sealed before any exploratory
analysis and verified unchanged by fingerprint — including the business-rule
baseline, so the comparison is like for like.

---

## Quickstart

### Nothing to install

**[https://churnguard-jluuzdkc7cndyetu9vpqvs.streamlit.app/](https://churnguard-jluuzdkc7cndyetu9vpqvs.streamlit.app/)** — deployed from this repository, redeployed on every push.

### Docker (nothing to install but Docker)

```bash
docker compose up --build
```

* API and interactive docs → <http://localhost:8000/docs>
* Dashboard → <http://localhost:8501>

### Local

```bash
# uv manages Python and dependencies: https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Reproduce everything from scratch — no manual downloads
uv run python -m churn_guard.data.ingest     # fetch + validate against a schema
uv run python -m churn_guard.data.split      # 70/15/15, sealed test set
uv run python -m churn_guard.eda             # analysis + figures (train only)
uv run python -m churn_guard.models.train    # tune 3 families, track in MLflow
uv run python -m churn_guard.models.evaluate # threshold, then the sealed test set
uv run python -m churn_guard.models.explain  # SHAP + fairness slices

# SQL analytics layer (optional — needs Docker)
docker compose up -d postgres
uv run python -m churn_guard.data.warehouse  # load raw + build views, print marts

uv run uvicorn churn_guard.api.main:app --reload   # API
uv run streamlit run streamlit_app.py              # dashboard
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment history
uv run pytest                                      # 78 tests
```

---

## The problem, framed properly

A telecom provider loses ~26.5% of its customer base annually. The retention
team currently reacts *after* a customer calls to cancel — by which point the
decision is made.

Three framing questions did more for this project than any modelling choice:

**1. What action follows a prediction?** A retention offer. If no action
existed, no accuracy would make the model worth building.

**2. What is the operational constraint?** The team can contact **500 customers
a month** out of 7,043. So the question is not *"who will churn?"* but
***"who are the 500 most worth calling?"*** — a **ranking** problem, which makes
Precision@K a first-class metric.

**3. What do mistakes cost?** Not the same amount, so the default 0.5 decision
threshold is wrong:

| | Predicted: no contact | Predicted: contact |
|---|---|---|
| **Stays** | $0 | **−$78** wasted offer |
| **Churns** | $0 | **+$156** = 0.30 × $780 − $78 |

The regret of a miss is **$156** — the intervention forgone — not the customer's
full $780 lifetime value, because an offer only succeeds 30% of the time. That
gives a **2:1** ratio and a derived threshold:

```
contact when   p × (acceptance_rate × CLV) > offer_cost
               p × (0.30 × 780)            > 78
               p                           > 0.333
```

Sweeping the validation set independently found **0.279**. Moving off 0.5 is
worth **+$2,361 per 1,000 customers** — the single highest return-per-effort
change in the project, and it involved no modelling at all.

Full write-up: [`docs/problem-definition.md`](docs/problem-definition.md).

---

## Architecture

```mermaid
flowchart TB
    subgraph training ["Training — offline, reproducible"]
        A[Public dataset<br/>fetched by script] --> B[Schema validation<br/>data contract]
        B --> C[Split 70/15/15<br/>test set sealed + fingerprinted]
        C --> D[EDA — train split only]
        D --> E[sklearn Pipeline<br/>clean · engineer · scale · encode]
        E --> F[RandomizedSearchCV<br/>3 model families × 30 candidates × 5 folds]
        F --> G[One-standard-error selection]
        G --> H[(MLflow<br/>runs + registry)]
        G --> I[Threshold from unit economics<br/>+ sensitivity analysis]
        I --> J[Sealed test set<br/>opened once]
    end

    subgraph serving ["Serving"]
        K[model.joblib<br/>the whole fitted pipeline] --> L[FastAPI<br/>/predict · /predict/batch · /health]
        L --> M[Streamlit dashboard]
        L --> N[Any other client]
    end

    G --> K
    O[Docker image] -.contains.-> L
    O -.contains.-> M
```

**The load-bearing decision:** the API serves the **entire fitted `Pipeline`**,
not a bare model. It receives raw customer records and reimplements zero
preprocessing — so train/serve skew is impossible by construction rather than by
discipline.

---

## The SQL analytics layer

A PostgreSQL warehouse sits beside the ML pipeline, organised in the standard
**medallion** pattern:

```
raw.customers            bronze — verbatim copy of the source, blanks and all
analytics.v_customers    silver — typed, renamed, cleaned; one source of truth
analytics.v_*            gold   — aggregated marts a BI tool reads directly
```

```bash
docker compose up -d postgres
uv run python -m churn_guard.data.warehouse
```

Then connect Power BI, Tableau, DBeaver or `psql` to `localhost:5432`
(database `churnguard`).

### Why a warehouse at all

| Reason | |
|---|---|
| **Scale** | 7k rows fit in pandas; 50M do not. Aggregation belongs in the database. |
| **One version of the truth** | The churn definition, the `TotalCharges` fix and the category collapsing live in *one view*. An analyst and a data scientist cannot disagree about the numbers. |
| **BI tools speak SQL** | Power BI cannot import a pandas script. It connects to a database. |

### What deliberately stays *out* of SQL

**Model feature engineering.** Serving receives a single JSON record with no
database to query, so features must be produced by the same fitted pipeline in
training and inference. Reimplementing them in SQL would create two copies of
the logic that drift apart — the exact train/serve skew the pipeline exists to
prevent.

> **SQL owns *what is true about customers*. The pipeline owns *what the model
> eats*.** Different jobs, no duplication.

### What the marts surface

| View | Answers |
|---|---|
| `v_kpi_summary` | headline KPIs for dashboard cards |
| `v_churn_by_segment` | churn by contract × internet, with `ROLLUP` subtotals |
| `v_retention_curve` | retention by tenure band, with `LAG` band-over-band deltas |
| `v_revenue_at_risk` | segments ranked by revenue lost, with running share |
| `v_addon_paradox` | Simpson's paradox, conditioned and unconditioned |
| `v_customer_risk` | transparent rules-based score, `NTILE` deciles |

### A finding only the SQL layer surfaced

```
churn rate (customers)   26.54%
churn rate (revenue)     30.50%   ← higher
```

**Churners are above-average spenders**, so the business loses proportionally
more revenue than customers. That reframes the case: the model is protecting
**30% of revenue**, not 26% of headcount.

And the loss is concentrated — the top 5 of 12 contract × payment segments carry
**91% of all revenue lost**, led by month-to-month fiber customers at **54.6%
churn** and **$1.2M/year at risk**. That is what makes targeted retention
worthwhile rather than a blanket discount.

The rules-based SQL score alone separates the base cleanly:

| Risk decile | Actual churn |
|---|---|
| 1 (highest) | **70.2%** |
| 5 | 23.2% |
| 10 (lowest) | **1.1%** |

A 64× spread from `CASE` expressions and `NTILE` — no model involved. It is the
transparent baseline the ML model has to beat, and it gives the dashboard a
ranking without a Python round trip.

---

## Results

Validation set used for model and threshold selection; test set opened once.

| Model | PR-AUC | ROC-AUC | Brier | Recall | P@75 | $/1,000 |
|---|---|---|---|---|---|---|
| B0 — majority class | 0.266 | 0.500 | 0.195 | 0.000 | 0.240 | $0 |
| B1 — existing business rule | 0.384 | 0.639 | 0.175 | 0.377 | 0.600 | $9,962 |
| Random forest (tuned) | 0.632 | 0.832 | 0.164 | 0.861 | 0.800 | $13,209 |
| Gradient boosting (tuned) | 0.639 | 0.836 | 0.139 | 0.701 | 0.760 | $17,711 |
| **Logistic regression (tuned)** | **0.640** | **0.835** | **0.139** | **0.765** | **0.787** | **$18,079** |

And on the sealed test set, model and current business rule scored side by side
at the same threshold:

| On the sealed test set | PR-AUC | Recall | P@75 | $/1,000 |
|---|---|---|---|---|
| B1 — existing business rule | 0.417 | 0.432 | 0.640 | $12,324 |
| **Logistic regression** | **0.662** | **0.796** | **0.840** | **$18,006** |
| | | | | **+46%** |

Note that **B0's PR-AUC of 0.266 equals the churn rate.** For PR-AUC, no-skill is
class prevalence, not 0.5 — comparing against 0.5 badly misjudges a model on
imbalanced data.

### Why the simplest model won

The three tuned families landed within **0.002 CV PR-AUC** of each other against
a fold-to-fold standard deviation of **±0.022**. That difference is noise; the
ranking would reshuffle under a different seed.

Applying the **one-standard-error rule** — among models within one standard
error of the best, take the simplest — selects logistic regression. Taking the
top number instead would have shipped the random forest, which is **worst on
business value** ($13,209 vs $18,079) and **worst calibrated** (Brier 0.164 vs
0.139), and overfits far more (train/CV gap +0.079 vs +0.009).

---

## Findings worth reading

### Simpson's paradox in the add-on services

Aggregated over all customers, churn appears to *rise* from 0 to 1 add-on, which
is nonsense:

| Add-ons | All customers | Internet customers only |
|---|---|---|
| 0 | 21.3% ⚠️ | **52.8%** |
| 1 | 44.9% | 44.9% |
| 6 | 6.2% | 6.2% |

The "0 add-ons" bucket mixes **1,074 customers with no internet at all** (7.1%
churn, structurally unable to buy add-ons) with **483 internet customers who
bought nothing** (52.8% churn — the single most at-risk group). The loyal
majority drowns out the at-risk minority and reverses the trend.

Conditioned on having internet, the relationship is clean and monotonic:
**52.8% → 6.2%**. Every add-on raises switching cost.

### Fiber optic: paying more, leaving more

| Internet | Churn |
|---|---|
| Fiber optic | **42.2%** |
| DSL | 19.0% |
| None | 7.1% |

The premium, higher-priced product has the worst retention. That is a
**price/quality problem worth escalating regardless of what the model does** —
a business finding the modelling happened to surface.

### The assumption that matters more than the model

Programme value across a plausible range of offer-acceptance rates:

| Acceptance rate | Threshold | Recall | $/1,000 |
|---|---|---|---|
| 15% | 0.721 | 0.228 | **$1,107** |
| 30% (assumed) | 0.279 | 0.765 | $18,079 |
| 50% | 0.234 | 0.811 | **$50,623** |

**The model is identical in every row.** Value swings **45×** on a number nobody
measured. The recommendation this produces is not a modelling one: *run a small
randomised pilot to measure the real acceptance rate before scaling*. That
measurement is worth more than any further tuning.

### A coefficient that flips sign

`MonthlyCharges` carries a **negative** coefficient (−0.429) while EDA shows
churners pay *more*. Both are correct — they answer different questions.
`has_internet` and `InternetService_Fiber optic` already absorb the
premium-service effect, so the residual variation in `MonthlyCharges` reflects
add-on count, which raises switching cost. **Coefficients are conditional, never
causal.**

### Fairness

| Slice | n | Base rate | Flagged | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|---|
| gender = Female | 529 | 0.270 | 0.399 | 0.797 | 0.540 | 0.850 |
| gender = Male | 528 | 0.259 | 0.405 | 0.796 | 0.509 | 0.858 |
| SeniorCitizen = 0 | 888 | 0.231 | 0.349 | 0.751 | 0.497 | 0.857 |
| SeniorCitizen = 1 | 169 | 0.444 | 0.680 | 0.920 | 0.600 | **0.774** |

**Gender: clean parity** — recall 0.797 vs 0.796. We could only verify this
because `gender` was deliberately retained despite carrying no predictive signal
(0.7pp spread).

**Seniors** are flagged at 68% vs 35%, but their base churn rate is also roughly
double (44% vs 23%), and **both recall and precision are higher** for them — they
are contacted more because they leave more, and the targeting is *more* accurate.
The genuine issue is **ranking quality** (ROC-AUC 0.774 vs 0.857) on a slice of
only 169 customers, so it is flagged for monitoring rather than treated as bias.

Demographic parity, equal opportunity and predictive parity **cannot all hold
simultaneously** when base rates differ — a proven impossibility, not a modelling
failure. This project prioritises **equal opportunity** (equal recall), so
everyone at risk gets a fair chance of being saved.

---

## Leakage prevention

The test set is split **before** any exploratory analysis, cleaning, imputation
or scaling — and never touched again until the final evaluation.

This is not caution for its own sake. Measured on this dataset, balancing classes
*before* splitting (a very common student shortcut) puts **713 customers on both
sides**, so 43% of the "unseen" test set has already been memorised:

| | ROC-AUC | PR-AUC |
|---|---|---|
| Balance → split (what you would report) | 0.9571 | 0.9547 |
| Split → balance train only (the truth) | 0.8102 | 0.5948 |
| **Inflation** | **+18%** | **+61%** |

You would publish a PR-AUC of 0.95 for a model worth 0.59, and nothing would look
wrong. **Leakage lies upward**, which is why it survives review — a bug that
worsens your score gets investigated; one that improves it gets celebrated.

Four structural defences:

1. **Split first**, immediately after schema validation
2. **All preprocessing inside a `Pipeline`**, so stateful steps are fitted per CV fold
3. **Test-set fingerprint** recorded at split time and reverified before evaluation
4. **A test that fails on leakage** — comparing the scaler's learned statistics
   against the training split's own, verified by mutation testing to fail when
   the pipeline is fitted on train+val

---

## Project structure

```
├── sql/                       the analytics warehouse
│   ├── 01_schema.sql          raw landing table + constraints  (bronze)
│   ├── 02_clean.sql           cleaned customer view            (silver)
│   └── 03_analytics.sql       aggregated marts for BI          (gold)
├── configs/config.yaml        every tunable value; nothing hardcoded
├── data/                      gitignored — regenerated by the ingest script
├── docs/
│   ├── problem-definition.md  framing, cost model, acceptance criteria
│   └── eda-findings.md        each finding paired with the decision it drove
├── src/churn_guard/
│   ├── config.py              typed config, paths resolved to the project root
│   ├── logger.py              console at INFO, rotating file at DEBUG
│   ├── exception.py           typed error hierarchy
│   ├── eda.py                 analysis + figures, training split only
│   ├── data/                  ingest · split (sealed) · warehouse (SQL loader)
│   ├── features/build.py      all 10 EDA decisions as sklearn transformers
│   ├── models/                metrics · baselines · train · evaluate · explain
│   └── api/                   schemas (Pydantic) · service · FastAPI app
├── tests/                     78 tests
├── streamlit_app.py           demo dashboard
├── Dockerfile                 multi-stage, non-root
└── .github/workflows/ci.yml   lint → pipeline → tests → image → smoke test
```

---

## Reproducibility

Every result regenerates from this repository alone. Verified by deleting
`models/` and `data/` entirely and re-running the pipeline:

```
Test seal verified: fingerprint 9ee2405ba4ee2531 unchanged since Day 1
Selected: logistic_regression (CV PR-AUC 0.6661)
PR-AUC 0.662 PASS   ROC-AUC 0.853 PASS   Recall 0.796 PASS
```

Identical fingerprint, identical model, identical metrics.

* **`uv.lock`** pins all ~180 transitive dependencies to exact versions
* **One seed** (`configs/config.yaml`) governs splitting, model init and CV shuffling
* **Data acquisition is a script**, so nothing depends on a manual download
* **CI runs the whole pipeline from a clean checkout**, so any dependency on a
  file that exists only on a developer's machine fails there first

---

## Honest limitations

**The dataset is a snapshot with no timestamps.** A true forward-looking
prediction window cannot be constructed or validated here, so this is a *current
risk score*, not a "will churn in the next 30 days" prediction. Production would
need point-in-time correct features and a **time-based** split — a random split
on temporal data leaks the future into the past.

**The cost assumptions are estimates.** Only average monthly revenue ($64.76) is
measured; CLV horizon, offer cost and acceptance rate are business assumptions.
The sensitivity table above shows how much rides on the last one.

**The test score is a point estimate.** PR-AUC 0.662 on 1,057 rows with a CV
standard deviation of ±0.022 is honestly reported as **≈0.65 ± 0.02**. Test
scoring higher than validation (0.662 vs 0.640) is split-to-split variation, not
improvement — both metrics are threshold-independent, so the threshold change
could not have moved them.

**Small slices are noisy.** The senior-citizen fairness gap rests on 169
customers.

## What production would add

| Gap | What it needs |
|---|---|
| Point-in-time features | A feature store (Feast/Tecton) with time-travel joins |
| Scheduled retraining | Airflow or Dagster, triggered on drift |
| Drift monitoring | Input/prediction distribution tracking with alerting |
| Proven business lift | A/B test against the existing rule — offline metrics only predict it |
| True acceptance rate | A randomised pilot; the single highest-value measurement available |
| Shadow deployment | Run alongside the current process before switching |

## Stack

Python 3.12 · **PostgreSQL / SQL** · uv · scikit-learn · pandas · MLflow · SHAP ·
FastAPI · Pydantic · Streamlit · Docker · pytest · ruff · GitHub Actions

## Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 21 columns, publicly available sample dataset. Downloaded
automatically by `churn_guard.data.ingest`.

## License

MIT
