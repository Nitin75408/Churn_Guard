# Problem Definition — ChurnGuard

> **Status:** Draft v1 · **Owner:** Nitin Madaan · **Last updated:** 2026-08-10
>
> In a real organisation this document is reviewed and signed off by the business
> stakeholder *before* any modelling work begins. If the framing here is wrong,
> no amount of model accuracy can rescue the project.

---

## 1. Business context

A telecom provider loses roughly a quarter of its customer base each year. Acquiring a
replacement customer costs several times more than retaining an existing one, so
reducing churn is the highest-leverage lever available to the retention team.

Today the retention team works reactively: they respond once a customer *calls to
cancel*. By that point the decision is usually already made. The opportunity is to
intervene **before** the customer decides to leave.

### Is ML actually the right tool here?

Asked deliberately, because most ML projects should be rules instead.

A simple rule (`contract == month-to-month AND tenure < 6`) does capture a meaningful
share of churners, and it will be our sanity check. But it cannot rank customers by
*how* at-risk they are, and ranking is exactly what a capacity-limited retention team
needs. It also cannot combine a dozen weak signals (payment method, support calls,
service mix) the way a model can. **ML is justified — but the rule is our floor, and
the model must clearly beat it.**

---

## 2. The ML problem

**Task type:** Binary classification, used as a **ranking** problem.

**Question:** *Given a customer's current profile and service usage, how likely are they
to churn?*

**Unit of prediction:** one active customer.

**Output:** a calibrated probability in `[0, 1]`, used to rank customers by risk.

### Definition of churn

This dataset is **contractual churn** — the customer actively cancels, producing a clear
event on a clear date. The `Churn` column is therefore given to us: `Yes` means the
customer left within the last month.

For contrast, in a **non-contractual** setting (e-commerce, retail) no cancellation event
exists and churn must be *defined*, e.g. "no purchase in 90 days". That choice materially
changes the dataset and must be justified explicitly.

### Prediction window — and a known limitation

**Intended framing:** predict churn far enough ahead that the retention team can act.
Assuming the team needs ~30 days to approve and deliver an offer, the useful target is
*"will this customer churn in the next 30 days?"*

**Actual limitation of this dataset:** the data is a **snapshot with no timestamps**. Each
row describes a customer's current state alongside a recent churn label. A true
forward-looking 30-day window therefore **cannot be constructed or validated here**.

**What we do instead:** treat this as a *current risk score* — "how at-risk is this customer
right now?" — and state the limitation openly.

**What production would require:**
- point-in-time correct features (only data available at the prediction date)
- a **time-based** train/test split, not a random one
- a label built from a real forward-looking window

---

## 3. The action (the "so what?" test)

A prediction with no action attached is worthless at any accuracy.

| Risk tier | Action taken |
|---|---|
| High | Proactive call from retention team + targeted discount offer |
| Medium | Automated email offer / plan-review nudge |
| Low | No action |

### Capacity constraint

> The retention team can contact roughly **500 customers per month**.

This is the single most important operational fact, and it reframes the problem:

We are **not** asking *"who will churn?"*
We are asking **"who are the 500 customers most worth contacting?"**

Consequence: **Precision@500** is a first-class metric, and ranking quality matters more
than any single yes/no decision.

---

## 4. Cost of errors

The two error types do **not** cost the same. All downstream threshold decisions follow
from this table.

### Assumptions

| Quantity | Assumed value | Source |
|---|---|---|
| Avg. revenue per customer / month | $65 | ✅ **verified 2026-08-10**: `MonthlyCharges` mean = $64.76 |
| Retained customer horizon | 12 months | business assumption |
| Customer lifetime value (CLV) | **$780** | 65 × 12 |
| Cost of a retention offer | **$78** | 20% discount for 6 months |
| Offer acceptance rate | **30%** | industry assumption |

### Cost matrix

|  | Predicted: stay | Predicted: churn |
|---|---|---|
| **Actually stays** | **TN** — $0 | **FP** — **−$78** (wasted discount) |
| **Actually churns** | **FN** — **−$780** (customer lost) | **TP** — **+$156** (0.30 × 780 − 78) |

### Implication

```
cost(False Negative)  =  10 ×  cost(False Positive)
```

A missed churner is **ten times** more expensive than a false alarm. Therefore the
default `0.5` decision threshold — which implicitly assumes the errors cost the same —
is **wrong for this problem**. The operating threshold will be chosen by maximising
expected value on the validation set, and is expected to land well below 0.5,
trading precision for recall.

---

## 5. Success metrics

### Primary (ML)

**PR-AUC (Average Precision).** Chosen over accuracy and ROC-AUC because the classes are
imbalanced (~26% positive). Accuracy is actively misleading here: predicting "nobody
churns" scores ~74% while being useless. PR-AUC measures performance specifically on the
minority class we care about.

### Secondary (ML)

ROC-AUC · Recall · Precision · F1 at the chosen threshold · **Precision@500**

### Business

**Expected net value per 1,000 customers scored**, computed from the cost matrix above.
This is the number reported to stakeholders.

### Guardrails

| Guardrail | Requirement | Why |
|---|---|---|
| Calibration | Brier score / reliability curve checked | probabilities drive money decisions |
| Latency | p95 < 200 ms per request | serving requirement |
| Fairness | no large performance gap by `SeniorCitizen`, `gender` | avoid discriminatory targeting |

---

## 6. Baselines and acceptance criteria

Defined **before** seeing any results, so the goalposts cannot move.

| Baseline | Description |
|---|---|
| B0 — Majority class | Predict "no churn" for everyone. Accuracy ~74%, recall 0. |
| B1 — Business rule | `month-to-month AND tenure < 6`. The current heuristic. |
| B2 — Logistic Regression | Simple, interpretable, well-tuned linear model. |

### Acceptance criteria

| Level | Target |
|---|---|
| Minimum | Beat B1 on PR-AUC by a clear margin |
| Target | **PR-AUC ≥ 0.65** and **ROC-AUC ≥ 0.84** |
| Business | **Recall ≥ 0.75** at the cost-optimal threshold |
| Stretch | Positive expected value maintained at Precision@500 |

---

## 7. Scope

**In scope:** tabular model on the public dataset · reproducible training pipeline ·
explainability (SHAP) · cost-based thresholding · REST API · demo UI · containerisation.

**Out of scope (and why):** real-time feature store, A/B testing, and automated retraining
all require production infrastructure and live traffic that a public snapshot cannot
provide. They are documented in the README as the natural next steps.

---

## 8. Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| Snapshot data, no time dimension | Cannot validate true forward-looking performance | Stated openly; time-based split specified as production requirement |
| Cost assumptions are estimates | Optimal threshold shifts if wrong | Sensitivity analysis across a range of FN:FP ratios |
| Target leakage from post-churn fields | Inflated offline scores, failure in production | Audit every feature for availability at prediction time |
| Small dataset (~7k rows) | Wide confidence intervals | Repeated stratified cross-validation; report variance, not just means |
