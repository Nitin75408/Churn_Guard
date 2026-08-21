---
title: ChurnGuard
emoji: 📉
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Churn prediction that optimises money, not accuracy
---

# ChurnGuard

**Customer churn prediction that optimises money, not accuracy.**

📦 **Source, tests and full write-up:** https://github.com/Nitin75408/Churn_Guard

---

## What this demo shows

**Score a customer** — move the sliders and watch the risk change, with a SHAP
explanation of *why*. Try 1 month tenure on a month-to-month fiber plan, then
switch to a two-year contract.

**Retention worklist** — the team can call ~7% of customers a month, so the model's
job is ranking, not classification. Of the top-ranked customers in this sample,
**95% genuinely churned** against a 23.7% base rate — a **4× lift**.

## The idea behind it

The decision threshold is **not 0.5**. Contacting a customer costs $78, a retention
offer works about 30% of the time, and a customer is worth $780 — so contacting is
worth it when `p × (0.30 × 780) > 78`, i.e. **p > 0.333**. Sweeping the validation
set put the empirical optimum at **0.279**.

Moving off the default threshold is worth about **$2,400 per 1,000 customers scored**
— no modelling involved, just arithmetic nobody did.

## Measured on a sealed test set

| | |
|---|---|
| PR-AUC | 0.662 |
| ROC-AUC | 0.853 |
| Recall | 0.796 |
| Precision@75 | 0.840 |
| Lift over the existing business rule | +46% |

The test set was split before any exploratory analysis and fingerprinted, then
opened exactly once.

## Honest limitations

The dataset is a snapshot with no timestamps, so a true forward-looking prediction
window cannot be validated — this is a *current risk score*. The 30% offer-acceptance
rate is an assumption, and programme value swings from $1,100 to $50,600 per 1,000
customers across a plausible 15–50% range. Measuring it with a pilot would matter
more than any further modelling.

Built with scikit-learn, FastAPI, Streamlit, PostgreSQL and Docker.
