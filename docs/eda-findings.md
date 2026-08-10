# EDA Findings — Training Split Only

> **Scope:** `data/interim/train.csv` — 4,929 rows. The validation and test splits
> were not opened. Reproduce with `uv run python -m churn_guard.eda`.
> Figures in `reports/figures/`.
>
> Each finding ends in a **decision**. An observation that does not change what
> gets built next did not need to be made.

---

## 1. The target is imbalanced — 26.5% churn

| Class | Count | Share |
|---|---|---|
| Stayed | 3,621 | 73.5% |
| Churned | 1,308 | 26.5% |

Imbalance ratio **2.77 : 1**.

A model predicting "nobody churns" scores **73.5% accuracy** while catching zero
churners.

> **Decision:** accuracy is not reported as a headline metric. Primary metric is
> **PR-AUC**. Use `class_weight="balanced"` rather than resampling — at 2.77:1 the
> imbalance is mild, and reweighting avoids the duplicate-row leakage risk that
> oversampling introduces.

---

## 2. Contract type is the single strongest predictor

| Contract | Customers | Churn rate |
|---|---|---|
| Month-to-month | 2,708 | **43.1%** |
| One year | 1,022 | 10.7% |
| Two year | 1,199 | **2.7%** |

A 16× difference between the extremes — the widest spread of any feature (40.4pp).

Causally unsurprising: a two-year contract makes leaving expensive. This is the
feature a simple business rule would already use, and it sets the bar the model
must clear.

> **Decision:** keep as-is, one-hot encoded. Expect it to dominate feature
> importance; if it does not, suspect a bug.

---

## 3. Tenure is the strongest numeric signal, and it is not linear

Correlation with churn: **−0.350** (strongest of the numerics).

| Tenure | Customers | Churn rate |
|---|---|---|
| 0–6 months | 1,050 | **51.4%** |
| 7–12 months | 492 | 37.4% |
| 13–24 months | 708 | 30.1% |
| 25–48 months | 1,099 | 20.1% |
| 49–72 months | 1,580 | **9.5%** |

Mean tenure: 37.7 months for stayers vs **18.1** for churners.

The decline is steep early and flattens later — the first six months carry far
more risk per month than months 49–72. A linear term cannot express that shape.

> **Decision:** engineer a bucketed `tenure_bucket` feature alongside raw tenure,
> so linear models can capture the curvature. Tree models find it unaided, but the
> explicit feature makes the effect legible in explanations.

---

## 4. Fiber optic customers churn more than twice as often as DSL

| InternetService | Customers | Churn rate |
|---|---|---|
| Fiber optic | 2,154 | **42.2%** |
| DSL | 1,701 | 19.0% |
| No internet | 1,074 | **7.1%** |

Fiber is the premium, higher-priced product — and it has the worst retention.

> **Business insight, independent of the model:** customers paying *more* are
> leaving *more*. That points at a price/quality mismatch or service-reliability
> problem in the fiber product. This is worth escalating to the business
> regardless of what the model does, and is the kind of finding that makes an EDA
> section worth reading.

> **Decision:** keep as a feature. Flag the fiber finding in the README.

---

## 5. Electronic check payment triples churn risk

| PaymentMethod | Customers | Churn rate |
|---|---|---|
| Electronic check | 1,660 | **45.7%** |
| Mailed check | 1,131 | 19.4% |
| Bank transfer (automatic) | 1,072 | 15.9% |
| Credit card (automatic) | 1,066 | 15.1% |

The two *automatic* methods sit around 15%; the two *manual* methods are far worse.

Plausible mechanism: automatic payment removes a recurring decision point. Every
manual payment is another moment to reconsider the subscription.

> **Decision:** keep as-is. Also engineer a binary `is_automatic_payment` feature
> — the manual/automatic distinction is the real signal and deserves its own column.

---

## 6. ⚠️ Simpson's paradox in the add-on services

Aggregated over all customers, churn appears to **rise** from 0 to 1 add-on, which
is nonsense:

| Add-ons | All customers | Internet customers only |
|---|---|---|
| 0 | 21.3% ⚠️ | **52.8%** |
| 1 | 44.9% | 44.9% |
| 2 | 36.2% | 36.2% |
| 3 | 28.3% | 28.3% |
| 4 | 22.2% | 22.2% |
| 5 | 13.0% | 13.0% |
| 6 | **6.2%** | **6.2%** |

The "0 add-ons" bucket mixes two unrelated populations:

| Inside "0 add-ons" | Count | Churn |
|---|---|---|
| No internet at all (cannot buy add-ons) | 1,074 | 7.1% |
| Has DSL, bought nothing | 214 | 42.5% |
| Has fiber, bought nothing | 269 | **61.0%** |

1,074 structurally-loyal customers drown out 483 highly at-risk ones, and the
average reverses the trend.

Conditioned on having internet the relationship is clean and monotonic:
**52.8% → 6.2%**. Each add-on service raises switching cost.

> **Decision:** engineer `num_addon_services`, but always paired with
> `has_internet`. The count alone is ambiguous and would mislead a linear model.

---

## 7. Six service columns encode "not applicable" as a category

`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`
and `StreamingMovies` each take `Yes` / `No` / `No internet service`.

`No internet service` is not a customer choice — it is implied by
`InternetService == "No"`, so the information is already present and duplicated
six times. One-hot encoding as-is produces six redundant columns, all perfectly
correlated with each other.

> **Decision:** collapse `"No internet service"` and `"No phone service"` into plain
> `"No"` before encoding. Same information, six fewer sparse columns.

---

## 8. TotalCharges is almost perfectly redundant

```
corr(tenure × MonthlyCharges, TotalCharges) = 0.9996
```

`TotalCharges` is essentially the product of the other two numerics — it carries
almost no independent information.

Consequences: unstable coefficients in linear models (collinearity), and split
importance in tree models, which makes all three features look individually
weaker than they are.

Data quality: 11 blank strings, all with `tenure == 0` — new customers never
billed. Missing-at-random, so the correct fill is **0**, not the median. Median
imputation would claim these brand-new customers had spent ~$1,400.

> **Decision:** convert to numeric with `errors="coerce"`, fill with 0. Keep the
> column, but use regularised linear models and read tree importances with the
> collinearity in mind. Engineer `avg_charges_per_month = TotalCharges / max(tenure, 1)`,
> which is closer to independent.

---

## 9. Features with little or no signal

| Feature | Spread | Read |
|---|---|---|
| `gender` | **0.7pp** | No signal. Female 26.9% vs Male 26.2%. |
| `PhoneService` | 2.0pp | Near-useless — 90% of customers have it. |
| `MultipleLines` | 3.4pp | Weak. |

> **Decision:** retain `gender` — not for prediction, but as a **fairness slice**.
> A model that performs materially worse for one gender is a problem we need to be
> able to detect, and that requires keeping the column. Retain the phone columns;
> at this dataset size the cost of a few extra one-hot columns is negligible and
> regularisation will handle them.

---

## Preprocessing decisions carried into the pipeline

| # | Step | Reason |
|---|---|---|
| 1 | Drop `customerID` | Identifier, no signal; lets the model memorise individuals |
| 2 | `TotalCharges` → numeric, fill 0 | Blank iff `tenure == 0`; MAR, true value is 0 |
| 3 | Collapse `"No internet/phone service"` → `"No"` | Redundant with `InternetService` |
| 4 | `tenure_bucket` | Relationship is non-linear |
| 5 | `num_addon_services` + `has_internet` | Monotonic once conditioned (§6) |
| 6 | `is_automatic_payment` | Manual vs automatic is the real signal (§5) |
| 7 | `avg_charges_per_month` | Less collinear than raw `TotalCharges` |
| 8 | One-hot encode categoricals | Required by linear models |
| 9 | Standard-scale numerics | Required by regularised linear models |
| 10 | `class_weight="balanced"` | Mild imbalance; avoids oversampling leakage |

**All ten are implemented inside a scikit-learn `Pipeline`, fitted on the training
split alone** — never on the full dataset. That is what keeps the val/test scores
honest.
