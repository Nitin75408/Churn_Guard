# Tableau Dashboard — Build Spec

> Plan the dashboard before opening the tool, exactly as we planned the model
> before writing training code. A dashboard built by dragging fields around
> until it looks busy is the most common portfolio mistake.

**Data source:** `data/exports/customers.csv` — 7,043 rows, one per customer,
31 columns. Regenerate with:

```bash
docker compose up -d postgres
uv run python -m churn_guard.data.warehouse
uv run python -m churn_guard.data.export_bi
```

---

## The one rule that separates this from every other churn dashboard

> **Every chart must end in a decision, not an observation.**

Most dashboards show *what happened* and stop. Yours states what to **do**, with
a number attached. That is the difference between "can use Tableau" and "thinks
like an analyst", and it is the thing an interviewer remembers.

Three recommendations go on the front page as text, not buried in a tooltip:

1. **Move month-to-month fiber customers onto annual contracts.** They churn at
   54.6% and carry $1.2M of annual revenue at risk — the single largest
   concentration in the base.
2. **Push electronic-check payers onto auto-pay.** Manual payment churns at
   45.3% (electronic check) against 15-17% for the two automatic methods. Every manual payment is another chance to
   reconsider.
3. **Bundle one add-on service at onboarding.** Internet customers with zero
   add-ons churn at 52.2%; with four or more, at most 22.3%. Each add-on raises
   switching cost.

---

## Colour palette — use these exact values

Set them once in Tableau (Edit Colors → Custom) and reuse throughout. Consistent
colour is most of what makes a dashboard look professional.

| Role | Hex | Use for |
|---|---|---|
| Primary | `#2a78d6` | main series, retained customers |
| Secondary | `#eb6834` | second series when comparing |
| Tertiary | `#1baf7a` | third series |
| **Critical** | `#d03b3b` | high risk, churned |
| **Warning** | `#fab219` | medium risk |
| **Good** | `#0ca30c` | low risk, retained |
| Muted ink | `#898781` | axis labels, captions |
| Gridline | `#e1e0d9` | gridlines — barely visible |

**Rules:**
- Sequential (one measure, low→high): one hue, light→dark. Never a rainbow.
- Diverging (above/below a midpoint): blue ↔ red with a grey middle.
- Status colours are reserved — never reuse red for "series 4".
- **Never let colour be the only signal.** Pair risk tiers with a word or icon.

---

## Dashboard 1 — Executive Overview

*Audience: a manager who has 30 seconds.*

```
┌────────────────────────────────────────────────────────────────┐
│  CHURNGUARD — Customer Retention Overview                      │
├──────────┬──────────┬──────────┬──────────────────────────────┤
│ 7,043    │ 26.5%    │ 30.5%    │ $1.67M                       │  ← KPI tiles
│ customers│ churn    │ REVENUE  │ annual revenue at risk       │
│          │ rate     │ churn ▲  │                              │
├──────────┴──────────┴──────────┴──────────────────────────────┤
│  ▸ 3 RECOMMENDATIONS (text, top-left, impossible to miss)      │
├──────────────────────────────┬─────────────────────────────────┤
│  Churn rate by contract      │  Revenue at risk by segment     │
│  (horizontal bar, sorted)    │  (horizontal bar, sorted desc)  │
├──────────────────────────────┼─────────────────────────────────┤
│  Retention curve by tenure   │  Churn by payment method        │
│  (line)                      │  (horizontal bar, sorted)       │
└──────────────────────────────┴─────────────────────────────────┘
```

### Sheet-by-sheet

**KPI tiles** (4 sheets, or one with Measure Names)
Big number + small label. `revenue_churn_rate` is the interesting one — call out
that it exceeds customer churn, because it means churners are above-average
spenders.
*Not charts.* A single number needs no axes.

**Churn rate by contract type**
- Rows: `contract_type` · Columns: `AVG(churned = "Yes")` as a percentage
- Horizontal bars, **sorted descending**, direct labels on
- Colour: single hue — one series needs no rainbow
- Title states the finding: *"Month-to-month customers churn 15× more than two-year (42.7% vs 2.8%)"*

**Revenue at risk by segment**
- Rows: `contract_type` + `payment_method` · Columns: `SUM(annual_revenue_lost)`
- Sorted descending, top 6 only — a long tail adds noise, not insight
- Colour: sequential blue by value

**Retention curve by tenure band**
- Columns: `tenure_band` (sort: 0-6mo → 49mo+, **not alphabetical**)
- Rows: churn rate · Line chart with markers
- Annotate the steep early drop: *"over half of new customers leave in 6 months"*

**Churn by payment method**
- Horizontal bars, sorted. Colour the two automatic methods differently from the
  two manual ones — that contrast *is* the insight.

---

## Dashboard 2 — Retention Worklist

*Audience: the retention team, who can call ~500 customers a month.*

```
┌────────────────────────────────────────────────────────────────┐
│  Who should we call this month?                                │
├───────────────────────────────┬────────────────────────────────┤
│  Churn rate by risk decile    │  Revenue at risk by decile     │
│  (bar, decile 1 → 10)         │  (bar)                         │
├───────────────────────────────┴────────────────────────────────┤
│  CALL LIST — top 500 by risk score                             │
│  customer_id │ contract │ tenure │ monthly │ risk │ tier       │
└────────────────────────────────────────────────────────────────┘
```

**Churn rate by risk decile**
- Columns: `risk_decile` · Rows: churn rate
- Colour: sequential red — darker means higher risk
- Title: *"The top decile churns at 70%; the bottom at 1%"*

**Call list table**
- Filter: `risk_rank <= 500`
- Columns: `customer_id`, `contract_type`, `tenure_months`, `monthly_charges`,
  `risk_score`, `risk_tier`
- Sort by `risk_rank`
- Colour `risk_tier` with the status palette — **and keep the word visible**

**Interactivity that earns its place:**
- Filter on `contract_type` and `internet_service`, in one row above the charts
- Make the decile chart a filter for the table (Dashboard → Use as Filter)

---

## Anti-patterns to avoid

| Don't | Why |
|---|---|
| Pie charts | People compare angles badly. Use a sorted bar. |
| Dual axes (two y-scales) | The single most misleading chart type. Two measures → two charts. |
| A number label on every point | Label selectively — the max, the min, the endpoints. |
| Rainbow colour scales | Not perceptually ordered. One hue, light→dark. |
| Default Tableau blue-orange everywhere | Set the palette above once and commit to it. |
| Titles like "Sheet 1" | Every title states the **finding**, not the field name. |
| 12 charts because you can | 4–6 charts that answer real questions beat 12 that decorate. |

---

## Publishing

1. **File → Save to Tableau Public** (needs a free account)
2. Set a clear name: *"ChurnGuard — Customer Retention Analytics"*
3. In the workbook's Tableau Public settings, **allow the workbook to be
   downloaded** so reviewers can inspect your calculated fields
4. Copy the public URL into the project README, next to the CI badge

> ⚠️ Tableau Public makes your data and workbook **public**. Fine here — this is
> an open dataset — but never publish anything confidential to it.

---

## Definition of done

- [ ] Both dashboards built and readable at 1920×1080 without scrolling
- [ ] Every chart title states a finding, not a field name
- [ ] The 3 recommendations are visible without scrolling on Dashboard 1
- [ ] Palette applied consistently; no default Tableau colours left
- [ ] Tooltips are clean — no `AGG(...)` or `SUM(...)` leaking into them
- [ ] Published to Tableau Public, link added to the README
