"""Exploratory data analysis — training split only.

Run it::

    uv run python -m churn_guard.eda

Reads ``data/interim/train.csv`` and nothing else. Studying the validation or
test data would leak information into every preprocessing decision that follows
— not through code, but through the analyst. Once you have *seen* the test
distribution you cannot unsee it, and the choices you make are quietly fitted
to it.

Written as a script rather than a notebook so the figures regenerate identically
on any machine and the analysis can be reviewed in a diff.

Every section ends in a **decision**, not just an observation. A plot that does
not change what you build next was not worth making.
"""

from __future__ import annotations

import matplotlib

# Non-interactive backend: renders to file without needing a display, which is
# what lets this run in CI or a container.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from churn_guard.config import ConfigSection, load_config
from churn_guard.data.split import load_split
from churn_guard.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

CHURN_COLORS = {"No": "#4C78A8", "Yes": "#E45756"}


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def _save(fig: plt.Figure, name: str, cfg: ConfigSection) -> None:
    path = cfg.paths.figures / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


def overview(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Shape, dtypes and the target balance."""
    _rule("1. OVERVIEW")
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")

    target = cfg.data.target_column
    counts = df[target].value_counts()
    rate = counts.get("Yes", 0) / len(df)
    print("\n  Target balance")
    print(f"    No  (stayed) : {counts['No']:>5,}  ({1 - rate:.1%})")
    print(f"    Yes (churned): {counts['Yes']:>5,}  ({rate:.1%})")
    print(f"\n  Imbalance ratio: {counts['No'] / counts['Yes']:.2f} : 1")
    print(f"  A model predicting 'nobody churns' scores {1 - rate:.1%} accuracy")
    print("  while catching zero churners. This is why accuracy is not our metric.")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=target, hue=target, palette=CHURN_COLORS, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    ax.set_title(f"Target balance — {rate:.1%} churn")
    ax.set_xlabel("Churn")
    _save(fig, "01_target_balance", cfg)


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric, filling new customers with 0.

    Applied here for analysis only. The same logic is reimplemented inside the
    modelling pipeline so it is fitted and applied per-split rather than once
    over a whole dataframe.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    n_blank = int(out["TotalCharges"].isna().sum())
    # Blank exactly when tenure == 0: the customer has never been billed, so
    # the true value is 0 rather than "unknown". Median imputation would claim
    # these brand-new customers had spent ~$1,400.
    out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
    logger.info("TotalCharges -> numeric, %d blank values filled with 0.", n_blank)
    return out


def numeric_analysis(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Distributions of the three numeric features, split by churn."""
    _rule("2. NUMERIC FEATURES")
    target = cfg.data.target_column
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    print(f"\n  {'feature':<16}{'mean|stayed':>14}{'mean|churned':>14}{'difference':>14}")
    print("  " + "-" * 56)
    for col in numeric_cols:
        stayed = df.loc[df[target] == "No", col].mean()
        churned = df.loc[df[target] == "Yes", col].mean()
        print(f"  {col:<16}{stayed:>14.2f}{churned:>14.2f}{churned - stayed:>+14.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df, x=col, hue=target, palette=CHURN_COLORS,
            bins=30, element="step", stat="density", common_norm=False, ax=axes[0, i],
        )
        axes[0, i].set_title(f"{col} — distribution")
        sns.boxplot(
            data=df, x=target, y=col, hue=target,
            palette=CHURN_COLORS, legend=False, ax=axes[1, i],
        )
        axes[1, i].set_title(f"{col} by churn")
    fig.suptitle("Numeric features (training set)", fontsize=14)
    _save(fig, "02_numeric_distributions", cfg)

    # Tenure is the strongest numeric signal; bucket it to see the shape of the
    # relationship rather than assuming it is linear.
    _rule("3. TENURE vs CHURN — is the relationship linear?")
    buckets = pd.cut(
        df["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72],
        labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"],
    )
    table = (
        df.assign(bucket=buckets)
        .groupby("bucket", observed=True)[target]
        .agg(customers="size", churn_rate=lambda s: (s == "Yes").mean())
    )
    print()
    for bucket, row in table.iterrows():
        bar = "█" * int(row["churn_rate"] * 50)
        print(f"  {bucket:<10} n={row['customers']:>5.0f}  {row['churn_rate']:>6.1%}  {bar}")
    print("\n  Churn collapses as tenure grows, but not in a straight line —")
    print("  the drop from 0-6mo to 7-12mo is far steeper than 25-48 to 49-72.")
    print("  DECISION: add a bucketed tenure feature so linear models can use it.")

    fig, ax = plt.subplots(figsize=(7, 4))
    table["churn_rate"].plot(kind="bar", color="#E45756", ax=ax)
    ax.set_title("Churn rate by tenure bucket")
    ax.set_ylabel("churn rate")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "03_tenure_buckets", cfg)


def categorical_analysis(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Churn rate per category — where the strongest signals live."""
    _rule("4. CATEGORICAL FEATURES — churn rate by category")
    target = cfg.data.target_column
    exclude = {cfg.data.id_column, target}
    cat_cols = [
        c for c in df.columns
        if c not in exclude and (df[c].dtype == object or str(df[c].dtype) == "str")
    ]

    overall = (df[target] == "Yes").mean()
    spreads: dict[str, float] = {}

    for col in sorted(cat_cols):
        grouped = df.groupby(col, observed=True)[target].agg(
            n="size", rate=lambda s: (s == "Yes").mean()
        ).sort_values("rate", ascending=False)
        spreads[col] = float(grouped["rate"].max() - grouped["rate"].min())

        print(f"\n  {col}   (spread {spreads[col]:.1%})")
        for value, row in grouped.iterrows():
            bar = "█" * int(row["rate"] * 45)
            flag = "  <<<" if row["rate"] > overall * 1.4 else ""
            print(f"    {str(value):<22} n={row['n']:>5.0f}  {row['rate']:>6.1%}  {bar}{flag}")

    _rule("5. RANKING — which categoricals separate churners best?")
    print("\n  Spread = (highest churn rate) - (lowest churn rate) across categories.")
    print("  A large spread means the column tells you a lot; near zero means it does not.\n")
    for col, spread in sorted(spreads.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * int(spread * 60)
        print(f"    {col:<22}{spread:>7.1%}  {bar}")

    top = sorted(spreads.items(), key=lambda kv: kv[1], reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # Not strict: the grid holds six axes but a dataset with fewer categorical
    # columns yields fewer entries, and leaving the spare axes empty is fine.
    for ax, (col, _) in zip(axes.ravel(), top, strict=False):
        rates = (
            df.groupby(col, observed=True)[target]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
        )
        rates.plot(kind="bar", ax=ax, color="#E45756")
        ax.axhline(overall, color="#4C78A8", linestyle="--", label=f"overall {overall:.0%}")
        ax.set_title(col)
        ax.set_ylabel("churn rate")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Six strongest categorical signals (training set)", fontsize=14)
    _save(fig, "04_top_categoricals", cfg)


def correlation_analysis(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Numeric correlations, including the redundancy in TotalCharges."""
    _rule("6. CORRELATION & REDUNDANCY")
    target = cfg.data.target_column
    work = df.copy()
    work["churn_flag"] = (work[target] == "Yes").astype(int)
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "churn_flag"]

    corr = work[numeric].corr()
    print("\n  Correlation with churn:")
    for col in numeric[:-1]:
        value = corr.loc[col, "churn_flag"]
        bar = "█" * int(abs(value) * 60)
        print(f"    {col:<18}{value:>+7.3f}  {bar}")

    # TotalCharges is very nearly tenure x MonthlyCharges. Two features carrying
    # the same information destabilise linear-model coefficients (collinearity)
    # and split tree importance between them, making both look weaker.
    implied = work["tenure"] * work["MonthlyCharges"]
    redundancy = float(np.corrcoef(implied, work["TotalCharges"])[0, 1])
    print(f"\n  corr(tenure x MonthlyCharges, TotalCharges) = {redundancy:.4f}")
    print("  TotalCharges is almost exactly the product of the other two.")
    print("  DECISION: keep it, but expect collinearity — prefer regularised")
    print("  linear models, and read tree importances with that in mind.")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix (training set)")
    _save(fig, "05_correlation_matrix", cfg)


def service_column_quirk(df: pd.DataFrame, cfg: ConfigSection) -> None:
    """Six service columns encode 'not applicable' as a third category."""
    _rule("7. A STRUCTURAL QUIRK IN THE SERVICE COLUMNS")
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    print()
    for col in service_cols:
        values = sorted(df[col].unique())
        print(f"    {col:<20} {values}")

    print("\n  'No internet service' is not a customer choice — it is a structural")
    print("  consequence of InternetService == 'No'. The information is already")
    print("  carried by InternetService, so these are duplicated encodings.")
    print("\n  DECISION: collapse 'No internet service' and 'No phone service' into")
    print("  plain 'No'. Fewer one-hot columns, same information, less sparsity.")

    target = cfg.data.target_column
    work = df.assign(n_addons=(df[service_cols] == "Yes").sum(axis=1))

    def _table(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby("n_addons", observed=True)[target].agg(
            customers="size", rate=lambda s: (s == "Yes").mean()
        )

    def _show(table: pd.DataFrame) -> None:
        for n, row in table.iterrows():
            bar = "█" * int(row["rate"] * 45)
            print(f"    {n} add-ons  n={row['customers']:>5.0f}  {row['rate']:>6.1%}  {bar}")

    # Simpson's paradox lives here. Aggregated over everyone, churn appears to
    # *rise* from 0 to 1 add-on, which makes no sense. The "0 add-ons" bucket is
    # two unrelated populations stacked together: customers with no internet at
    # all (structurally unable to buy add-ons, and very loyal) drown out
    # internet customers who bought nothing (the single most at-risk group).
    print("\n  (a) All customers — misleading:")
    _show(_table(work))

    print("\n  (b) Internet customers only — the real relationship:")
    internet_only = work[work["InternetService"] != "No"]
    _show(_table(internet_only))

    zero = work[work["n_addons"] == 0]
    breakdown = zero.groupby("InternetService", observed=True)[target].agg(
        customers="size", rate=lambda s: (s == "Yes").mean()
    )
    print("\n  Why (a) lies — inside the '0 add-ons' bucket:")
    for service, row in breakdown.iterrows():
        print(f"    {str(service):<14} n={row['customers']:>5.0f}  churn {row['rate']:>6.1%}")

    print("\n  DECISION: engineer 'num_addon_services', and always pair it with")
    print("  'has_internet'. Alone the count is ambiguous; together they are")
    print("  monotonic and strong (52.8% -> 6.2%).")

    fig, ax = plt.subplots(figsize=(7, 4))
    _table(work)["rate"].plot(kind="line", marker="o", ax=ax, label="all customers (misleading)")
    _table(internet_only)["rate"].plot(
        kind="line", marker="o", ax=ax, color="#E45756", label="internet customers only"
    )
    ax.set_title("Simpson's paradox: churn vs add-on count")
    ax.set_ylabel("churn rate")
    ax.set_xlabel("number of add-on services")
    ax.legend(fontsize=8)
    _save(fig, "06_addons_simpson", cfg)


def main() -> None:
    cfg = load_config()

    logger.info("EDA reads the TRAINING split only — val and test stay untouched.")
    df = load_split("train", cfg)
    df = fix_total_charges(df)

    overview(df, cfg)
    numeric_analysis(df, cfg)
    categorical_analysis(df, cfg)
    correlation_analysis(df, cfg)
    service_column_quirk(df, cfg)

    print(f"\n{'=' * 78}")
    logger.info("EDA complete — figures written to %s", cfg.paths.figures)


if __name__ == "__main__":
    main()
