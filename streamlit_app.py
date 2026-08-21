"""ChurnGuard demo dashboard.

Run it::

    uv run streamlit run streamlit_app.py

Architecture
------------
The dashboard talks to the FastAPI service over HTTP — one API, many clients,
which is how a real deployment is arranged. If the API is not reachable it falls
back to loading the model in-process, so a demo never dies because a second
terminal was not started.

That fallback costs about ten lines only because the scoring logic lives in
``churn_guard.api.service`` rather than inside the FastAPI route handlers. It is
the payoff for that separation.

Visual encoding
---------------
* Risk score is a **hero number**, not a chart — one value needs no axes.
* Risk tier uses the **status palette with an icon and a word**, so the state is
  never carried by colour alone.
* SHAP drivers use a **diverging bar** — the data has polarity (raises vs lowers
  risk), which is exactly what a two-hue-plus-neutral scale is for. Every bar
  carries a direct label, so the reading survives without colour.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Streamlit Community Cloud installs from requirements.txt and runs this file
# from the repository root — it never installs the project itself, so the src/
# layout would leave churn_guard unimportable. Adding src/ to the path is a
# no-op when the package is properly installed, as it is locally and in Docker.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from churn_guard.config import load_config
from churn_guard.data.split import load_split

# Configurable because the address differs by environment: localhost when both
# processes run on your machine, the compose service name inside Docker, a real
# hostname in a cluster. Hardcoding it would make the image environment-specific,
# which defeats the point of building one.
API_URL = os.getenv("CHURNGUARD_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 5

# Validated palette values (see the data-viz reference palette).
# Diverging pair for polarity, status palette for state.
COLOR_RAISES = "#e34948"     # red   — pushes risk up
COLOR_LOWERS = "#2a78d6"     # blue  — pulls risk down
COLOR_MUTED = "#898781"      # axis and label ink
COLOR_GRID = "#e1e0d9"
STATUS = {
    "high": ("#d03b3b", "🔴", "High risk"),
    "medium": ("#fab219", "🟠", "Medium risk"),
    "low": ("#0ca30c", "🟢", "Low risk"),
}

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")


# --------------------------------------------------------------------- client
class Client:
    """Scores customers via the API, falling back to an in-process model."""

    def __init__(self) -> None:
        self.mode = "api"
        self._service = None
        try:
            response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
            if not (response.ok and response.json().get("model_loaded")):
                raise RuntimeError("API reachable but no model loaded")
        except Exception:
            from churn_guard.api.service import ChurnService

            self._service = ChurnService()
            self._service.load()
            self.mode = "local"

    def info(self) -> dict:
        if self.mode == "api":
            return requests.get(f"{API_URL}/model/info", timeout=REQUEST_TIMEOUT).json()
        return {
            "model_family": self._service.metadata.get("model_family", "unknown"),
            "registered_version": str(
                self._service.metadata.get("registered_model_version", "")
            ),
            "decision_threshold": self._service.threshold,
            "trained_features": self._service.metadata.get("sklearn_feature_count", 0),
            "test_metrics": self._service.test_metrics,
        }

    def score(self, customers: list[dict]) -> list[dict]:
        if self.mode == "api":
            response = requests.post(
                f"{API_URL}/predict/batch",
                json={"customers": customers},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["predictions"]
        scored = self._service.score(pd.DataFrame(customers))
        for position, row in enumerate(scored):
            row["input_index"] = position
        ranked = sorted(scored, key=lambda r: r["churn_probability"], reverse=True)
        return [{**row, "rank": i + 1} for i, row in enumerate(ranked)]


@st.cache_resource
def get_client() -> Client:
    return Client()


@st.cache_data
def get_test_sample(n: int = 300) -> pd.DataFrame:
    cfg = load_config()
    df = load_split("test", cfg)
    return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)


@st.cache_data
def get_evaluation() -> dict:
    path = Path(load_config().paths.reports) / "final_evaluation.json"
    return json.loads(path.read_text()) if path.is_file() else {}


# ---------------------------------------------------------------------- chart
def driver_chart(drivers: list[dict]):
    """Diverging horizontal bars for one customer's SHAP contributions.

    Polarity is the whole point of the encoding, so the scale is diverging: one
    hue each side of a neutral zero line. Direct value labels mean the chart is
    still readable if the colours are indistinguishable to the viewer.
    """
    if not drivers:
        return None

    items = list(reversed(drivers))
    labels = [d["feature"] for d in items]
    values = [d["contribution"] for d in items]
    colors = [COLOR_RAISES if v > 0 else COLOR_LOWERS for v in values]

    fig, ax = plt.subplots(figsize=(7, 0.52 * len(items) + 1.1))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Thin bars leave a surface gap between neighbours rather than a solid block.
    ax.barh(labels, values, color=colors, height=0.62)

    span = max(abs(v) for v in values) or 1.0
    for y, value in enumerate(values):
        offset = span * 0.03
        ax.text(
            value + (offset if value > 0 else -offset),
            y,
            f"{value:+.2f}",
            va="center",
            ha="left" if value > 0 else "right",
            fontsize=9,
            color=COLOR_MUTED,
        )

    ax.axvline(0, color=COLOR_GRID, lw=1.4)
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_xlabel("contribution to risk (log-odds)", fontsize=9, color=COLOR_MUTED)
    ax.tick_params(axis="y", labelsize=9, colors="#52514e", length=0)
    ax.tick_params(axis="x", labelsize=8, colors=COLOR_MUTED, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- sidebar
client = get_client()
info = client.info()

with st.sidebar:
    st.title("📉 ChurnGuard")
    if client.mode == "api":
        st.success("Connected to the prediction API")
    else:
        # Not an error: the same scoring code runs in-process. Worth showing so
        # the reader knows which path is live, but not worth alarming them.
        st.info("Scoring in-process — model loaded directly")

    st.caption("Model in service")
    st.write(f"**{info.get('model_family', '?')}** · registry v{info.get('registered_version') or '-'}")
    st.metric("Decision threshold", f"{info.get('decision_threshold', 0):.3f}")

    metrics = info.get("test_metrics") or {}
    if metrics:
        st.caption("Sealed test set performance")
        st.write(
            f"PR-AUC **{metrics.get('pr_auc', 0):.3f}** · "
            f"ROC-AUC **{metrics.get('roc_auc', 0):.3f}**  \n"
            f"Recall **{metrics.get('recall', 0):.3f}** · "
            f"Precision@75 **{metrics.get('precision_at_k', 0):.3f}**"
        )
    st.caption(
        "Threshold is derived from unit economics "
        "(offer cost ÷ (acceptance rate × CLV)), not left at 0.5."
    )


# ---------------------------------------------------------------------- tabs
score_tab, worklist_tab, about_tab = st.tabs(
    ["Score a customer", "Retention worklist", "How it works"]
)

with score_tab:
    st.subheader("Score a single customer")
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 3)
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 79.0, step=0.5)
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

        st.caption("Add-on services")
        addons = {}
        a1, a2, a3 = st.columns(3)
        service_names = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        for i, name in enumerate(service_names):
            column = [a1, a2, a3][i % 3]
            with column:
                enabled = st.checkbox(name, value=False, key=f"addon_{name}")
            addons[name] = (
                "No internet service" if internet == "No" else ("Yes" if enabled else "No")
            )

        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": "No phone service" if phone == "No" else multiple,
            "InternetService": internet,
            **addons,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": round(float(monthly) * tenure, 2),
        }

    with right:
        result = client.score([payload])[0]
        probability = result["churn_probability"]
        color, icon, label = STATUS[result["risk_tier"]]

        st.markdown(
            f"<div style='font-size:4.2rem;line-height:1;font-weight:600;color:{color}'>"
            f"{probability:.0%}</div>"
            f"<div style='color:#52514e;margin-top:-.2rem'>probability of churn</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"### {icon} {label}")

        if result["should_contact"]:
            st.success(
                f"**Contact this customer.** Expected gain "
                f"${result['expected_value_of_contact']:,.2f} from a retention offer."
            )
        else:
            st.info(
                f"**No action.** Contacting would lose "
                f"${abs(result['expected_value_of_contact']):,.2f} on average."
            )

        st.markdown("**Why this score**")
        figure = driver_chart(result["top_drivers"])
        if figure:
            st.pyplot(figure, width="stretch")
            st.caption(
                "Red bars push risk up, blue bars pull it down; the number on each "
                "bar is its contribution in log-odds. Bars sum to the distance "
                "between this customer and the average customer."
            )
        else:
            st.caption("Explanations unavailable.")


with worklist_tab:
    st.subheader("Who should the retention team call this month?")
    cfg = load_config()
    capacity_share = int(cfg.costs.monthly_contact_capacity) / 7043

    sample = get_test_sample()
    feature_columns = [
        c for c in sample.columns
        if c not in (cfg.data.id_column, cfg.data.target_column)
    ]
    records = sample[feature_columns].to_dict(orient="records")
    for record in records:
        record["TotalCharges"] = float(
            pd.to_numeric(record["TotalCharges"], errors="coerce") or 0.0
        )
        record["SeniorCitizen"] = int(record["SeniorCitizen"])
        record["tenure"] = int(record["tenure"])

    predictions = client.score(records)
    capacity = max(1, round(len(predictions) * capacity_share))

    # The response is sorted by risk, so input_index is what joins it back to
    # the customer IDs and ground-truth labels held here.
    customer_ids = sample[cfg.data.id_column].tolist()
    churned = (sample[cfg.data.target_column] == cfg.data.positive_label).tolist()

    display = pd.DataFrame(
        {
            "rank": [p["rank"] for p in predictions],
            "customer": [customer_ids[p["input_index"]] for p in predictions],
            "risk": [p["churn_probability"] for p in predictions],
            "tier": [p["risk_tier"] for p in predictions],
            "expected value": [p["expected_value_of_contact"] for p in predictions],
            # Ground truth is shown only to demonstrate the hit rate. A live
            # system would not have this column at scoring time.
            "actually churned": [
                "yes" if churned[p["input_index"]] else "no" for p in predictions
            ],
        }
    )
    top = display.head(capacity)
    hit_rate = (top["actually churned"] == "yes").mean()
    base_rate = (display["actually churned"] == "yes").mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Call capacity", f"top {capacity} of {len(display)}", f"{capacity_share:.1%} of base")
    m2.metric("Hit rate in the worklist", f"{hit_rate:.1%}")
    m3.metric("Base rate if calling at random", f"{base_rate:.1%}",
              f"{hit_rate / base_rate:.1f}x lift", delta_color="off")

    st.caption(
        f"The team can contact {capacity_share:.1%} of customers per month, so the "
        "model's job is ranking, not classification. These are the top-ranked "
        f"{capacity} from a {len(display)}-customer sample of the sealed test set."
    )
    st.dataframe(
        top.style.format({"risk": "{:.1%}", "expected value": "${:,.2f}"}),
        width="stretch",
        hide_index=True,
    )


with about_tab:
    st.subheader("How this system works")
    st.markdown(
        """
**Pipeline.** Raw customer record → cleaning and feature engineering → scaling and
encoding → logistic regression. All of it is a single fitted scikit-learn
`Pipeline` object, so the API applies exactly the transformations training used.
Train/serve skew is impossible by construction.

**Why the threshold is not 0.5.** Contacting a customer costs $78. A retention
offer succeeds about 30% of the time and a customer is worth $780, so contacting
is worth it when `p × (0.30 × 780) > 78`, i.e. **p > 0.333**. Sweeping the
validation set put the empirical optimum at **0.279**. Moving off 0.5 is worth
about **$2,400 per 1,000 customers scored**.

**Why ranking beats classifying.** At the optimal threshold the model flags ~38%
of customers, but the retention team can call ~7%. The ordering is the product;
the cut-off is secondary.

**Honest limitations.** The dataset is a snapshot with no timestamps, so a true
forward-looking prediction window cannot be validated here. Production would
need point-in-time features and a time-based split. The 30% offer acceptance
rate is an assumption, not a measurement — and programme value swings from
\\$1,100 to \\$50,600 per 1,000 customers across a plausible 15–50% range, so
measuring it with a pilot matters more than any further modelling.
        """
    )
    evaluation = get_evaluation()
    if evaluation.get("sensitivity"):
        st.markdown("**Sensitivity to the acceptance-rate assumption**")
        sensitivity = pd.DataFrame(evaluation["sensitivity"])[
            ["acceptance_rate", "empirical_threshold", "recall", "value_per_1000"]
        ]
        st.dataframe(
            sensitivity.style.format(
                {
                    "acceptance_rate": "{:.0%}",
                    "empirical_threshold": "{:.3f}",
                    "recall": "{:.3f}",
                    "value_per_1000": "${:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
