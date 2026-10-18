from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from auth import render_auth_gate, render_role_info
from db import get_connection
from streamlit_autorefresh import st_autorefresh

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Results"
MODELS = ROOT / "Models"
DATA = ROOT / "Data" / "Raw" / "creditcard.csv"

LIVE = RESULTS / "live_predictions.csv"
GNN_CSV = RESULTS / "gnn_vs_random_forest.csv"
BASELINE_CSV = RESULTS / "model_comparison.csv"

st.set_page_config(
    page_title="FraudEye AI | Fraud Intelligence",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not render_auth_gate():
    st.stop()

render_role_info()

user = st.session_state.get("user", {})
is_admin = user.get("role") == "admin"

# ---------------- PROFESSIONAL CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #06111f;
    --panel: #0a1929;
    --border: #1a3850;
    --text: #e8f1ff;
    --muted: #93a9c2;
    --cyan: #28e0d0;
    --green: #19d6a0;
    --red: #ff496c;
    --purple: #9670ff;
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 70% -10%,
        rgba(0,199,190,.10), transparent 30%),
        radial-gradient(circle at 5% 10%,
        rgba(45,112,255,.08), transparent 25%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(6,17,31,.82);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#071625,#081321);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -.025em;
}

p, label {
    color: var(--muted);
}

.brand {
    padding: 8px 0 22px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

.brand-name {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -.04em;
}

.brand-name span {
    color: var(--cyan);
}

.brand-sub {
    color: var(--muted);
    font-size: .72rem;
    margin-top: 3px;
    letter-spacing: .08em;
}

.eyebrow {
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: .14em;
    font-size: .72rem;
    font-weight: 700;
}

.hero {
    background: linear-gradient(
        110deg,
        rgba(17,54,73,.9),
        rgba(10,25,41,.92) 55%,
        rgba(23,32,62,.86)
    );
    border: 1px solid #1b5362;
    border-radius: 18px;
    padding: 22px 25px;
    margin: 4px 0 20px;
    box-shadow: 0 12px 35px rgba(0,0,0,.12);
}

.hero-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin: 3px 0 5px;
}

.hero-copy {
    color: #afc7dc;
    font-size: .9rem;
}

.status-pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    border: 1px solid rgba(25,214,160,.45);
    background: rgba(25,214,160,.10);
    color: #45f2c1;
    font-size: .76rem;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(13,32,51,.98),
        rgba(8,23,38,.98)
    );
    border: 1px solid var(--border);
    padding: 16px 18px;
    border-radius: 15px;
    min-height: 108px;
    box-shadow: 0 8px 22px rgba(0,0,0,.10);
}

div[data-testid="stMetricLabel"] {
    color: #a9bfd5 !important;
    font-size: .82rem;
}

div[data-testid="stMetricValue"] {
    color: #f2f7ff !important;
    font-weight: 800;
}

.section-heading {
    font-size: 1.1rem;
    font-weight: 750;
    margin: 8px 0 4px;
}

.section-note {
    color: var(--muted);
    font-size: .82rem;
    margin-bottom: 12px;
}

.stButton > button {
    background: linear-gradient(100deg,#0bbda9,#13d8b3);
    color: #04131f;
    border: 0;
    border-radius: 10px;
    font-weight: 800;
    padding: .55rem 1rem;
}

.stButton > button:hover {
    color: #04131f;
    border: 0;
    filter: brightness(1.08);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #0b1c2d;
    border-color: var(--border);
    border-radius: 9px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: var(--border);
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HELPER FUNCTIONS ----------------
def read_csv_safe(path):
    try:
        return pd.read_csv(path) if path.exists() else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def style_fig(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#c6d7e8",
            size=11,
        ),
        margin=dict(l=8, r=8, t=48, b=8),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#203a50")
    fig.update_yaxes(
        gridcolor="rgba(92,130,160,.15)",
        zerolinecolor="#203a50",
    )
    return fig


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-note">{subtitle}</div>',
            unsafe_allow_html=True,
        )


# ---------------- UPI RISK ANALYSIS HELPER ----------------
def analyze_upi_transaction(row):
    """Explainable rule-based risk indicator for the UPI demo."""
    score = 0
    reasons = []

    try:
        amount = float(row.get("amount", 0))
    except (TypeError, ValueError):
        return {
            "risk_score": 0,
            "risk_level": "Invalid input",
            "risk_reasons": "Invalid transaction amount",
        }

    if amount < 0:
        return {
            "risk_score": 0,
            "risk_level": "Invalid input",
            "risk_reasons": "Amount cannot be negative",
        }

    if amount >= 50000:
        score += 2
        reasons.append("High transaction amount")
    elif amount >= 10000:
        score += 1
        reasons.append("Elevated transaction amount")

    hour = row.get("hour", None)

    if hour is None or pd.isna(hour):
        timestamp = pd.to_datetime(
            row.get("timestamp", None),
            errors="coerce",
        )
        if pd.notna(timestamp):
            hour = timestamp.hour

    try:
        hour = int(hour)
        if 0 <= hour <= 4:
            score += 1
            reasons.append("Transaction during late-night hours")
    except (TypeError, ValueError):
        pass

    new_recipient = str(
        row.get("new_recipient", False)
    ).strip().lower()

    if new_recipient in {"true", "yes", "y", "1", "new"}:
        score += 2
        reasons.append("New recipient")

    try:
        recent_transactions = int(
            float(row.get("transactions_last_hour", 0) or 0)
        )
    except (TypeError, ValueError):
        recent_transactions = 0

    if recent_transactions >= 5:
        score += 2
        reasons.append("Many transactions in the last hour")
    elif recent_transactions >= 3:
        score += 1
        reasons.append("Several transactions in the last hour")

    try:
        failed_attempts = int(
            float(row.get("failed_attempts_last_hour", 0) or 0)
        )
    except (TypeError, ValueError):
        failed_attempts = 0

    if failed_attempts >= 2:
        score += 3
        reasons.append("Multiple failed attempts")
    elif failed_attempts == 1:
        score += 1
        reasons.append("A failed attempt was recorded")

    if score >= 5:
        level = "High"
    elif score >= 3:
        level = "Medium"
    else:
        level = "Low"

    if not reasons:
        reasons.append("No configured risk indicators were triggered")

    return {
        "risk_score": score,
        "risk_level": level,
        "risk_reasons": "; ".join(reasons),
    }


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-name">'
        'ðŸ›¡ï¸ FraudEye <span>AI</span></div>'
        '<div class="brand-sub">REAL-TIME FRAUD INTELLIGENCE</div></div>',
        unsafe_allow_html=True
    )

    pages = [
        "Overview",
        "Live Monitoring",
        "Model Comparison",
        "Test Transaction",
        "UPI Risk Analyzer",
        "About Project",
    ]

    # Admin Panel sirf admin users ko dikhega.
    if is_admin:
        pages.insert(5, "Admin Panel")

    page = st.radio("NAVIGATION", pages)
    st.markdown("---")
    st.caption("Machine learning fraud detection prototype")


# ---------------- LOAD PROJECT FILES ----------------
model_path = MODELS / "fraudeye_random_forest.joblib"
config_path = MODELS / "fraudeye_config.joblib"

try:
    model = joblib.load(model_path) if model_path.exists() else None
    config = joblib.load(config_path) if config_path.exists() else {}
except Exception:
    model = None
    config = {}

threshold = (
    float(config.get("threshold", 0.20))
    if isinstance(config, dict)
    else 0.20
)


@st.cache_data(show_spinner=False)
def load_dataset(path):
    return pd.read_csv(path)


try:
    df = load_dataset(str(DATA)) if DATA.exists() else None
except Exception:
    df = None

live_df = read_csv_safe(LIVE)
gnn_df = read_csv_safe(GNN_CSV)
baseline_df = read_csv_safe(BASELINE_CSV)

colors = {
    "Random Forest": "#19d6a0",
    "GraphSAGE": "#9670ff",
    "Fraud": "#ff496c",
    "Legitimate": "#2788ff",
}


# ==========================================================
# PAGE 1: OVERVIEW
# ==========================================================
if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">AI-POWERED PROTECTION</div>
        <div class="hero-title">
            Detect suspicious activity. Protect every transaction.
        </div>
        <div class="hero-copy">
            A unified workspace for fraud classification,
            transaction monitoring, and model experiments.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and "Class" in df.columns:
        total = len(df)
        fraud = int((df["Class"] == 1).sum())
        legit = total - fraud
        rate = fraud / total * 100 if total else 0
    else:
        total = fraud = legit = rate = 0

    if (
        live_df is not None
        and not live_df.empty
        and "prediction" in live_df.columns
    ):
        live_total = len(live_df)
        live_fraud = int(
            (
                live_df["prediction"]
                .astype(str)
                .str.lower() == "fraud"
            ).sum()
        )
        lat = pd.to_numeric(
            live_df.get("latency_ms", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        avg_lat = lat.mean() if not lat.empty else None
    else:
        live_total = live_fraud = 0
        avg_lat = None

    a, b, c, d = st.columns(4)
    a.metric("Dataset Transactions", f"{total:,}" if total else "â€”")
    b.metric("Dataset Fraud Cases", f"{fraud:,}" if total else "â€”")
    c.metric("Fraud Rate", f"{rate:.3f}%" if total else "â€”")
    d.metric(
        "Avg API Latency",
        f"{avg_lat:.2f} ms" if avg_lat is not None else "Run simulator",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    x, y, z = st.columns([1.25, 1, 1])

    with x:
        section(
            "Transaction class distribution",
            "Historical dataset distribution",
        )
        if total:
            dist = pd.DataFrame({
                "Transaction Type": ["Legitimate", "Fraud"],
                "Count": [legit, fraud],
            })
            fig = px.pie(
                dist,
                names="Transaction Type",
                values="Count",
                hole=.72,
                color="Transaction Type",
                color_discrete_map={
                    "Legitimate": colors["Legitimate"],
                    "Fraud": colors["Fraud"],
                },
            )
            fig.update_traces(textinfo="percent")
            fig.add_annotation(
                text=f"{rate:.3f}%<br><sup>fraud rate</sup>",
                x=.5,
                y=.5,
                showarrow=False,
                font=dict(size=17, color="#f0f6ff"),
            )
            st.plotly_chart(
                style_fig(fig, 300),
                use_container_width=True,
            )
        else:
            st.info("Dataset not found: data/raw/creditcard.csv")

    with y:
        section(
            "Model performance",
            "F1-score on shared sampled test split",
        )
        if (
            gnn_df is not None
            and {"Model", "F1-score"}.issubset(gnn_df.columns)
        ):
            fig = px.bar(
                gnn_df,
                x="Model",
                y="F1-score",
                color="Model",
                color_discrete_map=colors,
                text_auto=".3f",
                range_y=[0, 1],
            )
            st.plotly_chart(
                style_fig(fig, 300),
                use_container_width=True,
            )
        else:
            st.info(
                "Run Src/compare_gnn_rf.py to create comparison results."
            )

    with z:
        section("Simulator activity", "Saved prediction log")
        st.metric("Transactions logged", f"{live_total:,}")
        st.metric("Fraud alerts in log", f"{live_fraud:,}")
        st.caption("Simulator log, not a live bank feed.")

    st.markdown("---")
    section(
        "Recent transactions",
        "Latest rows saved by the API-connected simulator",
    )

    if live_df is not None and not live_df.empty:
        st.dataframe(
            live_df.tail(8).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "â¬‡ Download transaction log",
            live_df.to_csv(index=False).encode("utf-8"),
            "fraudeye_live_predictions.csv",
            "text/csv",
        )
    else:
        st.info("No transaction rows available yet.")

    st.warning(
        "Compare models only when their metrics were calculated "
        "using comparable datasets, splits, and evaluation methods."
    )


# ==========================================================
# PAGE 2: LIVE MONITORING
# ==========================================================
elif page == "Live Monitoring":

    st.markdown("## Live Transaction Monitoring")
    st.caption(
        "Monitor predictions saved by your local FraudEye simulator/API."
    )

    refresh = st.toggle("Auto-refresh every 5 seconds", value=False)

    if refresh:
        st_autorefresh(interval=5000, key="live_monitor_refresh")

    if live_df is not None and not live_df.empty:
        st.dataframe(
            live_df.tail(100).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        if "prediction" in live_df.columns:
            prediction_text = (
                live_df["prediction"].astype(str).str.lower()
            )
            fraud_count = int((prediction_text == "fraud").sum())
            total_count = len(live_df)

            a, b, c = st.columns(3)
            a.metric("Transactions", f"{total_count:,}")
            b.metric("Fraud Alerts", f"{fraud_count:,}")
            c.metric(
                "Fraud Alert Rate",
                f"{fraud_count / total_count * 100:.2f}%"
                if total_count else "0.00%",
            )

        st.download_button(
            "Download monitoring data",
            live_df.to_csv(index=False).encode("utf-8"),
            "fraudeye_monitoring.csv",
            "text/csv",
        )
    else:
        st.info(
            "No simulator records found yet. Start your transaction "
            "simulator or send predictions to the API."
        )


# ==========================================================
# PAGE 3: ADMIN PANEL
# ==========================================================
elif page == "Admin Panel":

    if not is_admin:
        st.error("Admin access required.")
        st.stop()

    st.markdown("## Admin Panel")
    st.caption("User administration for the FraudEye AI prototype.")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, email, role, created_at
                    FROM users
                    ORDER BY id
                    """
                )
                users = cur.fetchall()

        users_df = pd.DataFrame(
            users,
            columns=["id", "name", "email", "role", "created_at"],
        )

        st.metric("Registered Users", len(users_df))
        st.dataframe(
            users_df[["id", "name", "email", "role", "created_at"]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error("Could not load the users table.")
        st.caption(str(exc))

    st.info(
        "This panel displays account information. Password hashes "
        "and login credentials should never be displayed here."
    )


# ==========================================================
# PAGE 4: MODEL COMPARISON
# ==========================================================
elif page == "Model Comparison":

    st.markdown("## Model Comparison")
    st.caption(
        "Compare the saved results from your fraud detection models."
    )

    if (
        gnn_df is not None
        and not gnn_df.empty
        and "Model" in gnn_df.columns
    ):
        st.dataframe(
            gnn_df,
            use_container_width=True,
            hide_index=True,
        )

        numeric_columns = [
            column
            for column in gnn_df.columns
            if pd.api.types.is_numeric_dtype(gnn_df[column])
        ]

        metric_column = next(
            (
                column
                for column in [
                    "F1-score",
                    "Precision",
                    "Recall",
                    "PR-AUC",
                    "Accuracy",
                ]
                if column in numeric_columns
            ),
            None,
        )

        if metric_column:
            fig = px.bar(
                gnn_df,
                x="Model",
                y=metric_column,
                color="Model",
                text_auto=".3f",
                range_y=[0, 1],
                title=f"Model comparison â€” {metric_column}",
                color_discrete_map=colors,
            )
            st.plotly_chart(
                style_fig(fig, 400),
                use_container_width=True,
            )
    elif (
        baseline_df is not None
        and not baseline_df.empty
    ):
        st.markdown("### Baseline model results")
        st.dataframe(
            baseline_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Comparison results are not available. Run your model "
            "comparison scripts to generate the results CSV files."
        )

    st.caption(
        "Use the same test split and evaluation metrics when comparing "
        "models. Accuracy alone can be misleading for imbalanced data."
    )


# ==========================================================
# PAGE 5: TEST TRANSACTION
# ==========================================================
elif page == "Test Transaction":

    st.markdown("## Test a Transaction")
    st.caption(
        "Enter sample transaction features and evaluate them with "
        "your saved Random Forest model."
    )

    if model is None:
        st.error(
            "Model file not found. Expected: "
            "models/fraudeye_random_forest.joblib"
        )
    else:
        st.info(
            "The credit-card model expects the same input features "
            "used during training. V1â€“V28 are anonymized dataset "
            "features, not real UPI transaction fields."
        )

        feature_names = (
            list(getattr(model, "feature_names_in_", []))
        )

        if not feature_names:
            feature_names = [
                f"V{i}" for i in range(1, 29)
            ] + ["Amount"]

        with st.form("test_transaction_form"):
            st.markdown("### Transaction features")

            values = {}

            columns = st.columns(3)

            for index, feature in enumerate(feature_names):
                with columns[index % 3]:
                    if feature.lower() == "amount":
                        values[feature] = st.number_input(
                            feature,
                            min_value=0.0,
                            value=100.0,
                            step=10.0,
                            key=f"test_{feature}",
                        )
                    else:
                        values[feature] = st.number_input(
                            feature,
                            value=0.0,
                            step=0.1,
                            format="%.4f",
                            key=f"test_{feature}",
                        )

            submitted = st.form_submit_button(
                "Analyze Transaction",
                type="primary",
            )

        if submitted:
            try:
                input_df = pd.DataFrame(
                    [values],
                    columns=feature_names,
                )

                probability = float(
                    model.predict_proba(input_df)[0][1]
                )

                prediction = (
                    "Fraud"
                    if probability >= threshold
                    else "Legitimate"
                )

                st.markdown("---")
                a, b = st.columns(2)

                a.metric("Prediction", prediction)
                b.metric(
                    "Fraud probability",
                    f"{probability * 100:.2f}%",
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                if prediction == "Fraud":
                    st.error(
                        "FRAUD ALERT: this model classified the "
                        "sample as potentially fraudulent."
                    )
                else:
                    st.success(
                        "The model classified this sample as "
                        "legitimate at the configured threshold."
                    )

                st.caption(
                    f"Decision threshold: {threshold:.2f}. "
                    "This is a model prediction, not proof of fraud."
                )

            except Exception as exc:
                st.error("Could not analyze this transaction.")
                st.caption(str(exc))


# ==========================================================
# PAGE 6: UPI RISK ANALYZER
# ==========================================================
elif page == "UPI Risk Analyzer":

    st.markdown("## UPI Transaction Risk Analyzer")
    st.caption(
        "Review sample UPI transaction data using transparent "
        "rule-based risk indicators."
    )

    st.warning(
        "Demo only: this page uses configured rules, not a trained "
        "UPI fraud model or a connection to a bank or UPI provider."
    )

    tab_upload, tab_manual = st.tabs(
        ["Upload CSV / Excel", "Manual Analysis"]
    )

    with tab_upload:
        st.markdown("### Upload transaction data")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="upi_transaction_upload",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    upi_df = pd.read_csv(uploaded_file)
                else:
                    upi_df = pd.read_excel(uploaded_file)

                st.markdown("#### Uploaded data preview")
                st.dataframe(
                    upi_df.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

                if "amount" not in upi_df.columns:
                    st.error(
                        "Required column missing: amount. "
                        "Please use a column named 'amount'."
                    )
                else:
                    if st.button(
                        "Analyze Uploaded Transactions",
                        type="primary",
                        key="analyze_upi_upload",
                    ):
                        results = []

                        for _, row in upi_df.iterrows():
                            result = analyze_upi_transaction(row)
                            results.append(result)

                        result_df = upi_df.copy()
                        result_df["risk_score"] = [
                            item["risk_score"] for item in results
                        ]
                        result_df["risk_level"] = [
                            item["risk_level"] for item in results
                        ]
                        result_df["risk_reasons"] = [
                            item["risk_reasons"] for item in results
                        ]

                        st.markdown("---")
                        st.markdown("### Analysis results")

                        a, b, c = st.columns(3)
                        a.metric(
                            "Transactions analyzed",
                            len(result_df),
                        )
                        b.metric(
                            "High risk",
                            int(
                                (
                                    result_df["risk_level"] == "High"
                                ).sum()
                            ),
                        )
                        c.metric(
                            "Medium risk",
                            int(
                                (
                                    result_df["risk_level"] == "Medium"
                                ).sum()
                            ),
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        chart_df = (
                            result_df["risk_level"]
                            .value_counts()
                            .rename_axis("Risk Level")
                            .reset_index(name="Transactions")
                        )

                        fig = px.bar(
                            chart_df,
                            x="Risk Level",
                            y="Transactions",
                            color="Risk Level",
                            title="UPI risk indicators",
                            color_discrete_map={
                                "Low": "#19d6a0",
                                "Medium": "#f5b942",
                                "High": "#ff496c",
                                "Invalid input": "#9670ff",
                            },
                        )

                        st.plotly_chart(
                            style_fig(fig, 350),
                            use_container_width=True,
                        )

                        st.download_button(
                            "Download analyzed CSV",
                            result_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            "fraudeye_upi_risk_results.csv",
                            "text/csv",
                        )

                        st.caption(
                            "Rule-based risk indicators are not "
                            "confirmed fraud labels."
                        )

            except Exception as exc:
                st.error(
                    "Could not read or analyze the uploaded file."
                )
                st.caption(str(exc))

    with tab_manual:

        st.markdown("### Enter transaction details")

        st.caption(
            "Use sample values only. Do not enter real customer "
            "identifiers or banking credentials."
        )

        with st.form("upi_manual_analysis_form"):

            amount = st.number_input(
                "Transaction amount (â‚¹)",
                min_value=0.0,
                max_value=100000000.0,
                value=1000.0,
                step=100.0,
            )

            hour = st.slider(
                "Transaction hour",
                min_value=0,
                max_value=23,
                value=12,
                help="0 = midnight, 12 = noon, 23 = 11 PM",
            )

            new_recipient = st.checkbox(
                "Is this a new recipient?",
                value=False,
            )

            transactions_last_hour = st.number_input(
                "Transactions in the last hour",
                min_value=0,
                max_value=1000,
                value=1,
                step=1,
            )

            failed_attempts_last_hour = st.number_input(
                "Failed attempts in the last hour",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
            )

            submitted = st.form_submit_button(
                "Analyze UPI Risk",
                type="primary",
            )

        if submitted:

            sample_transaction = {
                "amount": amount,
                "hour": hour,
                "new_recipient": new_recipient,
                "transactions_last_hour": transactions_last_hour,
                "failed_attempts_last_hour":
                    failed_attempts_last_hour,
            }

            risk = analyze_upi_transaction(sample_transaction)

            st.markdown("---")
            section("Analysis result")

            a, b = st.columns(2)

            a.metric(
                "Risk score",
                f"{risk['risk_score']} points",
            )

            b.metric(
                "Risk indicator",
                risk["risk_level"],
            )

            if risk["risk_level"] == "High":
                st.error(
                    "High risk indicator: review this transaction "
                    "carefully in this demonstration."
                )

            elif risk["risk_level"] == "Medium":
                st.warning(
                    "Medium risk indicator: additional review "
                    "may be appropriate."
                )

            elif risk["risk_level"] == "Low":
                st.success(
                    "Low risk indicator under the configured rules."
                )

            else:
                st.error("Invalid transaction input.")

            st.markdown("### Why was this score assigned?")

            reasons = risk["risk_reasons"].split("; ")

            for reason in reasons:
                st.write(f"- {reason}")

            st.caption(
                "This score is based on manually configured rules. "
                "It is not an ML probability, a bank decision, "
                "or proof that a transaction is fraudulent."
            )

    # ------------------------------------------------------
    # RISK RULES
    # ------------------------------------------------------
    with st.expander("View risk scoring rules"):

        st.markdown("""
        | Indicator | Points |
        |---|---:|
        | Amount â‚¹50,000 or more | +2 |
        | Amount â‚¹10,000 to â‚¹49,999.99 | +1 |
        | Transaction between 12 AM and 4 AM | +1 |
        | New recipient | +2 |
        | 3â€“4 transactions in the last hour | +1 |
        | 5 or more transactions in the last hour | +2 |
        | 1 failed attempt in the last hour | +1 |
        | 2 or more failed attempts in the last hour | +3 |

        **Score interpretation**

        - 0â€“2 points: Low
        - 3â€“4 points: Medium
        - 5 or more points: High

        These thresholds are illustrative rules for a student prototype.
        They have not been validated against real UPI fraud data.
        """)


# ==========================================================
# PAGE 7: ABOUT PROJECT
# ==========================================================
elif page == "About Project":

    st.markdown("## About FraudEye AI")

    st.markdown("""
    FraudEye AI is a student prototype for transaction fraud
    detection and model evaluation.

    ### Main components

    - **Random Forest:** baseline machine-learning classifier.
    - **Model comparison:** view saved model evaluation metrics.
    - **Live monitoring:** inspect predictions logged by the local
      simulator or API.
    - **UPI Risk Analyzer:** demonstration of transparent,
      rule-based risk indicators for uploaded sample data.
    - **Admin Panel:** view registered application users when logged
      in with an admin account.

    ### Dataset

    The primary credit-card dataset uses anonymized features
    such as `V1` through `V28`, `Amount`, and `Class`. It is not
    a real UPI transaction dataset.

    ### Important limitation

    This prototype is not connected to banks or UPI payment networks.
    Its predictions must not be treated as confirmed fraud findings
    or used as the sole basis for financial decisions.
    """)

    st.markdown("---")
    st.markdown("### Project status")

    a, b, c = st.columns(3)
    a.metric("ML Model", "Random Forest")
    b.metric("Input Dataset", "Credit Card CSV")
    c.metric("UPI Analyzer", "Rule-based demo")


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption(
    "FraudEye AI | Student Project Prototype | "
    "For demonstration and educational use only"
)
