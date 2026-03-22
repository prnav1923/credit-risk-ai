import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# --- Config ---
API_BASE_URL = "http://localhost:8000"
API_KEY = "credit-risk-secret-key-2024"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

# --- Page Config ---
st.set_page_config(
    page_title="Credit Risk AI System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .approve { border-left-color: #28a745 !important; }
    .review  { border-left-color: #ffc107 !important; }
    .decline { border-left-color: #dc3545 !important; }
    .fraud   { border-left-color: #6f42c1 !important; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def call_api(endpoint: str, payload: dict = None, method: str = "POST"):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=120)
        else:
            response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        return response.json(), response.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "API not running. Start uvicorn first."}, 503
    except Exception as e:
        return {"error": str(e)}, 500


def get_decision_color(decision: str) -> str:
    colors = {"APPROVE": "🟢", "REVIEW": "🟡", "DECLINE": "🔴"}
    return colors.get(decision, "⚪")


def create_risk_gauge(score: float, title: str = "Risk Score"):
    color = "#28a745" if score < 0.3 else "#ffc107" if score < 0.6 else "#dc3545"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 16}},
        number={"font": {"size": 24}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.3], "color": "#d4edda"},
                {"range": [0.3, 0.6], "color": "#fff3cd"},
                {"range": [0.6, 1], "color": "#f8d7da"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_shap_chart(risk_factors: list, protective_factors: list):
    features = []
    impacts = []
    colors = []

    for f in protective_factors[:5]:
        features.append(f["feature"])
        impacts.append(f["impact"])
        colors.append("#28a745")

    for f in risk_factors[:5]:
        features.append(f["feature"])
        impacts.append(f["impact"])
        colors.append("#dc3545")

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation="h",
        marker_color=colors
    ))
    fig.update_layout(
        title="SHAP Feature Impact",
        xaxis_title="Impact on Default Probability",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# --- Sidebar ---
st.sidebar.markdown("## 🏦 Credit Risk AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🎯 Single Assessment", "🤖 Multi-Agent Analysis", "📊 Model Info", "📋 Audit Log", "📈 Predictions History"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")

# Health check
health, status = call_api("/v1/health", method="GET")
if status == 200 and health.get("status") == "healthy":
    st.sidebar.success("✅ API Online")
    st.sidebar.info(f"Model: {'✅ Loaded' if health.get('model_loaded') else '❌ Not Loaded'}")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.warning("Run: uvicorn src.api.main:app --port 8000")

st.sidebar.markdown("---")
st.sidebar.markdown("**Stack**")
st.sidebar.markdown("XGBoost • Isolation Forest • XGBOD")
st.sidebar.markdown("LangChain • Pinecone • Groq")
st.sidebar.markdown("FastAPI • AWS ECS • Docker")


# --- Loan Application Form ---
def loan_application_form(key_prefix: str = ""):
    with st.expander("📝 Loan Application Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Financial Details**")
            loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 10000, key=f"{key_prefix}loan_amnt")
            int_rate = st.slider("Interest Rate (%)", 5.0, 31.0, 12.5, key=f"{key_prefix}int_rate")
            installment = st.number_input("Monthly Installment ($)", 50.0, 2000.0, 350.0, key=f"{key_prefix}installment")
            annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, key=f"{key_prefix}annual_inc")
            dti = st.slider("DTI Ratio (%)", 0.0, 50.0, 18.5, key=f"{key_prefix}dti")

        with col2:
            st.markdown("**Credit History**")
            fico_low = st.slider("FICO Score (Low)", 300, 850, 680, key=f"{key_prefix}fico_low")
            fico_high = st.slider("FICO Score (High)", 300, 850, 684, key=f"{key_prefix}fico_high")
            delinq_2yrs = st.number_input("Delinquencies (2yr)", 0, 20, 0, key=f"{key_prefix}delinq")
            pub_rec = st.number_input("Public Records", 0, 10, 0, key=f"{key_prefix}pub_rec")
            pub_rec_bankruptcies = st.number_input("Bankruptcies", 0, 5, 0, key=f"{key_prefix}bankruptcies")

        with col3:
            st.markdown("**Loan Details**")
            grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1, key=f"{key_prefix}grade")
            sub_grade = st.selectbox("Sub Grade", ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","D4","D5","E1","E2","E3","E4","E5","F1","F2","F3","F4","F5","G1","G2","G3","G4","G5"], index=7, key=f"{key_prefix}sub_grade")
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"], key=f"{key_prefix}home")
            purpose = st.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "car", "other"], key=f"{key_prefix}purpose")
            verification_status = st.selectbox("Verification", ["Verified", "Source Verified", "Not Verified"], key=f"{key_prefix}verification")

        col4, col5 = st.columns(2)
        with col4:
            open_acc = st.number_input("Open Accounts", 0, 50, 10, key=f"{key_prefix}open_acc")
            revol_bal = st.number_input("Revolving Balance ($)", 0, 100000, 15000, key=f"{key_prefix}revol_bal")
            revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, key=f"{key_prefix}revol_util")
        with col5:
            total_acc = st.number_input("Total Accounts", 0, 100, 25, key=f"{key_prefix}total_acc")
            emp_length = st.slider("Employment Length (yrs)", 0, 10, 5, key=f"{key_prefix}emp_length")
            mort_acc = st.number_input("Mortgage Accounts", 0, 20, 2, key=f"{key_prefix}mort_acc")

        application = {
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "installment": installment,
            "annual_inc": annual_inc,
            "dti": dti,
            "delinq_2yrs": float(delinq_2yrs),
            "fico_range_low": float(fico_low),
            "fico_range_high": float(fico_high),
            "open_acc": float(open_acc),
            "pub_rec": float(pub_rec),
            "revol_bal": float(revol_bal),
            "revol_util": revol_util,
            "total_acc": float(total_acc),
            "emp_length": float(emp_length),
            "mort_acc": float(mort_acc),
            "pub_rec_bankruptcies": float(pub_rec_bankruptcies),
            "num_actv_bc_tl": 4.0,
            "bc_util": 50.0,
            "percent_bc_gt_75": 25.0,
            "avg_cur_bal": 8000.0,
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "purpose": purpose,
            "grade": grade,
            "sub_grade": sub_grade,
            "initial_list_status": "w",
            "application_type": "Individual"
        }
    return application


# ============================================================
# PAGE 1 — SINGLE ASSESSMENT
# ============================================================
if page == "🎯 Single Assessment":
    st.markdown('<div class="main-header">🏦 Credit Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered credit risk scoring with fraud detection and explainability</div>', unsafe_allow_html=True)

    application = loan_application_form("single_")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        assess_btn = st.button("🎯 Full Assessment (Credit + Fraud)", use_container_width=True, type="primary")
    with col2:
        explain_btn = st.button("🔍 Explain Decision (SHAP)", use_container_width=True)
    with col3:
        agent_btn = st.button("🤖 Agent Analysis", use_container_width=True)

    if assess_btn:
        with st.spinner("Running combined assessment..."):
            result, status = call_api("/v1/assess", application)

        if status == 200:
            st.markdown("---")
            st.markdown("### 📊 Assessment Results")

            col1, col2, col3, col4 = st.columns(4)
            decision = result["combined_decision"]
            color_class = decision.lower() if decision in ["APPROVE", "REVIEW", "DECLINE"] else "review"

            with col1:
                st.metric("Combined Decision", f"{get_decision_color(decision)} {decision}")
            with col2:
                st.metric("Credit Risk Score", f"{result['credit_risk_score']:.4f}")
            with col3:
                st.metric("Fraud Score", f"{result['fraud_score']:.4f}")
            with col4:
                st.metric("Fraud Flag", "⚠️ YES" if result['fraud_flag'] else "✅ NO")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    create_risk_gauge(result["credit_risk_score"], "Credit Risk Score"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    create_risk_gauge(result["fraud_score"], "Fraud Risk Score"),
                    use_container_width=True
                )

            if result["fraud_indicators"]:
                st.warning("⚠️ Fraud Indicators Detected")
                for indicator in result["fraud_indicators"]:
                    st.markdown(f"- {indicator}")

            st.info(f"**{result['combined_risk']}**: {result['message']}")
        else:
            st.error(f"Error: {result.get('error', result)}")

    if explain_btn:
        with st.spinner("Running SHAP analysis..."):
            result, status = call_api("/v1/explain", application)

        if status == 200:
            st.markdown("---")
            st.markdown("### 🔍 SHAP Explanation")
            st.plotly_chart(
                create_shap_chart(
                    result["top_risk_factors"],
                    result["top_protective_factors"]
                ),
                use_container_width=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔴 Top Risk Factors**")
                for f in result["top_risk_factors"]:
                    st.markdown(f"- `{f['feature']}`: +{f['impact']:.4f}")
            with col2:
                st.markdown("**🟢 Top Protective Factors**")
                for f in result["top_protective_factors"]:
                    st.markdown(f"- `{f['feature']}`: {f['impact']:.4f}")
        else:
            st.error(f"Error: {result.get('error', result)}")

    if agent_btn:
        with st.spinner("Running ReAct agent analysis (this takes ~30 seconds)..."):
            result, status = call_api("/v1/agent", {"application": application})

        if status == 200:
            st.markdown("---")
            st.markdown("### 🤖 Agent Analysis")
            st.markdown(result["report"])
        else:
            st.error(f"Error: {result.get('error', result)}")


# ============================================================
# PAGE 2 — MULTI-AGENT ANALYSIS
# ============================================================
elif page == "🤖 Multi-Agent Analysis":
    st.markdown('<div class="main-header">🤖 Multi-Agent Credit Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">3 specialized agents: Risk → Compliance → Decision</div>', unsafe_allow_html=True)

    st.markdown("""
```
    Risk Agent          →    Compliance Agent    →    Decision Agent
    ─────────────────        ────────────────────      ──────────────────
    PredictRisk              RetrievePolicy            Synthesize all inputs
    ExplainDecision          CheckRegulations          Final recommendation
    DetectFraud              ECOA/FCRA check           Conditions & reasoning
```
    """)

    application = loan_application_form("multi_")

    if st.button("🚀 Run Multi-Agent Analysis", use_container_width=True, type="primary"):
        with st.spinner("Running 3-agent workflow (Risk → Compliance → Decision)... ~60 seconds"):
            result, status = call_api("/v1/multi-agent", {"application": application})

        if status == 200:
            st.markdown("---")
            st.markdown("### 📋 Multi-Agent Decision Report")
            st.markdown(result["report"])
        else:
            st.error(f"Error: {result.get('error', result)}")


# ============================================================
# PAGE 3 — MODEL INFO
# ============================================================
elif page == "📊 Model Info":
    st.markdown('<div class="main-header">📊 Model Information</div>', unsafe_allow_html=True)

    result, status = call_api("/v1/model-info", method="GET")

    if status == 200:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Credit Risk Model")
            st.metric("Model Type", result["model_type"])
            st.metric("AUC-ROC", result["auc_roc"])
            st.metric("PR-AUC", result["pr_auc"])
            st.metric("Features", result["features"])

        with col2:
            st.markdown("### Training Data")
            st.metric("Training Samples", f"{result['training_samples']:,}")
            st.metric("Test Samples", f"{result['test_samples']:,}")
            st.metric("Default Rate", result["default_rate"])
            st.metric("Dataset", result["dataset"])

        st.markdown("---")
        st.markdown("### Fraud Detection Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", "Isolation Forest + XGBOD")
        with col2:
            st.metric("AUC-ROC", "0.8841")
        with col3:
            st.metric("Fraud Rate", "7.31%")

        st.markdown("---")
        st.markdown("### RAG Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic Similarity", "0.6168")
        with col2:
            st.metric("Context Recall", "0.6168")

        # Drift report
        st.markdown("---")
        st.markdown("### Model Drift")
        drift_result, drift_status = call_api("/v1/monitor", method="GET")
        if drift_status == 200:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", drift_result["status"])
            with col2:
                st.metric("Current AUC", f"{drift_result['current_auc']:.4f}")
            with col3:
                st.metric("Drift", f"{drift_result['drift_pct']}%")
            st.success(f"✅ {drift_result['action']}")
    else:
        st.error("Could not fetch model info. Is the API running?")


# ============================================================
# PAGE 4 — AUDIT LOG
# ============================================================
elif page == "📋 Audit Log":
    st.markdown('<div class="main-header">📋 HITL Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Human override decisions logged for ECOA/FCRA compliance</div>', unsafe_allow_html=True)

    result, status = call_api("/v1/audit", method="GET")

    if status == 200:
        st.metric("Total Overrides", result["total_overrides"])

        if result["total_overrides"] == 0:
            st.info("No overrides recorded yet.")
        else:
            for i, override in enumerate(result["overrides"]):
                with st.expander(
                    f"Override #{i+1} — {override['override_decision']} "
                    f"by {override['reviewer_id']} at {override['timestamp'][:19]}"
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Decision", override["original_decision"])
                    with col2:
                        st.metric("Override Decision", override["override_decision"])
                    with col3:
                        st.metric("Risk Score", override["risk_score"])
                    st.markdown(f"**Reason:** {override['reason']}")
                    st.markdown(f"**Reviewer:** {override['reviewer_id']}")
                    st.markdown(f"**Timestamp:** {override['timestamp']}")
    else:
        st.error("Could not fetch audit log.")

    st.markdown("---")
    st.markdown("### Submit Manual Override")
    with st.form("override_form"):
        override_application = {
            "loan_amnt": 10000, "int_rate": 12.5, "installment": 350.0,
            "annual_inc": 60000, "dti": 18.5, "delinq_2yrs": 0.0,
            "fico_range_low": 680.0, "fico_range_high": 684.0, "open_acc": 10.0,
            "pub_rec": 0.0, "revol_bal": 15000.0, "revol_util": 45.0,
            "total_acc": 25.0, "emp_length": 5.0, "mort_acc": 2.0,
            "pub_rec_bankruptcies": 0.0, "num_actv_bc_tl": 4.0,
            "bc_util": 50.0, "percent_bc_gt_75": 25.0, "avg_cur_bal": 8000.0,
            "home_ownership": "RENT", "verification_status": "Verified",
            "purpose": "debt_consolidation", "grade": "B", "sub_grade": "B3",
            "initial_list_status": "w", "application_type": "Individual"
        }
        override_decision = st.selectbox("Override Decision", ["APPROVE", "REVIEW", "DECLINE"])
        reviewer_id = st.text_input("Reviewer ID", "underwriter_001")
        reason = st.text_area("Reason for Override")
        submitted = st.form_submit_button("Submit Override")

        if submitted and reason:
            payload = {
                "application": override_application,
                "override_decision": override_decision,
                "reviewer_id": reviewer_id,
                "reason": reason
            }
            result, status = call_api("/v1/override", payload)
            if status == 200:
                st.success(f"✅ Override recorded — {result['override_decision']} by {result['reviewer_id']}")
            else:
                st.error(f"Error: {result}")

# ============================================================
# PAGE 5 — PREDICTIONS HISTORY
# ============================================================
elif page == "📈 Predictions History":
    st.markdown('<div class="main-header">📈 Predictions History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All scored applications from API + Kafka streaming</div>', unsafe_allow_html=True)

    result, status = call_api("/v1/predictions", method="GET")

    if status == 200:
        st.metric("Total Predictions", result["total"])

        if result["total"] == 0:
            st.info("No predictions recorded yet.")
        else:
            # Convert to DataFrame
            df = pd.DataFrame(result["predictions"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                approve_count = len(df[df["combined_decision"] == "APPROVE"])
                st.metric("Approved", approve_count)
            with col2:
                review_count = len(df[df["combined_decision"] == "REVIEW"])
                st.metric("Review", review_count)
            with col3:
                decline_count = len(df[df["combined_decision"] == "DECLINE"])
                st.metric("Declined", decline_count)
            with col4:
                fraud_count = len(df[df["fraud_flag"] == True])
                st.metric("Fraud Detected", fraud_count)

            st.markdown("---")

            # Decision distribution chart
            col1, col2 = st.columns(2)
            with col1:
                decision_counts = df["combined_decision"].value_counts()
                fig = px.pie(
                    values=decision_counts.values,
                    names=decision_counts.index,
                    title="Decision Distribution",
                    color_discrete_map={
                        "APPROVE": "#28a745",
                        "REVIEW": "#ffc107",
                        "DECLINE": "#dc3545"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    df,
                    x="credit_risk_score",
                    title="Credit Risk Score Distribution",
                    color_discrete_sequence=["#1f77b4"],
                    nbins=20
                )
                fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Approve threshold")
                fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="Decline threshold")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### Recent Predictions")

            # Style the dataframe
            display_df = df[[
                "id", "timestamp", "grade",
                "credit_risk_score", "credit_decision",
                "fraud_score", "fraud_flag",
                "combined_decision"
            ]].copy()

            display_df["fraud_flag"] = display_df["fraud_flag"].map(
                {True: "⚠️ YES", False: "✅ NO"}
            )
            display_df["combined_decision"] = display_df["combined_decision"].map(
                {"APPROVE": "🟢 APPROVE", "REVIEW": "🟡 REVIEW", "DECLINE": "🔴 DECLINE"}
            )

            st.dataframe(display_df, use_container_width=True)

    else:
        st.error("Could not fetch predictions. Is the API running?")