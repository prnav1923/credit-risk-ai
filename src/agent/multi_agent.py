import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
import boto3
import shap
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config ---
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CATEGORICAL_COLS = [
    "home_ownership", "verification_status",
    "purpose", "grade", "sub_grade",
    "initial_list_status", "application_type"
]

NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "emp_length", "mort_acc", "pub_rec_bankruptcies",
    "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
    "loan_to_income", "fico_avg", "high_utilization"
]


# --- Load artifacts ---
def load_artifacts():
    logger.info("Loading artifacts from S3...")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/xgboost_model.pkl")
    credit_model = pickle.loads(model_obj["Body"].read())

    enc_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/encoders.pkl")
    encoders = pickle.loads(enc_obj["Body"].read())

    fraud_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/fraud_detection_models.pkl")
    fraud_models = pickle.loads(fraud_obj["Body"].read())

    explainer = shap.TreeExplainer(credit_model)
    logger.info("Artifacts loaded ✅")
    return credit_model, encoders, explainer, fraud_models


# --- Load vectorstore ---
def load_vectorstore():
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )


# --- Preprocess ---
def preprocess_input(data: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_to_income'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['high_utilization'] = (df['revol_util'] > 75).astype(int)

    for col in CATEGORICAL_COLS:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0

    return df[NUMERIC_COLS + CATEGORICAL_COLS]


def get_decision(risk_score: float):
    if risk_score < 0.3:
        return "APPROVE", "LOW RISK"
    elif risk_score < 0.6:
        return "REVIEW", "MEDIUM RISK"
    else:
        return "DECLINE", "HIGH RISK"


# --- Initialize artifacts ---
logger.info("Initializing multi-agent artifacts...")
credit_model, encoders, explainer, fraud_models = load_artifacts()
vectorstore = load_vectorstore()
iso_forest = fraud_models["iso_forest"]
fraud_scaler = fraud_models["scaler"]
xgbod = fraud_models["xgbod"]
fraud_feature_cols = fraud_models["feature_cols"]
logger.info("Multi-agent ready ✅")


# --- LLM ---
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )


# ============================================================
# AGENT 1 — RISK AGENT
# Tools: PredictRisk, ExplainDecision, DetectFraud
# ============================================================

@tool
def predict_credit_risk(application_json: str) -> str:
    """Predict credit risk score for a loan application.
    Returns risk score, decision, and risk level."""
    try:
        data = json.loads(application_json)
        X = preprocess_input(data, encoders)
        risk_score = float(credit_model.predict_proba(X)[:, 1][0])
        decision, risk_level = get_decision(risk_score)
        return json.dumps({
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "risk_level": risk_level
        })
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def explain_risk_decision(application_json: str) -> str:
    """Explain credit risk decision using SHAP values.
    Returns top risk and protective factors."""
    try:
        data = json.loads(application_json)
        X = preprocess_input(data, encoders)
        shap_values = explainer.shap_values(X)
        feature_names = NUMERIC_COLS + CATEGORICAL_COLS
        shap_dict = dict(zip(feature_names, shap_values[0]))
        sorted_shap = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_risk = [
            {"feature": k, "impact": round(float(v), 4)}
            for k, v in sorted_shap if v > 0
        ][:5]
        top_protective = [
            {"feature": k, "impact": round(float(v), 4)}
            for k, v in sorted_shap if v < 0
        ][:5]
        return json.dumps({
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective
        })
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def detect_fraud(application_json: str) -> str:
    """Detect fraud signals in a loan application.
    Returns fraud score, flag, and specific indicators."""
    try:
        data = json.loads(application_json)
        df = pd.DataFrame([data])

        if "loan_to_income" not in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
            df["loan_to_income"] = df["loan_to_income"].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
        if "fico_avg" not in df.columns:
            df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        if "high_utilization" not in df.columns:
            df["high_utilization"] = (df["revol_util"] > 75).astype(int)

        available_cols = [c for c in fraud_feature_cols if c in df.columns]
        X = df[available_cols].fillna(0).values

        # Isolation Forest score
        X_scaled = fraud_scaler.transform(X)
        iso_scores = iso_forest.score_samples(X_scaled)
        iso_scores_norm = 1 - (iso_scores - iso_scores.min()) / (
            iso_scores.max() - iso_scores.min() + 1e-10
        )

        # XGBOD score
        xgbod_proba = xgbod.predict_proba(X)
        xgbod_scores = xgbod_proba[:, 1] if xgbod_proba.ndim == 2 else xgbod_proba

        fraud_score = float(0.4 * iso_scores_norm[0] + 0.6 * xgbod_scores[0])
        fraud_flag = fraud_score > 0.5

        indicators = []
        if df["revol_util"].values[0] > 80:
            indicators.append("High revolving utilization (>80%)")
        if df["delinq_2yrs"].values[0] > 2:
            indicators.append("Multiple delinquencies in last 2 years")
        if df["pub_rec"].values[0] > 0:
            indicators.append("Public records present")
        if df["pub_rec_bankruptcies"].values[0] > 0:
            indicators.append("Bankruptcy history")
        if df["loan_to_income"].values[0] > 0.4:
            indicators.append("High loan-to-income ratio (>0.4)")
        if df["dti"].values[0] > 40:
            indicators.append("High DTI ratio (>40%)")

        return json.dumps({
            "fraud_score": round(fraud_score, 4),
            "fraud_flag": bool(fraud_flag),
            "fraud_risk": "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.4 else "LOW",
            "fraud_indicators": indicators
        })
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# AGENT 2 — COMPLIANCE AGENT
# Tools: RetrievePolicy, CheckRegulations
# ============================================================

@tool
def retrieve_credit_policy(query: str) -> str:
    """Retrieve relevant credit risk policy information.
    Use for questions about approval criteria, risk grades, decline rules."""
    try:
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def check_regulatory_compliance(application_json: str, decision: str) -> str:
    """Check if a credit decision complies with ECOA and FCRA regulations.
    Returns compliance status and required actions."""
    try:
        data = json.loads(application_json)

        compliance_issues = []
        required_actions = []

        # ECOA checks
        if decision == "DECLINE":
            required_actions.append(
                "Adverse action notice required within 30 days (ECOA)"
            )
            required_actions.append(
                "Must provide specific reasons for decline (FCRA)"
            )

        # FICO floor check
        if data.get("fico_range_low", 0) < 580:
            compliance_issues.append(
                "FICO below minimum threshold (580) — automatic decline required"
            )

        # DTI check
        if data.get("dti", 0) > 50:
            compliance_issues.append(
                "DTI exceeds 50% — automatic decline per policy Section 3"
            )

        # Bankruptcy check
        if data.get("pub_rec_bankruptcies", 0) > 0:
            compliance_issues.append(
                "Bankruptcy history present — review required per policy Section 3"
            )

        return json.dumps({
            "compliant": len(compliance_issues) == 0,
            "compliance_issues": compliance_issues,
            "required_actions": required_actions,
            "regulations": ["ECOA", "FCRA"],
            "audit_required": True
        })
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# MULTI-AGENT STATE
# ============================================================

class MultiAgentState(TypedDict):
    application: dict
    application_json: str
    risk_assessment: str
    fraud_assessment: str
    compliance_check: str
    final_decision: str
    messages: List


# ============================================================
# AGENT NODES
# ============================================================

def risk_agent_node(state: MultiAgentState) -> dict:
    """Risk Agent — assesses credit risk and fraud"""
    logger.info("Risk Agent running...")

    llm = get_llm()
    risk_tools = [predict_credit_risk, explain_risk_decision, detect_fraud]
    agent = create_react_agent(
        model=llm,
        tools=risk_tools,
        prompt=(
            "You are a credit risk analyst. "
            "For the given loan application:\n"
            "1. Predict the credit risk score\n"
            "2. Explain what factors drove the decision\n"
            "3. Check for fraud signals\n"
            "Be concise and factual. Return your findings as a structured summary."
        )
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content=f"Analyze this loan application for credit risk and fraud:\n{state['application_json']}"
        )]
    })

    risk_assessment = result["messages"][-1].content
    logger.info("Risk Agent complete ✅")

    return {
        "risk_assessment": risk_assessment,
        "messages": state["messages"] + [
            AIMessage(content=f"Risk Agent: {risk_assessment}")
        ]
    }


def compliance_agent_node(state: MultiAgentState) -> dict:
    """Compliance Agent — checks policy and regulatory compliance"""
    logger.info("Compliance Agent running...")

    llm = get_llm()
    compliance_tools = [retrieve_credit_policy, check_regulatory_compliance]
    agent = create_react_agent(
        model=llm,
        tools=compliance_tools,
        prompt=(
            "You are a credit compliance officer. "
            "For the given loan application and risk assessment:\n"
            "1. Retrieve relevant policy sections\n"
            "2. Check regulatory compliance (ECOA, FCRA)\n"
            "3. Identify any compliance issues\n"
            "4. List required actions\n"
            "Be thorough and cite specific policy sections."
        )
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            content=(
                f"Check compliance for this application:\n{state['application_json']}\n\n"
                f"Risk Assessment:\n{state['risk_assessment']}"
            )
        )]
    })

    compliance_check = result["messages"][-1].content
    logger.info("Compliance Agent complete ✅")

    return {
        "compliance_check": compliance_check,
        "messages": state["messages"] + [
            AIMessage(content=f"Compliance Agent: {compliance_check}")
        ]
    }


def decision_agent_node(state: MultiAgentState) -> dict:
    """Decision Agent — synthesizes final recommendation"""
    logger.info("Decision Agent running...")

    llm = get_llm()

    prompt = f"""You are a senior credit committee member making the final lending decision.

You have received assessments from two specialized agents:

RISK ASSESSMENT:
{state['risk_assessment']}

COMPLIANCE CHECK:
{state['compliance_check']}

APPLICATION:
{state['application_json']}

Based on ALL the above information, provide a final credit decision report with:

1. FINAL DECISION: (APPROVE / REVIEW / DECLINE)
2. RISK SUMMARY: Key risk factors
3. FRAUD SUMMARY: Fraud risk level and indicators
4. COMPLIANCE SUMMARY: Any compliance issues
5. RECOMMENDATION: Specific conditions or next steps
6. REASONING: Why this decision was made

Be decisive, clear, and justify every point."""

    response = llm.invoke([HumanMessage(content=prompt)])
    final_decision = response.content
    logger.info("Decision Agent complete ✅")

    return {
        "final_decision": final_decision,
        "messages": state["messages"] + [
            AIMessage(content=f"Decision Agent: {final_decision}")
        ]
    }


# ============================================================
# BUILD MULTI-AGENT GRAPH
# ============================================================

def build_multi_agent():
    graph = StateGraph(MultiAgentState)

    # Add agent nodes
    graph.add_node("risk_agent", risk_agent_node)
    graph.add_node("compliance_agent", compliance_agent_node)
    graph.add_node("decision_agent", decision_agent_node)

    # Define flow
    graph.set_entry_point("risk_agent")
    graph.add_edge("risk_agent", "compliance_agent")
    graph.add_edge("compliance_agent", "decision_agent")
    graph.add_edge("decision_agent", END)

    return graph.compile()


# ============================================================
# RUN MULTI-AGENT
# ============================================================

def run_multi_agent(application: dict) -> str:
    logger.info("=== Starting Multi-Agent Analysis ===")

    multi_agent = build_multi_agent()
    app_json = json.dumps(application)

    result = multi_agent.invoke({
        "application": application,
        "application_json": app_json,
        "risk_assessment": "",
        "fraud_assessment": "",
        "compliance_check": "",
        "final_decision": "",
        "messages": [HumanMessage(
            content="Analyze this loan application"
        )]
    })

    logger.info("=== Multi-Agent Analysis Complete ✅ ===")
    return result["final_decision"]


if __name__ == "__main__":
    test_application = {
        "loan_amnt": 10000,
        "int_rate": 12.5,
        "installment": 350.0,
        "annual_inc": 60000,
        "dti": 18.5,
        "delinq_2yrs": 0,
        "fico_range_low": 680,
        "fico_range_high": 684,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 15000,
        "revol_util": 45.0,
        "total_acc": 25,
        "emp_length": 5,
        "mort_acc": 2,
        "pub_rec_bankruptcies": 0,
        "num_actv_bc_tl": 4,
        "bc_util": 50.0,
        "percent_bc_gt_75": 25.0,
        "avg_cur_bal": 8000,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "grade": "B",
        "sub_grade": "B3",
        "initial_list_status": "w",
        "application_type": "Individual"
    }

    report = run_multi_agent(test_application)
    print(report)