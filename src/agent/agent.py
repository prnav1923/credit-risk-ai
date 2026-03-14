import os
import json
import logging
import pickle
import io
import numpy as np
import pandas as pd
import boto3
import shap
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline

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


# --- Load model + encoders from S3 ---
def load_artifacts():
    logger.info("Loading model artifacts from S3...")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/xgboost_model.pkl")
    model = pickle.loads(model_obj["Body"].read())

    enc_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/encoders.pkl")
    encoders = pickle.loads(enc_obj["Body"].read())

    explainer = shap.TreeExplainer(model)
    logger.info("Artifacts loaded ✅")
    return model, encoders, explainer


# --- Load vectorstore ---
def load_vectorstore():
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    return vectorstore


# --- Preprocess input ---
def preprocess_input(data: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_to_income'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['high_utilization'] = (df['revol_util'] > 75).astype(int)

    for col in CATEGORICAL_COLS:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return df[feature_cols]


def get_decision(risk_score: float):
    if risk_score < 0.3:
        return "APPROVE", "LOW RISK"
    elif risk_score < 0.6:
        return "REVIEW", "MEDIUM RISK"
    else:
        return "DECLINE", "HIGH RISK"


# --- Initialize global artifacts ---
logger.info("Initializing agent artifacts...")
model, encoders, explainer = load_artifacts()
vectorstore = load_vectorstore()
logger.info("Agent ready ✅")


# --- Tool 1: PredictRisk ---
@tool
def predict_risk(application_json: str) -> str:
    """Predict credit risk score for a loan application.
    Input: JSON string of loan application features.
    Output: Risk score, decision, and risk level."""
    try:
        data = json.loads(application_json)
        X = preprocess_input(data, encoders)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        decision, risk_level = get_decision(risk_score)
        return json.dumps({
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "risk_level": risk_level
        })
    except Exception as e:
        return f"Error predicting risk: {str(e)}"


# --- Tool 2: RetrievePolicy ---
@tool
def retrieve_policy(query: str) -> str:
    """Retrieve relevant credit risk policy information.
    Input: A question about credit risk policy.
    Output: Relevant policy text."""
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)
        return context
    except Exception as e:
        return f"Error retrieving policy: {str(e)}"


# --- Tool 3: ExplainDecision ---
@tool
def explain_decision(application_json: str) -> str:
    """Explain the credit risk decision using SHAP values.
    Input: JSON string of loan application features.
    Output: Top risk and protective factors."""
    try:
        data = json.loads(application_json)
        X = preprocess_input(data, encoders)
        shap_values = explainer.shap_values(X)
        feature_names = NUMERIC_COLS + CATEGORICAL_COLS
        shap_dict = dict(zip(feature_names, shap_values[0]))
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

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
        return f"Error explaining decision: {str(e)}"


# --- Tool 4: RetrieveSimilarCases ---
@tool
def retrieve_similar_cases(application_json: str) -> str:
    """Retrieve similar historical loan cases from the knowledge base.
    Input: JSON string of loan application features.
    Output: Similar cases with their outcomes."""
    try:
        data = json.loads(application_json)
        query = (
            f"loan amount {data.get('loan_amnt')} "
            f"interest rate {data.get('int_rate')} "
            f"DTI {data.get('dti')} "
            f"FICO {data.get('fico_range_low')} "
            f"purpose {data.get('purpose')} "
            f"grade {data.get('grade')}"
        )
        docs = vectorstore.similarity_search(query, k=2)
        cases = "\n\n".join(doc.page_content for doc in docs)
        return f"Similar policy context for this profile:\n{cases}"
    except Exception as e:
        return f"Error retrieving similar cases: {str(e)}"


# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    application: dict
    final_decision: str


# --- Tools list ---
tools = [predict_risk, retrieve_policy, explain_decision, retrieve_similar_cases]


# --- Agent node using TinyLlama ---
def agent_node(state: AgentState):
    application = state["application"]
    app_json = json.dumps(application)

    # Run all 4 tools sequentially for a complete analysis
    logger.info("Running PredictRisk...")
    risk_result = json.loads(predict_risk.invoke(app_json))

    logger.info("Running RetrievePolicy...")
    policy_result = retrieve_policy.invoke(
        f"loan approval criteria for {risk_result['risk_level']} borrower"
    )

    logger.info("Running ExplainDecision...")
    explain_result = json.loads(explain_decision.invoke(app_json))

    logger.info("Running RetrieveSimilarCases...")
    similar_result = retrieve_similar_cases.invoke(app_json)

    # Synthesize final decision
    risk_score = risk_result['risk_score']
    decision = risk_result['decision']
    risk_level = risk_result['risk_level']

    top_risks = ", ".join(
        [f"{r['feature']} (impact: {r['impact']})"
         for r in explain_result['top_risk_factors'][:3]]
    )
    top_protective = ", ".join(
        [f"{p['feature']} (impact: {p['impact']})"
         for p in explain_result['top_protective_factors'][:3]]
    )

    final_decision = f"""
=== CREDIT RISK ASSESSMENT REPORT ===

APPLICATION SUMMARY:
- Loan Amount: ${application.get('loan_amnt'):,}
- Purpose: {application.get('purpose')}
- Grade: {application.get('grade')} / {application.get('sub_grade')}

RISK ASSESSMENT:
- Risk Score: {risk_score} (0=Safe, 1=Default)
- Decision: {decision}
- Risk Level: {risk_level}

KEY RISK FACTORS:
- {top_risks}

PROTECTIVE FACTORS:
- {top_protective}

POLICY CONTEXT:
{policy_result[:300]}...

FINAL RECOMMENDATION:
{decision} — This application scored {risk_score:.4f}. 
{'Recommend approval with standard terms.' if decision == 'APPROVE' 
 else 'Recommend manual underwriter review.' if decision == 'REVIEW'
 else 'Recommend decline. Risk exceeds acceptable threshold.'}
=====================================
"""

    return {
        "messages": [AIMessage(content=final_decision)],
        "final_decision": final_decision
    }


# --- Build graph ---
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    return graph.compile()


# --- Run agent ---
def run_agent(application: dict) -> str:
    agent = build_agent()
    result = agent.invoke({
        "messages": [HumanMessage(content="Analyze this loan application")],
        "application": application,
        "final_decision": ""
    })
    return result["final_decision"]


if __name__ == "__main__":
    # Test application
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

    print(run_agent(test_application))