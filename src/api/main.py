import os
import io
import logging
import pickle
import numpy as np
import pandas as pd
import boto3
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from src.agent.agent import run_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Global model state ---
model = None
encoders = None
explainer = None

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

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


# --- Load model from S3 ---
def load_model_from_s3():
    global model, encoders, explainer
    logger.info("Loading model from S3...")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Load XGBoost model
    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/xgboost_model.pkl")
    model = pickle.loads(model_obj["Body"].read())

    # Load encoders
    enc_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/encoders.pkl")
    encoders = pickle.loads(enc_obj["Body"].read())

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)

    logger.info("Model loaded successfully ✅")


# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_s3()
    yield
    logger.info("Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk scoring using XGBoost + LangChain RAG",
    version="1.0.0",
    lifespan=lifespan
)


# --- Pydantic Schemas ---
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=10000.0, description="Loan amount in USD")
    int_rate: float = Field(..., example=12.5, description="Interest rate %")
    installment: float = Field(..., example=350.0, description="Monthly installment")
    annual_inc: float = Field(..., example=60000.0, description="Annual income")
    dti: float = Field(..., example=18.5, description="Debt to income ratio")
    delinq_2yrs: float = Field(..., example=0.0, description="Delinquencies in last 2 years")
    fico_range_low: float = Field(..., example=680.0, description="FICO score low")
    fico_range_high: float = Field(..., example=684.0, description="FICO score high")
    open_acc: float = Field(..., example=10.0, description="Number of open accounts")
    pub_rec: float = Field(..., example=0.0, description="Public records")
    revol_bal: float = Field(..., example=15000.0, description="Revolving balance")
    revol_util: float = Field(..., example=45.0, description="Revolving utilization %")
    total_acc: float = Field(..., example=25.0, description="Total accounts")
    emp_length: float = Field(..., example=5.0, description="Employment length in years")
    mort_acc: float = Field(..., example=2.0, description="Mortgage accounts")
    pub_rec_bankruptcies: float = Field(..., example=0.0, description="Bankruptcies")
    num_actv_bc_tl: float = Field(..., example=4.0, description="Active bankcard tradelines")
    bc_util: float = Field(..., example=50.0, description="Bankcard utilization %")
    percent_bc_gt_75: float = Field(..., example=25.0, description="% bankcards > 75% utilized")
    avg_cur_bal: float = Field(..., example=8000.0, description="Average current balance")
    home_ownership: str = Field(..., example="RENT", description="RENT/OWN/MORTGAGE")
    verification_status: str = Field(..., example="Verified", description="Income verification")
    purpose: str = Field(..., example="debt_consolidation", description="Loan purpose")
    grade: str = Field(..., example="B", description="Loan grade A-G")
    sub_grade: str = Field(..., example="B3", description="Loan sub grade")
    initial_list_status: str = Field(..., example="w", description="w or f")
    application_type: str = Field(..., example="Individual", description="Individual/Joint App")


class RiskResponse(BaseModel):
    risk_score: float
    decision: str
    risk_level: str
    confidence: str
    message: str


class ExplainResponse(BaseModel):
    risk_score: float
    decision: str
    top_risk_factors: list
    top_protective_factors: list

class AgentRequest(BaseModel):
    application: LoanApplication

class AgentResponse(BaseModel):
    report: str


# --- Preprocess input ---
def preprocess_input(data: LoanApplication) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    # Engineered features
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    df['high_utilization'] = (df['revol_util'] > 75).astype(int)

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0  # unseen category

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return df[feature_cols]


# --- Decision logic ---
def get_decision(risk_score: float):
    if risk_score < 0.3:
        return "APPROVE", "LOW RISK"
    elif risk_score < 0.6:
        return "REVIEW", "MEDIUM RISK"
    else:
        return "DECLINE", "HIGH RISK"


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": "XGBoost Classifier",
        "auc_roc": 0.7198,
        "pr_auc": 0.3875,
        "features": len(NUMERIC_COLS + CATEGORICAL_COLS),
        "training_samples": 1076248,
        "test_samples": 269062,
        "default_rate": "19.96%",
        "version": "1.0.0",
        "dataset": "Lending Club 2007-2018"
    }


@app.post("/predict", response_model=RiskResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = preprocess_input(application)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        decision, risk_level = get_decision(risk_score)

        return RiskResponse(
            risk_score=round(risk_score, 4),
            decision=decision,
            risk_level=risk_level,
            confidence=f"{round(risk_score * 100, 1)}%",
            message=f"Application {decision} — {risk_level} with score {round(risk_score, 4)}"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(application: LoanApplication):
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = preprocess_input(application)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        decision, _ = get_decision(risk_score)

        # SHAP values
        shap_values = explainer.shap_values(X)
        feature_names = NUMERIC_COLS + CATEGORICAL_COLS
        shap_dict = dict(zip(feature_names, shap_values[0]))

        # Sort by impact
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        # Top risk factors (positive SHAP = increases default probability)
        top_risk = [
            {"feature": k, "impact": round(float(v), 4)}
            for k, v in sorted_shap if v > 0
        ][:5]

        # Top protective factors (negative SHAP = decreases default probability)
        top_protective = [
            {"feature": k, "impact": round(float(v), 4)}
            for k, v in sorted_shap if v < 0
        ][:5]

        return ExplainResponse(
            risk_score=round(risk_score, 4),
            decision=decision,
            top_risk_factors=top_risk,
            top_protective_factors=top_protective
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        report = run_agent(request.application.dict())
        return AgentResponse(report=report)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))