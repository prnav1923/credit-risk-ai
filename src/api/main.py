import os
import io
import logging
import pickle
import numpy as np
import pandas as pd
import boto3
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from src.agent.agent import run_agent
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security
from fastapi import FastAPI, HTTPException, Depends
from datetime import datetime
import json
from src.monitoring.drift_detector import run_drift_detection
from src.model.fraud_detector import load_fraud_models_from_s3, predict_fraud
from src.agent.multi_agent import run_multi_agent
from sqlalchemy.orm import Session
from src.database import get_db, Prediction, AgentDecision, Override, DriftReport, init_db
from src.cache import get_cache_key, get_cached, set_cached, get_cache_stats

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
iso_forest = None
scaler = None
xgbod = None
fraud_feature_cols = None

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

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

def get_decision(risk_score: float):
    if risk_score < 0.3:
        return "APPROVE", "LOW RISK"
    elif risk_score < 0.6:
        return "REVIEW", "MEDIUM RISK"
    else:
        return "DECLINE", "HIGH RISK"

# --- Load model from S3 ---
def load_model_from_s3():
    global model, encoders, explainer, iso_forest, scaler, xgbod, fraud_feature_cols
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

    # Load fraud models
    iso_forest, scaler, xgbod, fraud_feature_cols = load_fraud_models_from_s3()

    logger.info("Model loaded successfully ✅")


# --- Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_s3()
    init_db()
    yield
    logger.info("Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk scoring using XGBoost + LangChain RAG",
    version="1.0.0",
    lifespan=lifespan
    # root_path="/v1"
)


# --- Pydantic Schemas ---


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=40000, example=10000.0, description="Loan amount in USD")
    int_rate: float = Field(..., gt=0, le=31, example=12.5, description="Interest rate %")
    installment: float = Field(..., gt=0, example=350.0, description="Monthly installment")
    annual_inc: float = Field(..., gt=0, example=60000.0, description="Annual income")
    dti: float = Field(..., ge=0, le=100, example=18.5, description="Debt to income ratio")
    delinq_2yrs: float = Field(..., ge=0, example=0.0, description="Delinquencies in last 2 years")
    fico_range_low: float = Field(..., ge=300, le=850, example=680.0, description="FICO score low")
    fico_range_high: float = Field(..., ge=300, le=850, example=684.0, description="FICO score high")
    open_acc: float = Field(..., ge=0, example=10.0, description="Number of open accounts")
    pub_rec: float = Field(..., ge=0, example=0.0, description="Public records")
    revol_bal: float = Field(..., ge=0, example=15000.0, description="Revolving balance")
    revol_util: float = Field(..., ge=0, le=100, example=45.0, description="Revolving utilization %")
    total_acc: float = Field(..., ge=0, example=25.0, description="Total accounts")
    emp_length: float = Field(..., ge=0, le=10, example=5.0, description="Employment length in years")
    mort_acc: float = Field(..., ge=0, example=2.0, description="Mortgage accounts")
    pub_rec_bankruptcies: float = Field(..., ge=0, example=0.0, description="Bankruptcies")
    num_actv_bc_tl: float = Field(..., ge=0, example=4.0, description="Active bankcard tradelines")
    bc_util: float = Field(..., ge=0, le=100, example=50.0, description="Bankcard utilization %")
    percent_bc_gt_75: float = Field(..., ge=0, le=100, example=25.0, description="% bankcards > 75% utilized")
    avg_cur_bal: float = Field(..., ge=0, example=8000.0, description="Average current balance")
    home_ownership: str = Field(..., example="RENT", description="RENT/OWN/MORTGAGE")
    verification_status: str = Field(..., example="Verified", description="Income verification")
    purpose: str = Field(..., example="debt_consolidation", description="Loan purpose")
    grade: str = Field(..., example="B", description="Loan grade A-G")
    sub_grade: str = Field(..., example="B3", description="Loan sub grade")
    initial_list_status: str = Field(..., example="w", description="w or f")
    application_type: str = Field(..., example="Individual", description="Individual/Joint App")

    @validator('fico_range_high')
    def fico_high_must_be_gte_low(cls, v, values):
        if 'fico_range_low' in values and v < values['fico_range_low']:
            raise ValueError('fico_range_high must be >= fico_range_low')
        return v

    @validator('home_ownership')
    def valid_home_ownership(cls, v):
        valid = ['RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE']
        if v.upper() not in valid:
            raise ValueError(f'home_ownership must be one of {valid}')
        return v.upper()

    @validator('grade')
    def valid_grade(cls, v):
        if v.upper() not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            raise ValueError('grade must be A-G')
        return v.upper()

    @validator('application_type')
    def valid_application_type(cls, v):
        valid = ['Individual', 'Joint App']
        if v not in valid:
            raise ValueError(f'application_type must be one of {valid}')
        return v

    @validator('initial_list_status')
    def valid_list_status(cls, v):
        if v.lower() not in ['w', 'f']:
            raise ValueError('initial_list_status must be w or f')
        return v.lower()

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

class FraudResponse(BaseModel):
    fraud_score: float
    fraud_flag: bool
    fraud_risk: str
    fraud_indicators: list

class AssessResponse(BaseModel):
    # Credit Risk
    credit_risk_score: float
    credit_decision: str
    credit_risk_level: str
    # Fraud
    fraud_score: float
    fraud_flag: bool
    fraud_risk: str
    fraud_indicators: list
    # Combined
    combined_decision: str
    combined_risk: str
    message: str

class MultiAgentResponse(BaseModel):
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

@app.get("/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/v1/model-info")
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


@app.post("/v1/predict", response_model=RiskResponse, dependencies=[Depends(verify_api_key)])
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Check cache first
        cache_key = get_cache_key("predict", application.dict())
        cached = get_cached(cache_key)
        if cached:
            return RiskResponse(**cached)

        X = preprocess_input(application)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        decision, risk_level = get_decision(risk_score)

        result = RiskResponse(
            risk_score=round(risk_score, 4),
            decision=decision,
            risk_level=risk_level,
            confidence=f"{round(risk_score * 100, 1)}%",
            message=f"Application {decision} — {risk_level} with score {round(risk_score, 4)}"
        )

        # Cache the result
        set_cached(cache_key, result.dict())
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/explain", response_model=ExplainResponse, dependencies=[Depends(verify_api_key)])
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

@app.post("/v1/agent", response_model=AgentResponse, dependencies=[Depends(verify_api_key)])
def agent_endpoint(request: AgentRequest, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        report = run_agent(request.application.dict())

        # Extract decision from report
        decision = "UNKNOWN"
        if "APPROVE" in report.upper():
            decision = "APPROVE"
        elif "DECLINE" in report.upper():
            decision = "DECLINE"
        elif "REVIEW" in report.upper():
            decision = "REVIEW"

        # Log to PostgreSQL
        agent_decision = AgentDecision(
            agent_type="single",
            application_json=request.application.dict(),
            report=report,
            decision=decision
        )
        db.add(agent_decision)
        db.commit()
        logger.info(f"Agent decision logged to PostgreSQL — ID: {agent_decision.id}")

        return AgentResponse(report=report)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

class OverrideRequest(BaseModel):
    application: LoanApplication
    override_decision: str
    reviewer_id: str
    reason: str

    @validator('override_decision')
    def valid_decision(cls, v):
        if v.upper() not in ['APPROVE', 'DECLINE', 'REVIEW']:
            raise ValueError('override_decision must be APPROVE, DECLINE or REVIEW')
        return v.upper()

class OverrideResponse(BaseModel):
    status: str
    original_decision: str
    override_decision: str
    reviewer_id: str
    reason: str
    timestamp: str

@app.post("/v1/override", response_model=OverrideResponse, dependencies=[Depends(verify_api_key)])
def manual_override(request: OverrideRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Get original prediction
        X = preprocess_input(request.application)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        original_decision, _ = get_decision(risk_score)

        # Log to PostgreSQL
        override = Override(
            reviewer_id=request.reviewer_id,
            original_decision=original_decision,
            override_decision=request.override_decision,
            reason=request.reason,
            risk_score=round(risk_score, 4),
            application_json=request.application.dict()
        )
        db.add(override)
        db.commit()
    

        # Log override to S3
        override_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "reviewer_id": request.reviewer_id,
            "original_decision": original_decision,
            "override_decision": request.override_decision,
            "reason": request.reason,
            "risk_score": round(risk_score, 4),
            "application": request.application.dict()
        }

        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        key = f"audit/overrides/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.reviewer_id}.json"
        s3.put_object(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Key=key,
            Body=json.dumps(override_record)
        )

        logger.info(f"Override logged to S3: {key}")

        return OverrideResponse(
            status="override_recorded",
            original_decision=original_decision,
            override_decision=request.override_decision,
            reviewer_id=request.reviewer_id,
            reason=request.reason,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Override error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audit", dependencies=[Depends(verify_api_key)])
def get_audit_log():
    try:
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        response = s3.list_objects_v2(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Prefix="audit/overrides/"
        )

        overrides = []
        if "Contents" in response:
            for obj in response["Contents"][:20]:  # last 20 overrides
                record = s3.get_object(
                    Bucket=os.getenv("S3_BUCKET_NAME"),
                    Key=obj["Key"]
                )
                overrides.append(json.loads(record["Body"].read()))

        return {
            "total_overrides": len(overrides),
            "overrides": overrides
        }
    except Exception as e:
        logger.error(f"Audit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/monitor", dependencies=[Depends(verify_api_key)])
def monitor():
    try:
        report = run_drift_detection()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/fraud", response_model=FraudResponse, dependencies=[Depends(verify_api_key)])
def fraud_detection(application: LoanApplication):
    if iso_forest is None:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")
    try:
        result = predict_fraud(
            application.dict(),
            iso_forest, scaler, xgbod, fraud_feature_cols
        )
        return FraudResponse(**result)
    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/assess", response_model=AssessResponse, dependencies=[Depends(verify_api_key)])
def assess(application: LoanApplication, db: Session = Depends(get_db)):
    if model is None or iso_forest is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        # Credit risk
        X = preprocess_input(application)
        risk_score = float(model.predict_proba(X)[:, 1][0])
        credit_decision, credit_risk_level = get_decision(risk_score)

        # Fraud detection
        fraud_result = predict_fraud(
            application.dict(),
            iso_forest, scaler, xgbod, fraud_feature_cols
        )

        # Combined decision
        if fraud_result["fraud_flag"]:
            combined_decision = "DECLINE"
            combined_risk = "FRAUD DETECTED"
        elif credit_decision == "DECLINE":
            combined_decision = "DECLINE"
            combined_risk = "HIGH CREDIT RISK"
        elif credit_decision == "REVIEW" or fraud_result["fraud_risk"] == "MEDIUM":
            combined_decision = "REVIEW"
            combined_risk = "MANUAL REVIEW REQUIRED"
        else:
            combined_decision = "APPROVE"
            combined_risk = "LOW RISK"
        
        # Log to PostgreSQL
        prediction = Prediction(
            loan_amnt=application.loan_amnt,
            int_rate=application.int_rate,
            annual_inc=application.annual_inc,
            dti=application.dti,
            fico_range_low=application.fico_range_low,
            fico_range_high=application.fico_range_high,
            grade=application.grade,
            sub_grade=application.sub_grade,
            home_ownership=application.home_ownership,
            purpose=application.purpose,
            credit_risk_score=round(risk_score, 4),
            credit_decision=credit_decision,
            credit_risk_level=credit_risk_level,
            fraud_score=fraud_result["fraud_score"],
            fraud_flag=fraud_result["fraud_flag"],
            fraud_risk=fraud_result["fraud_risk"],
            fraud_indicators=fraud_result["fraud_indicators"],
            combined_decision=combined_decision,
            combined_risk=combined_risk,
            application_json=application.dict()
        )
        db.add(prediction)
        db.commit()
        logger.info(f"Prediction logged to PostgreSQL — ID: {prediction.id}")

        return AssessResponse(
            credit_risk_score=round(risk_score, 4),
            credit_decision=credit_decision,
            credit_risk_level=credit_risk_level,
            fraud_score=fraud_result["fraud_score"],
            fraud_flag=fraud_result["fraud_flag"],
            fraud_risk=fraud_result["fraud_risk"],
            fraud_indicators=fraud_result["fraud_indicators"],
            combined_decision=combined_decision,
            combined_risk=combined_risk,
            message=f"{combined_decision} — Credit Risk: {round(risk_score, 4)}, Fraud Risk: {fraud_result['fraud_risk']}"
        )
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/multi-agent", response_model=MultiAgentResponse, dependencies=[Depends(verify_api_key)])
def multi_agent_endpoint(request: AgentRequest, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        report = run_multi_agent(request.application.dict())

        # Extract decision from report
        decision = "UNKNOWN"
        if "APPROVE" in report.upper():
            decision = "APPROVE"
        elif "DECLINE" in report.upper():
            decision = "DECLINE"
        elif "REVIEW" in report.upper():
            decision = "REVIEW"

        # Log to PostgreSQL
        agent_decision = AgentDecision(
            agent_type="multi",
            application_json=request.application.dict(),
            report=report,
            decision=decision
        )
        db.add(agent_decision)
        db.commit()
        logger.info(f"Multi-agent decision logged to PostgreSQL — ID: {agent_decision.id}")

        return MultiAgentResponse(report=report)
    except Exception as e:
        logger.error(f"Multi-agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/predictions", dependencies=[Depends(verify_api_key)])
def get_predictions(limit: int = 20, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(
        Prediction.timestamp.desc()
    ).limit(limit).all()
    return {
        "total": db.query(Prediction).count(),
        "predictions": [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat(),
                "loan_amnt": p.loan_amnt,
                "grade": p.grade,
                "credit_risk_score": p.credit_risk_score,
                "credit_decision": p.credit_decision,
                "fraud_score": p.fraud_score,
                "fraud_flag": p.fraud_flag,
                "combined_decision": p.combined_decision
            }
            for p in predictions
        ]
    }

@app.get("/v1/cache", dependencies=[Depends(verify_api_key)])
def cache_stats():
    return get_cache_stats()


@app.get("/v1/agent-decisions", dependencies=[Depends(verify_api_key)])
def get_agent_decisions(limit: int = 20, db: Session = Depends(get_db)):
    decisions = db.query(AgentDecision).order_by(
        AgentDecision.timestamp.desc()
    ).limit(limit).all()
    return {
        "total": db.query(AgentDecision).count(),
        "decisions": [
            {
                "id": d.id,
                "timestamp": d.timestamp.isoformat(),
                "agent_type": d.agent_type,
                "decision": d.decision,
                "report_preview": d.report[:200] + "..."
            }
            for d in decisions
        ]
    }