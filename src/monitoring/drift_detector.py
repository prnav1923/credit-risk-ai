import sys
sys.path.insert(0, '/Users/pranav/Code/credit-risk-ai')
from src.database import SessionLocal, DriftReport

import os
import json
import logging
import pickle
import io
import numpy as np
import pandas as pd
import boto3
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")
AUC_THRESHOLD = 0.70  # matches policy document
DRIFT_THRESHOLD = 0.05  # 5% drop triggers alert


def load_model_from_s3():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    model_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/xgboost_model.pkl")
    return pickle.loads(model_obj["Body"].read())


def load_test_data_from_s3():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="processed/test.csv")
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def compute_current_auc(model, test_df):
    logger.info("Computing current AUC on test set...")

    feature_cols = [c for c in test_df.columns if c != "target"]
    X = test_df[feature_cols]
    y = test_df["target"]

    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)

    logger.info(f"Current AUC: {auc:.4f}")
    return auc


def load_baseline_auc():
    # Baseline AUC from initial training
    return 0.7198


def check_drift(current_auc: float, baseline_auc: float):
    drift = baseline_auc - current_auc
    drift_pct = (drift / baseline_auc) * 100

    logger.info(f"Baseline AUC: {baseline_auc:.4f}")
    logger.info(f"Current AUC:  {current_auc:.4f}")
    logger.info(f"Drift:        {drift:.4f} ({drift_pct:.1f}%)")

    if current_auc < AUC_THRESHOLD:
        status = "CRITICAL"
        action = "Mandatory model review required per policy"
    elif drift > DRIFT_THRESHOLD:
        status = "WARNING"
        action = "Model retraining recommended"
    else:
        status = "HEALTHY"
        action = "No action required"

    return {
        "status": status,
        "baseline_auc": baseline_auc,
        "current_auc": current_auc,
        "drift": round(drift, 4),
        "drift_pct": round(drift_pct, 2),
        "action": action,
        "timestamp": datetime.now(timezone.utc)
.isoformat()
    }


def log_drift_to_s3(drift_report: dict):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"monitoring/drift/{datetime.now(timezone.utc)
.strftime('%Y%m%d_%H%M%S')}.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(drift_report)
    )
    logger.info(f"Drift report saved to S3: {key}")


def log_to_cloudwatch(drift_report: dict):
    try:
        cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
        cloudwatch.put_metric_data(
            Namespace="CreditRiskAI",
            MetricData=[
                {
                    "MetricName": "ModelAUC",
                    "Value": drift_report["current_auc"],
                    "Unit": "None",
                    "Timestamp": datetime.now(timezone.utc)

                },
                {
                    "MetricName": "ModelDrift",
                    "Value": drift_report["drift"],
                    "Unit": "None",
                    "Timestamp": datetime.now(timezone.utc)

                }
            ]
        )
        logger.info("Metrics logged to CloudWatch ✅")
    except Exception as e:
        logger.warning(f"CloudWatch logging failed: {e}")


def run_drift_detection():
    logger.info("=== Starting Drift Detection ===")

    model = load_model_from_s3()
    test_df = load_test_data_from_s3()
    current_auc = compute_current_auc(model, test_df)
    baseline_auc = load_baseline_auc()
    drift_report = check_drift(current_auc, baseline_auc)

    print("\n=== DRIFT DETECTION REPORT ===")
    print(f"Status:       {drift_report['status']}")
    print(f"Baseline AUC: {drift_report['baseline_auc']}")
    print(f"Current AUC:  {drift_report['current_auc']}")
    print(f"Drift:        {drift_report['drift']} ({drift_report['drift_pct']}%)")
    print(f"Action:       {drift_report['action']}")
    print("==============================\n")

    log_drift_to_s3(drift_report)
    log_to_cloudwatch(drift_report)

    # Log to PostgreSQL
    try:
        db = SessionLocal()
        drift_record = DriftReport(
            status=drift_report["status"],
            baseline_auc=drift_report["baseline_auc"],
            current_auc=drift_report["current_auc"],
            drift=drift_report["drift"],
            drift_pct=drift_report["drift_pct"],
            action=drift_report["action"]
        )
        db.add(drift_record)
        db.commit()
        db.close()
        logger.info("Drift report logged to PostgreSQL ✅")
    except Exception as e:
        logger.warning(f"PostgreSQL logging failed: {e}")

    logger.info("=== Drift Detection Complete ✅ ===")
    return drift_report

def compute_current_auc(model, test_df):
    logger.info("Computing current AUC on test set...")

    # Load encoders from S3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    enc_obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/encoders.pkl")
    encoders = pickle.loads(enc_obj["Body"].read())

    X = test_df.copy()
    y = X.pop("target")

    # Add engineered features if missing
    if 'loan_to_income' not in X.columns:
        X['loan_to_income'] = X['loan_amnt'] / X['annual_inc'].replace(0, np.nan)
        X['loan_to_income'] = X['loan_to_income'].replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'fico_avg' not in X.columns:
        X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2
    if 'high_utilization' not in X.columns:
        X['high_utilization'] = (X['revol_util'] > 75).astype(int)

    # Encode categoricals
    categorical_cols = [
        "home_ownership", "verification_status", "purpose",
        "grade", "sub_grade", "initial_list_status", "application_type"
    ]
    for col in categorical_cols:
        if col in X.columns and col in encoders:
            try:
                X[col] = encoders[col].transform(X[col].astype(str))
            except ValueError:
                X[col] = 0

    # Match exact feature order from training
    feature_cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "emp_length", "mort_acc", "pub_rec_bankruptcies",
        "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
        "loan_to_income", "fico_avg", "high_utilization",
        "home_ownership", "verification_status", "purpose",
        "grade", "sub_grade", "initial_list_status", "application_type"
    ]
    X = X[feature_cols]

    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)

    logger.info(f"Current AUC: {auc:.4f}")
    return auc

if __name__ == "__main__":
    run_drift_detection()