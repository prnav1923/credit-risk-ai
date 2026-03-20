import os
import io
import logging
import pickle
import numpy as np
import pandas as pd
import boto3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from pyod.models.xgbod import XGBOD
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

FEATURE_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "emp_length", "mort_acc", "pub_rec_bankruptcies",
    "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
    "loan_to_income", "fico_avg", "high_utilization"
]


# --- Load data from S3 ---
def load_data_from_s3(s3_key: str) -> pd.DataFrame:
    logger.info(f"Loading s3://{BUCKET_NAME}/{s3_key}")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info(f"Loaded shape: {df.shape}")
    return df


# --- Prepare features ---
def prepare_features(df: pd.DataFrame) -> np.ndarray:
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    return X.values, feature_cols


# --- Create fraud labels from defaults ---
def create_fraud_labels(df: pd.DataFrame) -> np.ndarray:
    """
    In absence of explicit fraud labels, we use a proxy:
    High-risk defaulted loans with suspicious patterns are flagged as fraud.
    
    Fraud indicators:
    - Charged off (defaulted)
    - High revolving utilization (> 80%)
    - Recent delinquencies (> 2 in 2 years)
    - Public records present
    - Low FICO with high loan amount
    """
    fraud_mask = (
        (df["target"] == 1) &  # defaulted
        (
            (df["revol_util"] > 80) |  # high utilization
            (df["delinq_2yrs"] > 2) |  # multiple delinquencies
            (df["pub_rec"] > 0) |      # public records
            (df["pub_rec_bankruptcies"] > 0)  # bankruptcies
        )
    )
    labels = fraud_mask.astype(int)
    fraud_rate = labels.mean()
    logger.info(f"Fraud rate (proxy): {fraud_rate:.2%}")
    return labels.values


# --- Train Isolation Forest ---
def train_isolation_forest(X: np.ndarray, contamination: float = 0.05):
    logger.info("Training Isolation Forest...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    logger.info("Isolation Forest trained ✅")
    return iso_forest, scaler


# --- Train XGBOD (supervised anomaly detection) ---
def train_xgbod(X: np.ndarray, y: np.ndarray):
    logger.info("Training XGBOD ensemble...")

    # Subsample for speed
    sample_size = min(50000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]
    y_sample = y[idx]

    xgbod = XGBOD(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    xgbod.fit(X_sample, y_sample)
    logger.info("XGBOD trained ✅")
    return xgbod


# --- Ensemble scoring ---
def ensemble_fraud_score(
    iso_forest, scaler, xgbod,
    X: np.ndarray,
    iso_weight: float = 0.4,
    xgbod_weight: float = 0.6
) -> np.ndarray:
    # Isolation Forest score
    X_scaled = scaler.transform(X)
    iso_scores = iso_forest.score_samples(X_scaled)
    iso_scores_norm = 1 - (iso_scores - iso_scores.min()) / (
        iso_scores.max() - iso_scores.min() + 1e-10
    )

    # XGBOD score
    xgbod_proba = xgbod.predict_proba(X)
    if xgbod_proba.ndim == 2:
        xgbod_scores = xgbod_proba[:, 1]
    else:
        xgbod_scores = xgbod_proba

    # Weighted ensemble
    ensemble_scores = (iso_weight * iso_scores_norm) + (xgbod_weight * xgbod_scores)
    return ensemble_scores


# --- Save models to S3 ---
def save_models_to_s3(iso_forest, scaler, xgbod, feature_cols):
    s3 = boto3.client("s3", region_name=AWS_REGION)

    models = {
        "iso_forest": iso_forest,
        "scaler": scaler,
        "xgbod": xgbod,
        "feature_cols": feature_cols
    }

    buffer = io.BytesIO()
    pickle.dump(models, buffer)
    buffer.seek(0)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key="models/fraud_detection_models.pkl",
        Body=buffer.getvalue()
    )
    logger.info("Fraud models saved to S3 ✅")


# --- Load models from S3 ---
def load_fraud_models_from_s3():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="models/fraud_detection_models.pkl")
    models = pickle.loads(obj["Body"].read())
    return (
        models["iso_forest"],
        models["scaler"],
        models["xgbod"],
        models["feature_cols"]
    )


# --- Predict fraud for single application ---
def predict_fraud(
    application: dict,
    iso_forest, scaler, xgbod,
    feature_cols: list,
    threshold: float = 0.5
) -> dict:
    df = pd.DataFrame([application])

    # Add engineered features if missing
    if "loan_to_income" not in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
        df["loan_to_income"] = df["loan_to_income"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
    if "fico_avg" not in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    if "high_utilization" not in df.columns:
        df["high_utilization"] = (df["revol_util"] > 75).astype(int)

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(0).values

    fraud_score = float(ensemble_fraud_score(
        iso_forest, scaler, xgbod, X
    )[0])

    fraud_flag = fraud_score > threshold

    # Fraud indicators
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

    return {
        "fraud_score": round(fraud_score, 4),
        "fraud_flag": bool(fraud_flag),
        "fraud_risk": "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.4 else "LOW",
        "fraud_indicators": indicators
    }


# --- Main training pipeline ---
def run_fraud_training():
    logger.info("=== Starting Fraud Detection Training ===")

    # Load data
    train_df = load_data_from_s3("processed/train.csv")

    # Prepare features
    X, feature_cols = prepare_features(train_df)
    y = create_fraud_labels(train_df)

    logger.info(f"Training samples: {len(X):,}")
    logger.info(f"Fraud samples: {y.sum():,} ({y.mean():.2%})")

    # Train models
    iso_forest, scaler = train_isolation_forest(X, contamination=y.mean())
    xgbod = train_xgbod(X, y)

    # Evaluate on sample
    logger.info("Evaluating ensemble...")
    sample_idx = np.random.choice(len(X), min(10000, len(X)), replace=False)
    X_sample = X[sample_idx]
    y_sample = y[sample_idx]

    scores = ensemble_fraud_score(iso_forest, scaler, xgbod, X_sample)
    auc = roc_auc_score(y_sample, scores)
    logger.info(f"Fraud Detection AUC-ROC: {auc:.4f}")

    # Save to S3
    save_models_to_s3(iso_forest, scaler, xgbod, feature_cols)

    logger.info("=== Fraud Detection Training Complete ✅ ===")
    return iso_forest, scaler, xgbod, feature_cols, auc


if __name__ == "__main__":
    run_fraud_training()