import os
import io
import logging
import numpy as np
import pandas as pd
import boto3
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pickle
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config ---
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
TARGET_COL = "target"
MLFLOW_EXPERIMENT = "credit-risk-xgboost"

CATEGORICAL_COLS = [
    "home_ownership", "verification_status",
    "purpose", "grade", "sub_grade", "initial_list_status",
    "application_type"
]

NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "emp_length", "mort_acc", "pub_rec_bankruptcies",
    "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
    "loan_to_income", "fico_avg", "high_utilization"  # ← make sure these are here
]


# --- Step 1: Load from S3 ---
def load_from_s3(s3_key: str) -> pd.DataFrame:
    logger.info(f"Loading s3://{BUCKET_NAME}/{s3_key}")
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=s3_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    logger.info(f"Loaded shape: {df.shape}")
    return df


# --- Step 2: Preprocess ---
def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    logger.info("Preprocessing...")
    df = df.copy()

    # Encode categoricals
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = encoders[col]
                df[col] = le.transform(df[col].astype(str))

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y, encoders


# --- Step 3: Train ---
def train_model(X_train, y_train):
    logger.info("Training XGBoost model...")

    # Split train into train + validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "gamma": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],  # ← real validation set now
        verbose=100
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model, params


# --- Step 4: Evaluate ---
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    logger.info(f"AUC-ROC:  {auc_roc:.4f}")
    logger.info(f"PR-AUC:   {pr_auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    metrics = {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc
    }

    return metrics


# --- Step 5: Run with MLflow ---
def run_training():
    # Set MLflow to log to S3
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load data from S3
    train_df = load_from_s3("processed/train.csv")
    test_df = load_from_s3("processed/test.csv")

    # Preprocess
    X_train, y_train, encoders = preprocess(train_df, fit=True)
    X_test, y_test, _ = preprocess(test_df, encoders=encoders, fit=False)

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    with mlflow.start_run(run_name="xgboost-baseline"):

        # Train
        model, params = train_model(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        logger.info("Training Complete ✅")

    # Save model locally
    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save encoders
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Upload to S3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    s3.upload_file("models/xgboost_model.pkl", BUCKET_NAME, "models/xgboost_model.pkl")
    s3.upload_file("models/encoders.pkl", BUCKET_NAME, "models/encoders.pkl")

    logger.info("Model saved to S3 ✅")

    return model, encoders, metrics


if __name__ == "__main__":
    run_training()