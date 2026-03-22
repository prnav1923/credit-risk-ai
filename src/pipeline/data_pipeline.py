import os
import io
import logging
import numpy as np
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.features import engineer_features, fill_nulls, LEAKAGE_COLS, FEATURE_COLS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# --- Config ---
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
TARGET_COL = "target"

FEATURE_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "emp_length", "mort_acc", "pub_rec_bankruptcies",
    "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
    "home_ownership", "verification_status", "purpose",
    "grade", "sub_grade", "initial_list_status", "application_type"
]

LEAKAGE_COLS = [
    # Original leakage
    "loan_status", "funded_amnt", "funded_amnt_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "recoveries", "collection_recovery_fee",
    "out_prncp", "out_prncp_inv", "last_pymnt_amnt",
    
    # Newly identified leakage
    "last_fico_range_high", "last_fico_range_low",
    "total_rec_late_fee", "hardship_dpd", "hardship_amount",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount"
]


# --- Step 1: Load ---
def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Raw shape: {df.shape}")

    # Keep only fully paid or charged off loans
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
    df[TARGET_COL] = (df["loan_status"] == "Charged Off").astype(int)

    logger.info(f"Filtered shape: {df.shape}")
    logger.info(f"Default rate: {df[TARGET_COL].mean():.2%}")
    return df


# --- Step 2: Clean ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
    cols_to_keep = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    # Clean emp_length
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].str.extract(r"(\d+)").astype(float)

    # Clean percentage columns
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace("%", "").astype(float)

    # Engineer features — single source of truth
    df = engineer_features(df)

    # Fill nulls — single source of truth
    df = fill_nulls(df)

    logger.info(f"Clean shape: {df.shape}")
    return df


# --- Step 3: Upload to S3 ---
def upload_to_s3(df: pd.DataFrame, s3_key: str):
    logger.info(f"Uploading to s3://{BUCKET_NAME}/{s3_key}")
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=s3_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Upload complete ✅ ({len(df):,} rows)")


# --- Step 4: Run Pipeline ---
def run_pipeline(raw_filepath: str):
    logger.info("=== Starting Data Pipeline ===")

    df = load_raw_data(raw_filepath)
    df = clean_data(df)

    # Split
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COL]
    )

    logger.info(f"Train size: {len(train):,} | Test size: {len(test):,}")

    # Upload all splits to S3
    upload_to_s3(df, "raw/lending_club_filtered.csv")
    upload_to_s3(train, "processed/train.csv")
    upload_to_s3(test, "processed/test.csv")

    logger.info("=== Pipeline Complete ✅ ===")
    return train, test


if __name__ == "__main__":
    # Update this path to where your CSV is
    run_pipeline("/Users/pranav/Code/credit-risk-ai/data/lending-club/accepted_2007_to_2018q4/accepted_2007_to_2018Q4.csv")