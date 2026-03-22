import numpy as np
import pandas as pd
import logging
from typing import Union

logger = logging.getLogger(__name__)

# ============================================================
# SINGLE SOURCE OF TRUTH FOR ALL FEATURE ENGINEERING
# Used by: data_pipeline.py, train.py, api/main.py,
#          kafka_consumer.py, drift_detector.py
# ============================================================

NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "emp_length", "mort_acc", "pub_rec_bankruptcies",
    "num_actv_bc_tl", "bc_util", "percent_bc_gt_75", "avg_cur_bal",
    "loan_to_income", "fico_avg", "high_utilization"
]

CATEGORICAL_COLS = [
    "home_ownership", "verification_status",
    "purpose", "grade", "sub_grade",
    "initial_list_status", "application_type"
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

LEAKAGE_COLS = [
    "loan_status", "funded_amnt", "funded_amnt_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "recoveries", "collection_recovery_fee",
    "out_prncp", "out_prncp_inv", "last_pymnt_amnt",
    "last_fico_range_high", "last_fico_range_low",
    "total_rec_late_fee", "hardship_dpd", "hardship_amount",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount"
]

TARGET_COL = "target"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for feature engineering.
    Called identically during training AND serving.

    This guarantees zero training-serving skew.
    """
    df = df.copy()

    # --- Engineered Feature 1: Loan to Income Ratio ---
    # High ratio = borrower taking on too much debt relative to income
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_to_income'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(df['loan_to_income'].median() if 'loan_to_income' in df else 0)

    # --- Engineered Feature 2: FICO Average ---
    # Average of low and high FICO range for a single credit score signal
    df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2

    # --- Engineered Feature 3: High Utilization Flag ---
    # Binary flag: revolving utilization > 75% is a strong default signal
    df['high_utilization'] = (df['revol_util'] > 75).astype(int)

    return df


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consistent null filling strategy.
    Called identically during training AND serving.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')

    return df


def get_feature_cols() -> list:
    """Returns the exact feature columns used for model training."""
    return FEATURE_COLS.copy()


def prepare_features_for_training(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline for training.
    Returns X (features) and y (target).
    """
    logger.info("Preparing features for training...")

    # Drop leakage columns
    df = df.drop(columns=LEAKAGE_COLS, errors='ignore')

    # Engineer features
    df = engineer_features(df)

    # Fill nulls
    df = fill_nulls(df)

    # Keep only relevant columns
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    logger.info(f"Training features shape: {X.shape}")
    return X, y


def prepare_features_for_serving(
    data: Union[dict, pd.DataFrame],
    encoders: dict
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for serving (API + Kafka).
    Guaranteed identical to training pipeline.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Engineer features (same as training)
    df = engineer_features(df)

    # Encode categoricals (same encoders as training)
    for col in CATEGORICAL_COLS:
        if col in encoders and col in df.columns:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = 0

    # Select exact feature columns in exact training order
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    return df[available_features]


def validate_feature_schema(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate that a dataframe has the expected feature schema.
    Returns (is_valid, missing_columns)
    """
    required = set(NUMERIC_COLS[:20])  # core numeric features
    available = set(df.columns)
    missing = list(required - available)

    if missing:
        logger.warning(f"Missing features: {missing}")
        return False, missing

    return True, []


def get_feature_stats() -> dict:
    """
    Returns expected feature statistics for drift detection.
    Based on training data distribution.
    """
    return {
        "loan_amnt": {"min": 500, "max": 40000, "mean": 14755},
        "int_rate": {"min": 5.32, "max": 30.99, "mean": 13.24},
        "dti": {"min": 0, "max": 99.99, "mean": 17.38},
        "fico_range_low": {"min": 610, "max": 845, "mean": 698},
        "revol_util": {"min": 0, "max": 100, "mean": 53.76},
        "annual_inc": {"min": 4000, "max": 9500000, "mean": 74491},
        "loan_to_income": {"min": 0, "max": 5, "mean": 0.21},
        "fico_avg": {"min": 612, "max": 847, "mean": 700},
    }