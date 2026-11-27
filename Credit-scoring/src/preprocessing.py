"""
Data Preprocessing Module for Credit Scoring Model.
Cleans corrupted strings/sentinels, parses credit history age, enforces realistic
domain bounds, splits datasets without customer leakage, and builds scikit-learn
preprocessing transformers.
"""

import os
import re
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import IDENTIFIER_COLUMNS, TARGET_COLUMN, TARGET_ENCODING
from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# Realistic domain bounds to filter injected sentinel/corrupted values to NaN
NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Age": (14.0, 100.0),
    "Annual_Income": (0.0, 300000.0),
    "Monthly_Inhand_Salary": (0.0, 30000.0),
    "Num_Bank_Accounts": (0.0, 15.0),
    "Num_Credit_Card": (0.0, 15.0),
    "Interest_Rate": (1.0, 40.0),
    "Num_of_Loan": (0.0, 15.0),
    "Delay_from_due_date": (-5.0, 100.0),
    "Num_of_Delayed_Payment": (0.0, 50.0),
    "Changed_Credit_Limit": (-20.0, 50.0),
    "Num_Credit_Inquiries": (0.0, 30.0),
    "Outstanding_Debt": (0.0, 10000.0),
    "Credit_Utilization_Ratio": (0.0, 100.0),
    "Total_EMI_per_month": (0.0, 5000.0),
    "Amount_invested_monthly": (0.0, 5000.0),
    "Monthly_Balance": (0.0, 5000.0),
}

CATEGORICAL_CORRUPT_TOKENS = {"_", "_______", "!@9#%8", "NM", "nan", "None", ""}

_HISTORY_PATTERN = re.compile(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", re.IGNORECASE)


def parse_credit_history_months(value) -> float:
    """
    Converts strings like '22 Years and 1 Months' (or numeric values) to total months.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = _HISTORY_PATTERN.search(str(value))
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return float(years * 12 + months)
    numeric = pd.to_numeric(str(value).strip(), errors="coerce")
    return float(numeric) if pd.notna(numeric) else np.nan


def _clean_numeric_series(series: pd.Series, bounds: Tuple[float, float]) -> pd.Series:
    """
    Strips trailing underscores/whitespace, converts to float, and replaces values
    outside domain bounds with NaN for downstream median imputation.
    """
    cleaned = series.astype(str).str.strip().str.strip("_")
    numeric = pd.to_numeric(cleaned, errors="coerce")
    low, high = bounds
    return numeric.where((numeric >= low) & (numeric <= high), np.nan)


def _count_loan_types(value) -> float:
    """
    Counts the number of loan types listed in Type_of_Loan (0 if missing or 'Not Specified').
    """
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() in {"not specified", "nan", "none"}:
        return 0.0
    normalized = text.replace(", and ", ", ").replace(" and ", ", ")
    parts = [p.strip() for p in normalized.split(",") if p.strip() and p.strip().lower() != "not specified"]
    return float(len(parts))


def clean_credit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes raw credit applicant data:
    - Cleans numeric strings and masks out-of-bounds sentinels to NaN
    - Parses Credit_History_Age into Credit_History_Months
    - Counts loan types from Type_of_Loan
    - Normalizes corrupted categorical tokens to 'Unknown'
    """
    out = df.copy()

    for col, bounds in NUMERIC_BOUNDS.items():
        if col in out.columns:
            out[col] = _clean_numeric_series(out[col], bounds)
        else:
            out[col] = np.nan

    if "Credit_History_Months" not in out.columns:
        if "Credit_History_Age" in out.columns:
            out["Credit_History_Months"] = out["Credit_History_Age"].apply(parse_credit_history_months)
        else:
            out["Credit_History_Months"] = np.nan

    if "Num_Loan_Types" not in out.columns:
        if "Type_of_Loan" in out.columns:
            out["Num_Loan_Types"] = out["Type_of_Loan"].apply(_count_loan_types)
        else:
            out["Num_Loan_Types"] = out["Num_of_Loan"].fillna(0.0)

    for cat_col in ["Month", "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]:
        if cat_col in out.columns:
            vals = out[cat_col].fillna("Unknown").astype(str).str.strip()
            out[cat_col] = vals.where(~vals.isin(CATEGORICAL_CORRUPT_TOKENS), "Unknown")
        else:
            out[cat_col] = "Unknown"

    return out


def build_preprocessor(
    numeric_features: Optional[list] = None,
    categorical_features: Optional[list] = None,
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer with median imputation + standard scaling
    for numeric features, and most-frequent imputation + one-hot encoding for categoricals.
    """
    num_cols = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits the dataframe into train and test sets.
    Uses GroupShuffleSplit on Customer_ID when available to prevent multi-month
    customer data leakage across train and test splits.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    y = df[TARGET_COLUMN].map(TARGET_ENCODING).to_numpy(dtype=int)

    if "Customer_ID" in df.columns and df["Customer_ID"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y, groups=df["Customer_ID"]))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y))

    drop_cols = [c for c in IDENTIFIER_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    X_train_raw = df.iloc[train_idx].drop(columns=drop_cols).reset_index(drop=True)
    X_test_raw = df.iloc[test_idx].drop(columns=drop_cols).reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train_raw, X_test_raw, y_train, y_test


def save_preprocessed_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    preprocessor: ColumnTransformer,
    output_path: str = "models/credit_features.joblib",
) -> None:
    """
    Persists preprocessed feature matrices, labels, feature names, and fitted preprocessor.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }
    joblib.dump(payload, output_path, compress=3)
    print(f"Preprocessed features saved to: {output_path}")


def load_preprocessed_data(filepath: str = "models/credit_features.joblib") -> dict:
    """
    Loads preprocessed feature arrays and metadata from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessed features not found at: {filepath}")
    data = joblib.load(filepath)
    print(
        f"Loaded preprocessed data from {filepath}: "
        f"X_train={data['X_train'].shape}, X_test={data['X_test'].shape}"
    )
    return data
