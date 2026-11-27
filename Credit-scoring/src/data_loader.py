"""
Data Loading Module for Credit Scoring Model.
Handles loading the raw financial dataset, label mappings, and exporting
representative sample applicant profiles for inference and testing.
"""

import json
import os
from typing import Dict, List

import pandas as pd


DEFAULT_RAW_DATA_PATH = "dataset/raw/train.csv"
DEFAULT_SAMPLES_JSON = "dataset/samples/sample_applicants.json"
DEFAULT_SAMPLES_CSV = "dataset/samples/sample_applicants.csv"

IDENTIFIER_COLUMNS: List[str] = ["ID", "Customer_ID", "Name", "SSN"]
TARGET_COLUMN = "Credit_Score"

TARGET_ENCODING: Dict[str, int] = {
    "Poor": 0,
    "Standard": 1,
    "Good": 2,
}
DEFAULT_LABEL_MAPPING: Dict[int, str] = {v: k for k, v in TARGET_ENCODING.items()}


def load_raw_data(data_path: str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw credit scoring CSV dataset from disk.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{data_path}'. "
            "Place train.csv inside dataset/raw/ before running."
        )

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded raw dataset from {data_path}: shape={df.shape}")
    return df


def save_label_mapping(mapping: Dict[int, str], filepath: str) -> None:
    """
    Saves integer-to-label mapping dictionary as JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable = {str(k): v for k, v in mapping.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Label mapping saved to: {filepath}")


def load_label_mapping(filepath: str) -> Dict[int, str]:
    """
    Loads integer-to-label mapping dictionary from JSON.
    """
    if not os.path.exists(filepath):
        return dict(DEFAULT_LABEL_MAPPING)
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def export_sample_applicants(
    df: pd.DataFrame,
    json_path: str = DEFAULT_SAMPLES_JSON,
    csv_path: str = DEFAULT_SAMPLES_CSV,
    samples_per_class: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Exports clean representative sample applicant rows (Poor, Standard, Good)
    to JSON and CSV under dataset/samples/ for testing and Streamlit demos.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataframe.")

    # Prefer clean, realistic rows without missing or corrupted core fields
    from src.preprocessing import clean_credit_dataframe

    cleaned = clean_credit_dataframe(df)
    valid_mask = cleaned[
        [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_History_Months",
            "Monthly_Balance",
        ]
    ].notna().all(axis=1)
    valid_mask &= cleaned["Age"] >= 18
    valid_mask &= cleaned["Occupation"] != "Unknown"
    valid_mask &= cleaned["Credit_Mix"] != "Unknown"
    valid_mask &= cleaned["Payment_Behaviour"] != "Unknown"

    clean_candidates = df[valid_mask].copy()

    sampled_frames = []
    for label in ["Poor", "Standard", "Good"]:
        subset = clean_candidates[clean_candidates[TARGET_COLUMN] == label]
        if len(subset) < samples_per_class:
            subset = df[df[TARGET_COLUMN] == label]
        sampled_frames.append(
            subset.sample(n=min(samples_per_class, len(subset)), random_state=random_state)
        )

    samples_df = pd.concat(sampled_frames, axis=0).reset_index(drop=True)
    drop_cols = [c for c in IDENTIFIER_COLUMNS if c in samples_df.columns]
    samples_export = samples_df.drop(columns=drop_cols)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    samples_export.to_csv(csv_path, index=False)

    records = []
    for idx, row in samples_export.iterrows():
        rec = row.to_dict()
        rec["Sample_Name"] = f"Applicant_{idx + 1}_{rec.get(TARGET_COLUMN, 'Unknown')}"
        records.append(rec)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Exported {len(samples_export)} sample applicants to {json_path} and {csv_path}")
    return samples_export
