"""
Inference Module for Credit Scoring Model.
Loads the end-to-end saved pipeline and predicts creditworthiness (Poor, Standard, Good)
along with class probabilities and key financial ratios for new applicant records.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data_loader import load_label_mapping
from src.feature_engineering import engineer_features
from src.models import load_trained_model
from src.preprocessing import clean_credit_dataframe


DEFAULT_MODEL_PATH = "models/credit_scoring_model.joblib"
DEFAULT_LABEL_MAPPING_PATH = "models/label_mapping.json"


def predict_credit_score(
    applicant_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH,
    label_mapping_path: str = DEFAULT_LABEL_MAPPING_PATH,
) -> Dict[str, Any]:
    """
    Predicts credit score class and probability distribution for one or more applicants.
    Returns a structured dictionary for single-row input, or a list of results under
    'predictions' for batch input.
    """
    pipeline = load_trained_model(model_path)
    label_map = load_label_mapping(label_mapping_path)

    if isinstance(applicant_data, dict):
        input_df = pd.DataFrame([applicant_data])
        single_mode = True
    elif isinstance(applicant_data, list):
        input_df = pd.DataFrame(applicant_data)
        single_mode = len(input_df) == 1
    elif isinstance(applicant_data, pd.DataFrame):
        input_df = applicant_data.copy()
        single_mode = len(input_df) == 1
    else:
        raise TypeError("applicant_data must be a dict, list of dicts, or pandas DataFrame.")

    pred_indices = pipeline.predict(input_df)
    pred_probas = pipeline.predict_proba(input_df)
    engineered_df = engineer_features(clean_credit_dataframe(input_df))

    results: List[Dict[str, Any]] = []
    for idx in range(len(input_df)):
        pred_cls = int(pred_indices[idx])
        probs = pred_probas[idx]
        prob_dict = {
            label_map[i]: round(float(probs[i]), 4)
            for i in sorted(label_map.keys())
        }
        row_eng = engineered_df.iloc[idx]
        results.append(
            {
                "predicted_class": pred_cls,
                "predicted_label": label_map.get(pred_cls, str(pred_cls)),
                "confidence": round(float(np.max(probs)), 4),
                "probabilities": prob_dict,
                "financial_ratios": {
                    "Debt_to_Income_Ratio": round(float(row_eng["Debt_to_Income_Ratio"]), 4),
                    "EMI_to_Salary_Ratio": round(float(row_eng["EMI_to_Salary_Ratio"]), 4),
                    "Savings_to_Salary_Ratio": round(float(row_eng["Savings_to_Salary_Ratio"]), 4),
                    "Delayed_Payment_Rate": round(float(row_eng["Delayed_Payment_Rate"]), 4),
                    "Credit_History_Months": (
                        None
                        if pd.isna(row_eng["Credit_History_Months"])
                        else int(row_eng["Credit_History_Months"])
                    ),
                },
            }
        )

    return results[0] if single_mode else {"predictions": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Credit Scoring inference on JSON input.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/samples/sample_applicants.json",
        help="Path to JSON file containing applicant dict or list of dicts.",
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--label-mapping-path", type=str, default=DEFAULT_LABEL_MAPPING_PATH)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    output = predict_credit_score(
        payload,
        model_path=args.model_path,
        label_mapping_path=args.label_mapping_path,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
