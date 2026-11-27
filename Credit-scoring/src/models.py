"""
Model Definitions Module for Credit Scoring Model.
Defines Logistic Regression, Decision Tree, and Random Forest classifiers,
along with end-to-end pipeline assembly and serialization helpers.
"""

import os
from typing import Any, Dict

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import CreditFeatureEngineer


def get_candidate_models(random_state: int = 42, n_estimators: int = 150) -> Dict[str, Any]:
    """
    Returns the candidate classification algorithms for credit scoring:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def build_inference_pipeline(preprocessor: Any, classifier: Any) -> Pipeline:
    """
    Combines cleaning/feature engineering, preprocessing, and the trained classifier
    into a single end-to-end scikit-learn Pipeline for raw applicant inference.
    """
    return Pipeline(
        steps=[
            ("feature_engineer", CreditFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model or pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath, compress=3)
    print(f"Model saved successfully to: {filepath}")


def load_trained_model(filepath: str = "models/credit_scoring_model.joblib") -> Any:
    """
    Loads a trained model or pipeline from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"Loaded trained model from: {filepath}")
    return model
