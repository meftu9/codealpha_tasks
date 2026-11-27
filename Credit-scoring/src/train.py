"""
Training Script for Credit Scoring Model.
Loads raw financial records, exports clean sample profiles, performs group-aware
train/test splitting, applies feature engineering and preprocessing, trains
Logistic Regression, Decision Tree, and Random Forest models, and persists the
best end-to-end pipeline.
"""

import argparse
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import (
    DEFAULT_LABEL_MAPPING,
    DEFAULT_RAW_DATA_PATH,
    export_sample_applicants,
    load_raw_data,
    save_label_mapping,
)
from src.evaluate import compute_classification_metrics
from src.feature_engineering import engineer_features
from src.models import build_inference_pipeline, get_candidate_models, save_model
from src.preprocessing import (
    build_preprocessor,
    clean_credit_dataframe,
    save_preprocessed_data,
    split_dataset,
)


def train(
    data_path: str = DEFAULT_RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 150,
    features_path: str = "models/credit_features.joblib",
    all_models_path: str = "models/all_models.joblib",
    best_model_path: str = "models/credit_scoring_model.joblib",
    label_mapping_path: str = "models/label_mapping.json",
) -> Dict[str, Any]:
    """
    Executes the complete credit scoring training pipeline.
    """
    print("=" * 60)
    print("STARTING CREDIT SCORING MODEL TRAINING")
    print("=" * 60)

    # 1. Load raw data & export representative sample applicants
    raw_df = load_raw_data(data_path)
    export_sample_applicants(raw_df, random_state=random_state)

    # 2. Save label mapping
    save_label_mapping(DEFAULT_LABEL_MAPPING, label_mapping_path)
    class_labels = [DEFAULT_LABEL_MAPPING[i] for i in sorted(DEFAULT_LABEL_MAPPING.keys())]

    # 3. Group-aware split on Customer_ID to prevent multi-month leakage
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(
        raw_df, test_size=test_size, random_state=random_state
    )
    print(f"Split complete: train={len(X_train_raw)}, test={len(X_test_raw)}")

    # 4. Clean & engineer financial history features
    X_train_eng = engineer_features(clean_credit_dataframe(X_train_raw))
    X_test_eng = engineer_features(clean_credit_dataframe(X_test_raw))

    # 5. Fit preprocessor strictly on training split
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_eng)
    X_test_proc = preprocessor.transform(X_test_eng)
    feature_names = list(preprocessor.get_feature_names_out())

    save_preprocessed_data(
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        feature_names,
        preprocessor,
        output_path=features_path,
    )

    # 6. Train all candidate classifiers
    candidates = get_candidate_models(random_state=random_state, n_estimators=n_estimators)
    trained_models: Dict[str, Any] = {}
    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    for name, clf in candidates.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_proc, y_train)
        y_pred = clf.predict(X_test_proc)
        y_proba = clf.predict_proba(X_test_proc)
        metrics = compute_classification_metrics(y_test, y_pred, y_proba, class_labels)
        trained_models[name] = clf
        metrics_by_model[name] = metrics
        print(
            f"  -> Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['weighted_f1_score']:.4f}, "
            f"ROC-AUC={metrics['roc_auc_ovr_weighted']:.4f}"
        )

    # 7. Select best classifier and save both all models & end-to-end pipeline
    best_model_name = max(
        metrics_by_model, key=lambda m: metrics_by_model[m]["weighted_f1_score"]
    )
    best_classifier = trained_models[best_model_name]
    inference_pipeline = build_inference_pipeline(preprocessor, best_classifier)

    save_model(trained_models, all_models_path)
    save_model(inference_pipeline, best_model_path)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — Best Model: {best_model_name}")
    print("=" * 60)
    return {
        "best_model_name": best_model_name,
        "metrics": metrics_by_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Credit Scoring classification models.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=150)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
