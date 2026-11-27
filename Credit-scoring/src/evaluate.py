"""
Evaluation Module for Credit Scoring Model.
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC (One-vs-Rest) across
Logistic Regression, Decision Tree, and Random Forest models, and generates
evaluation plots and JSON reports.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.data_loader import load_label_mapping
from src.models import load_trained_model
from src.preprocessing import load_preprocessed_data


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[str],
) -> Dict[str, Any]:
    """
    Computes Accuracy, Precision, Recall, F1-Score (weighted & macro),
    and multi-class One-vs-Rest ROC-AUC metrics.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    roc_auc_weighted = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    )
    roc_auc_macro = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    )

    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "weighted_precision": float(w_prec),
        "weighted_recall": float(w_rec),
        "weighted_f1_score": float(w_f1),
        "macro_precision": float(m_prec),
        "macro_recall": float(m_rec),
        "macro_f1_score": float(m_f1),
        "roc_auc_ovr_weighted": roc_auc_weighted,
        "roc_auc_ovr_macro": roc_auc_macro,
        "per_class_report": report,
    }


def plot_model_comparison(comparison_rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    Plots a grouped bar chart comparing candidate classifiers across key metrics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cmp = pd.DataFrame(comparison_rows)
    metrics = [
        ("accuracy", "Accuracy"),
        ("weighted_precision", "Precision"),
        ("weighted_recall", "Recall"),
        ("weighted_f1_score", "F1-Score"),
        ("roc_auc_ovr_weighted", "ROC-AUC"),
    ]

    melted = df_cmp.melt(
        id_vars="model",
        value_vars=[m[0] for m in metrics],
        var_name="Metric",
        value_name="Score",
    )
    label_lookup = dict(metrics)
    melted["Metric"] = melted["Metric"].map(label_lookup)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="model", palette="viridis")
    plt.ylim(0.0, 1.05)
    plt.title("Credit Scoring Model Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Classifier", loc="lower right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Model comparison plot saved to: {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str],
    model_name: str,
    output_path: str,
) -> None:
    """
    Generates and saves a confusion matrix heatmap for the selected model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Credit Score", fontsize=12)
    plt.ylabel("Actual Credit Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[str],
    model_name: str,
    output_path: str,
) -> None:
    """
    Plots One-vs-Rest ROC curves for each credit score class.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = list(range(len(class_labels)))
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    for idx, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[idx % len(colors)],
            lw=2,
            label=f"{label} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"One-vs-Rest ROC Curves — {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC curves plot saved to: {output_path}")


def plot_feature_importance(
    classifier: Any,
    feature_names: List[str],
    model_name: str,
    output_path: str,
    top_n: int = 15,
) -> List[Dict[str, float]]:
    """
    Extracts and plots the top-N feature importances (or absolute coefficients)
    for model interpretability.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        return []

    top_idx = np.argsort(importances)[-top_n:]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_vals, color="#2b6cb0", edgecolor="black")
    plt.xlabel("Importance Score", fontsize=12)
    plt.title(f"Top {len(top_names)} Financial Features — {model_name}", fontsize=14, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to: {output_path}")

    return [
        {"feature": name, "importance": float(val)}
        for name, val in zip(reversed(top_names), reversed(top_vals))
    ]


def evaluate_models(
    features_path: str = "models/credit_features.joblib",
    all_models_path: str = "models/all_models.joblib",
    label_mapping_path: str = "models/label_mapping.json",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Evaluates all trained candidate models on the held-out test set, selects the
    best model by weighted F1-score, generates visualizations, and exports a
    comprehensive JSON evaluation summary.
    """
    print("=" * 60)
    print("STARTING CREDIT SCORING MODEL EVALUATION")
    print("=" * 60)

    data = load_preprocessed_data(features_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = list(data["feature_names"])

    all_models: Dict[str, Any] = load_trained_model(all_models_path)
    label_map = load_label_mapping(label_mapping_path)
    class_labels = [label_map[i] for i in sorted(label_map.keys())]

    results: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []
    predictions_cache: Dict[str, tuple] = {}

    for name, clf in all_models.items():
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        metrics = compute_classification_metrics(y_test, y_pred, y_proba, class_labels)
        results[name] = metrics
        predictions_cache[name] = (y_pred, y_proba)
        comparison_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1_score": metrics["weighted_f1_score"],
                "roc_auc_ovr_weighted": metrics["roc_auc_ovr_weighted"],
            }
        )
        print(
            f"[{name}] Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['weighted_precision']:.4f} | "
            f"Rec={metrics['weighted_recall']:.4f} | "
            f"F1={metrics['weighted_f1_score']:.4f} | "
            f"ROC-AUC={metrics['roc_auc_ovr_weighted']:.4f}"
        )

    best_model_name = max(results, key=lambda m: results[m]["weighted_f1_score"])
    best_pred, best_proba = predictions_cache[best_model_name]
    best_clf = all_models[best_model_name]

    print(f"\nBest Performing Model: {best_model_name}")
    print(classification_report(y_test, best_pred, target_names=class_labels, zero_division=0))

    plot_model_comparison(comparison_rows, os.path.join(reports_dir, "model_comparison.png"))
    plot_confusion_matrix(
        y_test,
        best_pred,
        class_labels,
        best_model_name,
        os.path.join(reports_dir, "confusion_matrix.png"),
    )
    plot_roc_curves(
        y_test,
        best_proba,
        class_labels,
        best_model_name,
        os.path.join(reports_dir, "roc_curves.png"),
    )
    top_features = plot_feature_importance(
        best_clf,
        feature_names,
        best_model_name,
        os.path.join(reports_dir, "feature_importance.png"),
    )

    summary = {
        "best_model": best_model_name,
        "test_samples": int(len(y_test)),
        "num_features": int(X_test.shape[1]),
        "classes": class_labels,
        "models": results,
        "top_features": top_features,
    }

    summary_path = os.path.join(reports_dir, "evaluation_summary.json")
    os.makedirs(reports_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to: {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained credit scoring models.")
    parser.add_argument("--features-path", type=str, default="models/credit_features.joblib")
    parser.add_argument("--all-models-path", type=str, default="models/all_models.joblib")
    parser.add_argument("--label-mapping-path", type=str, default="models/label_mapping.json")
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    evaluate_models(
        features_path=args.features_path,
        all_models_path=args.all_models_path,
        label_mapping_path=args.label_mapping_path,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
