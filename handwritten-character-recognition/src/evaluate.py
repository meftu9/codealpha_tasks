"""
Evaluation Module for Handwritten Character Recognition.
Loads preprocessed test features and the trained model, computes predictions,
generates a confusion matrix heatmap, and exports evaluation metrics.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf

from src.models import load_trained_model
from src.preprocessing import load_label_mapping, load_preprocessed_data


def evaluate_model(
    model_path: str = "models/handwritten_cnn.keras",
    features_path: str = "models/character_features.npz",
    label_mapping_path: str = "models/label_mapping.json",
    cm_output_path: str = "reports/confusion_matrix.png",
    summary_output_path: str = "reports/evaluation_summary.json",
) -> Dict[str, Any]:
    """
    Evaluates the model on the test dataset, generates confusion matrix plot,
    and saves evaluation metrics summary to JSON.
    """
    print("=" * 60)
    print("STARTING MODEL EVALUATION")
    print("=" * 60)

    # 1. Load data and model
    _, _, x_test, y_test = load_preprocessed_data(features_path)
    model = load_trained_model(model_path)
    label_map = load_label_mapping(label_mapping_path)

    # 2. Evaluate loss and accuracy
    print("Running evaluation on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    # 3. Model predictions
    print("Generating predictions on test set...")
    y_pred_probs = model.predict(x_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    class_labels = [label_map.get(i, str(i)) for i in range(len(label_map))]
    report_dict = classification_report(
        y_test, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)

    fig_size = (14, 12) if len(class_labels) > 20 else (10, 8)
    show_annot = len(class_labels) <= 20
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm,
        annot=show_annot,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
    )
    plt.title(f"Confusion Matrix ({len(class_labels)} Classes - Letters & Digits)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Character Class", fontsize=12)
    plt.ylabel("Actual Character Class", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_output_path}")

    # 6. Evaluation Summary JSON
    summary = {
        "model_file": os.path.basename(model_path),
        "test_samples": int(len(y_test)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "per_class_metrics": {
            label: {
                "precision": report_dict[label]["precision"],
                "recall": report_dict[label]["recall"],
                "f1_score": report_dict[label]["f1-score"],
                "support": int(report_dict[label]["support"]),
            }
            for label in class_labels if label in report_dict
        },
    }

    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary JSON saved to: {summary_output_path}")

    print("=" * 60)
    print(f"Evaluation Complete!")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}% | Test Loss: {test_loss:.4f} | F1: {f1:.4f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Handwritten Character CNN Model")
    parser.add_argument("--model_path", type=str, default="models/handwritten_cnn.keras")
    parser.add_argument("--features_path", type=str, default="models/character_features.npz")
    parser.add_argument("--mapping_path", type=str, default="models/label_mapping.json")
    parser.add_argument("--cm_path", type=str, default="reports/confusion_matrix.png")
    parser.add_argument("--summary_path", type=str, default="reports/evaluation_summary.json")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        features_path=args.features_path,
        label_mapping_path=args.mapping_path,
        cm_output_path=args.cm_path,
        summary_output_path=args.summary_path,
    )
