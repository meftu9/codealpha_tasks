"""
Training Script for Handwritten Character Recognition.
Loads preprocessed data, builds and compiles the CNN model, sets up callbacks,
trains the model, and saves the final weights and training curve report.
"""

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.models import build_cnn_model, compile_model, save_model
from src.preprocessing import (
    create_default_mapping,
    load_raw_data,
    preprocess_pipeline,
    save_label_mapping,
    save_preprocessed_data,
)


def plot_training_curves(history: tf.keras.callbacks.History, output_path: str) -> None:
    """
    Plots training vs. validation accuracy and loss curves over epochs and saves as PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy Plot
    ax1.plot(epochs, history.history["accuracy"], "bo-", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history.history:
        ax1.plot(epochs, history.history["val_accuracy"], "r^-", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs, history.history["loss"], "bo-", label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        ax2.plot(epochs, history.history["val_loss"], "r^-", label="Validation Loss", linewidth=2)
    ax2.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {output_path}")


def train(
    dataset: str = "emnist",
    epochs: int = 5,
    batch_size: int = 128,
    validation_split: float = 0.1,
    learning_rate: float = 0.001,
    raw_data_path: Optional[str] = None,
    features_path: str = "models/character_features.npz",
    model_path: str = "models/handwritten_cnn.keras",
    label_mapping_path: str = "models/label_mapping.json",
    reports_curve_path: str = "reports/training_curves.png",
) -> tf.keras.callbacks.History:
    """
    Orchestrates the complete training workflow for EMNIST (letters + numbers) or MNIST.
    """
    print("=" * 60)
    print(f"STARTING HANDWRITTEN CHARACTER RECOGNITION TRAINING ({dataset.upper()})")
    print("=" * 60)

    if raw_data_path is None:
        raw_data_path = "dataset/raw/emnist_balanced.npz" if dataset == "emnist" else "dataset/raw/mnist.npz"

    # 1. Load raw dataset
    (x_train_raw, y_train), (x_test_raw, y_test) = load_raw_data(raw_data_path, dataset_type=dataset)

    # 2. Preprocess data
    print("Preprocessing images (normalizing and reshaping)...")
    x_train, x_test = preprocess_pipeline(x_train_raw, x_test_raw)

    # 3. Save preprocessed arrays
    save_preprocessed_data(x_train, y_train, x_test, y_test, features_path)

    # 4. Save label mapping reference
    num_classes = len(np.unique(y_train))
    mapping = create_default_mapping(num_classes)
    save_label_mapping(mapping, label_mapping_path)
    print(f"Detected {num_classes} distinct character classes (digits and letters).")

    # 5. Build and compile model
    print("Building and compiling CNN architecture...")
    model = build_cnn_model(input_shape=(28, 28, 1), num_classes=num_classes)
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()

    # 6. Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # 7. Model Training
    print(f"\nTraining model for up to {epochs} epochs (batch size: {batch_size})...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Save model
    save_model(model, model_path)

    # 9. Plot and save training curves
    plot_training_curves(history, reports_curve_path)

    # 10. Initial test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("=" * 60)
    print(f"Training Complete! Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Handwritten Character CNN Model")
    parser.add_argument("--dataset", type=str, default="emnist", choices=["emnist", "mnist"], help="Dataset (emnist for letters+numbers, mnist for numbers)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw dataset")
    parser.add_argument("--features_path", type=str, default="models/character_features.npz", help="Path to save features")
    parser.add_argument("--model_path", type=str, default="models/handwritten_cnn.keras", help="Path to save model")
    parser.add_argument("--mapping_path", type=str, default="models/label_mapping.json", help="Path to save label mapping")
    parser.add_argument("--reports_path", type=str, default="reports/training_curves.png", help="Path to save curves")

    args = parser.parse_args()
    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        raw_data_path=args.data_path,
        features_path=args.features_path,
        model_path=args.model_path,
        label_mapping_path=args.mapping_path,
        reports_curve_path=args.reports_path,
    )
