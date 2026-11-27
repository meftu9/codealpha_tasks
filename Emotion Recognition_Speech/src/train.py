import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models import build_cnn_model, build_lstm_model, build_crnn_model
from preprocessing import extract_and_prepare_features

def plot_history(history, model_name, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", lw=2)
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", lw=2)
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curve to {save_path}")

def train_all_models(features_path="models/audio_features.npz", epochs=50, batch_size=32):
    if not os.path.exists(features_path):
        print(f"Features file {features_path} not found. Running feature extraction...")
        extract_and_prepare_features()

    data = np.load(features_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    groups = data["groups"]
    train_idx = data["train_idx"]
    classes = data["classes"]

    # Split train into train_sub and validation using GroupShuffleSplit (speaker-independent)
    train_groups = groups[train_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sub_train_idx, val_idx = next(splitter.split(X_train, y_train, train_groups))

    X_train_sub = X_train[sub_train_idx]
    y_train_sub = y_train[sub_train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    print(f"Dataset split summary:")
    print(f"  Train:      {X_train_sub.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
    print(f"  Num Classes: {len(classes)} ({list(classes)})")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    results = {}

    # 1. Train CNN Model
    print("\n" + "="*50)
    print("1. TRAINING 2D CNN MODEL")
    print("="*50)
    cnn = build_cnn_model(input_shape=X_train_sub.shape[1:], num_classes=len(classes))
    cnn_history = cnn.fit(
        X_train_sub, y_train_sub,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    cnn.save("models/audio_emotion_cnn.keras")
    cnn.save("notebook/audio_emotion_cnn.keras")
    plot_history(cnn_history, "CNN Model", "reports/cnn_training_curve.png")

    val_loss_cnn, val_acc_cnn = cnn.evaluate(X_val, y_val, verbose=0)
    test_loss_cnn, test_acc_cnn = cnn.evaluate(X_test, y_test, verbose=0)
    results["CNN"] = {
        "val_accuracy": float(val_acc_cnn),
        "val_loss": float(val_loss_cnn),
        "test_accuracy": float(test_acc_cnn),
        "test_loss": float(test_loss_cnn)
    }

    # 2. Train Bi-LSTM Model
    print("\n" + "="*50)
    print("2. TRAINING BI-DIRECTIONAL LSTM MODEL")
    print("="*50)
    # Shape transformation for LSTM: (N, 40, 228, 1) -> (N, 228, 40)
    X_train_sub_seq = X_train_sub.squeeze(-1).transpose(0, 2, 1)
    X_val_seq = X_val.squeeze(-1).transpose(0, 2, 1)
    X_test_seq = X_test.squeeze(-1).transpose(0, 2, 1)

    lstm = build_lstm_model(input_shape=X_train_sub_seq.shape[1:], num_classes=len(classes))
    lstm_history = lstm.fit(
        X_train_sub_seq, y_train_sub,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    lstm.save("models/audio_emotion_lstm.keras")
    lstm.save("notebook/audio_emotion_lstm.keras")
    plot_history(lstm_history, "Bi-LSTM Model", "reports/lstm_training_curve.png")

    val_loss_lstm, val_acc_lstm = lstm.evaluate(X_val_seq, y_val, verbose=0)
    test_loss_lstm, test_acc_lstm = lstm.evaluate(X_test_seq, y_test, verbose=0)
    results["Bi-LSTM"] = {
        "val_accuracy": float(val_acc_lstm),
        "val_loss": float(val_loss_lstm),
        "test_accuracy": float(test_acc_lstm),
        "test_loss": float(test_loss_lstm)
    }

    # 3. Train CRNN Hybrid Model
    print("\n" + "="*50)
    print("3. TRAINING CRNN (CNN + Bi-LSTM) HYBRID MODEL")
    print("="*50)
    crnn = build_crnn_model(input_shape=X_train_sub.shape[1:], num_classes=len(classes))
    crnn_history = crnn.fit(
        X_train_sub, y_train_sub,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    crnn.save("models/audio_emotion_crnn.keras")
    crnn.save("notebook/audio_emotion_crnn.keras")
    plot_history(crnn_history, "CRNN Model", "reports/crnn_training_curve.png")

    val_loss_crnn, val_acc_crnn = crnn.evaluate(X_val, y_val, verbose=0)
    test_loss_crnn, test_acc_crnn = crnn.evaluate(X_test, y_test, verbose=0)
    results["CRNN"] = {
        "val_accuracy": float(val_acc_crnn),
        "val_loss": float(val_loss_crnn),
        "test_accuracy": float(test_acc_crnn),
        "test_loss": float(test_loss_crnn)
    }

    # Save training comparison summary
    with open("reports/model_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*50)
    print("TRAINING COMPARISON SUMMARY:")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name:10s} | Test Acc: {metrics['test_accuracy']*100:.2f}% | Test Loss: {metrics['test_loss']:.4f}")

    return results

if __name__ == "__main__":
    train_all_models()

