import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

def evaluate_models(features_path="models/audio_features.npz"):
    if not os.path.exists(features_path):
        features_path = "notebook/audio_features.npz"
    if not os.path.exists(features_path):
        raise FileNotFoundError("audio_features.npz not found.")

    data = np.load(features_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]

    # Prepared formats
    X_test_2d = X_test # (N, 40, 228, 1)
    X_test_seq = X_test.squeeze(-1).transpose(0, 2, 1) # (N, 228, 40)

    os.makedirs("reports", exist_ok=True)

    models_to_eval = [
        ("CNN", "models/audio_emotion_cnn.keras", X_test_2d),
        ("Bi-LSTM", "models/audio_emotion_lstm.keras", X_test_seq),
        ("CRNN", "models/audio_emotion_crnn.keras", X_test_2d)
    ]

    all_reports = {}

    for name, model_file, test_features in models_to_eval:
        if not os.path.exists(model_file):
            # check in notebook/
            alt_path = os.path.join("notebook", os.path.basename(model_file))
            if os.path.exists(alt_path):
                model_file = alt_path
            else:
                print(f"Skipping {name}: model file {model_file} not found.")
                continue

        print("\n" + "="*50)
        print(f"EVALUATING {name} MODEL: {model_file}")
        print("="*50)
        model = load_model(model_file)

        loss, acc = model.evaluate(test_features, y_test, verbose=0)
        print(f"Test Accuracy: {acc * 100:.2f}%")
        print(f"Test Loss:     {loss:.4f}")

        y_pred = np.argmax(model.predict(test_features, verbose=0), axis=1)

        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        report_text = classification_report(y_test, y_pred, target_names=classes)
        print("\nClassification Report:\n", report_text)

        all_reports[name] = {
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "report": report
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        plt.title(f"Speech Emotion Recognition - {name} Confusion Matrix")
        plt.tight_layout()
        cm_path = f"reports/confusion_matrix_{name.lower().replace('-', '_')}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")

    with open("reports/evaluation_summary.json", "w") as f:
        json.dump(all_reports, f, indent=4)

    return all_reports

if __name__ == "__main__":
    evaluate_models()

