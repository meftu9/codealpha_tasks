import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def load_ravdess_metadata(dataset_path="dataset/ravdess"):
    """
    Parses RAVDESS directory structure and extracts metadata from filenames.
    """
    data = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    for actor in sorted(os.listdir(dataset_path)):
        actor_path = os.path.join(dataset_path, actor)
        if not os.path.isdir(actor_path):
            continue

        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):
                parts = filename.split(".")[0].split("-")
                if len(parts) >= 7:
                    emotion_code = parts[2]
                    actor_id = int(parts[6])
                    emotion = EMOTION_LABELS.get(emotion_code, "unknown")
                    file_path = os.path.join(actor_path, filename)
                    data.append({
                        "file_path": file_path,
                        "filename": filename,
                        "actor_id": actor_id,
                        "emotion": emotion
                    })

    df = pd.DataFrame(data)
    return df

def extract_mfcc(file_path, target_sr=22050, n_mfcc=40, max_frames=228):
    """
    Loads an audio file and extracts 40-band MFCC feature matrix padded/truncated to max_frames.
    """
    audio, sr = librosa.load(file_path, sr=target_sr)
    # Trim leading and trailing silence
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed_audio) < 1000:
        trimmed_audio = audio

    mfcc = librosa.feature.mfcc(y=trimmed_audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to max_frames
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc

def extract_and_prepare_features(dataset_path="dataset/ravdess", output_path="models/audio_features.npz", test_size=0.2, random_state=42):
    """
    Extracts features for all files, splits with GroupShuffleSplit by actor ID,
    standardizes, and saves the prepared datasets.
    """
    print(f"Loading metadata from: {dataset_path}")
    df = load_ravdess_metadata(dataset_path)
    print(f"Total audio files found: {len(df)}")

    X = []
    y = []

    print("Extracting MFCC features from all audio files...")
    for idx, row in df.iterrows():
        mfcc = extract_mfcc(row["file_path"])
        X.append(mfcc)
        y.append(row["emotion"])
        if (idx + 1) % 200 == 0 or (idx + 1) == len(df):
            print(f"  Processed {idx + 1}/{len(df)} files")

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    groups = df["actor_id"].to_numpy()

    print(f"Emotion classes ({len(classes)}): {list(classes)}")
    print(f"Features shape: {X.shape}")

    # Speaker-independent train/test split (no actor overlap)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y_encoded, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Standardize using training statistics only
    mean = X_train.mean()
    std = X_train.std() + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Add channel dimension for 2D CNN (N, n_mfcc, max_frames, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups=groups,
        train_idx=train_idx,
        test_idx=test_idx,
        classes=classes,
        norm_mean=mean,
        norm_std=std
    )
    print(f"Saved prepared features to: {output_path}")

    # Also save a copy inside notebook/ for notebook compatibility
    os.makedirs("notebook", exist_ok=True)
    np.savez(
        "notebook/audio_features.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        groups=groups,
        train_idx=train_idx,
        test_idx=test_idx,
        classes=classes,
        norm_mean=mean,
        norm_std=std
    )
    print("Saved copy to: notebook/audio_features.npz")

    return X_train, X_test, y_train, y_test, classes

if __name__ == "__main__":
    extract_and_prepare_features()

