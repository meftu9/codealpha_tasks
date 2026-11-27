import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition (SER)",
    page_icon="🎙️",
    layout="wide"
)

EMOTION_EMOJIS = {
    "neutral": "😐 Neutral",
    "calm": "😌 Calm",
    "happy": "😊 Happy",
    "sad": "😢 Sad",
    "angry": "😠 Angry",
    "fearful": "😨 Fearful",
    "disgust": "🤢 Disgust",
    "surprised": "😲 Surprised"
}

EMOTION_COLORS = {
    "neutral": "#9E9E9E",
    "calm": "#4CAF50",
    "happy": "#FFEB3B",
    "sad": "#2196F3",
    "angry": "#F44336",
    "fearful": "#9C27B0",
    "disgust": "#795548",
    "surprised": "#FF9800"
}

@st.cache_resource
def load_all_models_and_metadata():
    models = {}
    classes = None
    mean = 0.0
    std = 1.0

    # Load metadata and normalization stats
    feat_paths = ["models/audio_features.npz", "notebook/audio_features.npz"]
    for fp in feat_paths:
        if os.path.exists(fp):
            data = np.load(fp)
            classes = list(data["classes"])
            if "norm_mean" in data:
                mean = float(data["norm_mean"])
                std = float(data["norm_std"])
            break

    if classes is None:
        classes = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    # Model paths
    model_paths = {
        "2D CNN": ["models/audio_emotion_cnn.keras", "notebook/audio_emotion_cnn.keras"],
        "Bi-LSTM": ["models/audio_emotion_lstm.keras", "notebook/audio_emotion_lstm.keras"],
        "CRNN (Hybrid)": ["models/audio_emotion_crnn.keras", "notebook/audio_emotion_crnn.keras"]
    }

    for name, paths in model_paths.items():
        for p in paths:
            if os.path.exists(p):
                try:
                    models[name] = load_model(p)
                    break
                except Exception as e:
                    print(f"Error loading {p}: {e}")

    return models, classes, mean, std

def extract_features(audio, sr, target_sr=22050, n_mfcc=40, max_frames=228, mean=0.0, std=1.0):
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Silence trimming
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed_audio) >= 1000:
        audio = trimmed_audio

    # MFCC extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_frames]

    # Normalize
    mfcc_norm = (mfcc - mean) / (std + 1e-8)

    # Return CNN input (1, 40, 228, 1) and Sequential input (1, 228, 40)
    feat_cnn = mfcc_norm[np.newaxis, ..., np.newaxis]
    feat_seq = mfcc_norm.T[np.newaxis, ...]

    return audio, sr, mfcc, feat_cnn, feat_seq

# App header
st.title("🎙️ Speech Emotion Recognition (SER) System")
st.markdown(
    """
    Recognize human emotional states (**Happy, Sad, Angry, Neutral, Fearful, Disgust, Surprised, Calm**) 
    from speech audio signals using deep learning and speech signal processing (MFCCs).
    """
)

models, classes, norm_mean, norm_std = load_all_models_and_metadata()

# Sidebar
st.sidebar.header("⚙️ Configuration")
if not models:
    st.sidebar.error("⚠️ No trained models found in `models/`. Please run model training first.")
    selected_model_name = None
else:
    selected_model_name = st.sidebar.selectbox("Choose Model Architecture", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Details")
st.sidebar.info(
    "**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)\n\n"
    "**Features:** 40 Mel-Frequency Cepstral Coefficients (MFCCs)\n\n"
    "**Sample Rate:** 22,050 Hz\n\n"
    "**Actors:** 24 Professional Actors (Speaker-Independent Split)"
)

# Main input selection
tab_upload, tab_sample, tab_record, tab_eval = st.tabs(["📁 Upload Audio", "🎵 Sample Library", "🎤 Record Voice", "📊 Model Evaluation"])

audio_bytes = None
audio_source_label = ""

with tab_upload:
    uploaded_file = st.file_uploader("Upload a speech WAV file", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_source_label = f"Uploaded: {uploaded_file.name}"

with tab_sample:
    st.markdown("Choose a verified audio file from the RAVDESS dataset:")
    dataset_dir = "dataset/ravdess"
    sample_options = []
    if os.path.exists(dataset_dir):
        actors = sorted(os.listdir(dataset_dir))
        for actor in actors[:4]: # Check first 4 actors
            actor_path = os.path.join(dataset_dir, actor)
            if os.path.isdir(actor_path):
                for f in sorted(os.listdir(actor_path))[:6]:
                    sample_options.append(os.path.join(actor_path, f))

    if sample_options:
        selected_sample = st.selectbox("Select Sample Audio", sample_options)
        if selected_sample:
            with open(selected_sample, "rb") as f:
                audio_bytes = f.read()
            audio_source_label = f"Sample: {os.path.basename(selected_sample)}"
    else:
        st.write("No sample files found in `dataset/ravdess`.")

with tab_record:
    st.markdown("Record your voice directly using your microphone:")
    recorded_audio = st.audio_input("Record audio")
    if recorded_audio is not None:
        audio_bytes = recorded_audio.read()
        audio_source_label = "Microphone Recording"

with tab_eval:
    st.subheader("Model Performance & Confusion Matrix")
    report_img_cnn = "reports/confusion_matrix_cnn.png"
    report_img_lstm = "reports/confusion_matrix_bilstm.png"
    report_img_crnn = "reports/confusion_matrix_crnn.png"
    curves_cnn = "reports/cnn_training_curve.png"

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if os.path.exists(report_img_cnn):
            st.image(report_img_cnn, caption="2D CNN Confusion Matrix")
        elif os.path.exists(curves_cnn):
            st.image(curves_cnn, caption="CNN Learning Curves")
        else:
            st.info("Evaluation plots will appear once evaluation runs.")
    with col_e2:
        if os.path.exists(report_img_crnn):
            st.image(report_img_crnn, caption="CRNN Hybrid Confusion Matrix")
        elif os.path.exists(report_img_lstm):
            st.image(report_img_lstm, caption="Bi-LSTM Confusion Matrix")

    if os.path.exists("reports/model_comparison.json"):
        import json
        with open("reports/model_comparison.json", "r") as f:
            comp_data = json.load(f)
        st.markdown("### 🏆 Architecture Comparison")
        comp_df = pd.DataFrame(comp_data).T
        comp_df["test_accuracy"] = (comp_df["test_accuracy"] * 100).round(2).astype(str) + "%"
        comp_df["val_accuracy"] = (comp_df["val_accuracy"] * 100).round(2).astype(str) + "%"
        comp_df["test_loss"] = comp_df["test_loss"].round(4)
        comp_df["val_loss"] = comp_df["val_loss"].round(4)
        st.dataframe(comp_df, use_container_width=True)

# Audio processing & prediction section
if audio_bytes is not None:
    st.markdown("---")
    st.subheader(f"🔊 Audio Analysis: {audio_source_label}")
    st.audio(audio_bytes)

    # Save to temporary scratch location for librosa loading
    os.makedirs("temp", exist_ok=True)
    temp_wav_path = "temp/current_audio.wav"
    with open(temp_wav_path, "wb") as f:
        f.write(audio_bytes)

    try:
        y, sr = librosa.load(temp_wav_path, sr=22050)
        duration = len(y) / sr

        # Extract features
        audio_proc, sr_proc, mfcc_feat, feat_cnn, feat_seq = extract_features(
            y, sr, mean=norm_mean, std=norm_std
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📈 Acoustic Visualizations")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))

            # Waveform
            librosa.display.waveshow(y, sr=sr, ax=ax[0], color="#2196F3")
            ax[0].set_title(f"Waveform (Duration: {duration:.2f}s, SR: {sr}Hz)")
            ax[0].set_xlabel("Time (seconds)")
            ax[0].set_ylabel("Amplitude")
            ax[0].grid(True, alpha=0.3)

            # MFCC Heatmap
            img = librosa.display.specshow(mfcc_feat, sr=sr_proc, x_axis="time", ax=ax[1], cmap="magma")
            fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
            ax[1].set_title("40 Mel-Frequency Cepstral Coefficients (MFCCs)")
            ax[1].set_xlabel("Time")
            ax[1].set_ylabel("MFCC Index")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 🎯 Predicted Emotion")
            if selected_model_name and selected_model_name in models:
                active_model = models[selected_model_name]

                # Determine input format
                if "LSTM" in selected_model_name:
                    preds = active_model.predict(feat_seq, verbose=0)[0]
                else:
                    preds = active_model.predict(feat_cnn, verbose=0)[0]

                top_idx = int(np.argmax(preds))
                top_emotion = classes[top_idx]
                top_prob = preds[top_idx] * 100

                emoji_label = EMOTION_EMOJIS.get(top_emotion, top_emotion.capitalize())

                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; border-radius: 12px; padding: 20px; text-align: center; border-left: 8px solid #4CAF50; margin-bottom: 20px;">
                        <h1 style="margin: 0; font-size: 3rem;">{emoji_label}</h1>
                        <h3 style="margin-top: 10px; color: #333;">Confidence: <strong>{top_prob:.1f}%</strong></h3>
                        <p style="color: #666; margin: 0;">Evaluated via <b>{selected_model_name}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Probability Distribution Chart
                st.markdown("#### Confidence Breakdown Across All Emotions")
                prob_df = pd.DataFrame({
                    "Emotion": [EMOTION_EMOJIS.get(c, c.capitalize()) for c in classes],
                    "Confidence (%)": [round(float(p) * 100, 2) for p in preds]
                }).sort_values(by="Confidence (%)", ascending=True)

                st.bar_chart(prob_df.set_index("Emotion"))
            else:
                st.warning("Please train models or ensure saved models are placed in the `models/` directory.")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")

