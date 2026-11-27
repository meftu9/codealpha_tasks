"""
Interactive Web Application for Handwritten Character and Sentence Recognition.
Supports both Number Recognition (0-9) and Letter Recognition (A-Z, a-z),
as well as Multi-Character Sentence Recognition with automatic segmentation and space detection.
"""

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from src.models import load_trained_model
from src.preprocessing import load_label_mapping, preprocess_single_image
from src.sentence_recognition import (
    create_sample_sentence_image,
    recognize_sentence,
    segment_characters,
)

import importlib

# Optional canvas support
try:
    _canvas_module = importlib.import_module("streamlit_drawable_canvas")
    st_canvas = getattr(_canvas_module, "st_canvas", None)
    CANVAS_AVAILABLE = st_canvas is not None
except Exception:
    st_canvas = None
    CANVAS_AVAILABLE = False


# Page Configuration
st.set_page_config(
    page_title="Handwritten Character & Sentence Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants & Paths
MODEL_PATH = "models/handwritten_cnn.keras"
LABEL_MAPPING_PATH = "models/label_mapping.json"
TEMP_IMAGE_PATH = "temp/current_character.png"
SAMPLES_DIR = "dataset/samples"
REPORTS_DIR = "reports"


@st.cache_resource
def get_model(path: str = MODEL_PATH):
    """Loads and caches the trained CNN model."""
    if os.path.exists(path):
        return load_trained_model(path)
    return None


@st.cache_data
def get_label_mapping(path: str = LABEL_MAPPING_PATH) -> Dict[int, str]:
    """Loads and caches the label mapping."""
    return load_label_mapping(path)


@st.cache_data
def get_evaluation_summary() -> Optional[dict]:
    """Loads evaluation summary if available."""
    summary_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_top_probabilities(probs: np.ndarray, label_map: Dict[int, str], top_k: int = 8):
    """Renders a clean horizontal bar chart of the top-k class probabilities."""
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = [probs[i] * 100 for i in top_indices]
    top_labels = [f"'{label_map.get(i, str(i))}' (class {i})" for i in top_indices]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    y_pos = np.arange(len(top_labels))
    colors = ["#2ca02c" if i == 0 else "#1f77b4" for i in range(len(top_labels))]

    bars = ax.barh(y_pos, top_probs, color=colors, edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Highest probability on top
    ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_xlim([0, 105])
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, p in zip(bars, top_probs):
        if p > 1.0:
            ax.text(
                bar.get_width() + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold" if bar == bars[0] else "normal",
            )

    plt.tight_layout()
    return fig


def main():
    # Header Section
    st.title("✍️ Handwritten Character & Sentence Recognition")
    st.markdown(
        "A multi-purpose deep learning system trained on **EMNIST Balanced** (47 classes: Digits 0–9 and Letters A–Z). "
        "Supports **Single Character Recognition** and **Multi-Character Sentence/Word Recognition** with automated segmentation."
    )

    # Sidebar - Model Info & Diagnostics
    st.sidebar.header("📊 Model Diagnostics & Metrics")
    model = get_model()
    label_map = get_label_mapping()
    summary = get_evaluation_summary()

    if model is not None:
        st.sidebar.success(f"✅ Model Loaded: `handwritten_cnn.keras` ({len(label_map)} Classes)")
        if summary:
            col_m1, col_m2 = st.sidebar.columns(2)
            col_m1.metric("Test Accuracy", f"{summary.get('test_accuracy', 0)*100:.2f}%")
            col_m2.metric("Test Loss", f"{summary.get('test_loss', 0):.4f}")
            st.sidebar.metric("Weighted F1-Score", f"{summary.get('weighted_f1_score', 0):.4f}")
    else:
        st.sidebar.error("⚠️ Trained model not found. Please run `python -m src.train`.")

    # Sidebar Reports Viewer
    st.sidebar.subheader("📈 Training & Evaluation Reports")
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    tc_path = os.path.join(REPORTS_DIR, "training_curves.png")

    with st.sidebar.expander("📉 View Training Curves"):
        if os.path.exists(tc_path):
            st.image(tc_path, caption="Accuracy & Loss Curves", use_container_width=True)
        else:
            st.info("No training curves found in `reports/`.")

    with st.sidebar.expander("🔍 View Confusion Matrix"):
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix ({len(label_map)} Classes)", use_container_width=True)
        else:
            st.info("No confusion matrix found in `reports/`.")

    # Main Navigation Mode
    mode = st.radio(
        "Select Recognition Mode:",
        ["🔤 Single Character (Digits & Letters)", "📝 Sentence & Word Recognition (Extendable Pipeline)"],
        horizontal=True,
    )

    # =========================================================================
    # MODE 1: Single Character Recognition
    # =========================================================================
    if mode.startswith("🔤"):
        st.markdown("---")
        tab_titles = ["📁 Upload Character", "🖼️ Sample Characters (Digits & Letters)"]
        if CANVAS_AVAILABLE:
            tab_titles.append("✏️ Draw Canvas")

        tabs = st.tabs(tab_titles)
        input_image = None
        source_name = ""

        # Tab 1: Upload
        with tabs[0]:
            st.subheader("Upload an Individual Handwritten Letter or Digit")
            uploaded_file = st.file_uploader(
                "Choose an image file (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                key="single_file_uploader",
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                source_name = uploaded_file.name

        # Tab 2: Sample Characters
        with tabs[1]:
            st.subheader("Select from Sample Digits & Letters")
            sample_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "sample_*.png")))
            # Exclude sentence sample from single character picker
            char_samples = [p for p in sample_files if not p.endswith("sample_sentence.png")]

            if char_samples:
                # Group by Digits vs Letters for quick selection
                digit_samples = [os.path.basename(p) for p in char_samples if any(c.isdigit() for c in os.path.basename(p))]
                letter_samples = [os.path.basename(p) for p in char_samples if not any(c.isdigit() for c in os.path.basename(p))]

                col_sel1, col_sel2 = st.columns(2)
                with col_sel1:
                    chosen_digit = st.selectbox("Sample Digits (0–9):", digit_samples, key="sel_digit")
                with col_sel2:
                    chosen_letter = st.selectbox("Sample Letters (A–Z):", letter_samples, key="sel_letter")

                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("Load Digit Sample", key="btn_load_digit"):
                    input_image = Image.open(os.path.join(SAMPLES_DIR, chosen_digit))
                    source_name = chosen_digit
                elif col_btn2.button("Load Letter Sample", key="btn_load_letter"):
                    input_image = Image.open(os.path.join(SAMPLES_DIR, chosen_letter))
                    source_name = chosen_letter
                elif uploaded_file is None:
                    # Default selection
                    input_image = Image.open(os.path.join(SAMPLES_DIR, chosen_digit))
                    source_name = chosen_digit
            else:
                st.info("No sample images found in `dataset/samples/`.")

        # Tab 3: Canvas (if library available)
        if CANVAS_AVAILABLE:
            with tabs[2]:
                st.subheader("Draw a Character on Canvas")
                stroke_width = st.slider("Brush stroke width", 10, 35, 20)
                canvas_result = st_canvas(
                    fill_color="#000000",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="single_canvas",
                )
                if canvas_result.image_data is not None:
                    drawn_array = canvas_result.image_data.astype("uint8")
                    if drawn_array[:, :, :3].sum() > 0:
                        input_image = Image.fromarray(drawn_array)
                        source_name = "Drawn on Canvas"

        # Prediction Display
        if input_image is not None and model is not None:
            st.markdown("---")
            st.subheader("🔬 Image Preprocessing & Real-Time Prediction")

            # Save current character to temp
            os.makedirs(os.path.dirname(TEMP_IMAGE_PATH), exist_ok=True)
            input_image.save(TEMP_IMAGE_PATH)

            # Preprocess
            tensor, processed_pil = preprocess_single_image(input_image)

            # Layout: Images on Left, Prediction on Right
            col_img, col_proc, col_pred = st.columns([1, 1, 1.6])

            with col_img:
                st.markdown(f"**Original Input** (`{source_name}`)")
                st.image(input_image, use_container_width=True)

            with col_proc:
                st.markdown("**Processed (28×28 Tensor)**")
                st.image(processed_pil, use_container_width=True, clamp=True)
                st.caption("Normalized to [0, 1], centered, and color-inverted if drawn on light paper.")

            # Inference
            probs = model.predict(tensor, verbose=0)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class] * 100)
            predicted_char = label_map.get(pred_class, str(pred_class))
            char_type = "Digit" if predicted_char.isdigit() else "Letter"

            with col_pred:
                st.markdown(f"**Model Output ({char_type})**")
                st.metric(
                    label=f"Predicted {char_type}",
                    value=f"'{predicted_char}'",
                    delta=f"{confidence:.2f}% Confidence",
                )

                # Top 3 Candidates
                top3_idx = np.argsort(probs)[-3:][::-1]
                st.markdown("**Top-3 Candidates:**")
                for rank, idx in enumerate(top3_idx, 1):
                    char = label_map.get(idx, str(idx))
                    prob_val = probs[idx] * 100
                    st.write(f"**{rank}.** Character `{char}` — **{prob_val:.2f}%**")

            # Top Probabilities Chart
            st.markdown("### 📊 Top Class Confidence Distribution")
            fig = plot_top_probabilities(probs, label_map, top_k=8)
            st.pyplot(fig)
            plt.close(fig)

    # =========================================================================
    # MODE 2: Sentence & Word Recognition (Extendable Architecture)
    # =========================================================================
    else:
        st.markdown("---")
        st.subheader("📝 Handwritten Sentence & Word Recognition")
        st.markdown(
            "This module demonstrates the extensible pipeline for recognizing full words and sentences. "
            "It automatically segments individual characters, detects spacing between words, classifies each letter/digit, "
            "and reconstructs the complete sentence string."
        )

        sent_col1, sent_col2 = st.columns([1, 1])

        sentence_image = None
        sent_source = ""

        with sent_col1:
            sent_option = st.radio(
                "Choose Sentence Source:",
                ["🖼️ Load Sample Sentence ('AI 2026')", "📁 Upload Sentence Image", "✍️ Synthesize Custom Text"],
            )

        with sent_col2:
            if sent_option.startswith("🖼️"):
                sample_sent_path = os.path.join(SAMPLES_DIR, "sample_sentence.png")
                if not os.path.exists(sample_sent_path):
                    create_sample_sentence_image("AI 2026", samples_dir=SAMPLES_DIR, output_path=sample_sent_path)
                sentence_image = Image.open(sample_sent_path)
                sent_source = "Sample Sentence ('AI 2026')"

            elif sent_option.startswith("📁"):
                sent_upload = st.file_uploader(
                    "Upload a handwritten sentence/word image:",
                    type=["png", "jpg", "jpeg"],
                    key="sent_file_uploader",
                )
                if sent_upload is not None:
                    sentence_image = Image.open(sent_upload)
                    sent_source = sent_upload.name

            elif sent_option.startswith("✍️"):
                custom_text = st.text_input("Enter text to synthesize from handwritten samples:", "HELLO 2026")
                if st.button("Synthesize Handwritten Sentence Image"):
                    sentence_image = create_sample_sentence_image(custom_text, samples_dir=SAMPLES_DIR)
                    sent_source = f"Synthesized: '{custom_text}'"

        if sentence_image is not None and model is not None:
            st.markdown("---")
            st.subheader("🔍 Segmentation & Sentence Reconstruction")

            # Run sentence recognition pipeline
            with st.spinner("Segmenting characters and running recognition..."):
                recognized_text, char_details, annotated_img = recognize_sentence(
                    sentence_image, model, label_map
                )

            # Display side-by-side
            c_orig, c_annot = st.columns(2)
            with c_orig:
                st.markdown(f"**Original Sentence Image** (`{sent_source}`)")
                st.image(sentence_image, use_container_width=True)

            with c_annot:
                st.markdown("**Segmented Character Bounding Boxes**")
                st.image(annotated_img, use_container_width=True)
                st.caption("Green boxes show detected character contours. Yellow labels show CNN predictions.")

            # Recognized Result Banner
            st.success(f"### 🎉 Recognized Sentence: **`{recognized_text}`**")

            # Detailed character breakdown table
            if char_details:
                st.markdown("#### 📋 Character-by-Character Recognition Breakdown")
                table_data = []
                for i, c in enumerate(char_details, 1):
                    table_data.append({
                        "Char #": i,
                        "Predicted Character": f"'{c['character']}'",
                        "Confidence": f"{c['confidence']:.2f}%",
                        "Bounding Box (X1, Y1, X2, Y2)": str(c["bbox"]),
                        "Space After": "Yes" if c["space_after"] else "No",
                    })
                st.dataframe(table_data, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("Handwritten Character & Sentence Recognition System | EMNIST & MNIST Deep Learning Pipeline")


if __name__ == "__main__":
    main()
