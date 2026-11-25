# Handwritten Character & Sentence Recognition ✍️

A comprehensive, production-ready Deep Learning project for recognizing handwritten **numbers (digits 0–9)**, **letters (A–Z / a–z)**, and **complete sentences/words** using an advanced **Convolutional Neural Network (CNN)** trained on **EMNIST Balanced** (47 classes). Includes modular Python source code, sequential Jupyter notebooks, evaluation reports, and an interactive **Streamlit web application**.

---

## 📁 Project Directory Structure

```text
Handwritten_Character_Recognition/
├── .vscode/
│   └── settings.json                  # Editor settings, formatter, and Python interpreter path
├── dataset/
│   ├── raw/
│   │   ├── emnist_balanced.npz        # Cached EMNIST Balanced dataset (112,800 train, 18,800 test)
│   │   └── mnist.npz                  # Cached standard MNIST digits dataset
│   └── samples/                       # Sample handwritten test images (digits 0–9, letters A–Z, sample_sentence.png)
├── models/
│   ├── handwritten_cnn.keras          # Fully trained 47-class CNN model file (Numbers & Letters)
│   ├── label_mapping.json             # Class index to character mapping dictionary (47 classes)
│   └── character_features.npz         # Preprocessed NumPy train and test arrays
├── notebook/
│   ├── 01_data_understanding.ipynb    # Data loading, shape verification, and 47-class distribution plots
│   ├── 02_image_preprocessing.ipynb   # Pixel normalization, channel reshaping, and orientation/inversion handling
│   ├── 03_model_training.ipynb        # CNN architecture definition, compilation, callbacks, and training
│   ├── 04_model_evaluation.ipynb      # Test accuracy/loss evaluation, metrics, and 47-class confusion matrix
│   └── 05_inference_and_testing.ipynb # Custom character inference, probability charts, and sentence recognition
├── reports/
│   ├── training_curves.png            # Graphs of training vs. validation accuracy and loss over epochs
│   ├── confusion_matrix.png           # Confusion matrix heatmap across all 47 character classes
│   └── evaluation_summary.json        # Test accuracy, loss, precision, recall, and per-class metrics
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── preprocessing.py               # EMNIST/MNIST data loading, normalization, reshaping, and single-image pipeline
│   ├── emnist_loader.py               # Stream-extractor for EMNIST from official NIST repository
│   ├── models.py                      # Multi-class CNN architecture definition, compilation, and save/load utils
│   ├── train.py                       # Complete training script supporting both EMNIST and MNIST
│   ├── evaluate.py                    # Evaluation script, classification report, and confusion matrix
│   └── sentence_recognition.py        # Segmentation, spacing detection, and sentence reconstruction
├── temp/
│   ├── current_character.png          # Temporary holding spot for uploaded/drawn character
│   └── annotated_sentence.png         # Annotated sentence visualization with bounding boxes
├── app.py                             # Interactive Streamlit application (Single Char & Sentence Recognition)
├── requirements.txt                   # List of dependencies (TensorFlow, NumPy, SciPy, Streamlit, etc.)
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 Key Features

1. **Both Number and Letter Recognition (47 Classes)**:
   - Recognizes digits `0` through `9`.
   - Recognizes uppercase letters `A` through `Z` and distinctive lowercase letters (`a`, `b`, `d`, `e`, `f`, `g`, `h`, `n`, `q`, `r`, `t`).
   - Trained on the balanced 112,800-sample EMNIST dataset.

2. **Extendable Sentence & Word Recognition**:
   - Accepts multi-character handwritten words or full sentences.
   - Automatically segments individual characters using connected-component bounding box detection.
   - Detects natural spaces between words based on character width ratios.
   - Classifies each segmented character with the CNN and reconstructs the full text string.

3. **High-Performance CNN Architecture**:
   - Dual convolutional blocks with Batch Normalization, Max Pooling, and Dropout.
   - 256-unit fully-connected classification head.
   - Evaluated on 18,800 test images.

4. **Interactive Streamlit Web App**:
   - **Mode 1**: Single Character Recognition with live drawing canvas, file uploader, and sample letter/number selector.
   - **Mode 2**: Sentence & Word Recognition with live bounding box segmentation, text reconstruction, and character breakdown table.

5. **Sequential Jupyter Notebooks**:
   - Clean, five-stage workflow from data understanding and image preprocessing to model training, evaluation, and end-to-end sentence inference.

---

## ⚙️ Installation & Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd Handwritten_Character_Recognition
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Project

### 1. Training the Model
Train the 47-class CNN model on EMNIST (numbers and letters):
```bash
python -m src.train --dataset emnist --epochs 5 --batch_size 128
```
*(To train exclusively on standard 10-class MNIST digits, pass `--dataset mnist`)*

### 2. Evaluating the Model
Evaluate the model on the 18,800 test images:
```bash
python -m src.evaluate
```
This generates:
- `reports/confusion_matrix.png`: Heatmap across all 47 classes.
- `reports/evaluation_summary.json`: Comprehensive precision, recall, and F1-score report.

### 3. Testing Sentence Recognition
To test the sentence recognition module on a sample image:
```bash
python -c "from src.sentence_recognition import recognize_sentence; from src.models import load_trained_model; from src.preprocessing import load_label_mapping; m = load_trained_model('models/handwritten_cnn.keras'); lm = load_label_mapping('models/label_mapping.json'); text, _, _ = recognize_sentence('dataset/samples/sample_sentence.png', m, lm); print('Recognized Text:', text)"
```

### 4. Launching the Interactive Web Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`:
- Toggle between **🔤 Single Character Mode** (numbers 0–9 and letters A–Z) and **📝 Sentence & Word Recognition Mode**.
- Upload custom handwritten images, draw on the canvas, or select bundled sample characters.
- View real-time predictions, bounding box segmentations, and confidence breakdowns.

---

## 📓 Sequential Notebooks Guide

The `notebook/` directory provides five sequential, standalone Jupyter notebooks:

1. [01_data_understanding.ipynb](file:///d:/ML-Projects/Handwritten_Character_Recognition/notebook/01_data_understanding.ipynb):
   - Loading 47-class EMNIST Balanced dataset.
   - Inspecting dataset sizes (112,800 train, 18,800 test) and class distributions.
   - Previewing handwritten numbers and letters in a 4×5 sample grid.

2. [02_image_preprocessing.ipynb](file:///d:/ML-Projects/Handwritten_Character_Recognition/notebook/02_image_preprocessing.ipynb):
   - Pixel normalization ($[0.0, 1.0]$) and channel reshaping (`(N, 28, 28, 1)`).
   - EMNIST column-major orientation handling and automatic background color inversion.
   - Saving features to `models/character_features.npz`.

3. [03_model_training.ipynb](file:///d:/ML-Projects/Handwritten_Character_Recognition/notebook/03_model_training.ipynb):
   - Defining the 47-class CNN architecture with Batch Normalization and Dropout.
   - Compiling with Adam optimizer and training callbacks (`EarlyStopping`, `ReduceLROnPlateau`).
   - Exporting weights to `models/handwritten_cnn.keras` and curves to `reports/training_curves.png`.

4. [04_model_evaluation.ipynb](file:///d:/ML-Projects/Handwritten_Character_Recognition/notebook/04_model_evaluation.ipynb):
   - Test accuracy and loss evaluation on 18,800 test images.
   - Full classification report (precision, recall, F1 per class).
   - Plotting the 47×47 confusion matrix heatmap in `reports/confusion_matrix.png`.

5. [05_inference_and_testing.ipynb](file:///d:/ML-Projects/Handwritten_Character_Recognition/notebook/05_inference_and_testing.ipynb):
   - Single character inference on digits and letters from `dataset/samples/`.
   - Multi-character sentence segmentation, bounding box extraction, and text reconstruction.

---

## 📊 Performance Benchmark (EMNIST Balanced - 47 Classes)

| Metric | Score |
| :--- | :--- |
| **Total Classes** | **47 (0–9 digits, A–Z uppercase, distinctive lowercases)** |
| **Test Accuracy** | **87.74%** |
| **Test Loss** | **0.3572** |
| **Weighted Precision** | **87.74%** |
| **Weighted Recall** | **87.74%** |
| **Weighted F1-Score** | **0.8753** |
| **Test Samples** | 18,800 images |

Detailed per-class metrics can be found in [evaluation_summary.json](file:///d:/ML-Projects/Handwritten_Character_Recognition/reports/evaluation_summary.json).

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x / Keras
- **Dataset**: EMNIST Balanced (Extended MNIST) & MNIST
- **Computer Vision & Segmentation**: SciPy (`scipy.ndimage`) & Pillow (PIL)
- **Numerical Computing**: NumPy
- **Data Visualization**: Matplotlib & Seaborn
- **Metrics**: scikit-learn
- **Frontend / Web UI**: Streamlit
