# Credit Scoring Model 💳

A clean, modular Machine Learning project for predicting an individual's **creditworthiness** (**Poor**, **Standard**, or **Good**) from past financial history using **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers. Built with modular Python source code, sequential Jupyter notebooks, automated evaluation reports, and an interactive **Streamlit web application**.

---

## 📁 Project Directory Structure

```text
Credit-scoring/
├── .vscode/
│   └── settings.json                                  # Editor settings and .venv interpreter path
├── dataset/
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── train.csv                                  # Raw financial dataset (100,000 records, 28 columns)
│   └── samples/
│       ├── sample_applicants.csv                      # Representative Poor / Standard / Good profiles (CSV)
│       └── sample_applicants.json                     # Representative sample applicant profiles (JSON)
├── models/
│   ├── credit_scoring_model.joblib                    # Best end-to-end inference pipeline (Random Forest)
│   ├── all_models.joblib                              # Trained Logistic Regression, Decision Tree, and Random Forest
│   ├── credit_features.joblib                         # Preprocessed train/test matrices (80,000 / 20,000) & preprocessor
│   └── label_mapping.json                             # Class index to label mapping (0: Poor, 1: Standard, 2: Good)
├── notebook/
│   ├── 01_data_understanding.ipynb                    # Schema, class distribution, and data quality audit
│   ├── 02_feature_engineering_and_preprocessing.ipynb # Cleaning, financial ratios, and group-aware split
│   ├── 03_model_training.ipynb                        # Training Logistic Regression, Decision Tree, and Random Forest
│   ├── 04_model_evaluation.ipynb                      # Precision, Recall, F1, ROC-AUC, confusion matrix, and plots
│   └── 05_inference_and_testing.ipynb                 # Sample and custom applicant credit score inference
├── reports/
│   ├── model_comparison.png                           # Bar chart comparing all 3 classifiers across metrics
│   ├── confusion_matrix.png                           # Confusion matrix heatmap for the best model
│   ├── roc_curves.png                                 # One-vs-Rest multi-class ROC curves
│   ├── feature_importance.png                         # Top 15 most predictive raw and engineered features
│   └── evaluation_summary.json                        # Full test metrics and per-class report
├── src/
│   ├── __init__.py                                    # Package initializer
│   ├── data_loader.py                                 # Raw CSV loader, label mapping, and sample exporter
│   ├── preprocessing.py                               # Noise cleaning, bounds validation, group split, ColumnTransformer
│   ├── feature_engineering.py                         # Financial ratio engineering & sklearn Pipeline transformer
│   ├── models.py                                      # Logistic Regression, Decision Tree, Random Forest definitions
│   ├── train.py                                       # End-to-end training CLI script
│   ├── evaluate.py                                    # Evaluation CLI script and report generator
│   └── predict.py                                     # Single & batch inference CLI and Python API
├── temp/
│   ├── .gitkeep
│   └── current_applicant.json                         # Temporary payload buffer for latest applicant test
├── .gitignore
├── app.py                                             # Interactive Streamlit application
├── requirements.txt                                   # Project dependencies
└── README.md                                          # Project documentation
```

---

## 🚀 Key Features

1. **Leakage-Free Group Splitting**:
   - Uses `GroupShuffleSplit` on `Customer_ID` (80,000 train / 20,000 test) so multi-month records for the same individual never leak across train and test sets.
2. **Robust Financial Data Cleaning**:
   - Strips trailing underscore noise (`28_`, `1328.93_`), normalizes corrupted placeholders (`_______`, `!@9#%8`, `_`, `__10000__`, `__-333...__`), enforces realistic domain bounds, and parses `Credit_History_Age` into `Credit_History_Months`.
3. **Domain Feature Engineering (`src/feature_engineering.py`)**:
   - `Debt_to_Income_Ratio`
   - `EMI_to_Salary_Ratio`
   - `Savings_to_Salary_Ratio`
   - `Investment_to_Salary_Ratio`
   - `Credit_Accounts_Total`
   - `Delayed_Payment_Rate`
   - `Inquiries_per_Account`
   - `Delay_Severity_Score`
   - `Spending_Level` & `Payment_Value_Size` (parsed from `Payment_Behaviour`)
4. **Multi-Model Comparison & Evaluation**:
   - Evaluates **Logistic Regression**, **Decision Tree**, and **Random Forest** on **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **One-vs-Rest ROC-AUC**.

---

## 📊 Observed Test Results (20,000 Held-Out Records)

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OvR Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6519 | 0.6891 | 0.6519 | 0.6541 | 0.7852 |
| **Decision Tree** | 0.6586 | 0.6981 | 0.6586 | 0.6611 | 0.7984 |
| **Random Forest (Best)** | **0.6868** | **0.7268** | **0.6868** | **0.6884** | **0.8409** |

---

## ⚙️ Installation & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train all models**:
   ```bash
   python -m src.train
   ```

3. **Evaluate and generate reports**:
   ```bash
   python -m src.evaluate
   ```

4. **Run CLI inference on sample profiles**:
   ```bash
   python -m src.predict --input dataset/samples/sample_applicants.json
   ```

5. **Launch the Streamlit web app**:
   ```bash
   streamlit run app.py
   ```
