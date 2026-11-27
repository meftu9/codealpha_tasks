"""
Interactive Web Application for Credit Scoring Model.
Supports single-applicant creditworthiness assessment (manual entry or sample profiles),
batch CSV scoring, probability visualization, financial ratio diagnostics, and
model evaluation reports.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import load_label_mapping
from src.feature_engineering import engineer_features
from src.models import load_trained_model
from src.preprocessing import clean_credit_dataframe


st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/credit_scoring_model.joblib"
LABEL_MAPPING_PATH = "models/label_mapping.json"
SAMPLES_JSON_PATH = "dataset/samples/sample_applicants.json"
TEMP_APPLICANT_PATH = "temp/current_applicant.json"
REPORTS_DIR = "reports"


@st.cache_resource
def get_pipeline(path: str = MODEL_PATH):
    """Loads and caches the trained end-to-end credit scoring pipeline."""
    if os.path.exists(path):
        return load_trained_model(path)
    return None


@st.cache_data
def get_label_mapping(path: str = LABEL_MAPPING_PATH) -> Dict[int, str]:
    """Loads and caches the class label mapping."""
    return load_label_mapping(path)


@st.cache_data
def get_evaluation_summary() -> Optional[dict]:
    """Loads evaluation summary JSON if available."""
    summary_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_sample_applicants() -> List[Dict[str, Any]]:
    """Loads exported sample applicants for one-click demo testing."""
    if os.path.exists(SAMPLES_JSON_PATH):
        with open(SAMPLES_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def plot_class_probabilities(probs: np.ndarray, label_map: Dict[int, str]):
    """Renders a clean horizontal bar chart of predicted credit score probabilities."""
    indices = sorted(label_map.keys())
    labels = [label_map[i] for i in indices]
    scores = [float(probs[i]) * 100.0 for i in indices]
    color_palette = {"Poor": "#e53e3e", "Standard": "#dd6b20", "Good": "#2f855a"}
    colors = [color_palette.get(lbl, "#3182ce") for lbl in labels]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    bars = ax.barh(labels, scores, color=colors, edgecolor="black", height=0.55)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_title("Creditworthiness Probability Breakdown", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, pct in zip(bars, scores):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def save_current_applicant(applicant: Dict[str, Any]) -> None:
    """Stores the latest evaluated applicant payload in temp/current_applicant.json."""
    os.makedirs(os.path.dirname(TEMP_APPLICANT_PATH), exist_ok=True)
    with open(TEMP_APPLICANT_PATH, "w", encoding="utf-8") as f:
        json.dump(applicant, f, indent=2)


def main() -> None:
    st.title("💳 Credit Scoring & Creditworthiness Assessment")
    st.markdown(
        "Predicts an individual's creditworthiness (**Poor**, **Standard**, or **Good**) from "
        "past financial history using engineered debt, payment-delay, and liquidity features."
    )

    pipeline = get_pipeline()
    label_map = get_label_mapping()
    eval_summary = get_evaluation_summary()
    samples = get_sample_applicants()

    if pipeline is None:
        st.error(
            "Trained model not found at `models/credit_scoring_model.joblib`. "
            "Run `python -m src.train` and `python -m src.evaluate` first."
        )
        st.stop()

    with st.sidebar:
        st.header("📌 Model Overview")
        if eval_summary:
            best_name = eval_summary.get("best_model", "Random Forest")
            best_metrics = eval_summary.get("models", {}).get(best_name, {})
            st.success(f"Active Model: **{best_name}**")
            st.metric("Test Accuracy", f"{best_metrics.get('accuracy', 0.0) * 100:.2f}%")
            st.metric("Weighted F1-Score", f"{best_metrics.get('weighted_f1_score', 0.0):.4f}")
            st.metric("ROC-AUC (OvR)", f"{best_metrics.get('roc_auc_ovr_weighted', 0.0):.4f}")
        else:
            st.info("Evaluation summary not generated yet. Run `python -m src.evaluate`.")

        st.divider()
        mode = st.radio(
            "Select Mode",
            ["Single Applicant Assessment", "Batch CSV Scoring", "Model Evaluation Reports"],
        )

    if mode == "Single Applicant Assessment":
        col_form, col_result = st.columns([1.2, 1.0], gap="large")

        defaults: Dict[str, Any] = {
            "Month": "January",
            "Age": 34,
            "Occupation": "Engineer",
            "Annual_Income": 65000.0,
            "Monthly_Inhand_Salary": 5200.0,
            "Num_Bank_Accounts": 3,
            "Num_Credit_Card": 4,
            "Interest_Rate": 10,
            "Num_of_Loan": 2,
            "Type_of_Loan": "Auto Loan, Personal Loan",
            "Delay_from_due_date": 5,
            "Num_of_Delayed_Payment": 3,
            "Changed_Credit_Limit": 8.5,
            "Num_Credit_Inquiries": 2,
            "Credit_Mix": "Good",
            "Outstanding_Debt": 950.0,
            "Credit_Utilization_Ratio": 28.5,
            "Credit_History_Age": "15 Years and 6 Months",
            "Payment_of_Min_Amount": "No",
            "Total_EMI_per_month": 180.0,
            "Amount_invested_monthly": 320.0,
            "Payment_Behaviour": "High_spent_Medium_value_payments",
            "Monthly_Balance": 650.0,
        }

        with col_form:
            st.subheader("Applicant Financial Profile")
            if samples:
                sample_names = ["Custom Input"] + [s.get("Sample_Name", f"Sample {i}") for i, s in enumerate(samples)]
                chosen_sample = st.selectbox("Load Sample Applicant Profile", sample_names)
                if chosen_sample != "Custom Input":
                    matched = next(
                        (s for s in samples if s.get("Sample_Name") == chosen_sample),
                        None,
                    )
                    if matched:
                        for k, v in matched.items():
                            if k in defaults and pd.notna(v):
                                defaults[k] = v

            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=int(float(str(defaults["Age"]).strip("_"))))
                annual_income = st.number_input("Annual Income ($)", min_value=0.0, max_value=300000.0, value=float(str(defaults["Annual_Income"]).strip("_")), step=1000.0)
                monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, max_value=30000.0, value=float(defaults["Monthly_Inhand_Salary"]), step=100.0)
                outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, max_value=10000.0, value=float(str(defaults["Outstanding_Debt"]).strip("_")), step=50.0)
                total_emi = st.number_input("Total EMI / Month ($)", min_value=0.0, max_value=5000.0, value=float(defaults["Total_EMI_per_month"]), step=20.0)

            with c2:
                num_bank_accounts = st.number_input("Bank Accounts", min_value=0, max_value=15, value=int(defaults["Num_Bank_Accounts"]))
                num_credit_cards = st.number_input("Credit Cards", min_value=0, max_value=15, value=int(defaults["Num_Credit_Card"]))
                num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=15, value=int(float(str(defaults["Num_of_Loan"]).strip("_"))))
                interest_rate = st.number_input("Interest Rate (%)", min_value=1, max_value=40, value=int(defaults["Interest_Rate"]))
                credit_util = st.slider("Credit Utilization (%)", min_value=0.0, max_value=100.0, value=float(defaults["Credit_Utilization_Ratio"]))

            with c3:
                delay_days = st.number_input("Delay From Due Date (days)", min_value=0, max_value=100, value=max(0, int(defaults["Delay_from_due_date"])))
                delayed_payments = st.number_input("Delayed Payments Count", min_value=0, max_value=50, value=int(float(str(defaults["Num_of_Delayed_Payment"]).strip("_"))))
                credit_inquiries = st.number_input("Credit Inquiries", min_value=0, max_value=30, value=int(float(defaults["Num_Credit_Inquiries"])))
                amount_invested = st.number_input("Invested Monthly ($)", min_value=0.0, max_value=5000.0, value=float(defaults["Amount_invested_monthly"]), step=25.0)
                monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, max_value=5000.0, value=float(defaults["Monthly_Balance"]), step=25.0)

            c4, c5 = st.columns(2)
            with c4:
                credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad", "Unknown"], index=["Good", "Standard", "Bad", "Unknown"].index(defaults["Credit_Mix"]) if defaults["Credit_Mix"] in ["Good", "Standard", "Bad", "Unknown"] else 0)
                payment_min = st.selectbox("Payment of Min Amount", ["No", "Yes", "Unknown"], index=["No", "Yes", "Unknown"].index(defaults["Payment_of_Min_Amount"]) if defaults["Payment_of_Min_Amount"] in ["No", "Yes", "Unknown"] else 0)
                occupation = st.text_input("Occupation", value=str(defaults["Occupation"]))
            with c5:
                credit_history_age = st.text_input("Credit History Age", value=str(defaults["Credit_History_Age"]))
                payment_behaviour = st.selectbox(
                    "Payment Behaviour",
                    [
                        "High_spent_Large_value_payments",
                        "High_spent_Medium_value_payments",
                        "High_spent_Small_value_payments",
                        "Low_spent_Large_value_payments",
                        "Low_spent_Medium_value_payments",
                        "Low_spent_Small_value_payments",
                        "Unknown",
                    ],
                    index=1,
                )
                changed_limit = st.number_input("Changed Credit Limit (%)", min_value=-20.0, max_value=50.0, value=float(defaults["Changed_Credit_Limit"]))

            applicant_payload = {
                "Month": defaults["Month"],
                "Age": age,
                "Occupation": occupation,
                "Annual_Income": annual_income,
                "Monthly_Inhand_Salary": monthly_salary,
                "Num_Bank_Accounts": num_bank_accounts,
                "Num_Credit_Card": num_credit_cards,
                "Interest_Rate": interest_rate,
                "Num_of_Loan": num_of_loan,
                "Type_of_Loan": defaults.get("Type_of_Loan", "Personal Loan"),
                "Delay_from_due_date": delay_days,
                "Num_of_Delayed_Payment": delayed_payments,
                "Changed_Credit_Limit": changed_limit,
                "Num_Credit_Inquiries": credit_inquiries,
                "Credit_Mix": credit_mix,
                "Outstanding_Debt": outstanding_debt,
                "Credit_Utilization_Ratio": credit_util,
                "Credit_History_Age": credit_history_age,
                "Payment_of_Min_Amount": payment_min,
                "Total_EMI_per_month": total_emi,
                "Amount_invested_monthly": amount_invested,
                "Payment_Behaviour": payment_behaviour,
                "Monthly_Balance": monthly_balance,
            }

        with col_result:
            st.subheader("Prediction & Risk Diagnostics")
            save_current_applicant(applicant_payload)
            input_df = pd.DataFrame([applicant_payload])
            pred_idx = int(pipeline.predict(input_df)[0])
            pred_probs = pipeline.predict_proba(input_df)[0]
            pred_label = label_map.get(pred_idx, str(pred_idx))
            confidence = float(np.max(pred_probs)) * 100.0

            if pred_label == "Good":
                st.success(f"### Predicted Credit Score: **{pred_label}** ({confidence:.1f}% confidence)")
            elif pred_label == "Standard":
                st.warning(f"### Predicted Credit Score: **{pred_label}** ({confidence:.1f}% confidence)")
            else:
                st.error(f"### Predicted Credit Score: **{pred_label}** ({confidence:.1f}% confidence)")

            fig = plot_class_probabilities(pred_probs, label_map)
            st.pyplot(fig)
            plt.close(fig)

            eng = engineer_features(clean_credit_dataframe(input_df)).iloc[0]
            st.markdown("#### Engineered Financial Indicators")
            r1, r2 = st.columns(2)
            r1.metric("Debt-to-Income Ratio", f"{eng['Debt_to_Income_Ratio']:.3f}")
            r2.metric("EMI-to-Salary Ratio", f"{eng['EMI_to_Salary_Ratio']:.3f}")
            r1.metric("Savings-to-Salary Ratio", f"{eng['Savings_to_Salary_Ratio']:.3f}")
            r2.metric("Delayed Payment Rate", f"{eng['Delayed_Payment_Rate']:.2f}")

    elif mode == "Batch CSV Scoring":
        st.subheader("Batch Applicant Scoring")
        uploaded = st.file_uploader("Upload an applicant CSV file", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded, low_memory=False)
            preds = pipeline.predict(batch_df)
            probs = pipeline.predict_proba(batch_df)

            out_df = batch_df.copy()
            out_df["Predicted_Credit_Score"] = [label_map.get(int(i), str(i)) for i in preds]
            out_df["Confidence"] = np.round(np.max(probs, axis=1), 4)

            st.dataframe(out_df.head(50), use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="credit_score_predictions.csv",
                mime="text/csv",
            )

    else:
        st.subheader("Model Evaluation & Comparison Reports")
        if eval_summary:
            rows = []
            for m_name, m_vals in eval_summary.get("models", {}).items():
                rows.append(
                    {
                        "Model": m_name,
                        "Accuracy": round(m_vals["accuracy"], 4),
                        "Precision (Weighted)": round(m_vals["weighted_precision"], 4),
                        "Recall (Weighted)": round(m_vals["weighted_recall"], 4),
                        "F1-Score (Weighted)": round(m_vals["weighted_f1_score"], 4),
                        "ROC-AUC (OvR Weighted)": round(m_vals["roc_auc_ovr_weighted"], 4),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        img_specs = [
            ("Model Comparison", os.path.join(REPORTS_DIR, "model_comparison.png")),
            ("Confusion Matrix", os.path.join(REPORTS_DIR, "confusion_matrix.png")),
            ("One-vs-Rest ROC Curves", os.path.join(REPORTS_DIR, "roc_curves.png")),
            ("Top Feature Importances", os.path.join(REPORTS_DIR, "feature_importance.png")),
        ]
        c_left, c_right = st.columns(2)
        for idx, (title, img_path) in enumerate(img_specs):
            target_col = c_left if idx % 2 == 0 else c_right
            if os.path.exists(img_path):
                target_col.markdown(f"**{title}**")
                target_col.image(img_path, use_container_width=True)


if __name__ == "__main__":
    main()
