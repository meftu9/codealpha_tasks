"""
Feature Engineering Module for Credit Scoring Model.
Derives domain-specific financial ratios and behavioral risk indicators from
cleaned applicant financial history.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NUMERIC_FEATURES: List[str] = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Num_Loan_Types",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    # Engineered financial ratios & risk features
    "Debt_to_Income_Ratio",
    "EMI_to_Salary_Ratio",
    "Savings_to_Salary_Ratio",
    "Investment_to_Salary_Ratio",
    "Credit_Accounts_Total",
    "Delayed_Payment_Rate",
    "Inquiries_per_Account",
    "Delay_Severity_Score",
]

CATEGORICAL_FEATURES: List[str] = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Spending_Level",
    "Payment_Value_Size",
]


def _parse_payment_behaviour(series: pd.Series) -> pd.DataFrame:
    """
    Splits Payment_Behaviour strings (e.g., 'High_spent_Small_value_payments')
    into Spending_Level and Payment_Value_Size columns.
    """
    text = series.fillna("Unknown").astype(str)
    spending = np.where(
        text.str.startswith("High_spent"),
        "High",
        np.where(text.str.startswith("Low_spent"), "Low", "Unknown"),
    )
    value_size = np.where(
        text.str.contains("Large_value"),
        "Large",
        np.where(
            text.str.contains("Medium_value"),
            "Medium",
            np.where(text.str.contains("Small_value"), "Small", "Unknown"),
        ),
    )
    return pd.DataFrame(
        {"Spending_Level": spending, "Payment_Value_Size": value_size},
        index=series.index,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates financial-history features such as debt-to-income ratio, EMI burden,
    savings rate, delayed payment rate, credit inquiry intensity, and spending behavior.
    Works identically on batch datasets and single-row inference inputs.
    """
    out = df.copy()

    annual_income = out["Annual_Income"].astype(float)
    monthly_salary = out["Monthly_Inhand_Salary"].astype(float).fillna(annual_income / 12.0)
    annual_income = annual_income.fillna(monthly_salary * 12.0)

    out["Annual_Income"] = annual_income
    out["Monthly_Inhand_Salary"] = monthly_salary

    safe_annual = annual_income.where(annual_income > 0.0, np.nan)
    safe_monthly = monthly_salary.where(monthly_salary > 0.0, np.nan)

    out["Debt_to_Income_Ratio"] = out["Outstanding_Debt"].astype(float) / safe_annual
    out["EMI_to_Salary_Ratio"] = out["Total_EMI_per_month"].astype(float) / safe_monthly
    out["Savings_to_Salary_Ratio"] = out["Monthly_Balance"].astype(float) / safe_monthly
    out["Investment_to_Salary_Ratio"] = out["Amount_invested_monthly"].astype(float) / safe_monthly

    num_bank = out["Num_Bank_Accounts"].astype(float)
    num_cards = out["Num_Credit_Card"].astype(float)
    num_loans = out["Num_of_Loan"].astype(float)
    total_credit_lines = np.maximum(num_cards.fillna(0.0) + num_loans.fillna(0.0), 1.0)

    out["Credit_Accounts_Total"] = num_bank + num_cards + num_loans
    out["Delayed_Payment_Rate"] = out["Num_of_Delayed_Payment"].astype(float) / total_credit_lines
    out["Inquiries_per_Account"] = out["Num_Credit_Inquiries"].astype(float) / total_credit_lines
    out["Delay_Severity_Score"] = (
        np.maximum(out["Delay_from_due_date"].astype(float), 0.0)
        * np.maximum(out["Num_of_Delayed_Payment"].astype(float), 0.0)
    )

    behaviour_parts = _parse_payment_behaviour(out["Payment_Behaviour"])
    out["Spending_Level"] = behaviour_parts["Spending_Level"]
    out["Payment_Value_Size"] = behaviour_parts["Payment_Value_Size"]

    return out


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer wrapping clean_credit_dataframe and engineer_features
    for seamless end-to-end Pipeline persistence.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from src.preprocessing import clean_credit_dataframe

        cleaned = clean_credit_dataframe(X)
        engineered = engineer_features(cleaned)
        selected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        return engineered[selected_cols]
