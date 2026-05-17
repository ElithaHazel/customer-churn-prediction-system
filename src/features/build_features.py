"""
Feature engineering and preprocessing for the Telco Churn project.

This module has two responsibilities:

1. `add_engineered_features(df)`:
   Adds NEW columns derived from the raw ones — these go INTO the model.
   Every feature here exists because of a hypothesis tested in EDA.

2. `build_preprocessor()`:
   Returns an sklearn ColumnTransformer that:
     - Scales numeric features (only matters for LogReg; XGBoost is invariant)
     - One-hot encodes categorical features
   The transformer is wrapped inside a Pipeline downstream so it learns
   from train folds only — preventing leakage during cross-validation.

WHY SEPARATE FROM CLEANING:
Cleaning is "make the data not broken" — it happens once, deterministically.
Feature engineering is "create signal for the model" — it's a modeling choice
and may be revised. Keeping them separate keeps each file focused and testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------------------------------
# 1. Feature engineering
# --------------------------------------------------------------------------

# Service columns counted into the engagement feature
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with engineered columns added.

    Engineered features:

      - `tenure_bucket` (categorical): tenure binned into business-meaningful
        groups. The EDA showed a sharp churn drop after month 6, plus a long
        loyalty tail past month 48. Buckets make this non-linearity explicit
        for any model — and easier to explain in a business meeting.

      - `services_count` (int): number of "Yes" service signups (0–9).
        Captures total product engagement. EDA showed customers without
        OnlineSecurity / TechSupport churned much more — this rolls that
        up into one engagement signal.

      - `avg_monthly_spend` (float): TotalCharges / tenure, with safe handling
        for tenure=0. Smoother than MonthlyCharges for long-tenure customers
        whose plan changed over time.

      - `is_auto_pay` (Yes/No): True for "Bank transfer (automatic)" or
        "Credit card (automatic)". EDA showed electronic-check users churned
        3x more — auto-pay is a strong commitment signal.
    """
    df = df.copy()

    # 1. Tenure buckets — business-meaningful cuts based on EDA findings
    bins = [-0.1, 12, 24, 48, 1000]
    labels = ["0-12m", "12-24m", "24-48m", "48m+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    # 2. Services count — total engagement signal
    def _count_services(row: pd.Series) -> int:
        count = 0
        for col in SERVICE_COLUMNS:
            val = str(row[col]).strip()
            # "Yes" counts; "No internet service" and "No phone service" don't
            if val == "Yes":
                count += 1
        return count

    df["services_count"] = df[SERVICE_COLUMNS].apply(_count_services, axis=1)

    # 3. Average monthly spend — divide-by-zero-safe
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].replace(0, np.nan),
        df["MonthlyCharges"],  # for tenure=0, fall back to monthly
    )
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["MonthlyCharges"])

    # 4. Auto-pay binary
    auto_methods = {"Bank transfer (automatic)", "Credit card (automatic)"}
    df["is_auto_pay"] = df["PaymentMethod"].apply(
        lambda x: "Yes" if str(x).strip() in auto_methods else "No"
    )

    return df


# --------------------------------------------------------------------------
# 2. Column groupings AFTER engineering
# --------------------------------------------------------------------------

def get_columns_for_modeling() -> dict[str, list[str]]:
    """Return the canonical column groupings for the preprocessing pipeline.

    These include the engineered columns. Centralized so notebooks and the
    Streamlit app see the same definitions.
    """
    numeric = ["tenure", "MonthlyCharges", "TotalCharges",
               "services_count", "avg_monthly_spend"]
    categorical = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure_bucket", "is_auto_pay",
    ]
    return {"numeric": numeric, "categorical": categorical, "target": "Churn"}


# --------------------------------------------------------------------------
# 3. The preprocessor — used inside model Pipelines
# --------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numeric and one-hot encodes categorical.

    WHY ColumnTransformer:
    - Single object that handles heterogenous columns.
    - Wraps cleanly inside an sklearn Pipeline.
    - The Pipeline ensures `fit` happens on training fold only — no leakage.

    Why StandardScaler:
    - LogReg requires it (coefficients are scale-sensitive; regularization
      penalizes large coefficients unfairly otherwise).
    - XGBoost is scale-invariant; scaling doesn't help but doesn't hurt.

    Why OneHotEncoder (drop=None, handle_unknown='ignore'):
    - One-hot is the safe default for nominal categoricals.
    - drop=None keeps full information for tree models (no info loss).
    - handle_unknown='ignore' means at inference time, an unseen category
      becomes all zeros instead of crashing — important for production.
    """
    cols = get_columns_for_modeling()

    numeric_pipe = StandardScaler()
    categorical_pipe = OneHotEncoder(
        drop=None,
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, cols["numeric"]),
            ("cat", categorical_pipe, cols["categorical"]),
        ],
        remainder="drop",  # any unlisted column is dropped — fail loud not silent
        verbose_feature_names_out=True,
    )
