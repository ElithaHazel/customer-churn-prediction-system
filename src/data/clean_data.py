"""
Data cleaning module for the Telco Customer Churn dataset.

WHY THIS IS A MODULE (not notebook code):
- In production, training and inference both call the SAME cleaning function.
- If we duplicate logic in notebooks, train/serve skew creeps in silently.
- We can unit-test these functions and version them in git.

The cleaning decisions made here are documented inline so they survive
in interviews: every line below is a decision you should be ready to defend.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# Use the same project root resolution as load_data.py for consistency.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco DataFrame.

    Cleaning steps applied (in order):
      1. Drop `customerID` — it's a unique identifier with no predictive value.
         Including it would let a tree model "memorize" rows or pick up
         spurious patterns from the ID structure.

      2. Coerce `TotalCharges` to numeric. The raw CSV has 11 rows where this
         value is a single space character " ". We've confirmed all 11 rows
         correspond to tenure=0 customers (brand new — never billed). We
         impute these with 0 because zero history is the *correct* value,
         not "missing/unknown."

      3. Map the target `Churn` to integer: Yes -> 1, No -> 0. Models expect
         numeric targets; we do this once, here, and never touch it again.

      4. Encode `SeniorCitizen` as a Yes/No string. The raw column is already
         0/1, but downstream we treat it like other binary categoricals — so
         we normalize it here for consistency.

      5. Strip whitespace from object columns (defensive — guards against
         hidden trailing spaces in some categorical values).

    We do NOT do here:
      - One-hot encoding or scaling (those belong in the modeling pipeline
        so they participate in cross-validation and don't leak from train
        to test).
      - "No internet service" -> "No" remapping. Decision: we keep the
        distinction because customers with no internet are structurally
        different from internet customers who declined a service. Tree
        models will learn the difference; for logistic regression it
        becomes another one-hot column, which is fine.
    """
    df = df.copy()

    # Step 1: drop the identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Step 2: fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # All blanks correspond to tenure=0 — verified in EDA notebook.
    # Sanity check, not a silent assumption:
    null_mask = df["TotalCharges"].isna()
    if null_mask.any():
        bad = df.loc[null_mask & (df["tenure"] != 0)]
        if len(bad) > 0:
            raise ValueError(
                f"Found {len(bad)} rows with null TotalCharges and tenure != 0. "
                "Cleaning assumption broken — investigate before proceeding."
            )
        df.loc[null_mask, "TotalCharges"] = 0.0

    # Step 3: encode target as int
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Step 4: normalize SeniorCitizen
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})

    # Step 5: defensive whitespace strip on object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def save_processed(df: pd.DataFrame, filename: str = "telco_clean.csv") -> Path:
    """Save the cleaned DataFrame to data/processed/ and return the path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def get_feature_columns() -> dict[str, list[str]]:
    """Return the canonical feature groupings used downstream.

    Centralizing this here means every notebook and script sees the same
    definitions. Update once, fix everywhere.
    """
    return {
        "numeric": ["tenure", "MonthlyCharges", "TotalCharges"],
        "binary_categorical": [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "PaperlessBilling",
        ],
        "multi_categorical": [
            "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
        ],
        "target": "Churn",
    }
