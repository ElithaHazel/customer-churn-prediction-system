"""
Data loading module.

Why this is a module and not notebook code:
- Every notebook that loads data will call the same function -> consistency
- If the schema changes, we fix it in ONE place
- Easier to unit test
- This is how production code is organized
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Resolve project root relative to this file. This avoids "it works on my laptop
# but breaks in the notebook" problems caused by working directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_telco(filename: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load the raw IBM Telco Customer Churn dataset.

    The CSV has a known quirk: `TotalCharges` is typed as object because some
    rows have a single space " " instead of a number (these are brand-new
    customers whose billing cycle has not yet completed). We surface that here
    by loading naively — cleaning happens in the cleaning module, not here.
    Loading and cleaning are separate responsibilities.
    """
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected the Telco dataset at {path}.\n"
            f"Download it from Kaggle: "
            f"https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            f"and place the CSV in data/raw/."
        )
    return pd.read_csv(path)


def quick_summary(df: pd.DataFrame) -> None:
    """Print a sanity-check summary. Use this as the first cell of any new notebook."""
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
