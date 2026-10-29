"""Clean the raw insurance data and engineer features."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "insurance.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "insurance_processed.csv"


def clean_and_process_data(raw_path=RAW_PATH, processed_path=PROCESSED_PATH):
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")

    df = pd.read_csv(raw_path).drop_duplicates(keep="first")

    # Feature engineering
    df["is_obese"] = (df["bmi"] >= 30).astype(int)
    df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
    df["obese_smoker"] = df["is_obese"] * df["smoker_yes"]

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data ({df.shape[0]} rows, {df.shape[1]} cols) saved to {processed_path}")
    return df


if __name__ == "__main__":
    clean_and_process_data()
