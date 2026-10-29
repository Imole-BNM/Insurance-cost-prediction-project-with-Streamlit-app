import os
import pandas as pd
from data_processing import clean_data_and_process_data

#define the path to the raw data and the processed data directory
raw_path = os.path.join("data", "raw", "insurance.csv")
processed_dir = os.path.join("data", "processed")
os.makedirs(processed_dir, exist_ok=True)


#load and clean the data
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Raw data file not found at {raw_path}. Please ensure the file exists.")
df = pd.read_csv(raw_path)
df = df.drop_duplicates(keep='first')

#handle feature engineering and preprocessing
df["is_obese"] = (df["bmi"] >= 30).astype(int)
df_encoded = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

#ensure the interactive term uses the correct column names
if "smoker_yes" in df_encoded.columns:
    df_encoded["obese_smoker"] = df_encoded["is_obese"] * df_encoded["smoker_yes"]

#save the processed data to a new CSV file
processed_path = os.path.join(processed_dir, "insurance_processed.csv")
df_encoded.to_csv(processed_path, index=False)
print(f"Processed data saved to {processed_path}")

if __name__ == "__main__":
    clean_data_and_process_data()