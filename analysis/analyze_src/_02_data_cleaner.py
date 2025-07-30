import pandas as pd
from pathlib import Path
from config_manager import ConfigManager

def clean_data():
    cm = ConfigManager()
    raw_data_dir = Path(cm.get("data.raw_data_dir"))
    processed_data_dir = Path(cm.get("data.processed_data_dir"))
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(raw_data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No raw CSV files found in {raw_data_dir}")
        return

    data_path = csv_files[0]
    df = pd.read_csv(data_path)
    print(f"Loaded raw data: {data_path.name} | Shape: {df.shape}")

    # Clean 'TotalCharges' — convert to numeric, coerce errors
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Optional: Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()

    # Save to processed data folder
    output_path = Path(cm.get("data.train_data_file"))
    df_cleaned.to_csv(output_path, index=False)
    print(f" Cleaned data saved to {output_path} | New shape: {df_cleaned.shape}")

if __name__ == "__main__":
    clean_data()
