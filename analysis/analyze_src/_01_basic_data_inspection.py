import pandas as pd
from pathlib import Path
from config_manager import ConfigManager

def basic_inspection():
    cm = ConfigManager()
    raw_data_dir = Path(cm.get("data.raw_data_dir"))
    csv_files = list(raw_data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        return

    # Load first CSV file
    data_path = csv_files[0]
    print(f"Inspecting file: {data_path.name}")
    df = pd.read_csv(data_path)

    print("\n DataFrame Shape:", df.shape)
    print("\n Columns:", df.columns.tolist())
    print("\n Null Values:\n", df.isnull().sum())
    print("\n Sample Rows:\n", df.head())

if __name__ == "__main__":
    basic_inspection()
