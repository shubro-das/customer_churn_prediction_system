import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path

from analyze_src.config_manager import ConfigManager

def load_schema(schema_path):
    with open(schema_path, 'r') as file:
        return yaml.safe_load(file)

def validate_schema(df, schema):
    expected_columns = schema['columns']
    for col, dtype in expected_columns.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if df[col].dtype != dtype and dtype != "object":
            print(f"Warning: Column {col} has type {df[col].dtype}, expected {dtype}")
    return True

def encode_categoricals(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def train_pipeline():
    # Load config
    cm = ConfigManager()
    config = cm
    train_data_path = Path(config.get("data.train_data_file"))
    model_dir = Path(config.get("artifacts.model_dir"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(train_data_path)

    # Load and validate schema
    schema = load_schema("schema.yaml")
    validate_schema(df, schema)

    # Separate features and label
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Encode categorical features
    X_encoded, label_encoders = encode_categoricals(X)
    y_encoded = LabelEncoder().fit_transform(y)

    # Load params
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    test_size = params['model']['test_size']
    random_state = params['model']['random_state']
    xgb_params = params['xgboost_params']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded,
        test_size=test_size,
        random_state=random_state
    )

    # Train model
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    # Save model
    model_path = model_dir / "churn_model.pkl"
    joblib.dump(model, model_path)

    print(f"✅ Model trained and saved at: {model_path}")
    print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_pipeline()
