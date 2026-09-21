"""
Script to train the salary prediction model.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from src.ml.model import create_model, save_model

DATA_PATH = Path("data/salaries.csv")
MODEL_PATH = Path("models/salary_model.joblib")

def load_data() -> pd.DataFrame:
    """Load the dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please place the dataset in the data/ directory."
        )
    return pd.read_csv(DATA_PATH)

def preprocess_data(df: pd.DataFrame):
    """Preprocess the data for training."""
    # We assume the dataset has the following columns:
    # job_title, experience_level, employment_type, remote_ratio, company_size, salary_in_usd
    target = "salary_in_usd"
    features = ["job_title", "experience_level", "employment_type", "remote_ratio", "company_size"]
    
    X = df[features]
    y = df[target]
    
    return X, y

def train_model():
    """Train the model and save it."""
    # Load data
    df = load_data()
    X, y = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained. MAE: {mae:.2f}, R2: {r2:.2f}")
    
    # Save model
    save_model(model)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
