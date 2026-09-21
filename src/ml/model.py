"""
Model definition and loading utilities.
"""
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

MODEL_PATH = Path("models/salary_model.joblib")

def load_model():
    """Load the trained model from disk."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")

def save_model(model: Pipeline):
    """Save the trained model to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def create_model() -> Pipeline:
    """
    Create a Random Forest model with preprocessing for categorical features.
    This mirrors the preprocessing used in training.
    """
    # Categorical features to one-hot encode
    categorical_features = ["job_title", "experience_level", "employment_type", "company_size"]
    # Note: remote_ratio is numeric but we treat as categorical for simplicity
    categorical_features.append("remote_ratio")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    return pipeline
