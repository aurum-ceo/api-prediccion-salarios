from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

router = APIRouter()

# Load the model
MODEL_PATH = Path("models/salary_model.joblib")
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None  # In a real app, you might raise an error or load a default

class SalaryInput(BaseModel):
    job_title: str
    experience_level: str  # EN, MI, SE, EX
    employment_type: str   # FT, PT, CT, FL
    remote_ratio: int      # 0, 50, 100
    company_size: str      # S, M, L

class SalaryOutput(BaseModel):
    predicted_salary_usd: float
    confidence_interval: dict
    model_version: str
    timestamp: str

# Mock preprocessing function (in a real app, you would use the same preprocessing as in training)
def preprocess_input(data: SalaryInput) -> pd.DataFrame:
    # For simplicity, we just return a DataFrame with the input.
    # In a real scenario, you would encode categorical variables, etc.
    df = pd.DataFrame([data.dict()])
    return df

@router.post("/predict", response_model=SalaryOutput)
async def predict_salary(input_data: SalaryInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Preprocess
    processed = preprocess_input(input_data)
    
    # Predict
    prediction = model.predict(processed)[0]
    
    # For confidence interval, we can use the standard deviation of the predictions
    # or use a quantile regression forest. For simplicity, we'll use a fixed percentage.
    lower = prediction * 0.9
    upper = prediction * 1.1
    
    return SalaryOutput(
        predicted_salary_usd=float(prediction),
        confidence_interval={"lower": lower, "upper": upper},
        model_version="1.0.0",
        timestamp=pd.Timestamp.now().isoformat()
    )

@router.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return {
        "model_type": "RandomForestRegressor",
        "version": "1.0.0",
        "features": ["job_title", "experience_level", "employment_type", "remote_ratio", "company_size"]
    }
