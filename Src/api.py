
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "fraudeye_random_forest.joblib"
CONFIG_PATH = ROOT / "models" / "fraudeye_config.joblib"

app = FastAPI(
    title="FraudEye AI API",
    description="Transaction fraud scoring API",
    version="1.0.0",
)

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    raise RuntimeError("Model or configuration file is missing.")

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(CONFIG_PATH)["threshold"])

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.get("/")
def home():
    return {"project": "FraudEye AI", "status": "API online"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "Random Forest"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        row = transaction.model_dump()
        X = pd.DataFrame([row], columns=FEATURES)

        probability = float(model.predict_proba(X)[0, 1])
        is_fraud = probability >= threshold

        return {
            "fraud_probability": round(probability, 6),
            "threshold": threshold,
            "prediction": "Fraud" if is_fraud else "Legitimate",
            "alert": "FRAUD_ALERT" if is_fraud else "NORMAL",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Transaction prediction failed.",
        ) from exc
