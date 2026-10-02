from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI(title="Loan Approval Prediction API")

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Find the trained model in the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "loan_approval_model.pkl")

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    applicant = pd.DataFrame([{
        "no_of_dependents": data["no_of_dependents"],
        "education": data["education"],
        "self_employed": data["self_employed"],
        "income_annum": data["income_annum"],
        "loan_amount": data["loan_amount"],
        "loan_term": data["loan_term"],
        "cibil_score": data["cibil_score"],
        "residential_assets_value": data["residential_assets_value"],
        "commercial_assets_value": data["commercial_assets_value"],
        "luxury_assets_value": data["luxury_assets_value"],
        "bank_asset_value": data["bank_asset_value"]
    }])

    prediction = model.predict(applicant)[0]
    probabilities = model.predict_proba(applicant)[0]

    if prediction == 1:
        status = "Approved"
        confidence = probabilities[1] * 100
    else:
        status = "Rejected"
        confidence = probabilities[0] * 100

    return {
        "status": status,
        "confidence": round(confidence, 2)
    }