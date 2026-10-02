import pandas as pd
import joblib

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# Create one sample loan application
applicant = pd.DataFrame([{
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5000000,
    "loan_amount": 15000000,
    "loan_term": 20,
    "cibil_score": 750,
    "residential_assets_value": 8000000,
    "commercial_assets_value": 2000000,
    "luxury_assets_value": 5000000,
    "bank_asset_value": 4000000
}])

# Make prediction
prediction = model.predict(applicant)[0]

# Get prediction probability
probability = model.predict_proba(applicant)[0]

# Display result
if prediction == 1:
    status = "APPROVED"
    confidence = probability[1] * 100
else:
    status = "REJECTED"
    confidence = probability[0] * 100

print("\n==============================")
print("     LOAN PREDICTION")
print("==============================")

print("Loan Status :", status)
print(f"Confidence  : {confidence:.2f}%")

print("==============================")