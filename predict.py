import joblib
import pandas as pd

# Load trained model
model = joblib.load("credit_scoring_model.pkl")

# Example applicant
applicant = pd.DataFrame([{
    "checking_account": "A11",
    "duration": 12,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 2000,
    "savings": "A61",
    "employment": "A73",
    "installment_rate": 4,
    "personal_status": "A93",
    "other_debtors": "A101",
    "residence": 4,
    "property": "A121",
    "age": 30,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "dependents": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
}])

# Prediction
prediction = model.predict(applicant)[0]

if prediction == 1:
    print("Credit Risk: GOOD")
else:
    print("Credit Risk: BAD")