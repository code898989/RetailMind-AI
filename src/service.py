import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "churn_pipeline_v1.joblib")

class ChurnPredictorService:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artefact not found at {model_path}. Run training pipeline first.")
        self.model = joblib.load(model_path)
        self.version = "v1-baseline-logistic-regression"

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        predictions = self.model.predict(records)
        probabilities = self.model.predict_proba(records)[:, 1]

        result_df = records.copy()
        result_df["predicted_churn"] = predictions
        result_df["churn_probability"] = probabilities.round(4)
        result_df["risk_level"] = result_df["churn_probability"].apply(
            lambda p: "High Risk" if p >= 0.50 else "Low Risk"
        )
        return result_df

if __name__ == "__main__":
    sample_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }])

    service = ChurnPredictorService()
    print(f"[SERVICE VERSION] {service.version}")
    response = service.predict(sample_customer)
    print(response[["MonthlyCharges", "tenure", "churn_probability", "risk_level"]])
