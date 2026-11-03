import os
import joblib
import pandas as pd

DAY4_PATH = os.path.join("models", "churn_production_v4.joblib")
DAY3_PATH = os.path.join("models", "churn_v3.joblib")
DAY2_PATH = os.path.join("models", "churn_champion_v2.joblib")
DAY1_PATH = os.path.join("models", "churn_pipeline_v1.joblib")

class ChurnPredictorService:
    def __init__(self, model_path: str = None):
        if model_path is None:
            for candidate in [DAY4_PATH, DAY3_PATH, DAY2_PATH, DAY1_PATH]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError("No trained model artefact found.")

        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        if "v4" in model_path:
            self.version = "Day 4 Final Production Model (Engineered + Ensembles)"
        elif "v3" in model_path:
            self.version = "Day 3 Tuned Champion (Random Forest)"
        elif "v2" in model_path:
            self.version = "Day 2 Champion (Random Forest)"
        else:
            self.version = "Day 1 Baseline (Logistic Regression)"

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        preds = self.pipeline.predict(records)
        probas = self.pipeline.predict_proba(records)[:, 1]

        output = records.copy()
        output["predicted_churn"] = preds
        output["churn_risk_probability"] = probas.round(4)
        output["risk_flag"] = output["churn_risk_probability"].apply(
            lambda p: "High Risk" if p >= 0.5 else "Low Risk"
        )
        return output

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
    print(f"[SERVICE VERSION ACTIVE] {service.version}")
    result = service.predict(sample_customer)
    print(result[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])
