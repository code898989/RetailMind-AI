import os
import joblib
import pandas as pd
import numpy as np

DAY5_ANN_PATH = os.path.join("models", "churn_ann_v5.keras")
DAY5_PREP_PATH = os.path.join("models", "ann_preprocessor_v5.joblib")
DAY4_PATH = os.path.join("models", "churn_production_v4.joblib")
DAY3_PATH = os.path.join("models", "churn_v3.joblib")
DAY2_PATH = os.path.join("models", "churn_champion_v2.joblib")
DAY1_PATH = os.path.join("models", "churn_pipeline_v1.joblib")

class ChurnPredictorService:
    def __init__(self, use_ann: bool = False):
        self.use_ann = use_ann
        if self.use_ann and os.path.exists(DAY5_ANN_PATH) and os.path.exists(DAY5_PREP_PATH):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(DAY5_ANN_PATH)
            self.preprocessor = joblib.load(DAY5_PREP_PATH)
            self.version = "Day 5 Deep Learning ANN (Keras)"
        else:
            # Fallback to Phase 1 Classical Champion
            model_path = None
            for candidate in [DAY4_PATH, DAY3_PATH, DAY2_PATH, DAY1_PATH]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            if not model_path:
                raise FileNotFoundError("No trained model artefact found.")
            self.pipeline = joblib.load(model_path)
            self.version = "Phase 1 Classical Champion (XGBoost/Ensemble)"

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        if self.use_ann:
            features = self.preprocessor.transform(records)
            probas = self.model.predict(features).ravel()
            preds = (probas >= 0.5).astype(int)
        else:
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

    # Benchmark Phase 1 Classical Champion vs Day 5 ANN
    service_classical = ChurnPredictorService(use_ann=False)
    print(f"[ACTIVE SERVICE] {service_classical.version}")
    print(service_classical.predict(sample_customer)[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])

    if os.path.exists(DAY5_ANN_PATH):
        service_ann = ChurnPredictorService(use_ann=True)
        print(f"\n[ACTIVE SERVICE] {service_ann.version}")
        print(service_ann.predict(sample_customer)[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])
