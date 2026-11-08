import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Transformer definition required for unpickling Day 4 pipeline
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Monthly_to_Total_Ratio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1.0)
        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 60, 100],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"]
        ).astype(str)
        X["IsMonthToMonth"] = (X["Contract"] == "Month-to-month").astype(int)
        X["IsAutoPayment"] = X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
        has_tv = (X["StreamingTV"] == "Yes").astype(int)
        has_mov = (X["StreamingMovies"] == "Yes").astype(int)
        X["StreamBundle"] = has_tv + has_mov
        return X

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
            model_path = None
            for candidate in [DAY4_PATH, DAY3_PATH, DAY2_PATH, DAY1_PATH]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            if not model_path:
                raise FileNotFoundError("No trained model artefact found in models/.")
            self.pipeline = joblib.load(model_path)
            if "v4" in model_path:
                self.version = "Day 4 Production Model (XGBoost/Ensemble)"
            elif "v3" in model_path:
                self.version = "Day 3 Tuned Champion (Random Forest)"
            elif "v2" in model_path:
                self.version = "Day 2 Champion (Random Forest)"
            else:
                self.version = "Day 1 Baseline (Logistic Regression)"

    def predict(self, records: pd.DataFrame) -> pd.DataFrame:
        if self.use_ann:
            features = self.preprocessor.transform(records)
            probas = self.model.predict(features, verbose=0).ravel()
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

    service = ChurnPredictorService(use_ann=False)
    print(f"[ACTIVE SERVICE] {service.version}")
    result = service.predict(sample_customer)
    print(result[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])

    if os.path.exists(DAY5_ANN_PATH):
        service_ann = ChurnPredictorService(use_ann=True)
        print(f"\n[ACTIVE SERVICE] {service_ann.version}")
        result_ann = service_ann.predict(sample_customer)
        print(result_ann[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])
