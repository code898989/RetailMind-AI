import os
import joblib
import pandas as pd
import numpy as np

DAY6_CNN_PATH = os.path.join("models", "catalog_cnn_v6.keras")
DAY6_LABELS_PATH = os.path.join("models", "catalog_labels_v6.joblib")
DAY5_ANN_PATH = os.path.join("models", "churn_ann_v5.keras")
DAY5_PREP_PATH = os.path.join("models", "ann_preprocessor_v5.joblib")
DAY4_PATH = os.path.join("models", "churn_production_v4.joblib")

class ChurnPredictorService:
    def __init__(self, use_ann: bool = False):
        self.use_ann = use_ann
        if self.use_ann and os.path.exists(DAY5_ANN_PATH) and os.path.exists(DAY5_PREP_PATH):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(DAY5_ANN_PATH)
            self.preprocessor = joblib.load(DAY5_PREP_PATH)
            self.version = "Day 5 Deep Learning ANN (Keras)"
        else:
            self.pipeline = joblib.load(DAY4_PATH)
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

class CatalogVisionService:
    """Day 6: Computer Vision Auto-tagging Service for Retail Products."""
    def __init__(self):
        if not os.path.exists(DAY6_CNN_PATH) or not os.path.exists(DAY6_LABELS_PATH):
            raise FileNotFoundError("Catalog CNN model not found. Run train_day6_cnn.py first.")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(DAY6_CNN_PATH)
        self.labels = joblib.load(DAY6_LABELS_PATH)
        self.version = "Day 6 Catalog Classifier (CNN)"

    def predict_image(self, image_array: np.ndarray) -> dict:
        \"\"\"Accepts (H, W, 3) normalized [0, 1] float32 image array.\"\"\"
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        probs = self.model.predict(image_array, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        
        return {
            "predicted_category": self.labels[top_idx],
            "confidence": float(round(probs[top_idx], 4)),
            "all_probabilities": {self.labels[i]: float(round(probs[i], 4)) for i in range(len(self.labels))}
        }

if __name__ == "__main__":
    # Test Churn Service
    sample_customer = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
    }])
    churn_service = ChurnPredictorService()
    print(f"[ACTIVE SERVICE] {churn_service.version}")
    print(churn_service.predict(sample_customer)[["MonthlyCharges", "tenure", "churn_risk_probability", "risk_flag"]])

    # Test Vision Service if model exists
    if os.path.exists(DAY6_CNN_PATH):
        vision_service = CatalogVisionService()
        print(f"\n[ACTIVE SERVICE] {vision_service.version}")
        dummy_img = np.random.uniform(0.0, 1.0, size=(64, 64, 3)).astype("float32")
        result = vision_service.predict_image(dummy_img)
        print("Auto-Tagged Image Output:", result)
