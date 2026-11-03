import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_pipeline_v1.joblib")

def load_and_clean_data(file_path: str):
    df = pd.read_csv(file_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def build_pipeline(X: pd.DataFrame):
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

def run_training():
    print("[INFO] Loading & cleaning data...")
    X, y = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 40)
    print("DAY 1 BASELINE MODEL METRICS")
    print("=" * 40)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("=" * 40 + "\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[SUCCESS] Pipeline persisted at: {MODEL_PATH}")

if __name__ == "__main__":
    run_training()
