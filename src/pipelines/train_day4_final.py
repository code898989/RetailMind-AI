import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score, accuracy_score

DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
CHAMPION_PATH = os.path.join(MODEL_DIR, "churn_production_v4.joblib")

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

def load_clean_data(path: str):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def run_day4_pipeline():
    print("[INFO] Loading dataset for Day 4 Feature Engineering & Ensembles...")
    X, y = load_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    fe = DomainFeatureEngineer()
    X_fe_sample = fe.transform(X_train)
    num_cols = X_fe_sample.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_fe_sample.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        ]
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    ensembles = {
        "Random Forest (Day 4 Bagging)": Pipeline([
            ("fe", DomainFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=180, max_depth=7, class_weight="balanced", random_state=42))
        ]),
        "XGBoost (Day 4 Boosting Champion)": Pipeline([
            ("fe", DomainFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            ))
        ])
    }

    trained_pipelines = {}
    print(f"\n{'Model':<34} | {'Accuracy':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'ROC-AUC':<8}")
    print("-" * 88)

    for name, pipe in ensembles.items():
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_proba)

        print(f"{name:<34} | {acc:.4f}   | {rec:.4f}   | {prec:.4f}    | {f1:.4f} | {auc:.4f}")

    champion_name = "XGBoost (Day 4 Boosting Champion)"
    production_pipe = trained_pipelines[champion_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(production_pipe, CHAMPION_PATH)
    print(f"\n[WINNER] Selected '{champion_name}' as Final Production Model.")
    print(f"[SUCCESS] Model and feature pipeline persisted to: {CHAMPION_PATH}")

if __name__ == "__main__":
    run_day4_pipeline()
