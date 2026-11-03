import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, recall_score, accuracy_score

DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
CHAMPION_PATH = os.path.join(MODEL_DIR, "churn_v3.joblib")

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

def run_day3_tuning():
    print("[INFO] Loading dataset for Day 3 hyperparameter search...")
    X, y = load_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ]
    )

    # 5-fold stratified CV focusing on ROC-AUC & balanced churn recall
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grids = {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "params": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"]
            }
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [6, 10, None],
                "clf__min_samples_split": [2, 5],
                "clf__class_weight": ["balanced", "balanced_subsample"]
            }
        },
        "SVM (RBF)": {
            "pipeline": Pipeline([
                ("preprocessor", preprocessor),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42))
            ]),
            "params": {
                "clf__C": [0.5, 1.0, 2.0],
                "clf__gamma": ["scale", "auto"],
                "clf__class_weight": ["balanced"]
            }
        }
    }

    best_estimators = {}
    print(f"\n{'Model':<22} | {'Best CV ROC-AUC':<15} | {'Test Accuracy':<13} | {'Test Recall':<11} | {'Test ROC-AUC':<12}")
    print("-" * 82)

    for name, config in grids.items():
        search = GridSearchCV(
            config["pipeline"],
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            return_train_score=True
        )
        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_
        best_estimators[name] = best_pipe
        train_score = search.cv_results_['mean_train_score'][search.best_index_]
        val_score = search.best_score_

        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, y_pred)
        test_rec = recall_score(y_test, y_pred, pos_label=1)
        test_auc = roc_auc_score(y_test, y_proba)

        print(f"{name:<22} | {val_score:<15.4f} | {test_acc:<13.4f} | {test_rec:<11.4f} | {test_auc:<12.4f}")
        print(f"   -> Train CV AUC: {train_score:.4f} | Bias-Variance Gap: {abs(train_score - val_score):.4f}")
        print(f"   -> Best Params: {search.best_params_}\n")

    # Lock Tuned Random Forest as Day 3 Champion
    champion_pipe = best_estimators["Random Forest"]
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(champion_pipe, CHAMPION_PATH)
    print(f"[WINNER] Tuned Random Forest locked as Day 3 Champion.")
    print(f"[SUCCESS] Persisted to: {CHAMPION_PATH}")

if __name__ == "__main__":
    run_day3_tuning()
