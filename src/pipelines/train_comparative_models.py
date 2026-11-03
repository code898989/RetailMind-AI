import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
CHAMPION_PATH = os.path.join(MODEL_DIR, "churn_champion_v2.joblib")

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

def run_benchmark():
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

    models = {
        "Logistic Regression (Day 1)": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight="balanced"),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    }

    trained_pipelines = {}
    print(f"{'Model':<28} | {'Accuracy':<8} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'ROC-AUC':<8}")
    print("-" * 80)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_proba)

        print(f"{name:<28} | {acc:.4f}   | {rec:.4f}   | {prec:.4f}    | {f1:.4f} | {auc:.4f}")

    champion_name = "Random Forest"
    champion_pipeline = trained_pipelines[champion_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(champion_pipeline, CHAMPION_PATH)
    print(f"\n[WINNER] Selected '{champion_name}' as Day 2 Champion.")
    print(f"[SUCCESS] Champion pipeline persisted to: {CHAMPION_PATH}")

if __name__ == "__main__":
    run_benchmark()
