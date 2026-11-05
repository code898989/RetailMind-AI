import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, accuracy_score

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
ANN_MODEL_PATH = os.path.join(MODEL_DIR, "churn_ann_v5.keras")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "ann_preprocessor_v5.joblib")

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

def build_ann_model(input_dim: int):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def run_day5_training():
    print("[INFO] Loading dataset for Day 5 Keras ANN benchmark...")
    X, y = load_clean_data(DATA_PATH)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
        ]
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Calculate class weights to handle imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weights = {
        0: 1.0,
        1: neg_count / pos_count
    }

    print(f"[INFO] Training Keras ANN on input shape: {X_train.shape}...")
    ann = build_ann_model(X_train.shape[1])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        mode="min"
    )

    ann.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=80,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate ANN
    y_proba = ann.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 55)
    print("DAY 5 KERAS ANN EVALUATION RESULTS")
    print("=" * 55)
    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("=" * 55 + "\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    ann.save(ANN_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[SUCCESS] Saved ANN model to: {ANN_MODEL_PATH}")
    print(f"[SUCCESS] Saved preprocessor to: {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    run_day5_training()
