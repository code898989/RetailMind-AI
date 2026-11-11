import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = os.path.join("data", "raw", "store_demand.csv")
MODEL_PATH = os.path.join("models", "demand_lstm_v9.keras")
SCALER_PATH = os.path.join("models", "demand_scaler_v9.joblib")
LOOKBACK = 14  # 2 weeks of lookback to predict next day

def generate_synthetic_demand_data():
    """Generates 2 years of realistic daily sales data with weekly seasonality."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    base_sales = 200
    
    # Add weekly seasonality (higher on weekends) + random noise
    sales = []
    for d in dates:
        weekend_boost = 150 if d.weekday() >= 5 else 0
        trend = np.sin(d.dayofyear / 365 * 2 * np.pi) * 50
        noise = np.random.normal(0, 20)
        sales.append(max(0, int(base_sales + weekend_boost + trend + noise)))
        
    df = pd.DataFrame({"date": dates, "sales": sales})
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Generated synthetic demand dataset at {DATA_PATH}")

def create_windows(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def run_pipeline():
    if not os.path.exists(DATA_PATH):
        generate_synthetic_demand_data()

    print("[INFO] Loading and preprocessing demand data...")
    df = pd.read_csv(DATA_PATH)
    sales_data = df["sales"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_sales = scaler.fit_transform(sales_data)

    X, y = create_windows(scaled_sales, LOOKBACK)
    
    # Hold out the last 60 days for testing
    split = len(X) - 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[INFO] Building LSTM... Train shape: {X_train.shape}")
    model = Sequential([
        LSTM(64, activation="relu", return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    print("[INFO] Training LSTM...")
    model.fit(X_train, y_train, epochs=25, batch_size=16, validation_split=0.1, verbose=1)

    print("[INFO] Evaluating on hold-out test set...")
    preds_scaled = model.predict(X_test)
    preds = scaler.inverse_transform(preds_scaled)
    actuals = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    print("\n" + "="*50)
    print("DAY 9 LSTM DEMAND FORECASTING METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE) : {mae:.2f} units")
    print(f"Root Mean Squared Error   : {rmse:.2f} units")
    print("="*50 + "\n")

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SUCCESS] Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

if __name__ == "__main__":
    run_pipeline()
