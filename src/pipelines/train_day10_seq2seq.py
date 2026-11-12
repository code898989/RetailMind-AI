import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Reshape, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join("data", "raw", "store_demand.csv")
MODEL_PATH = os.path.join("models", "demand_seq2seq_v10.keras")
SCALER_PATH = os.path.join("models", "demand_scaler_v10.joblib")

def run_pipeline():
    df = pd.read_csv(DATA_PATH)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["sales"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - 20):
        X.append(scaled[i:i+14])
        y.append(scaled[i+14:i+21])
    X, y = np.array(X), np.array(y)
    split = len(X) - 60
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    
    inp = Input(shape=(14, 1))
    out, sh, sc = LSTM(64, return_sequences=True, return_state=True)(inp)
    
    # Keras 3 compliant dimension mapping
    sh_expanded = Reshape((1, 64))(sh)
    cv, _ = Attention()([sh_expanded, out], return_attention_scores=True)
    cv_flat = Flatten()(cv)
    
    concat = Concatenate()([cv_flat, sh])
    dense = Dense(32, activation="relu")(concat)
    dec = Dense(7)(dense)
    
    m = Model(inp, dec)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    print("[INFO] Training Seq2Seq Attention Model...")
    m.fit(X_tr, y_tr.squeeze(), epochs=25, batch_size=16, validation_split=0.1, verbose=1)
    
    preds = scaler.inverse_transform(m.predict(X_te)).flatten()
    actuals = scaler.inverse_transform(y_te.squeeze()).flatten()
    print("\nDAY 10 MAE:", mean_absolute_error(actuals, preds))
    m.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[SUCCESS] Saved Day 10 Model")

if __name__ == "__main__":
    run_pipeline()
