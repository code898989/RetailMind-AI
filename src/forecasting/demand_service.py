import os
import joblib
import numpy as np

class DemandForecastService:
    def __init__(self):
        self.model_path = os.path.join("models", "demand_lstm_v9.keras")
        self.scaler_path = os.path.join("models", "demand_scaler_v9.joblib")
        self.model = None
        self.scaler = None
        self.lookback = 14
        self._load_assets()

    def _load_assets(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

    def predict_next_day_demand(self, recent_sales: list) -> dict:
        """Takes exactly 14 days of recent sales and forecasts day 15."""
        if not self.model or not self.scaler:
            return {"error": "Model artifacts not found. Run training pipeline first."}
            
        if len(recent_sales) != self.lookback:
            return {"error": f"Expected exactly {self.lookback} days of history, got {len(recent_sales)}"}

        sales_array = np.array(recent_sales).reshape(-1, 1)
        scaled_input = self.scaler.transform(sales_array)
        
        # Reshape to (batch_size, timesteps, features)
        model_input = scaled_input.reshape(1, self.lookback, 1)
        
        scaled_pred = self.model.predict(model_input, verbose=0)
        actual_pred = self.scaler.inverse_transform(scaled_pred)[0][0]
        
        return {
            "historical_input": recent_sales,
            "forecasted_demand_units": max(0, int(round(actual_pred))),
            "status": "Success"
        }

if __name__ == "__main__":
    service = DemandForecastService()
    # Mocking the last 14 days of sales for a single SKU
    recent_history = [210, 195, 205, 215, 340, 360, 200, 190, 205, 210, 220, 350, 370, 215]
    
    print("\n" + "="*50)
    print("INVENTORY PLANNING: NEXT-DAY DEMAND FORECAST")
    print("="*50)
    res = service.predict_next_day_demand(recent_history)
    for k, v in res.items():
        print(f"{k:<25}: {v}")
    print("="*50 + "\n")
