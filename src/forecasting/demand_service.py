import os
import joblib
import numpy as np

class DemandForecastService:
    def __init__(self):
        self.m, self.s = None, None
        if os.path.exists("models/demand_seq2seq_v10.keras"):
            import tensorflow as tf
            self.m = tf.keras.models.load_model("models/demand_seq2seq_v10.keras")
            self.s = joblib.load("models/demand_scaler_v10.joblib")

    def predict_procurement_horizon(self, hist):
        if not self.m: return {"error": "No model"}
        scaled = self.s.transform(np.array(hist).reshape(-1,1)).reshape(1, 14, 1)
        preds = self.s.inverse_transform(self.m.predict(scaled, verbose=0))[0]
        f = [max(0, int(round(p))) for p in preds]
        return {"forecasted_demand_7d": f, "total_weekly_procurement": sum(f), "status": "Success"}

if __name__ == "__main__":
    svc = DemandForecastService()
    print("\nPROCUREMENT FORECAST: 7-DAY SEQ2SEQ HORIZON")
    res = svc.predict_procurement_horizon([210,195,205,215,340,360,200,190,205,210,220,350,370,215])
    for k, v in res.items(): print(f"{k:<28}: {v}")
