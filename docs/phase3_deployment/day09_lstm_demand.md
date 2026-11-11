# Day 9: RNN/LSTM Demand Forecasting

## 1. Scenario Context
Kicking off Phase 3 by shifting from Vision to Time-Series Sequential Modeling. Over-stocking traps working capital, while stockouts result in lost revenue. We built an LSTM to predict T+1 demand based on T-14 sliding lookback windows.

## 2. Technical Blueprint
- **Architecture**: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1, Linear)
- **Data Engineering**: MinMax scaling applied to historical sales. Data is transformed into (batch_size, 14, 1) tensors.
- **Metric**: MAE (Mean Absolute Error) is used to evaluate how many physical units the prediction misses by on average.
