# RetailMind AI — Enterprise Retail Intelligence Platform

RetailMind AI is an incremental, production-oriented retail intelligence platform built across 35 daily engineering milestones.

## Day 1 Increment: Churn ML Backbone & Baseline Service

### Product Vision
Establish an operational churn-prediction baseline. Customer retention teams need early signals for high-risk accounts to take proactive retention measures before contract cancellation.

### Baseline Benchmark Ledger
| Metric | Baseline Score (Day 1) | Notes |
| :--- | :--- | :--- |
| **Model Architecture** | Logistic Regression | Regularized L2 solver (\max_iter=1000\) |
| **Accuracy** | 0.8053 | Test split (20% stratified hold-out) |
| **Precision (Churn=1)** | 0.6544 | Accounts flagged that actually churned |
| **Recall (Churn=1)** | 0.5722 | Proportion of churners successfully identified |
| **F1-Score** | 0.6105 | Harmonic mean |
| **ROC-AUC** | 0.8361 | Baseline threshold-independent discriminative power |

*All subsequent increments (Tree models, Ensembles, ANN) measure relative improvement against these figures.*

### Quick Start
\\\ash
# 1. Train and save baseline pipeline
python src/pipelines/train_pipeline.py

# 2. Run inference service
python src/service.py
\\\
"@

Set-Content -Path docs/PROJECT_LOG.md -Value @"
# RetailMind AI — Daily Engineering Log

## Day 1: End-to-End ML Pipeline & Service
- **Delivered**:
  - Structured repository (\data/\, \src/\, \models/\, \
otebooks/\, \docs/\).
  - Implemented data sanitization handling non-numeric \TotalCharges\ strings.
  - Built unified \ColumnTransformer\ with \StandardScaler\ and \OneHotEncoder\.
  - Serialized fitted pipeline to \models/churn_pipeline_v1.joblib\.
  - Built \ChurnPredictorService\ in \src/service.py\ returning probability and risk tier.
- **Baseline Metrics**:
  - ROC-AUC: 0.8361, Accuracy: 80.53%, Churn Recall: 57.22%.
