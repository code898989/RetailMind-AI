# RetailMind AI — Daily Development Log

## Day 1: End-to-End Baseline Pipeline & ML-as-a-Service
- **Changes Added**:
  - Initialized repo structure (data/, src/, models/, notebooks/, docs/).
  - Built scikit-learn Pipeline with ColumnTransformer (scaling + one-hot encoding).
  - Trained baseline Logistic Regression model and exported artefact via joblib.
  - Implemented service wrapper (src/service.py) for inference.
- **Business Value Added**:
  - Provides immediate baseline churn risk scoring for retail account teams.
- **Open Blockers / Points for Doubt-Clearing**:
  - Recall on churn is 53%, missing at-risk users due to class imbalance.

## Day 2: Decision Tree, Random Forest & SVM
- **Changes Added**:
  - Implemented multi-model comparative training harness (src/pipelines/train_comparative_models.py).
  - Evaluated Decision Tree, Random Forest, and SVM models on identical preprocessed features.
  - Selected Random Forest as Champion and persisted to models/churn_champion_v2.joblib.
  - Updated src/service.py to auto-resolve and serve Champion v2.
  - Added comparison table in README.md and created day02 notebook.
- **Business Value Added**:
  - Retention teams now capture ~75.7% of all churning accounts (up from 57.2%), preserving recurring revenue by triggering proactive retention outreach.
- **Open Blockers / Points for Doubt-Clearing**:
  - SVM training runtime is noticeably higher due to quadratic scaling; evaluate whether hyperparameter search or gradient boosting offers a better speed-to-recall trade-off.

## Day 3: Hyperparameter Tuning & Bias-Variance Diagnosis
- **Changes Added**:
  - Implemented 5-fold cross-validation grid search (\src/pipelines/train_day3_tuning.py\).
  - Tuned hyperparameter spaces for Logistic Regression, Random Forest, and SVM.
  - Diagnosed variance: default Random Forest showed a 12% train-validation gap; restricting depth to 6 reduced the gap to 3.5% without sacrificing recall.
  - Locked and saved tuned champion to \models/churn_v3.joblib\.
  - Added notebook \
otebooks/day03_tuning_bias_variance.ipynb\.
- **Business Value Added**:
  - The model generalises stably to unseen cohorts, preventing performance decay once deployed to live retail batches.
- **Open Blockers / Points for Doubt-Clearing**:
  - Model relies strictly on raw features without interaction terms (e.g. tenure vs charges ratio).

## Day 4: Feature Engineering, PCA & XGBoost Ensembles
- **Changes Added**:
  - Implemented \DomainFeatureEngineer\ in \src/pipelines/train_day4_final.py\.
  - Added PCA dimensionality reduction and compared Bagging (Random Forest) vs. Boosting (XGBoost).
  - Selected XGBoost with \scale_pos_weight\ as the final Phase 1 Production Model and saved to \models/churn_production_v4.joblib\.
  - Updated \src/service.py\ to auto-serve the Day 4 production model.
  - Added Day 4 notebook and updated the Phase 1 benchmark table in \README.md\.
- **Business Value Added**:
  - Reached **80.5% Churn Recall** and **0.8612 ROC-AUC**. Four out of five churning retail accounts are caught ahead of contract expiration, minimizing lost MRR.
- **Open Blockers / Points for Doubt-Clearing**:
  - Phase 1 complete. Ready for Phase 2: Deep Learning baseline (Keras ANN benchmark).
