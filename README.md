# RetailMind AI

RetailMind AI is an enterprise-grade retail intelligence platform developed incrementally across 35 production milestones.

## Phase 1 Milestone Assessment: Classical ML Churn Engine Progression

| Milestone | Model Architecture | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Day 1** | Baseline Logistic Regression (StdScaler + OHE) | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated Baseline |
| **Day 2** | Decision Tree / Random Forest / SVM | 0.7854 | 0.7567 | 0.5739 | 0.8491 | Tree Benchmark |
| **Day 3** | Tuned Random Forest (5-Fold Stratified CV, depth=6) | 0.7925 | 0.7701 | 0.5842 | 0.8498 | Generalised Model |
| **Day 4** | **Domain Features + Ensembles (XGBoost + scale_pos_weight)** | **0.8010** | **0.8048** | **0.5960** | **0.8612** | **Phase 1 Production Champion** |

### Key Milestones Delivered Through Day 4
1. **Domain Feature Engineering**: Synthesized \Monthly_to_Total_Ratio\, \TenureGroup\, \IsMonthToMonth\, and \StreamBundle\ indicators inside an automated scikit-learn transformer.
2. **Dimension Reduction**: Evaluated PCA preserving 95% variance to eliminate collinearity in service attributes.
3. **Boosting Ensembles**: Shifted from bagging to gradient-boosted decision trees using XGBoost with inverse class weighting, boosting churn recall to **> 80%** with a peak ROC-AUC of **0.8612**.
4. **Service Freezing**: Preprocessing steps and model serialized as a single immutable pipeline in \models/churn_production_v4.joblib\.
