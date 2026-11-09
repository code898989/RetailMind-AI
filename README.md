# RetailMind AI

RetailMind AI is an enterprise-grade retail intelligence platform developed incrementally across 35 production milestones.

## Phase 2: Deep Learning Progression & Model Benchmarking

| Day / Milestone | Model Architecture | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Day 1** | Baseline Logistic Regression | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated |
| **Day 2** | Decision Tree / Random Forest / SVM | 0.7854 | 0.7567 | 0.5739 | 0.8491 | Evaluated |
| **Day 3** | Tuned Random Forest (5-Fold CV) | 0.7925 | 0.7701 | 0.5842 | 0.8498 | Audited |
| **Day 4** | Domain Features + Ensembles (XGBoost) | 0.8010 | 0.8048 | 0.5960 | 0.8612 | Production Champion |
| **Day 5** | Deep ANN (Keras 3-Layer + BatchNorm + Dropout) | 0.7583 | 0.7834 | 0.5288 | 0.8415 | DL Benchmark |
| **Day 6** | Catalog Auto-Tagger CNN (Conv2D + Augmentation) | 0.8840 | N/A (Multi-Class) | 0.8710 | 0.9420 | CV Module |

## Honest Diagnostic: When to Use Which?
1. **Classical Ensembles (XGBoost / Random Forest) — The Clear Winner on Tabular Data**:
   - **Why XGBoost Remains Production Champion**: On medium-sized tabular retail records (~7,000 rows), gradient boosted decision trees naturally handle mixed feature distributions, discrete boundaries, and non-linear interactions without requiring heavy parameter search or architectural tuning. XGBoost delivered higher ROC-AUC (0.8612 vs. 0.8415) and significantly better precision with zero risk of gradient vanishing.
   - **Operational Footprint**: Near-instant inference latency, minimal memory usage, and no native GPU runtime dependency.

2. **Deep Learning (Keras Feed-Forward ANN & CNN) — When Depth Actually Adds Value**:
   - **Observations on Tabular Telco**: The deep ANN learned meaningful decision surfaces and high churn recall (0.7834), but required aggressive regularization (Dropout, BatchNormalization, EarlyStopping) to prevent overfitting tabular noise.
   - **Computer Vision (Day 6)**: Where spatial hierarchies and pixel arrays dominate (like SKU image auto-tagging), classical tabular algorithms cannot compete. Convolutions extract spatial features that scale directly to retail catalog categorization.
   - **When to Deploy Neural Nets in RetailMind AI**:
     - High-cardinality entity embeddings (e.g., millions of customer IDs or SKU categories).
     - Multi-modal retail data (e.g., fusing tabular transactional history with product images or customer support audio/text transcripts).
     - Massive streaming datasets ($> 10^7$ rows) where continuous online gradient descent out-scales tree memory caches.

### Day 7: Deep Architecture Optimization & Full-Dataset Evaluation

| Day / Milestone | Model Architecture | Accuracy | Recall (Churn=1) | Precision (Churn=1) | ROC-AUC | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Day 1** | Baseline Logistic Regression | 0.8053 | 0.5722 | 0.6544 | 0.8361 | Deprecated |
| **Day 2** | Decision Tree / Random Forest / SVM | 0.7854 | 0.7567 | 0.5739 | 0.8491 | Tree Benchmark |
| **Day 3** | Tuned Random Forest (5-Fold CV) | 0.7925 | 0.7701 | 0.5842 | 0.8498 | Generalised Model |
| **Day 4** | **Domain Features + XGBoost Ensemble** | **0.8010** | **0.8048** | **0.5960** | **0.8612** | **Production Champion** |
| **Day 5** | Deep Feed-Forward ANN (Keras) | 0.7583 | 0.7834 | 0.5288 | 0.8415 | DL Baseline |
| **Day 7** | **Optimized Deep Architecture (Full Holdout Evaluation)** | **0.7820** | **0.7915** | **0.5620** | **0.8540** | **Phase 2 DL Benchmark** |

#### Day 7 Technical Highlights:
- **Full Dataset Holdout Evaluation**: Benchmarked inference across the entire holdout test set (~1,409 records) rather than single-row mock payloads, measuring true production latency (~85 ms per batch).
- **Training vs. Inference Demarcation**: Full-dataset training runs via training pipelines across all stratified folds with adaptive learning rates (ReduceLROnPlateau), while the production wrapper (src/service.py) handles real-time single-record scoring.
- **Empirical Takeaway**: Deep representation learning narrows the performance gap to XGBoost on dense categorical combinations, though gradient boosted trees retain the edge in operational simplicity and inference throughput.
