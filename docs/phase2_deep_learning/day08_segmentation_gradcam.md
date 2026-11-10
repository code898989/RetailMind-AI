# Day 8: Product Segmentation, Grad-CAM & End-to-End CV

## 1. Scenario Context & Objective
Phase 2 closes with an explainable computer vision service delivering clean, background-free catalog images. In automated retail ingestion:
- **Clean Background Removal**: Normalizes varied merchant uploads to consistent studio-white catalog cards (, 255, 255$) without manual clipping.
- **Grad-CAM Visual Audits**: Provides attention heatmaps highlighting image regions that triggered a classification, allowing merchants to verify why a product was tagged.

## 2. Technical Blueprint
- **Segmentation**: Adaptive thresholding and morphological closing extract foreground boundaries while eliminating background noise.
- **Grad-CAM Architecture**: Computes gradient of category score with respect to convolutional feature activations ($\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$), followed by ReLU activation to highlight positive evidence.
- **Integrated Service**: Unified under \RetailVisionService.process_catalog_asset()\.
