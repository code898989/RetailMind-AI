# Day 7: Transfer Learning (ResNet50) & YOLO Object Detection

## 1. Objective
Automate e-commerce catalog image quality assurance (QA) and shelf auditing using:
- **ResNet50 Backbone**: Feature reuse from ImageNet for zero-shot visual representation.
- **Ultralytics YOLO**: Bounding-box detection to flag empty, single-item, or cluttered multi-item catalog uploads.

## 2. Architecture Details
- **Classifier**: ResNet50 (Frozen) -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(5, Softmax).
- **Detector**: YOLOv8 Nano for fast, real-time bounding box extraction.
