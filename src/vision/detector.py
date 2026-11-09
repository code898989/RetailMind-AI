import cv2
import numpy as np
from ultralytics import YOLO

class RetailProductDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect_and_locate(self, image_input, conf_threshold: float = 0.25):
        results = self.model.predict(source=image_input, conf=conf_threshold, verbose=False)
        first_result = results[0]

        detections = []
        for box in first_result.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0].item())

            detections.append({
                "label": class_name,
                "confidence": round(conf, 4),
                "bbox": [round(c, 2) for c in coords]
            })

        return {
            "total_items_detected": len(detections),
            "detections": detections
        }

if __name__ == "__main__":
    detector = RetailProductDetector()
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (50, 50), (250, 300), (255, 255, 255), -1)
    res = detector.detect_and_locate(dummy_img)
    print("[DETECTOR TEST COMPLETE]")
    print(res)
