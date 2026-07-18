import numpy as np
from typing import List

from config import HeadDetectionConfig
from utils import Detection

class HeadDetector:
    def __init__(self, cfg: HeadDetectionConfig):
        self.cfg = cfg
        self.backend = cfg.backend

        if self.backend == "yolo":
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError("Could not load YOLO (`pip install ultralytics`)") from e
            self._model = YOLO(cfg.yolo_weights)
            
        else:
            raise ValueError(f"Unknown head-detection backend: {self.backend}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source = frame, 
            conf=self.cfg.yolo_conf, 
            save=False,
            verbose=False,
            classes=[0]
        )

        detections = []
        
        # For standard COCO-trained YOLO models (yolov8n.pt, yolov11n.pt, etc.), class 0 = person
        result = results[0]

        for box in result.boxes:

            cls_id = int(box.cls[0])
            class_name = self._model.names[cls_id]

            if class_name == "person":
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), confidence=conf)
                )
        
        return detections