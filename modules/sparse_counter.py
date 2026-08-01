from typing import List
import cv2

from utils import Detection

class BoundingBoxCounter:
    def __init__(self, nms_iou_threshold: float = 0.35, score_threshold: float = 0.0):
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

    def count(self, detections: List[Detection]) -> (float, List[Detection]):
        if not detections:
            return 0.0, []

        boxes = [d.box for d in detections]
        scores = [d.confidence for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_iou_threshold)
        # NMSBoxes is a greedy algorithm

        if len(indices) == 0:
            return 0.0, []

        indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
        kept = [detections[i] for i in indices]
        return float(len(kept)), kept