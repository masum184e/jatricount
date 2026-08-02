from config import DensityDecisionConfig
from typing import List
import itertools

from utils import Detection, iou

class DensityDecisionModule:
    def __init__(self, cfg: DensityDecisionConfig):
        self.cfg = cfg

    def decide(self, detections: List[Detection], frame_shape) -> str:
        h, w = frame_shape[:2]
        frame_area_units = (h * w) / 1000.0
        n = len(detections)

        if n < self.cfg.min_heads_for_dense_check:
            return "sparse"

        density = n / frame_area_units if frame_area_units > 0 else 0.0
        overlap_ratio = self._mean_overlap_ratio(detections)

        if density >= self.cfg.heads_per_1000px2_threshold or overlap_ratio >= self.cfg.overlap_ratio_threshold:
            return "dense"
        return "sparse"

    def _mean_overlap_ratio(self, detections: List[Detection]) -> float:
        if len(detections) < 2:
            return 0.0
        pairs = list(itertools.combinations(detections, 2))
        overlaps = [iou(a.box, b.box) for a, b in pairs]
        overlapping = [o for o in overlaps if o > 0]
        return len(overlapping) / len(pairs)