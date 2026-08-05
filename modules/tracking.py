"""
tracking.py
------------
Box 8 in the diagram: "Multi Object Tracking"

Turns per-frame counts into a single video-level "how many distinct people
appeared" number by tracking heads across frames (a lightweight SORT-style
tracker: IoU-based matching solved with the Hungarian algorithm, plus a
missed-frame grace period). Without tracking, the same person standing in
view for 100 frames would otherwise be counted 100 times.
"""

from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import TrackingConfig
from utils import Detection, iou


class Track:
    def __init__(self, track_id: int, detection: Detection):
        self.id = track_id
        self.box = detection.box
        self.hits = 1
        self.missed = 0
        self.confirmed = False


class MultiObjectTracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.total_unique_confirmed = 0

    def update(self, detections: List[Detection]) -> List[int]:
        """Update tracks with this frame's detections.
        Returns a list of track IDs parallel to `detections` (order preserved).
        """
        track_ids = list(self.tracks.keys())
        assigned_det_ids = [-1] * len(detections)

        if track_ids and detections:
            cost = np.ones((len(track_ids), len(detections)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - iou(self.tracks[tid].box, det.box)

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks, matched_dets = set(), set()

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= (1.0 - self.cfg.iou_match_threshold):
                    tid = track_ids[r]
                    self._update_track(tid, detections[c])
                    assigned_det_ids[c] = tid
                    matched_tracks.add(tid)
                    matched_dets.add(c)

            self._age_unmatched(set(track_ids) - matched_tracks)
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        else:
            self._age_unmatched(set(track_ids))
            unmatched_dets = list(range(len(detections)))

        for j in unmatched_dets:
            tid = self._create_track(detections[j])
            assigned_det_ids[j] = tid

        self._purge_dead_tracks()
        return assigned_det_ids

    # -- internals ----------------------------------------------------------

    def _update_track(self, tid: int, det: Detection):
        t = self.tracks[tid]
        t.box = det.box
        t.hits += 1
        t.missed = 0
        if not t.confirmed and t.hits >= self.cfg.min_hits_to_confirm:
            t.confirmed = True
            self.total_unique_confirmed += 1

    def _create_track(self, det: Detection) -> int:
        tid = self._next_id
        self._next_id += 1
        t = Track(tid, det)
        if self.cfg.min_hits_to_confirm <= 1:
            t.confirmed = True
            self.total_unique_confirmed += 1
        self.tracks[tid] = t
        return tid

    def _age_unmatched(self, unmatched_ids):
        for tid in unmatched_ids:
            self.tracks[tid].missed += 1

    def _purge_dead_tracks(self):
        dead = [tid for tid, t in self.tracks.items() if t.missed > self.cfg.max_missed_frames]
        for tid in dead:
            del self.tracks[tid]
