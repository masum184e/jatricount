"""
frame_extraction.py
--------------------
Box 2 in the diagram: "Frame Extraction"

Reads an input video and yields frames sampled at a target FPS, optionally
resized. Keeping this isolated means swapping in an RTSP stream or an
image-folder source later only requires changing this one module.
"""

import cv2
from config import FrameExtractionConfig


class FrameExtractor:
    def __init__(self, video_path: str, cfg: FrameExtractionConfig):
        self.video_path = video_path
        self.cfg = cfg
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_interval = max(1, round(self.source_fps / cfg.sample_fps))

    def __iter__(self):
        frame_idx = 0
        yielded = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            if frame_idx % self.frame_interval == 0:
                if self.cfg.resize_width:
                    h, w = frame.shape[:2]
                    scale = self.cfg.resize_width / w
                    frame = cv2.resize(frame, (self.cfg.resize_width, int(h * scale)))

                timestamp = frame_idx / self.source_fps
                yield frame_idx, timestamp, frame
                yielded += 1

                if self.cfg.max_frames and yielded >= self.cfg.max_frames:
                    break

            frame_idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
