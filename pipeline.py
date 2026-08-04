"""
pipeline.py
"""
import os
import cv2
from utils import show_images
from config import PreprocessingConfig, PipelineConfig
from modules.preprocessing import Preprocessor
from modules.head_detection import HeadDetector
from modules.density_decision import DensityDecisionModule
from modules.sparse_counter import BoundingBoxCounter
from modules.density_map import DensityMapPredictor
from modules.fusion import FusionModule
from modules.frame_extraction import FrameExtractor
from logger import get_logger, StepTimer

log = get_logger(__name__)


class CrowdCountingPipeline:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        self.preprocessor = Preprocessor(self.cfg.preprocessing)
        self.head_detector = HeadDetector(self.cfg.head_detection)
        self.density_decider = DensityDecisionModule(self.cfg.density_decision)
        self.bbox_counter = BoundingBoxCounter()
        self.density_predictor = DensityMapPredictor(self.cfg.density_map)
        self.fusion = FusionModule(self.cfg.fusion)

    def run_image(self, raw_frame, tag: str = "frame", save_visuals: bool = True, verbose: bool = True):
        """
        Runs the full pipeline on a single frame.

        tag: used to namespace output files, so video frames don't overwrite
             each other (e.g. "frame_000123").
        save_visuals: set False for video mode to avoid writing an image per
                       sampled frame.
        verbose: set False to demote per-frame detail from INFO to DEBUG
                 (keeps it out of the console but still in the log file).
        """
        level = log.info if verbose else log.debug

        level(f"Processing {tag} ...")

        with StepTimer(log, f"{tag} | Preprocess"):
            process_frame = self.preprocessor.process(raw_frame)

        if save_visuals:
            show_images(
                (raw_frame, "Original"),
                (process_frame, "Process Frame"),
                output_path=f"output/{tag}_process.png"
            )

        sparse_input = process_frame.copy()
        dense_input = process_frame.copy()

        with StepTimer(log, f"{tag} | Head Detection"):
            detection_result = self.head_detector.detect(sparse_input)

        scene_type = self.density_decider.decide(detection_result, sparse_input.shape)
        level(f"Scene Type: {scene_type}")

        sparse_count, kept_dets = self.bbox_counter.count(detection_result)
        level(f"Sparse Count: {sparse_count}")
        log.debug(f"Kept Dets: {kept_dets}")  # noisy raw data -> always debug

        if save_visuals:
            for det in detection_result:
                sparse_input = det.plot(sparse_input)
            show_images(
                (raw_frame, "Original"),
                (sparse_input, "Process Frame"),
                output_path=f"output/{tag}_detection.png"
            )

        with StepTimer(log, f"{tag} | Density Prediction"):
            result = self.density_predictor.predict(dense_input, detection_result)
            log.debug(f"predict() returned: {result!r}")
            if result is None:
                log.error(
                    f"predict() returned None for backend={self.density_predictor.backend!r} "
                    f"-- check _predict_{self.density_predictor.backend} for a missing return"
                )
                raise RuntimeError("DensityMapPredictor.predict() returned None")
            dense_count, density_map = result

        if save_visuals:
            density_overlay = self.density_predictor.visualize(dense_input, density_map)
            show_images(
                (raw_frame, "Original"),
                (density_overlay, "Density Map"),
                output_path=f"output/{tag}_density.png"
            )
        level(f"Dense Count: {dense_count}")

        fused_count = self.fusion.fuse(scene_type, sparse_count, dense_count)
        level(f"Fusion Count: {fused_count}")

        summary = {
            "tag": tag,
            "sparse_count": sparse_count,
            "dense_count": round(dense_count, 2),
            "fused_count": round(fused_count, 2),
            "scene_type": scene_type,
            "detected_heads": len(detection_result),
            "kept_detections": len(kept_dets),
        }

        level(f"Frame Summary: {summary}")

        return summary

    def run_video(self, video_path: str, save_visuals: bool = False, verbose_per_frame: bool = False):
        """
        Runs the pipeline over sampled frames of a video, using FrameExtractor
        (box 2) to handle decoding/sampling/resizing.
        """
        os.makedirs("output", exist_ok=True)
        results = []

        log.info(f"Processing Video: {video_path} ...")
        with FrameExtractor(video_path, self.cfg.frame_extraction) as extractor:
            for frame_idx, timestamp, frame in extractor:
                tag = f"frame_{frame_idx:06d}"
                summary = self.run_image(
                    frame,
                    tag=tag,
                    save_visuals=save_visuals,
                    verbose=verbose_per_frame,
                )
                summary["timestamp"] = round(timestamp, 2)
                results.append(summary)
                log.info(
                    f"[t={timestamp:6.2f}s] frame {frame_idx:6d} -> "
                    f"fused_count={summary['fused_count']}"
                )

        if not results:
            log.warning("No frames processed.")
            return results

        avg_fused = sum(r["fused_count"] for r in results) / len(results)
        max_fused = max(r["fused_count"] for r in results)

        log.info("=== Crowd Counting Video Summary ===")
        log.info({
            "frames_processed": len(results),
            "avg_fused_count": round(avg_fused, 2),
            "max_fused_count": round(max_fused, 2),
        })

        return results