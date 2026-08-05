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
        save_visuals: if True, writes {tag}_process.png / _detection.png /
                       _density.png to disk (single-image mode). Set False
                       for video mode -- no per-frame PNGs are written, but
                       the annotated frames are still built and returned
                       (see below) so run_video() can write them into video
                       files instead.
        verbose: set False to demote per-frame detail from INFO to DEBUG
                 (keeps it out of the console but still in the log file).

        Returns a summary dict. Also includes two internal keys, popped by
        run_video() before results are stored/logged:
          - "_sparse_frame": the detection-boxes frame (green boxes + conf).
          - "_dense_frame" : the density-map overlay frame.
        These are always built (regardless of save_visuals) so video mode
        never needs to touch disk per frame.
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

            for det in detection_result:
                x, y, w, h = det.box

                cv2.rectangle(
                    sparse_input,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),   # Green
                    4              # Thickness
                )

                cv2.putText(
                    sparse_input,
                    f"{det.confidence:.2f}",
                    (x, y - 8),                    # Bottom-left corner of the text (slightly above the box)
                    cv2.FONT_HERSHEY_SIMPLEX,      # Font type
                    4,                             # Font scale (text size)
                    (0, 255, 0),                   # Green
                    10,                            # Thickness
                )

        scene_type = self.density_decider.decide(detection_result, sparse_input.shape)
        level(f"Scene Type: {scene_type}")

        sparse_count, kept_dets = self.bbox_counter.count(detection_result)
        level(f"Sparse Count: {sparse_count}")
        log.debug(f"Kept Dets: {kept_dets}")  # noisy raw data -> always debug

        for det in detection_result:
            sparse_input = det.plot(sparse_input)

        if save_visuals:
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

        # Always build the density overlay (needed for video mode too), and
        # only write it to disk when save_visuals is on.
        density_overlay = self.density_predictor.visualize(dense_input, density_map)

        if save_visuals:
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
            # Internal only -- popped by run_video() before results are
            # returned/logged.
            "_sparse_frame": sparse_input,
            "_dense_frame": density_overlay,
        }

        level("=" * 75)
        level("Frame Summary:")
        level(f"Input Type        : {summary['tag']}")
        level(f"Scene Type        : {summary['scene_type'].title()}")
        level(f"Detected Heads    : {summary['detected_heads']}")
        level(f"Valid Detections  : {summary['kept_detections']}")
        level(f"Sparse Count      : {summary['sparse_count']:.2f}")
        level(f"Dense Count       : {summary['dense_count']:.2f}")
        level(f"Final Crowd Count : {summary['fused_count']:.2f}")

        return summary

    def run_video(
        self,
        video_path: str,
        verbose_per_frame: bool = False,
        write_video: bool = True,
        sparse_video_path: str = "output/sparse_video.mp4",
        dense_video_path: str = "output/dense_video.mp4",
    ):
        """
        Runs the pipeline over sampled frames of a video, using FrameExtractor
        (box 2) to handle decoding/sampling/resizing.

        No per-frame PNGs are written in video mode. Instead, if
        write_video=True, two annotated videos are produced:
          - sparse_video_path: detection boxes + running count overlay.
          - dense_video_path : density-map overlay + running count overlay.
        """
        os.makedirs("output", exist_ok=True)
        results = []

        sparse_writer = None
        dense_writer = None

        video_fps = (
            getattr(self.cfg.frame_extraction, "output_fps", None)
            or getattr(self.cfg.frame_extraction, "sample_fps", None)
            or getattr(self.cfg.frame_extraction, "target_fps", None)
            or 25.0
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        def _open_writer(path, frame):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(path, fourcc, video_fps, (w, h))
            if not writer.isOpened():
                log.error(
                    f"Failed to open VideoWriter at {path} "
                    f"(fps={video_fps}, size={w}x{h}). Check codec/path."
                )
                return None
            log.info(f"Video -> {path} (fps={video_fps}, size={w}x{h})")
            return writer

        log.info(f"Processing Video: {video_path} ...")
        with FrameExtractor(video_path, self.cfg.frame_extraction) as extractor:
            for frame_idx, timestamp, frame in extractor:
                tag = f"frame_{frame_idx:06d}"
                summary = self.run_image(
                    frame,
                    tag=tag,
                    save_visuals=False,
                    verbose=verbose_per_frame,
                )

                sparse_frame = summary.pop("_sparse_frame", None)
                dense_frame = summary.pop("_dense_frame", None)

                if write_video:
                    label = f"Count: {summary['fused_count']:.1f} | {summary['scene_type']}"

                    if sparse_frame is not None:
                        cv2.putText(
                            sparse_frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        )
                        if sparse_writer is None:
                            sparse_writer = _open_writer(sparse_video_path, sparse_frame)
                        if sparse_writer is not None:
                            sparse_writer.write(sparse_frame)
                    else:
                        log.warning(f"{tag}: missing '_sparse_frame', skipping sparse video write.")

                    if dense_frame is not None:
                        cv2.putText(
                            dense_frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        )
                        if dense_writer is None:
                            dense_writer = _open_writer(dense_video_path, dense_frame)
                        if dense_writer is not None:
                            dense_writer.write(dense_frame)
                    else:
                        log.warning(f"{tag}: missing '_dense_frame', skipping dense video write.")

                summary["timestamp"] = round(timestamp, 2)
                results.append(summary)
                log.info(
                    f"[t={timestamp:6.2f}s] frame {frame_idx:6d} -> "
                    f"fused_count={summary['fused_count']}"
                )

        if sparse_writer is not None:
            sparse_writer.release()
            log.info(f"Sparse (detection) video saved to {sparse_video_path}")
        if dense_writer is not None:
            dense_writer.release()
            log.info(f"Dense (density map) video saved to {dense_video_path}")

        if not results:
            log.warning("No frames processed.")
            return results

        avg_fused = sum(r["fused_count"] for r in results) / len(results)
        max_fused = max(r["fused_count"] for r in results)

        log.info("=" * 75)
        log.info("Crowd Counting Video Summary")
        log.info(f"Frames Processed : {len(results)}")
        log.info(f"Average Count    : {avg_fused:.2f}")
        log.info(f"Maximum Count    : {max_fused:.2f}")

        return results