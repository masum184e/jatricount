import cv2

from utils import show_images

from config import PreprocessingConfig, PipelineConfig
from logger import get_logger, StepTimer

from modules.preprocessing import Preprocessor
from modules.head_detection import HeadDetector
from modules.density_decision import DensityDecisionModule
from modules.sparse_counter import BoundingBoxCounter
from modules.density_map import DensityMapPredictor
from modules.fusion import FusionModule

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

        log.info("CrowdCountingPipeline initialized")
 
    def run_image(self, raw_frame):
        log.info("=" * 60)
        log.info("Processing image | shape=%s", raw_frame.shape)
        process_frame = self.preprocessor.process(raw_frame)
        show_images(
            (raw_frame, "Original"),
            (process_frame, "Process Frame"),
            output_path="output/process.png"
        )
        sparse_input = process_frame.copy()
        dense_input = process_frame.copy()
        print("===========Preprocess End===========\n\n\n")
        print("Detecting Head ...")

        detection_result = self.head_detector.detect(sparse_input)
        scene_type = self.density_decider.decide(detection_result, sparse_input.shape)
        print(f"Scene Type: {scene_type}")
        sparse_count, kept_dets = self.bbox_counter.count(detection_result)
        print(f"Sparse Count: {sparse_count}")
        print(f"Kept Dets: {kept_dets}")

        for det in detection_result:
            sparse_input = det.plot(sparse_input)

        show_images(
            (raw_frame, "Original"),
            (sparse_input, "Process Frame"),
            output_path="output/detection.png"
        )
        print("===========Head Detected===========\n\n\n")

        print("Detecting Dense ...")
        dense_count, density_map = self.density_predictor.predict(dense_input, detection_result)
        density_overlay = self.density_predictor.visualize(dense_input, density_map)
        
        show_images(
            (raw_frame, "Original"),
            (density_overlay, "Density Map"),
            output_path="output/density.png"
        )
        print(f"Dense Count: {dense_count}")
        print("===========Dense Counted===========\n\n\n")

        fused_count = self.fusion.fuse(scene_type, sparse_count, dense_count)
        print(f"Fusion Count: {fused_count}")

        print("\n\n\n=== Crowd Counting Summary ===\n")

        print({
            "frames_processed": 1,
            "instantaneous_sparse_count": sparse_count,
            "instantaneous_dense_count": round(dense_count, 2),
            "final_fused_count": round(fused_count, 2),
            "scene_type": scene_type,
            "detected_heads": len(detection_result),
            "kept_detections": len(kept_dets),
        })