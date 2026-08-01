import cv2

from utils import show_images

from config import PreprocessingConfig, PipelineConfig

from modules.preprocessing import Preprocessor
from modules.head_detection import HeadDetector
from modules.density_decision import DensityDecisionModule
from modules.sparse_counter import BoundingBoxCounter
from modules.density_map import DensityMapPredictor

class CrowdCountingPipeline:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()

        self.preprocessor = Preprocessor(self.cfg.preprocessing)
        self.head_detector = HeadDetector(self.cfg.head_detection)
        self.density_decider = DensityDecisionModule(self.cfg.density_decision)
        self.bbox_counter = BoundingBoxCounter()
        self.density_predictor = DensityMapPredictor(self.cfg.density_map)

    def run_image(self, raw_frame):
        print("Processing Image ...")
        process_frame = self.preprocessor.process(raw_frame)
        show_images(
            (raw_frame, "Original"),
            (process_frame, "Process Frame"),
            output_path="output/process.png"
        )
        print("===========Preprocess End===========")
        print("Detecting Head ...")

        detection_result = self.head_detector.detect(process_frame)
        scene_type = self.density_decider.decide(detection_result, process_frame.shape)
        print(f"Scene Type: {scene_type}")
        sparse_count, kept_dets = self.bbox_counter.count(detection_result)
        print(f"Sparse Count: {sparse_count}")
        print(f"Kept Dets: {kept_dets}")

        for det in detection_result:
            process_frame = det.plot(process_frame)

        show_images(
            (raw_frame, "Original"),
            (process_frame, "Process Frame"),
            output_path="output/detection.png"
        )
        print("===========Head Detected===========")

        print("Detecting Dense ...")
        dense_count, density_map = self.density_predictor.predict(process_frame, detection_result)
        density_overlay = self.density_predictor.visualize(process_frame, density_map)
        
        show_images(
            (raw_frame, "Original"),
            (density_overlay, "Density Map"),
            output_path="output/density.png"
        )
        print(f"Dense Count: {dense_count}")
        print("===========Dense Counted===========")