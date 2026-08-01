import cv2

from utils import show_images

from config import PreprocessingConfig, PipelineConfig

from modules.preprocessing import Preprocessor
from modules.head_detection import HeadDetector
from modules.density_decision import DensityDecisionModule

class CrowdCountingPipeline:
    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()

        self.preprocessor = Preprocessor(self.cfg.preprocessing)
        self.head_detector = HeadDetector(self.cfg.head_detection)
        self.density_decider = DensityDecisionModule(self.cfg.density_decision)

    def run_image(self, raw_frame):
        print("Processing Image ...")
        process_frame = self.preprocessor.process(raw_frame)
        show_images(
            (raw_frame, "Original"),
            (process_frame, "Process Frame"),
            output_path="output/process.png"
        )
        print("Preprocess End")
        print("Detecting Head ...")

        detection_result = self.head_detector.detect(process_frame)
        scene_type = self.density_decider.decide(detection_result, process_frame.shape)
        print(f"Scene Type: {scene_type}")

        for det in detection_result:
            process_frame = det.plot(process_frame)

        show_images(
            (raw_frame, "Original"),
            (process_frame, "Process Frame"),
            output_path="output/detection.png"
        )
        print("Head Detected")