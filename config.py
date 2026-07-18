from dataclasses import dataclass, field

@dataclass
class PreprocessingConfig:
    do_denoise: bool = True
    do_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)
    blur_kernel: tuple = (3, 3)

@dataclass
class HeadDetectionConfig:
    backend: str = "yolo"            # "haar" | "yolo"
    haar_cascade: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.08
    min_neighbors: int = 4
    min_size: tuple = (18, 18)
    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.25

@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    head_detection: HeadDetectionConfig = field(default_factory=HeadDetectionConfig)
