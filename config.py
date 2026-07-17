from dataclasses import dataclass

@dataclass
class PreprocessingConfig:
    do_denoise: bool = True
    do_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)
    blur_kernel: tuple = (3, 3)