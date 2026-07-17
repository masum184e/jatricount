import numpy as np
import cv2

from config import PreprocessingConfig

class Preprocessor:
    def __init__(self, cfg: PreprocessingConfig):
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)

    def process(self, frame: np.ndarray) -> np.ndarray:
        out = frame

        if self.cfg.do_denoise:
            out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

        if self.cfg.do_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        if self.cfg.blur_kernel:
            out = cv2.GaussianBlur(out, self.cfg.blur_kernel, 0)

        return out
