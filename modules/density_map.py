from typing import List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

from config import DensityMapConfig
from utils import Detection

class DensityMapPredictor:
    def __init__(self, cfg: DensityMapConfig):
        self.cfg = cfg
        self.backend = cfg.backend

        if self.backend == "mdnn":
            try:
                from models.fourcolumn_mdnn import FourColumnMDNN
            except ImportError as e:
                raise ImportError("Could not load FourColumnMDNN model definition") from e

            self.device = self.cfg.device
            self._model = FourColumnMDNN()
            self._model.load_state_dict(
                torch.load(self.cfg.mdnn_weights, map_location=self.device)
            )
            self._model.to(self.device)
            self._model.eval()

            self._transform = transforms.Compose([transforms.ToTensor()])
        elif self.backend == "classical":
            self._model = None
        else:
            raise ValueError(f"Unknown density-map backend: {self.backend}")

    def predict(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[float, np.ndarray]:
        if self.backend == "mdnn":
            return self._predict_mdnn(frame)
        return self._predict_classical(frame, detections)

    def _predict_mdnn(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_img = self._transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self._model(input_img)

        density_map = output.squeeze().cpu().numpy()
        count = float(density_map.sum())
        return count, density_map

    def _predict_classical(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[float, np.ndarray]:
        h, w = frame.shape[:2]
        ds = self.cfg.downsample
        dh, dw = max(1, h // ds), max(1, w // ds)
        density = np.zeros((dh, dw), dtype=np.float32)

        for det in detections:
            cx, cy = det.centroid
            dx, dy = int(cx / ds), int(cy / ds)
            if 0 <= dy < dh and 0 <= dx < dw:
                density[dy, dx] += 1.0

        sigma = max(1.0, self.cfg.gaussian_sigma / ds)
        density = cv2.GaussianBlur(density, (0, 0), sigma)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        edge_density = cv2.resize(edges, (dw, dh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        edge_density = cv2.GaussianBlur(edge_density, (0, 0), sigma)

        detection_mass = density.sum()
        if detection_mass > 0 and edge_density.sum() > 0:
            correction_budget = 0.15 * detection_mass
            edge_density *= correction_budget / edge_density.sum()
            density += edge_density

        return float(density.sum()), density

    def visualize(self, frame: np.ndarray, density_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay density map as a heatmap on top of the frame."""
        h, w = frame.shape[:2]
        dmap = cv2.resize(density_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        dmax = dmap.max()
        dmap_norm = (dmap / dmax * 255.0).astype(np.uint8) if dmax > 0 else dmap.astype(np.uint8)

        heatmap = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)