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

    def visualize(self, frame: np.ndarray, density_map: np.ndarray,
                  alpha: float = 0.8, radius: int = 30) -> np.ndarray:

        h, w = frame.shape[:2]

        # Heatmap
        dmap = cv2.resize(
            density_map.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        dmax = dmap.max()
        if dmax > 0:
            dmap_norm = (dmap / dmax * 255).astype(np.uint8)
        else:
            dmap_norm = np.zeros_like(dmap, dtype=np.uint8)

        heatmap = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

        # -------------------------
        # Find local peaks
        # -------------------------
        kernel = np.ones((31, 31), np.uint8)
        local_max = cv2.dilate(dmap, kernel)

        peaks = (dmap == local_max) & (dmap > 0.2 * dmax)

        ys, xs = np.where(peaks)

        # Draw large dots
        for x, y in zip(xs, ys):
            cv2.circle(overlay, (x, y), radius, (255, 255, 255), -1)

        return overlay