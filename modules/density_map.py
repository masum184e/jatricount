import time
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

from config import DensityMapConfig
from utils import Detection


# ==========================================================
# IMPORTANT: prevent OpenCV / PyTorch thread-pool deadlock.
#
# When cv2 calls (cvtColor, resize, etc.) run in the same
# process as torch inference, their internal thread pools
# (OpenCV's worker threads + PyTorch's OpenMP threads) can
# contend for the same CPU cores and deadlock silently --
# no exception, no traceback, the process just hangs.
#
# This must run before any heavy cv2/torch calls happen,
# so it's set at import time here.
# ==========================================================
cv2.setNumThreads(0)
torch.set_num_threads(1)


class DensityMapPredictor:

    def __init__(self, cfg: DensityMapConfig):

        self.cfg = cfg
        self.backend = cfg.backend

        # Optional: resize the long side of the frame to this many
        # pixels before running it through the mdnn model. Large
        # multi-column CNNs (FourColumnMDNN uses parallel branches
        # with big kernels) can take a very long time on CPU if fed
        # full-resolution frames, and that can look identical to a
        # hang. Set to None to disable resizing.
        self.mdnn_max_side: Optional[int] = getattr(
            cfg, "mdnn_max_side", 768
        )

        # ==================================================
        # PyTorch MDNN backend
        # ==================================================

        if self.backend == "mdnn":

            try:
                from mdnn.fourcolumn import FourColumnMDNN

            except ImportError as e:
                raise ImportError(
                    "Could not import FourColumnMDNN"
                ) from e

            # Device
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            print(f"[DensityMapPredictor] Using device: {self.device}")

            # Create model
            self._model = FourColumnMDNN()

            # Load weights
            checkpoint = torch.load(
                self.cfg.model_path,
                map_location=self.device
            )

            # Handle checkpoints saved with extra keys
            if isinstance(checkpoint, dict):

                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                elif "model" in checkpoint:
                    checkpoint = checkpoint["model"]

            self._model.load_state_dict(
                checkpoint
            )

            self._model.to(
                self.device
            )

            self._model.eval()

            # Image preprocessing
            self._transform = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

            # Warm up the model once so the first real call isn't
            # slowed down by lazy CUDA context init / kernel JIT
            # compilation (only matters on GPU, harmless on CPU).
            self._warmup()

        # ==================================================
        # Classical backend
        # ==================================================

        elif self.backend == "classical":

            self._model = None

        else:

            raise ValueError(
                f"Unknown density-map backend: {self.backend}"
            )

    # ======================================================
    # Warmup (mdnn only)
    # ======================================================

    def _warmup(self):
        print("[DensityMapPredictor] Warming up mdnn model...")
        t0 = time.time()

        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        try:
            with torch.no_grad():
                image = Image.fromarray(dummy)
                tensor = self._transform(image).unsqueeze(0).to(self.device)
                self._model(tensor)
        except Exception as e:
            # Don't crash startup on a warmup failure; the real
            # call will surface a proper error if something is
            # actually wrong.
            print(f"[DensityMapPredictor] Warmup failed (non-fatal): {e}")

        print(f"[DensityMapPredictor] Warmup done in {time.time() - t0:.2f}s")

    # ======================================================
    # Main prediction interface
    # ======================================================

    def predict(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> Tuple[float, np.ndarray]:

        if self.backend == "mdnn":

            return self._predict_mdnn(
                frame
            )

        return self._predict_classical(
            frame,
            detections
        )

    # ======================================================
    # PyTorch MDNN inference
    # ======================================================

    def _resize_for_mdnn(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize frame so its long side is at most self.mdnn_max_side.
        Returns the resized frame and the scale factor applied
        (resized / original), so callers can rescale outputs if needed.
        """
        if not self.mdnn_max_side:
            return frame, 1.0

        h, w = frame.shape[:2]
        long_side = max(h, w)

        if long_side <= self.mdnn_max_side:
            return frame, 1.0

        scale = self.mdnn_max_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        return resized, scale

    def _predict_mdnn(
        self,
        frame: np.ndarray
    ) -> Tuple[float, np.ndarray]:

        print(f"[mdnn] input frame shape={frame.shape}")

        # Downscale large frames before inference -- big multi-column
        # CNNs can be extremely slow on full-res CPU input, which
        # looks like a hang if left unaddressed.
        infer_frame, scale = self._resize_for_mdnn(frame)
        if scale != 1.0:
            print(
                f"[mdnn] resized to {infer_frame.shape} "
                f"(scale={scale:.3f}) for inference"
            )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            infer_frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image
        )

        # Tensor shape:
        # [C,H,W]
        input_tensor = self._transform(
            image
        )

        # Add batch dimension:
        # [1,C,H,W]
        input_tensor = input_tensor.unsqueeze(
            0
        )

        input_tensor = input_tensor.to(
            self.device
        )

        print(
            f"[mdnn] input tensor shape={tuple(input_tensor.shape)}, "
            f"starting forward pass..."
        )
        t0 = time.time()

        with torch.no_grad():

            output = self._model(
                input_tensor
            )

        print(f"[mdnn] forward pass done in {time.time() - t0:.2f}s")

        # Remove batch/channel dimensions
        density_map = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )

        count = float(
            density_map.sum()
        )

        return (
            count,
            density_map
        )

    # ======================================================
    # Classical density map
    # ======================================================

    def _predict_classical(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> Tuple[float, np.ndarray]:

        h, w = frame.shape[:2]

        ds = self.cfg.downsample

        dh = max(
            1,
            h // ds
        )

        dw = max(
            1,
            w // ds
        )

        density = np.zeros(
            (dh, dw),
            dtype=np.float32
        )

        # Add detection points

        for det in detections:

            cx, cy = det.centroid

            dx = int(
                cx / ds
            )

            dy = int(
                cy / ds
            )

            if (
                0 <= dx < dw and
                0 <= dy < dh
            ):

                density[dy, dx] += 1.0

        sigma = max(
            1.0,
            self.cfg.gaussian_sigma / ds
        )

        density = cv2.GaussianBlur(
            density,
            (0, 0),
            sigma
        )

        # Edge correction

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            60,
            150
        )

        edge_density = cv2.resize(
            edges,
            (dw, dh),
            interpolation=cv2.INTER_AREA
        ).astype(
            np.float32
        ) / 255.0

        edge_density = cv2.GaussianBlur(
            edge_density,
            (0, 0),
            sigma
        )

        detection_mass = density.sum()

        if (
            detection_mass > 0
            and edge_density.sum() > 0
        ):

            correction_budget = (
                0.15 *
                detection_mass
            )

            edge_density *= (
                correction_budget /
                edge_density.sum()
            )

            density += edge_density

        return (
            float(density.sum()),
            density
        )

    # ======================================================
    # Visualization
    # ======================================================

    def visualize(
        self,
        frame: np.ndarray,
        density_map: np.ndarray,
        alpha: float = 0.8,
        radius: int = 30
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        dmap = cv2.resize(
            density_map.astype(
                np.float32
            ),
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        dmax = dmap.max()

        if dmax > 0:

            dmap_norm = (
                dmap /
                dmax *
                255
            ).astype(
                np.uint8
            )

        else:

            dmap_norm = np.zeros_like(
                dmap,
                dtype=np.uint8
            )

        heatmap = cv2.applyColorMap(
            dmap_norm,
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            frame,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        # Peak detection

        kernel = np.ones(
            (31, 31),
            np.uint8
        )

        local_max = cv2.dilate(
            dmap,
            kernel
        )

        peaks = (
            (dmap == local_max)
            &
            (dmap > 0.2 * dmax)
        )

        ys, xs = np.where(
            peaks
        )

        for x, y in zip(xs, ys):

            cv2.circle(
                overlay,
                (x, y),
                radius,
                (255, 255, 255),
                -1
            )

        return overlay