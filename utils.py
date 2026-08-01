import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Detection:
    
    box: Tuple[int, int, int, int]   # (x, y, w, h)
    confidence: float = 1.0

    @property
    def centroid(self) -> Tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    def plot(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:

        x, y, w, h = self.box

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            2,
        )

        label = f"{self.confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        return image

def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def show_images(
    *images: Tuple[np.ndarray, str],
    output_path: str = "output/output.png",
    figsize: Tuple[int, int] = (15, 5)
) -> None:

    n = len(images)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=figsize,
        gridspec_kw={"wspace": 0, "hspace": 0}
    )

    if n == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    # Remove all margins around the figure
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)