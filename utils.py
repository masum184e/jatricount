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


def show_images(
    *images: Tuple[np.ndarray, str],
    output_path: str = "output/output.png",
    figsize: Tuple[int, int] = (15, 5)
) -> None:

    n = len(images)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=figsize
    )

    if n == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)