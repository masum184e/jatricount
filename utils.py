import matplotlib.pyplot as plt
import cv2

def show_images(*images, output_path="output/output.png", figsize=(15, 5)):
    
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)