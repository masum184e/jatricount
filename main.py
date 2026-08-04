"""
main.py
"""
import os
import cv2
from pipeline import CrowdCountingPipeline
from logger import get_logger

log = get_logger(__name__)


def print_intro():
    log.info("=" * 50)
    log.info("       Crowd Counting Pipeline")
    log.info("=" * 50)
    log.info("This tool estimates crowd counts from an image or a video "
              "using a hybrid sparse/dense approach.")


def get_mode() -> str:
    while True:
        log.info("Select input type:")
        log.info("  1. Image")
        log.info("  2. Video")
        choice = input("Enter choice (1/2): ").strip()
        if choice == "1":
            return "image"
        if choice == "2":
            return "video"
        log.warning("Invalid choice, please enter 1 or 2.")


def get_path(mode: str) -> str:
    while True:
        path = input(f"Enter path to {mode} file: ").strip().strip('"')
        if os.path.isfile(path):
            return path
        log.warning(f"File not found: {path}")


def main():
    print_intro()
    log.info("Session started")
    pipeline = CrowdCountingPipeline()
    mode = get_mode()
    path = get_path(mode)

    if mode == "image":
        frame = cv2.imread(path)
        if frame is None:
            log.error(f"Could not read image: {path}")
            return
        pipeline.run_image(frame, tag="image")
    else:
        pipeline.run_video(path)


if __name__ == "__main__":
    main()