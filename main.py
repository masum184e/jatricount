import cv2

from pipeline import CrowdCountingPipeline

def main():
    pipeline = CrowdCountingPipeline()
    
    image_path = r"test_data\test_9.jpg"
    frame = cv2.imread(image_path)
    pipeline.run_image(frame)

if __name__ == "__main__":
    main()
