import os
from PIL import Image
from src.utils.resize_utils import  batch_resize_and_process

# --- Configuration ---
if __name__ == "__main__":
    # 1. Root folder for original data (images and labels should be in subfolders here)
    BASE_INPUT = "/mnt/c/Datasets/OlivePG/bbox_ground_truth"
    INPUT_IMAGE_FOLDER = os.path.join(BASE_INPUT, "images") 
    INPUT_LABEL_FOLDER = os.path.join(BASE_INPUT, "labels")

    # 2. Root folder for output data (images and labels will be saved in subfolders here)
    BASE_OUTPUT = "/mnt/c/Datasets/OlivePG/bbox_ground_truth_640"
    OUTPUT_IMAGE_FOLDER = os.path.join(BASE_OUTPUT, "images")
    OUTPUT_LABEL_FOLDER = os.path.join(BASE_OUTPUT, "labels")

    TARGET_SIZE = (640, 480)

    # Define the specific dimensions to include: (3648x2736) or (2736x3648)
    REQUIRED_DIMENSIONS = {(3648, 2736), (2736, 3648)}
    # ---------------------

    # Run the function
    batch_resize_and_process(INPUT_IMAGE_FOLDER, INPUT_LABEL_FOLDER, 
                            OUTPUT_IMAGE_FOLDER, OUTPUT_LABEL_FOLDER, 
                            TARGET_SIZE, REQUIRED_DIMENSIONS)
    print("\nBatch image and label processing complete.")