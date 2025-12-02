from src.utils.slice_utils import slice_folder
import sys
import os


if __name__ == "__main__":

    # Define your folder paths
    BASE_DATA_DIR = "/home/girobat/OlivePG/optim_dataset/bbox_ground_truth_half_original"  # Input folder
    OUTPUT_DIR = "/home/girobat/OlivePG/optim_dataset/bbox_ground_truth_half_tiled"  # Output folder

    # Example: Running the function for the entire folder
    # Assuming BASE_DATA_DIR has 'images' and 'labels' subfolders
    slice_folder(
        data_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        slice_size=640,  # The target size for YOLO model
        overlap_ratio=0.2,  # 20% overlap
    )
