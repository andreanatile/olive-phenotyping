from src.utils.slice_utils import slicer
import sys
import os


if __name__ == "__main__":

    # Define your folder paths
    BASE_DATA_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_70"  # Input folder
    OUTPUT_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_70_patch_nkeep"  # Output folder

    # List of dataset splits
    split_names = ["train", "val", "test"]
    slicer(
        base_data_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        split_names=split_names,
        slice_size=640,
        overlap=0.20,
        keep_empty_patch=True,
    )
