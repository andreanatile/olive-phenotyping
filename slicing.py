from src.utils.slice_utils import slice_folder
import sys
import os


if __name__ == "__main__":

    # Define your folder paths
    BASE_DATA_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_70"  # Input folder
    OUTPUT_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_70_patch_nkeep"  # Output folder
    
    # List of dataset splits
    #split_names = ["train", "val", "test"]
    split_names = ["train"]
    for split in split_names:
        input_dir = os.path.join(BASE_DATA_DIR, split)
        output_dir = os.path.join(OUTPUT_DIR, split)

        # Call the slicing function
        slice_folder(
            data_dir=input_dir,
            output_dir=output_dir,
            slice_size=640,     # The target size for YOLO model
            overlap_ratio=0.2,  # 20% overlap
            keep_empty_patch=False  
        )
