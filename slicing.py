from src.utils.slice_detection_utils import slicer

if __name__ == "__main__":

    # Define your folder paths
    BASE_DATA_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_80"  # Input folder
    OUTPUT_DIR = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_640"  # Output folder

    # List of dataset splits
    split_names = ["train", "val"]
    slicer(
        BASE_DATA_DIR=BASE_DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        split_names=split_names,
        slice_size=640,
        overlap_ratio=0.20,
        keep_empty_patch=True,
    )
