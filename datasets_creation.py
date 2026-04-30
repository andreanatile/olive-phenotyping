from src.utils.kfold import KFoldSplitter
from pathlib import Path


dataset_path=Path("mnt/c/Datasets/OliveProva/OliveProva/bbox_ground_truth_new")
dataset_yaml=Path("mnt/c/Datasets/OliveProva/OliveProva/bbox_ground_truth_new/bbox_ground_truth_new.yaml")
save_dir=Path("mnt/c/Datasets/OliveProva/kfold")

from src.utils.slice_detection_utils import slicer_kfold

#
splitter=KFoldSplitter()
# Note: create_kfold_splits returns the list of dataset.yaml paths if needed
splitter.create_kfold_splits(dataset_path, dataset_yaml, save_dir)

# This creates the patching dataset from the kfold splits
sliced_save_dir = Path("mnt/c/Datasets/OliveProva/kfold_sliced")
slicer_kfold(
    kfold_dir=str(save_dir),
    output_dir=str(sliced_save_dir),
    slice_size=640,
    overlap_ratio=0.2,
    keep_empty_patch=False,
    area_threshold=0.1
)

# --- Segmentation ---
from src.utils.slice_segmentation_utils import slicer_seg_kfold

# Please adjust these paths for your segmentation dataset
dataset_path_seg = Path("mnt/c/Datasets/OliveProva/OliveProva/seg_ground_truth_new")
dataset_yaml_seg = Path("mnt/c/Datasets/OliveProva/OliveProva/seg_ground_truth_new/seg_ground_truth_new.yaml")
save_dir_seg = Path("mnt/c/Datasets/OliveProva/kfold_seg")

# Create kfold splits for segmentation
splitter_seg = KFoldSplitter()
splitter_seg.create_kfold_splits(dataset_path_seg, dataset_yaml_seg, save_dir_seg)

# This creates the patching dataset from the segmentation kfold splits
sliced_save_dir_seg = Path("mnt/c/Datasets/OliveProva/kfold_seg_sliced")
slicer_seg_kfold(
    kfold_dir=str(save_dir_seg),
    output_dir=str(sliced_save_dir_seg),
    slice_size=640,
    overlap_ratio=0.2,
    keep_empty_patch=False,
    area_threshold=0.1
)
