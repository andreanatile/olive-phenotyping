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

