from src.models.segmenter import OliveSegmenter
from src.models.utils import  times_analyzer
from src.models.evaluator import OliveEvaluator
import os
from src.utils.slice_detection_utils import save_config_file

model_path = "checkpoints/best.pt"
# Initialize Segmenter
segmenter = OliveSegmenter(model_path)

# Configuration
folder_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_80/val"
conf=0.5
overlap_ratio=0.2
slice_size=1280
output_path="/mnt/c/Datasets/OlivePG/segmentation_result_best_val_0.5_0.2_1280"
outputs_labels_dir=os.path.join(output_path, "labels")
times_path=os.path.join(output_path, "times_summary.json")
metrics_path=os.path.join(output_path, "evaluation_summary.json")
config_path=os.path.join(output_path, "segmentation_config.json")


# Segment the olives in the folder and save the final labels and times summary
print("Starting segmentation...")
results_summary, times_summary = segmenter.segment_folder(
    folder_path=folder_path, 
    conf=conf, 
    overlap_ratio=overlap_ratio, 
    slice_size=slice_size, 
    output_label_dir=outputs_labels_dir
)

# Analyze times and save to JSON
print("Analyzing times...")
times_analyzer(times_summary, save_json=True, output_path=times_path)

# Evaluate the final boxes against ground truth and save metrics to JSON
# Note: Evaluator converts polygon labels to boxes for evaluation.
print("Evaluating results...")
evaluator = OliveEvaluator(
    gt_folder=os.path.join(folder_path, "labels"),
    pred_folder=outputs_labels_dir
)
evaluator.evaluate(save_json=True, output_path=metrics_path)

config = {
    "model_path": model_path,
    "folder_path": folder_path,
    "conf": conf,
    "overlap_ratio": overlap_ratio,
    "slice_size": slice_size,
    "output_path": output_path,
    "outputs_labels_dir": outputs_labels_dir,
    "times_path": times_path,
    "metrics_path": metrics_path
}
save_config_file(config, config_path)
print("Done.")
