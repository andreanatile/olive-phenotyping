from src.models.counter import OliveCounter
from src.models.utils import  times_analyzer
from src.models.evaluator import OliveEvaluator
import os
from src.utils.slice_detection_utils import save_config_file
model_path = "runs/detection_patch_comparison/fb_comp_yolo11n/weights/best.pt"
counter = OliveCounter(model_path)

# Configuration
folder_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_80/val"
conf=0.5
overlap_ratio=0.2
slice_size=640
output_path="/mnt/c/Datasets/OlivePG/count_result_best_val_0.5_0.2_640"
outputs_labels_dir=os.path.join(output_path, "labels")
times_path=os.path.join(output_path, "times_summary.json")
metrics_path=os.path.join(output_path, "evaluation_summary.json")
config_path=os.path.join(output_path, "counting_config.json")



# Count the olives in the folder and save the final labels and times summary
results_summary,times_summary = counter.count_folder(folder_path=folder_path, conf=conf, overlap_ratio=overlap_ratio, slice_size=slice_size, output_label_dir=outputs_labels_dir)

# Analyze times and save to JSON
times_analyzer(times_summary,save_json=True, output_path=times_path)

# Evaluate the final boxes against ground truth and save metrics to JSON
evaluator=OliveEvaluator(
    gt_folder=os.path.join(folder_path, "labels"),
    pred_folder=outputs_labels_dir
)
evaluator.evaluate(save_json=True, output_path=metrics_path)

config={
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