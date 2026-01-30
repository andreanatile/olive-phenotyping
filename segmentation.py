from src.models.segmenter import OliveSegmenter
from src.models.utils import  times_analyzer
from src.models.evaluator import OliveEvaluator
import os
from src.utils.slice_detection_utils import save_config_file
from src.utils.segmentation_utils import visualize_segmentation_comparison

model_path = "checkpoints/best_seg.pt"
# Initialize Segmenter
segmenter = OliveSegmenter(model_path)

# Configuration
folder_path="/mnt/c/Datasets/OlivePG/olive_dataset_yolo/val"
conf=0.5
overlap_ratio=0.2
slice_size=1280
iou_threshold=0.5
output_path="/mnt/c/Datasets/OlivePG/segmentation_result_nano_0.5_0.2_1280"
outputs_labels_dir=os.path.join(output_path, "labels")
times_path=os.path.join(output_path, "times_summary.json")
metrics_path=os.path.join(output_path, "evaluation_summary.json")
config_path=os.path.join(output_path, "segmentation_config.json")
gt_folder_path=os.path.join(folder_path, "labels")
images_gt_path=os.path.join(folder_path, "images")
visualize_predict_gt_folder=os.path.join(output_path, "segmentation_comparison")


# Segment the olives in the folder and save the final labels and times summary
print("Starting segmentation...")
results_summary, times_summary = segmenter.segment_folder(
    folder_path=folder_path, 
    conf=conf, 
    overlap_ratio=overlap_ratio, 
    slice_size=slice_size, 
    output_label_dir=outputs_labels_dir,
    iou_threshold=iou_threshold
)

# Analyze times and save to JSON
print("Analyzing times...")
times_analyzer(times_summary, save_json=True, output_path=times_path)

# Evaluate the final boxes against ground truth and save metrics to JSON
# Note: Evaluator converts polygon labels to boxes for evaluation.
print("Evaluating results...")
evaluator = OliveEvaluator(
    gt_folder=gt_folder_path,
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
    "metrics_path": metrics_path,
    "iou_threshold": iou_threshold
}
save_config_file(config, config_path)
print("Done.")



visualize_segmentation_comparison(pred_dir=outputs_labels_dir, gt_dir=gt_folder_path, img_dir=images_gt_path, output_dir=visualize_predict_gt_folder)