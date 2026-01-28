from src.models.counter import OliveCounter
from src.models.utils import  times_analyzer
from src.models.evaluator import OliveEvaluator
import os

model_path = "checkpoints/best.pt"
counter = OliveCounter(model_path)

folder_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_80/val"
conf=0.3
overlap_ratio=0.2
slice_size=640
output_path="/mnt/c/Datasets/OlivePG/count_result_best_val"
outputs_labels_dir=os.path.join(output_path, "labels")
times_path=os.path.join(output_path, "times_summary.json")

results_summary,times_summary = counter.count_folder(folder_path=folder_path, conf=conf, overlap_ratio=overlap_ratio, slice_size=slice_size, output_label_dir=outputs_labels_dir)
times_analyzer(times_summary,save_json=True, output_path=times_path)

evaluator=OliveEvaluator(
    gt_folder=os.path.join(folder_path, "labels"),
    pred_folder=outputs_labels_dir
)
evaluator.evaluate()