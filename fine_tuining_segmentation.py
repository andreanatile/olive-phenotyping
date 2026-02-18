from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer
import pandas as pd
import os

# 1. Configuration - Use Segmentation models (-seg)
# Options: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt
model_variants = ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt']
dataset_path = "/mnt/c/Datasets/OlivePG/olive_dataset_yolo/data.yaml"
comparison_results = []

for model_name in model_variants:
    run_name = f"fb_seg_comp_{model_name.split('-')[0]}"
    print(f"\n🚀 Training Segmentation Model: {model_name}")
    
    # 2. Initialize and Train
    model = YOLO(model_name)
    results = model.train(
        data=dataset_path,
        epochs=100,
        patience=10,
        imgsz=640,
        project="SegmentComparison",
        name=run_name,
        exist_ok=True
    )
    
    # 3. Optimize Saved Weights
    weights_dir = f"SegmentComparison/{run_name}/weights"
    if os.path.exists(weights_dir):
        for weight_file in os.listdir(weights_dir):
            if weight_file.endswith(".pt"):
                strip_optimizer(os.path.join(weights_dir, weight_file))

    # 4. Collect Stats - Segmentation uses 'metrics/mAP50(M)' for Masks
    comparison_results.append({
        "Model": model_name,
        "Box_mAP50": results.results_dict.get('metrics/mAP50(B)', 0),
        "Mask_mAP50": results.results_dict.get('metrics/mAP50(M)', 0), # (M) is for Mask
        "Best_Epoch": results.fitness_epoch + 1 if hasattr(results, 'fitness_epoch') else "N/A"
    })

# 5. Summary Table
df = pd.DataFrame(comparison_results)
print("\n" + "="*40)
print("SEGMENTATION PERFORMANCE SUMMARY")
print("="*40)
print(df)