from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer
import pandas as pd
import os

# 1. Configuration
model_variants = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']
dataset_path = "/mnt/c/Datasets/OlivePG/bbox_gt_ul_640/bbox_gt_ul_640.yaml"
comparison_results = []

for model_name in model_variants:
    run_name = f"fb_comp_{model_name.split('.')[0]}"
    print(f"\n🚀 Training: {model_name}")
    
    # 2. Initialize and Train
    model = YOLO(model_name)
    results = model.train(
        data=dataset_path,
        epochs=100,
        patience=10,
        imgsz=640,
        project="detection_patch_comparison",
        name=run_name,
        exist_ok=True        # Overwrites folder if it exists
    )
    
    # 3. Optimize Saved Weights (Optional)
    # This reduces file size for all saved checkpoints in the weights folder
    weights_dir = f"detection_patch_comparison/{run_name}/weights"
    for weight_file in os.listdir(weights_dir):
        if weight_file.endswith(".pt"):
            strip_optimizer(os.path.join(weights_dir, weight_file))

    # 4. Collect Stats
    comparison_results.append({
        "Model": model_name,
        "Best_mAP50": results.results_dict['metrics/mAP50(B)'],
        "Weights_Saved": f"Every 10 epochs in {weights_dir}"
    })

# 5. Summary Table
df = pd.DataFrame(comparison_results)
print("\n" + "="*30)
print("STRATEGIC COMPARISON SUMMARY")
print("="*30)
print(df)