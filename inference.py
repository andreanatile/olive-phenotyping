from ultralytics import YOLO
import numpy as np

# 1. Load your specific model
model = YOLO('ModelComparison/fb_comp_yolo11n/weights/best.pt')

# 2. Run inference on a source (folder or single image)
results = model.predict(source='/mnt/c/Datasets/OlivePG/bbox_gt_ul_80/val/images', imgsz=640, device=0) # 0 for GPU

# 3. Extract speed metrics
# 'results' is a list of Result objects (one per image)
inference_times = [r.speed['inference'] for r in results]
average_time = np.mean(inference_times)

print(f"Average Inference Time: {average_time:.2f} ms per image")
print(f"Theoretical Throughput: {1000/average_time:.2f} FPS")