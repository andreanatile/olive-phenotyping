from ultralytics import YOLO

model=YOLO("checkpoints/best_final.pt")

yaml_640_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_640/bbox_gt_ul_640.yaml"
yaml_1280_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_1280/bbox_gt_ul_1280.yaml"
yaml_path="/mnt/c/Datasets/OlivePG/bbox_gt_ul_80/bbox_gt_ul_80.yaml"

#model.val(data=yaml_640_path,split="val",imgsz=640,batch=8,conf=0.5,iou=0.6)
metrics=model.val(data=yaml_1280_path,split="val",imgsz=1280,batch=8,conf=0.5,iou=0.5,verbose=True)



from ultralytics import YOLO

# Define the models and resolutions you want to test
configs = [
    {'name': 'Nano-640',  'path': 'yolo11n.pt', 'imgsz': 640},
    {'name': 'Nano-1280', 'path': 'yolo11n.pt', 'imgsz': 1280},
    {'name': 'Small-640', 'path': 'yolo11s.pt', 'imgsz': 640},
    {'name': 'Small-1280','path': 'yolo11s.pt', 'imgsz': 1280},
]

results_table = []

for config in configs:
    # Load your model (replace with your fine-tuned .pt paths)
    model = YOLO(config['path'])
    
    # Run validation
    metrics = model.val(data='your_olive_data.yaml', imgsz=config['imgsz'], split='val')
    
    # Extract specific data
    results_table.append({
        "Config": config['name'],
        "mAP50-95": metrics.results_dict['metrics/mAP50-95(B)'],
        "mAP_Small": metrics.results_dict['metrics/mAP50-95(small)'], # Critical for olives
        "Preprocess (ms)": metrics.speed['preprocess'],
        "Inference (ms)": metrics.speed['inference'],
        "Postprocess (ms)": metrics.speed['postprocess']
    })

# Print a simple summary
print(f"{'Model':<12} | {'mAP':<8} | {'mAP(S)':<8} | {'Inf (ms)':<8}")
for r in results_table:
    print(f"{r['Config']:<12} | {r['mAP50-95']:<8.4f} | {r['mAP_Small']:<8.4f} | {r['Inference (ms)']:<8.2f}")
