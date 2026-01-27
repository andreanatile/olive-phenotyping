from ultralytics import YOLO

config_path = "/mnt/c/Datasets/OlivePG/seg_tiled_640/config_seg_640.yaml"

# --- 1. Fine-tune YOLO11-Nano Segmentation ---
model_nano = YOLO("checkpoints/yolo11n-seg.pt")
print("🚀 Starting Nano Training...")

results_nano = model_nano.train(
    data=config_path,
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    patience=20,
    project="runs/olive_seg_finetuning",
    name="yolo11n_olives", # Unique name for Nano
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    lr0=0.01,
    single_cls=True
)

# --- 2. Fine-tune YOLO11-Small Segmentation ---
# Note: You need to have yolo11s-seg.pt in your checkpoints folder
model_small = YOLO("checkpoints/yolo11s-seg.pt")
print("🚀 Starting Small Training...")

results_small = model_small.train(
    data=config_path,
    epochs=300,
    imgsz=640,
    batch=16, # If you get "Out of Memory", lower this for the 'Small' model
    device=0,
    workers=8,
    patience=20,
    project="runs/olive_seg_finetuning",
    name="yolo11s_olives", # Unique name for Small
    exist_ok=True,
    pretrained=True,
    optimizer="auto",
    lr0=0.01,
    single_cls=True
)