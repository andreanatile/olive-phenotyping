from ultralytics import YOLOE
import json

# Create a YOLOE model
model = YOLOE("checkpoints/yoloe-11s-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes

# Conduct model validation on the COCO128-seg example dataset
config_path="/mnt/c/Datasets/OlivePG/seg_tiled_640/config_seg_640.yaml"

metrics = model.val(data=config_path,save_json=True,conf=0.01)

maps={
    " map50-95(B)": metrics.box.map,
    " map50(B)": metrics.box.map50,
    "maps(B)": metrics.box.maps,
    " map50-95(M)": metrics.seg.map,
    " map50(M)": metrics.seg.map50,
    "maps(M)": metrics.seg.maps,
}  
print(maps)
# Extracting the metrics dictionary
stats = metrics.results_dict
# Save to a custom JSON file
with open("my_metrics.json", "w") as f:
    json.dump(stats, f, indent=4)
