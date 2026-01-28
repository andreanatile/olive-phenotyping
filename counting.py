from src.models.counter import OliveCounter
from src.models.utils import plot_results

model_path = "checkpoints/bestv2.pt"
counter = OliveCounter(model_path)
output_path = "/runs"
# img_path = "data/Normalized/corrected/IMG_20241021_092715.jpg"
# img_path = "data/Normalized/corrected/IMG_20240828_090602.jpg"
img_path = "data/Normalized/corrected/IMG_20240807_082537.jpg"
# img_path = "data/foto olivo del  07.08.24/da0r1ar/IMG_20240807_082540.jpg"
# img_path = "data/Normalized/corrected/IMG_20240807_083339.jpg"
# results = counter.count(img_path=img_path, conf=0.50, overlap_ratio=0.2)
total_count, final_boxes, times = counter.count(
    img_path=img_path, 
    conf=0.50, 
    overlap_ratio=0.2,
    slice_size=640,
    save_final_boxes=True,
    output_label_dir="/home/girobat/Olive/runs/custom_labels"
)
plot_results(img_path=img_path, boxes=final_boxes, show=True, output_path=output_path)
