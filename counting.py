from src.models.counting import OliveCounter
from src.models.utils import plot_results

model_path = "checkpoints/best.pt"
counter = OliveCounter(model_path)
output_path = "/home/girobat/Pictures/cnr"
img_path = "data/foto olivo del  07.08.24/da0r1ar/IMG_20240807_082540.jpg"
# img_path = "data/Normalized/corrected/IMG_20240807_083339.jpg"
# results = counter.count(img_path=img_path, conf=0.50, overlap_ratio=0.2)
total_count, final_boxes = counter.count(
    img_path=img_path, conf=0.50, overlap_ratio=0.2
)
plot_results(img_path=img_path, boxes=final_boxes, show=True, output_path=output_path)
