from src.models.counting import OliveCounter

counter = OliveCounter("best.pt")

img_path = "data/foto olivo del  07.08.24/da0r1ar/IMG_20240807_082540.jpg"
results = counter.count(img_path=img_path, conf=0.25, overlap_ratio=0.2)
