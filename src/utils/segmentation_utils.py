import os
import json
import shutil
import random
from pathlib import Path

def prepare_olive_dataset(json_file, image_dir, output_dir, train_ratio=0.8):
    # 1. Load COCO JSON
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # COCO structure uses lists for images and annotations
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # 2. Setup final directory structure
    final_path = Path(output_dir)
    for split in ['train', 'val']:
        (final_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (final_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 3. Shuffle the IMAGES list (not the whole dictionary)
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    for i, img_task in enumerate(images):
        split = 'train' if i < split_idx else 'val'
        
        img_id = img_task['id']
        img_name = img_task['file_name']
        label_name = Path(img_name).stem + '.txt'
        
        # 4. Find all annotations for this specific image ID
        yolo_lines = []
        img_annos = [a for a in annotations if a['image_id'] == img_id]
        
        for ann in img_annos:
            # COCO segmentation is a list of polygons
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                class_id = 0  # Your olive class
                
                for seg in ann['segmentation']:
                    # COCO coordinates are in pixels, YOLO needs normalized 0-1
                    # Format: x1, y1, x2, y2...
                    normalized = []
                    for i in range(0, len(seg), 2):
                        nx = seg[i] / img_task['width']
                        ny = seg[i+1] / img_task['height']
                        normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])
                    
                    yolo_lines.append(f"{class_id} " + " ".join(normalized))

        # 5. Move files
        src_img = os.path.join(image_dir, img_name)
        if os.path.exists(src_img) and yolo_lines:
            shutil.copy(src_img, final_path / split / 'images' / img_name)
            with open(final_path / split / 'labels' / label_name, 'w') as f:
                f.write("\n".join(yolo_lines))

    # 6. Create data.yaml
    yaml_content = f"path: {os.path.abspath(output_dir)}\ntrain: train/images\nval: val/images\n\nnames:\n  0: olive"
    with open(final_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"Dataset created successfully in: {output_dir}")

# --- SETTINGS ---

json_file="/mnt/c/Datasets/OlivePG/old/prova/result_coco.json"
image_dir="/mnt/c/Datasets/OlivePG/old/prova/images"
output_path="/mnt/c/Datasets/OlivePG/olive_dataset_yolo"
prepare_olive_dataset(
    json_file=json_file, # Path to your Label Studio JSON
    image_dir=image_dir,   # Folder where original images are
    output_dir=output_path,       # Where you want the final dataset
    train_ratio=0.8                        # 80% Train, 20% Val
)