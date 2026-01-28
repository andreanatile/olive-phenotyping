import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

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

def yolo_to_mask(txt_path, shape):
    """
    Converts YOLO segmentation format (txt) to a binary mask.
    shape: tuple (height, width)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not os.path.exists(txt_path):
        return mask
        
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: 
            continue
            
        # Parse coordinates (skip class_id at index 0)
        coords = [float(x) for x in parts[1:]]
        
        # Convert to pixels
        points = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                px = int(coords[i] * w)
                py = int(coords[i+1] * h)
                points.append([px, py])
            
        if points:
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw polygon
            cv2.fillPoly(mask, [pts], (255))
            
    return mask

def visualize_segmentation_comparison(pred_dir, gt_dir, img_dir, output_dir, file_extension="*.jpg"):
    """
    Visualizes comparison between Predicted and Ground Truth masks overlaid on original images.
    
    Args:
        pred_dir (str): Directory containing predicted masks.
        gt_dir (str): Directory containing ground truth masks.
        img_dir (str): Directory containing original images.
        output_dir (str): Directory to save the comparison plots.
        file_extension (str): File extension to search for (default: "*.jpg").
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # search for files in img_dir with the given extension
    search_path = os.path.join(img_dir, file_extension)
    img_files = glob.glob(search_path)
    
    if not img_files:
        print(f"No files found in {img_dir} with extension {file_extension}")
        return

    print(f"Found {len(img_files)} images. Starting visualization...")

    for img_path in img_files:
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        
        # Try finding corresponding masks. 
        # Strategy: Look for name_no_ext.* in pred and gt dirs to handle different extensions (e.g. png vs jpg)
        pred_candidates = glob.glob(os.path.join(pred_dir, f"{name_no_ext}.*"))
        gt_candidates = glob.glob(os.path.join(gt_dir, f"{name_no_ext}.*"))
        
        if not pred_candidates:
            print(f"Warning: No predicted mask found for {basename}")
            continue
        if not gt_candidates:
            print(f"Warning: No GT mask found for {basename}")
            continue
            
        pred_path = pred_candidates[0]
        gt_path = gt_candidates[0]
        
        # Load Images
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Helper to load mask (handles images and YOLO txt)
        def load_mask(path, target_shape):
            if path.endswith('.txt'):
                return yolo_to_mask(path, target_shape)
            else:
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if m is None: return None
                if m.shape != target_shape:
                    m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                return m

        pred_mask = load_mask(pred_path, img.shape[:2])
        gt_mask = load_mask(gt_path, img.shape[:2])
        
        if pred_mask is None or gt_mask is None:
            print(f"Error loading masks for {basename}")
            continue

        # Create overlays
        def create_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
            overlay = image.copy()
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            
            mask_bool = mask > 0
            # Ensure safe broadcasting/indexing
            if mask_bool.any():
                overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 1-alpha, colored_mask[mask_bool], alpha, 0)
            return overlay

        gt_overlay = create_overlay(img, gt_mask, color=(0, 255, 0)) # Green for GT
        pred_overlay = create_overlay(img, pred_mask, color=(255, 0, 0)) # Red for Pred

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth (Green)")
        axes[1].axis('off')
        
        axes[2].imshow(pred_overlay)
        axes[2].set_title("Prediction (Red)")
        axes[2].axis('off')
        
        plt.tight_layout()
        out_file = os.path.join(output_dir, f"vis_{basename}")
        plt.savefig(out_file)
        plt.close(fig)
        
    print(f"Visualization complete. Saved to {output_dir}")