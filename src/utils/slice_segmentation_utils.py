import cv2
import numpy as np
import os
import cv2
from shapely.geometry import Polygon, box

def process_segmentation_labels(original_labels, tile_bbox, slice_size, x_start, y_start, area_threshold=0.1):
    """
    Clips polygon segmentation labels to the tile boundary using shapely.
    """
    new_labels = []
    # Create a box for the tile boundary [x_min, y_min, x_max, y_max]
    tile_poly = box(*tile_bbox)

    for class_id, poly_coords in original_labels:
        # Create (x, y) pairs from flat list
        points = [(poly_coords[i], poly_coords[i+1]) for i in range(0, len(poly_coords), 2)]
        
        if len(points) < 3:
            continue
            
        obj_poly = Polygon(points)
        if not obj_poly.is_valid:
            obj_poly = obj_poly.buffer(0) # Fix self-intersecting geometries

        # 1. Intersection (Clipping)
        if obj_poly.intersects(tile_poly):
            inter = obj_poly.intersection(tile_poly)
            
            # Handle cases where intersection results in multiple pieces
            if inter.geom_type == 'MultiPolygon':
                polys = list(inter.geoms)
            elif inter.geom_type == 'Polygon':
                polys = [inter]
            else:
                continue

            for p in polys:
                # 2. Area check (ignore tiny slivers)
                if p.area / obj_poly.area < area_threshold:
                    continue

                # 3. Translate to tile relative and normalize to [0, 1]
                coords = np.array(p.exterior.coords)
                normalized_coords = []
                for x, y in coords[:-1]: # YOLO doesn't need the repeated last point
                    nx = (x - x_start) / slice_size
                    ny = (y - y_start) / slice_size
                    normalized_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

                new_labels.append(f"{class_id} " + " ".join(normalized_coords))

    return new_labels

def slicer_seg(
    BASE_DATA_DIR,
    OUTPUT_DIR,
    split_names=['train', 'val'],
    slice_size=640,
    overlap_ratio=0.2,
    keep_empty_patch=False,
    area_threshold=0.1
):
    """
    Coordinator function to process different splits (train, val, test) for segmentation.
    """
    for split in split_names:
        input_split_dir = os.path.join(BASE_DATA_DIR, split)
        output_split_dir = os.path.join(OUTPUT_DIR, split)
        
        print(f"--- Processing Split: {split} ---")
        
        # Call the folder-level slicer
        slice_folder_seg(
            data_dir=input_split_dir,
            output_dir=output_split_dir,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            area_threshold=area_threshold,
            keep_empty_patch=keep_empty_patch
        )

def slice_folder_seg(data_dir, output_dir, slice_size=640, overlap_ratio=0.2, area_threshold=0.1, keep_empty_patch=False):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    
    output_img_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    stride = int(slice_size * (1 - overlap_ratio))

    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img = cv2.imread(os.path.join(image_dir, file_name))
        if img is None: continue
        
        H, W, _ = img.shape
        base_name = os.path.splitext(file_name)[0]
        
        # Load Original Segmentation Labels (denormalize to pixels)
        original_polygons = []
        label_path = os.path.join(label_dir, base_name + ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    coords = [p * W if i % 2 == 0 else p * H for i, p in enumerate(parts[1:])]
                    original_polygons.append((class_id, coords))

        # Tiling logic
        num_x = int(np.ceil((W - slice_size) / stride)) + 1
        num_y = int(np.ceil((H - slice_size) / stride)) + 1

        for i in range(num_x):
            for j in range(num_y):
                x_start = min(i * stride, W - slice_size)
                y_start = min(j * stride, H - slice_size)
                tile_bbox = [x_start, y_start, x_start + slice_size, y_start + slice_size]

                new_seg_labels = process_segmentation_labels(
                    original_polygons, tile_bbox, slice_size, x_start, y_start, area_threshold
                )

                if not new_seg_labels and not keep_empty_patch:
                    continue 

                tile_name = f"{base_name}_tile_{i}_{j}"
                crop = img[y_start:y_start+slice_size, x_start:x_start+slice_size]
                
                cv2.imwrite(os.path.join(output_img_dir, f"{tile_name}.jpg"), crop)
                with open(os.path.join(output_label_dir, f"{tile_name}.txt"), "w") as f:
                    f.write("\n".join(new_seg_labels))
        
        print(f"✅ Sliced: {file_name}")


def visualize_yolo_segmentation(base_folder):
    """
    Reads images and YOLO segmentation labels (polygons) from subfolders 
    'images' and 'labels' and saves images with masks drawn.
    """
    # --- 1. Define Paths and Setup ---
    image_dir = os.path.join(base_folder, "images")
    label_dir = os.path.join(base_folder, "labels")
    output_dir = os.path.join(base_folder, "labeled_masks")

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(f"Error: Required subfolders 'images' and 'labels' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Starting Segmentation Visualization in {base_folder} ---")

    # --- 2. Process Files ---
    processed_count = 0

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            base_name = os.path.splitext(file_name)[0]
            label_name = base_name + ".txt"  # <--- FIXED: Defined here
            
            image_path = os.path.join(image_dir, file_name)
            label_path = os.path.join(label_dir, label_name)
            output_path = os.path.join(output_dir, file_name)

            img = cv2.imread(image_path)
            if img is None:
                continue

            H, W, _ = img.shape
            overlay = img.copy()

            # --- 3. Read and Draw Polygons ---
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line_num, line in enumerate(f):
                        try:
                            parts = list(map(float, line.strip().split()))
                            if len(parts) < 3:
                                continue
                            
                            # YOLO format: class_id x1 y1 x2 y2 ...
                            class_id = int(parts[0])
                            coords = parts[1:]
                            
                            # Reshape and denormalize
                            points = []
                            for i in range(0, len(coords), 2):
                                px = int(coords[i] * W)
                                py = int(coords[i+1] * H)
                                points.append([px, py])
                            
                            pts = np.array(points, np.int32).reshape((-1, 1, 2))

                            # Blue color for olives (BGR: 255, 0, 0)
                            cv2.fillPoly(overlay, [pts], (255, 0, 0))
                            # White border for contrast
                            cv2.polylines(img, [pts], True, (255, 255, 255), 1)

                        except Exception as e:
                            print(f"Error processing line {line_num} in {label_name}: {e}")
                            continue

            # --- 4. Blend and Save ---
            alpha = 0.4 # Transparency factor
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
            cv2.imwrite(output_path, img)
            processed_count += 1

    print(f"--- Complete: {processed_count} images saved to {output_dir} ---")