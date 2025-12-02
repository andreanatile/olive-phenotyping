import cv2
import numpy as np
import os

# --- Core Modules ---


def denormalize_yolo_to_abs(line, W, H):
    """
    Converts a single YOLO label line (normalized) to absolute pixel coordinates (x_min, y_min, x_max, y_max).
    """
    parts = list(map(float, line.strip().split()))
    if len(parts) != 5:
        return None, None  # Invalid label format

    class_id = int(parts[0])
    x_c_norm, y_c_norm, w_norm, h_norm = parts[1:]

    # Denormalize center, width, and height
    w = w_norm * W
    h = h_norm * H
    x_c = x_c_norm * W
    y_c = y_c_norm * H

    # Convert to x_min, y_min, x_max, y_max
    x_min = int(x_c - w / 2)
    y_min = int(y_c - h / 2)
    x_max = int(x_c + w / 2)
    y_max = int(y_c + h / 2)

    # Ensure coordinates are within original image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)

    return [class_id, x_min, y_min, x_max, y_max], class_id


def process_tile_labels(
    original_labels, tile_bbox, tile_size, x_start, y_start, area_threshold=0.1
):
    """
    Calculates the new YOLO labels for a single tile based on object intersection.
    """
    new_labels = []

    for class_id, x_min_orig, y_min_orig, x_max_orig, y_max_orig in original_labels:

        # 1. Intersection (Clipping)
        x_min_clip = max(x_min_orig, tile_bbox[0])
        y_min_clip = max(y_min_orig, tile_bbox[1])
        x_max_clip = min(x_max_orig, tile_bbox[2])
        y_max_clip = min(y_max_orig, tile_bbox[3])

        intersection_w = x_max_clip - x_min_clip
        intersection_h = y_max_clip - y_min_clip

        if intersection_w > 0 and intersection_h > 0:

            # 2. Truncation Threshold Check
            orig_area = (x_max_orig - x_min_orig) * (y_max_orig - y_min_orig)
            clipped_area = intersection_w * intersection_h

            if clipped_area / orig_area < area_threshold:
                # Discard object (e.g., a tiny sliver of a facebook logo) if it's too truncated
                continue

            # 3. Translation to Tile Coordinates
            x_min_tile = x_min_clip - x_start
            y_min_tile = y_min_clip - y_start
            x_max_tile = x_max_clip - x_start
            y_max_tile = y_max_clip - y_start

            # 4. Renormalization to YOLO Format
            w_tile = x_max_tile - x_min_tile
            h_tile = y_max_tile - y_min_tile
            x_c_tile = x_min_tile + w_tile / 2
            y_c_tile = y_min_tile + h_tile / 2

            x_c_norm = x_c_tile / tile_size
            y_c_norm = y_c_tile / tile_size
            w_norm = w_tile / tile_size
            h_norm = h_tile / tile_size

            # Append the new label string
            new_labels.append(
                f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    return new_labels


# --- Main Folder Processing Function ---


def process_folder(data_dir, output_dir, tile_size=640, overlap_ratio=0.2):
    """
    The main command to tile all images and labels in a folder.

    Args:
        data_dir (str): Base directory containing 'images' and 'labels' subfolders.
        output_dir (str): Directory to save the new tiled dataset.
        tile_size (int): The side length of the square tiles (e.g., 640).
        overlap_ratio (float): The fractional overlap (e.g., 0.2 for 20%).
    """

    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Define and create output subdirectories
    output_img_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Calculate required parameters
    stride = int(tile_size * (1 - overlap_ratio))
    processed_count = 0

    print(f"--- Starting Tiling Process: Tile Size={tile_size}, Stride={stride} ---")

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):

            # --- File Path Setup ---
            image_path = os.path.join(image_dir, file_name)
            base_name = os.path.splitext(file_name)[0]
            label_file_name = base_name + ".txt"
            label_path = os.path.join(label_dir, label_file_name)

            # --- Load Image and Labels ---
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {file_name}: Could not load image.")
                continue

            H, W, _ = img.shape

            # Calculate the number of tiles needed
            num_x = int(np.ceil((W - tile_size) / stride)) + 1
            num_y = int(np.ceil((H - tile_size) / stride)) + 1

            original_labels = []
            try:
                with open(label_path, "r") as f:
                    for line in f:
                        abs_bbox, _ = denormalize_yolo_to_abs(line, W, H)
                        if abs_bbox:
                            original_labels.append(abs_bbox)
            except FileNotFoundError:
                print(
                    f"Warning: Label file {label_file_name} not found. Tiling image only."
                )

            # --- Tiling Loop ---
            for i in range(num_x):
                for j in range(num_y):
                    # Calculate tile boundaries
                    x_start = min(i * stride, W - tile_size)
                    y_start = min(j * stride, H - tile_size)
                    x_end = x_start + tile_size
                    y_end = y_start + tile_size

                    # Crop image
                    tile_img = img[y_start:y_end, x_start:x_end]
                    tile_name = f"{base_name}_tile_{i}_{j}"

                    # Save image tile
                    cv2.imwrite(
                        os.path.join(output_img_dir, f"{tile_name}.jpg"), tile_img
                    )

                    # Process and save labels
                    tile_bbox = [x_start, y_start, x_end, y_end]
                    new_labels = process_tile_labels(
                        original_labels, tile_bbox, tile_size, x_start, y_start
                    )

                    if new_labels:
                        with open(
                            os.path.join(output_label_dir, f"{tile_name}.txt"), "w"
                        ) as f:
                            f.write("\n".join(new_labels))

            processed_count += 1
            print(f"✅ Processed {file_name} ({W}x{H}) into {num_x * num_y} tiles.")

    print(f"--- Tiling Complete: {processed_count} images processed. ---")


# --- Example Execution (The Single Command) ---

if __name__ == "__main__":
    # Define your folder paths
    BASE_DATA_DIR = "my_yolo_dataset_original"  # Input folder
    OUTPUT_DIR = "tiled_yolo_dataset"  # Output folder

    # Example: Running the function for the entire folder
    # Assuming BASE_DATA_DIR has 'images' and 'labels' subfolders
    process_folder(
        data_dir=BASE_DATA_DIR,
        output_dir=OUTPUT_DIR,
        tile_size=640,  # The target size for YOLO model
        overlap_ratio=0.2,  # 20% overlap
    )
