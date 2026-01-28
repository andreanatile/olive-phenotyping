import json
import cv2
import numpy as np
import os
import shutil


def visualize_yolo_labels(base_folder):
    """
    Reads images and YOLO labels from subfolders 'images' and 'labels'
    and saves images with bounding boxes drawn in a new 'labeled' subfolder.

    Args:
        base_folder (str): The directory containing the 'images' and 'labels' subfolders
                           (e.g., 'tiled_yolo_dataset').
    """

    # --- 1. Define Paths and Setup ---
    image_dir = os.path.join(base_folder, "images")
    label_dir = os.path.join(base_folder, "labels")
    output_dir = os.path.join(base_folder, "labeled")  # New output subfolder

    # Check if input folders exist
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(
            f"Error: Required subfolders 'images' and 'labels' not found in {base_folder}."
        )
        return

    # Create the output folder
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Label Visualization in {base_folder} ---")

    # --- 2. Process Files ---
    processed_count = 0

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):

            base_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(image_dir, file_name)
            label_path = os.path.join(label_dir, base_name + ".txt")
            output_path = os.path.join(output_dir, file_name)

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Skipping {file_name}: Could not load image.")
                continue

            H, W, _ = img.shape

            # --- 3. Read and Draw Labels ---
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        try:
                            # YOLO format: class_id x_center y_center width height (all normalized 0-1)
                            _, x_c, y_c, w, h = map(float, line.strip().split())

                            # Denormalize to absolute pixel coordinates
                            x_center = int(x_c * W)
                            y_center = int(y_c * H)
                            box_w = int(w * W)
                            box_h = int(h * H)

                            # Calculate corners (x_min, y_min, x_max, y_max)
                            x_min = int(x_center - box_w / 2)
                            y_min = int(y_center - box_h / 2)
                            x_max = int(x_center + box_w / 2)
                            y_max = int(y_center + box_h / 2)

                            # Draw the rectangle (color is BGR format: Blue is [255, 0, 0])
                            # Thickness is set to 2
                            cv2.rectangle(
                                img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2
                            )

                            # Optional: Draw the class ID label (just using a small black filled rectangle)
                            # cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        except ValueError:
                            print(f"Skipping malformed line in {base_name}.txt")
                            continue

            # --- 4. Save Annotated Image ---
            cv2.imwrite(output_path, img)
            processed_count += 1

    print(
        f"--- Visualization Complete: {processed_count} images saved to {output_dir} ---"
    )


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


def process_sclice_labels(
    original_labels, slice_bbox, slice_size, x_start, y_start, area_threshold=0.1
):
    """
    Calculates the new YOLO labels for a single tile based on object intersection.
    """
    new_labels = []

    for class_id, x_min_orig, y_min_orig, x_max_orig, y_max_orig in original_labels:

        # 1. Intersection (Clipping)
        x_min_clip = max(x_min_orig, slice_bbox[0])
        y_min_clip = max(y_min_orig, slice_bbox[1])
        x_max_clip = min(x_max_orig, slice_bbox[2])
        y_max_clip = min(y_max_orig, slice_bbox[3])

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

            x_c_norm = x_c_tile / slice_size
            y_c_norm = y_c_tile / slice_size
            w_norm = w_tile / slice_size
            h_norm = h_tile / slice_size

            # Append the new label string
            new_labels.append(
                f"{class_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    return new_labels


# --- Main Folder Processing Function ---
def slicer(
    BASE_DATA_DIR,
    OUTPUT_DIR,
    split_names,
    slice_size=640,
    overlap_ratio=0.2,
    keep_empty_patch=True,
    area_threshold=0.1,
    save_visualization=False,
):
    """
    Function for automatize slicing of multiple dataset splits.

    Args:
        BASE_DATA_DIR (str): Base directory containing 'images' and 'labels' subfolders.
        OUTPUT_DIR (str): Directory to save the new tiled dataset.
        split_names (list): List of dataset splits to process (e.g., ['train', 'val', 'test']).
        slice_size (int): The side length of the square tiles (e.g., 640).
        overlap_ratio (float): The fractional overlap (e.g., 0.2 for 20%).
        keep_empty_patch (bool): Whether to keep tiles with no objects.
        area_threshold (float): Minimun area of the bounding boxes label to keep after slicing the image.
        save_visualization (bool): Whether to save visualization of labeled slices.
    """

    for split in split_names:
        input_dir = os.path.join(BASE_DATA_DIR, split)
        output_dir = os.path.join(OUTPUT_DIR, split)

        # Call the slicing function
        slice_folder(
            data_dir=input_dir,
            output_dir=output_dir,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            keep_empty_patch=keep_empty_patch,
        )


def slice_folder(
    data_dir,
    output_dir,
    slice_size=640,
    overlap_ratio=0.2,
    area_threshold=0.1,
    keep_empty_patch=False,
    save_visualization=False,
):
    """
    The main command to slice all images and labels in a folder.

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
    stride = int(slice_size * (1 - overlap_ratio))
    processed_count = 0

    print(f"--- Starting Tiling Process: Tile Size={slice_size}, Stride={stride} ---")

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
            num_x = int(np.ceil((W - slice_size) / stride)) + 1
            num_y = int(np.ceil((H - slice_size) / stride)) + 1

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
                    x_start = min(i * stride, W - slice_size)
                    y_start = min(j * stride, H - slice_size)
                    x_end = x_start + slice_size
                    y_end = y_start + slice_size

                    # Process and save labels
                    tile_bbox = [x_start, y_start, x_end, y_end]
                    new_labels = process_sclice_labels(
                        original_labels,
                        tile_bbox,
                        slice_size,
                        x_start,
                        y_start,
                        area_threshold,
                    )

                    # Set slice name and crop the image
                    slice_img = img[y_start:y_end, x_start:x_end]
                    slice_name = f"{base_name}_tile_{i}_{j}"

                    # Save label file
                    with open(
                        os.path.join(output_label_dir, f"{slice_name}.txt"), "w"
                    ) as f:
                        f.write("\n".join(new_labels))

                    if not new_labels and not keep_empty_patch:
                        print(f"Skipping empty tile at ({i}, {j}) for {file_name}")
                        continue  # Skip tiles with no valid labels

                    # Save image slice
                    cv2.imwrite(
                        os.path.join(output_img_dir, f"{slice_name}.jpg"), slice_img
                    )

            processed_count += 1
            print(f"✅ Processed {file_name} ({W}x{H}) into {num_x * num_y} tiles.")

    print(f"--- Tiling Complete: {processed_count} images processed. ---")
    if save_visualization:
        visualize_yolo_labels(output_dir)
        print(f"--- Label Visualization Complete ---")


def slice_img(
    img_path: str = None,
    img: list = None,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
):
    """
    Slices a single image into smaller tiles.

    Args:
        img_path (str): Path to the input image file.
        img (list): Input image as a numpy array. If provided, img_path is ignored.
        slice_size (int): The side length of the square tiles (e.g., 640).
        overlap_ratio (float): The fractional overlap (e.g., 0.2 for 20%).

    Returns:
        list: A list of image tiles as numpy arrays.
    """
    if img is None:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {img_path}")

    H, W, _ = img.shape
    stride = int(slice_size * (1 - overlap_ratio))

    tiles = []
    coordinates = []
    num_x = int(np.ceil((W - slice_size) / stride)) + 1
    num_y = int(np.ceil((H - slice_size) / stride)) + 1

    for i in range(num_x):
        for j in range(num_y):
            x_start = min(i * stride, W - slice_size)
            y_start = min(j * stride, H - slice_size)
            x_end = x_start + slice_size
            y_end = y_start + slice_size

            tile_img = img[y_start:y_end, x_start:x_end]
            tiles.append(tile_img)
            coordinates.append((x_start, y_start, x_end, y_end))

    return tiles, coordinates


def find_folder_difference(folder_a, folder_b, destination_folder=None):
    # Create the destination folder if it doesn't exist
    if destination_folder is not None and not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created directory: {destination_folder}")

    # Get sets of filenames
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))

    # Find files in A that are NOT in B
    unique_to_a = files_a - files_b

    print(f"Moving {len(unique_to_a)} unique files...")

    for filename in unique_to_a:
        
        source_path = os.path.join(folder_a, filename)
        print(source_path)
        if destination_folder is not None:
            dest_path = os.path.join(destination_folder, filename)
            
            # Check if it's actually a file (to avoid copying subfolders)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path) # copy2 preserves metadata
                print(f"Copied: {filename}")


import os

def get_all_jpg_paths(directory):
    """
    Scans the directory for all .jpg and .jpeg files 
    and returns a list of their absolute paths.
    """
    jpg_path_list = []
    
    # os.walk yields a 3-tuple (dirpath, dirnames, filenames)
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check for jpg or jpeg (case-insensitive)
            if file.lower().endswith((".jpg", ".jpeg")):
                # Create the full absolute path
                full_path = os.path.join(root, file)
                jpg_path_list.append(full_path)
                
    return jpg_path_list


def save_config_file(config, path):
    """
    Saves the configuration dictionary to a valid JSON file.
    """
    try:
        with open(path, "w") as f:
            # json.dump handles all the formatting, quotes, and braces for you
            json.dump(config, f, indent=4)
            
        print(f"Configuration saved to: {path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")