import cv2
import numpy as np
import os


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
