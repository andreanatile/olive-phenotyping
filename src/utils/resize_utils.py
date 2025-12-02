import os
from PIL import Image


def rotate_and_resize_yolo_labels(input_file):
    """
    Transforms normalized YOLO coordinates for a 90-degree clockwise rotation.
    If the image is rotated from Portrait (2736x3648) to Landscape (3648x2736).
    
    Formula: x_new = y_old, y_new = 1.0 - x_old, w_new = h_old, h_new = w_old
    """
    rotated_lines = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])

            # Apply the 90-degree clockwise rotation logic:
            x_c_new = y_c
            y_c_new = 1.0 - x_c
            w_new = h
            h_new = w

            new_line = f"{class_id} {x_c_new:.4f} {y_c_new:.4f} {w_new:.4f} {h_new:.4f}\n"
            rotated_lines.append(new_line)

        except Exception:
            # Skip malformed lines silently
            continue
            
    return rotated_lines


def batch_resize_and_process(input_img_folder, input_lbl_folder, output_img_folder, output_lbl_folder, target_size, required_dims):
    """
    Rotates/resizes images and transforms corresponding labels only for specified dimensions.
    """
    # 1. Create output folders if they don't exist
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_lbl_folder, exist_ok=True)
    
    target_width, target_height = target_size
    print(f"Starting batch process for target size: {target_width}x{target_height}...")

    for filename in os.listdir(input_img_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_stem = os.path.splitext(filename)[0]
            input_path = os.path.join(input_img_folder, filename)
            label_path = os.path.join(input_lbl_folder, name_stem + ".txt")
            
            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Skipping {filename}: Corresponding label file not found.")
                continue

            try:
                img = Image.open(input_path)
                original_width, original_height = img.size
                
                # --- CHECK 1: Filter by required dimensions ---
                if (original_width, original_height) not in required_dims:
                    print(f"Skipping {filename}: Dimensions {original_width}x{original_height} do not match required sizes.")
                    continue 
                
                processed_img = img
                label_lines = None
                
                # --- CHECK 2: Rotation & Label Transformation ---
                # Check for rotation: If the original image is Portrait (Height > Width) AND matches a required portrait size
                if original_height > original_width:
                    # Rotation: 90 degrees clockwise (Image.ROTATE_270 is 270 deg clockwise, which is 90 deg counter-clockwise, or 90 deg clockwise with Image.ROTATE_90)
                    # We use ROTATE_270 to ensure the image becomes landscape (Width > Height)
                    processed_img = img.transpose(Image.ROTATE_270)
                    
                    # Transform the label coordinates
                    label_lines = rotate_and_resize_yolo_labels(label_path)
                    
                    print(f"Processing {filename}: Rotated {original_width}x{original_height} -> {processed_img.size[0]}x{processed_img.size[1]} and TRANSFORMED labels.")
                
                else:
                    # No Rotation needed (image is already landscape)
                    processed_img = img
                    
                    # Read original labels (no transformation needed)
                    with open(label_path, 'r') as f:
                        label_lines = f.readlines()
                        
                    print(f"Processing {filename}: Resizing {original_width}x{original_height} (Labels preserved).")

                # --- Final Resize and Save ---
                # Resize image (Applied to both rotated and non-rotated images)
                resized_img = processed_img.resize(target_size)
                
                # Save the new image
                output_img_path = os.path.join(output_img_folder, filename)
                resized_img.save(output_img_path)
                
                # Save the new/transformed label file
                output_lbl_path = os.path.join(output_lbl_folder, name_stem + ".txt")
                with open(output_lbl_path, 'w') as f:
                    f.writelines(label_lines)
                
                print(f"  -> Saved successfully to {output_img_folder} at {target_width}x{target_height}")
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")