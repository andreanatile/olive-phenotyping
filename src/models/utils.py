from pathlib import Path
import yaml
import torch
import cv2
import numpy as np


def parse_cfg(yaml_file_path):
    """
    Parses a YAML file and extracts the absolute paths for specified keys.
    """
    try:
        with open(yaml_file_path, "r") as file:
            config = yaml.safe_load(file)

        base_path = Path(config.get("path"))
        if not base_path:
            raise ValueError("The 'path' key is missing or empty in the YAML file.")

        keys_to_process = ["bbox_gt", "seg_gt"]

        for key in keys_to_process:
            relative_path = config.get(key)
            if relative_path:
                # Join base and relative paths, then get the absolute path
                gt_path = base_path / relative_path
                config[key] = gt_path.resolve()
            else:
                config[key] = None

        return config

    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None


def plot_results(
    img_path: str,
    boxes: torch.Tensor,
    show: bool = True,
    output_path: str = None,
):
    # 1. Convert string path to Path object
    path_obj = Path(img_path)
    path_output = Path(output_path) if output_path else None

    # 2. Create the new filename: "sum_" + original name
    # .parent gets the folder, .name gets 'image.jpg'
    output_path = path_output / f"sum_{path_obj.name}"

    # 3. Load the image
    img = cv2.imread(str(path_obj))  # cv2 needs string, not Path object
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    canvas = img.copy()

    # 4. Convert tensor to numpy
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()

    # 5. Draw the boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 6. Add counter label
    count_text = f"Total Olives: {len(boxes)}"
    cv2.putText(
        canvas, count_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4
    )

    # 7. Save the image using the new Pathlib path
    cv2.imwrite(str(output_path), canvas)
    print(f"Result saved to: {output_path}")

    # 8. Display logic
    if show:
        h, w = canvas.shape[:2]
        display_img = cv2.resize(canvas, (w // 3, h // 3)) if w > 2500 else canvas
        cv2.imshow("Olive Detection Results", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def times_analyzer(times_summary: dict):
    """
    Analyzes the times summary dictionary to compute average times for each stage.
    """
    total_images = len(times_summary)
    if total_images == 0:
        print("No images processed.")
        return {}

    # Initialize accumulators
    total_preprocessing = 0.0
    total_inference = 0.0
    total_nms = 0.0
    total_counting = 0.0
    max_total_counting_time = {"preprocessing_time": 0.0,
            "inference_time": 0.0,
            "nms_time": 0.0,
            "total_counting_time": 0.0}

    # Accumulate times
    for times in times_summary.values():
        total_preprocessing += times.get("preprocessing_time", 0.0)
        total_inference += times.get("inference_time", 0.0)
        total_nms += times.get("nms_time", 0.0)
        total_counting += (
            times.get("preprocessing_time", 0.0) +
            times.get("inference_time", 0.0) +
            times.get("nms_time", 0.0)
        )
        if times.get("total_counting_time", 0.0) > max_total_counting_time["total_counting_time"]:
            max_total_counting_time = times

    # Compute averages
    avg_times = {
        "average_preprocessing_time": total_preprocessing / total_images,
        "average_inference_time": total_inference / total_images,
        "average_nms_time": total_nms / total_images,
        "average_counting_time": total_counting / total_images,
    }

    return avg_times,max_total_counting_time