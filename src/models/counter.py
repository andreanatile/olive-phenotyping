from ultralytics import YOLO
from src.utils.slice_detection_utils import slice_img
import numpy as np
from torchvision.ops import nms
import torch
from pathlib import Path
import cv2
import time


class OliveCounter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """
        Loads the YOLOWorld model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        print(f"Loading model from {model_path}")
        model = YOLO(model_path)
        return model

    def predict(
        self,
        img: list = None,
        img_path: str = None,
        conf: int = 0.25,
        overlap_ratio: float = 0.2,
        slice_size: int = 640,
    ):
        """
        Count the number of olives inside the image using YOLO model.
        Args:
            img (list): image in numpy array format.
            img_path (str): Path to the image file.
            conf (int): Confidence threshold for detections.
        Returns:
            counts (list): List of counts of detected olives for each image.
        """
        if img is None and img_path is None:
            raise ValueError("Either img or img_path must be provided.")
        try:
            tiles, coordinates = slice_img(
                img=img,
                img_path=img_path,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return None

        # result[7].show()
        results = self.model.predict(
            tiles, conf=conf, verbose=True, save=True, visualize=True
        )
        return results

    def count(
        self,
        img: list = None,
        img_path: str = None,
        conf: int = 0.25,
        overlap_ratio: float = 0.2,
        slice_size: int = 640,
        save_final_boxes: bool = False,
        output_label_dir: str = "labels"
    ):
        """
        Count the number of olives inside the image using YOLO model.
        Args:
            img (list): image in numpy array format.
            img_path (str): Path to the image file.
            conf (int): Confidence threshold for detections.
        Returns:
            counts (list): List of counts of detected olives for each image.
        """
        path_obj = Path(img_path)
        if img is None and img_path is None:
            raise ValueError("Either img or img_path must be provided.")
        try:
            start_preprocessing_time=time.time()
            tiles, coordinates = slice_img(
                img=img,
                img_path=img_path,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            end_preprocessing_time=time.time()
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return None

        # result[7].show()
        start_inference_time=time.time()
        results = self.model.predict(tiles, conf=conf, verbose=True)
        end_inference_time=time.time()
        
        

        start_nms_time=time.time()
        # Apply NMS and get final count
        total_count, final_boxes = self.nms_count(
            results, coordinates, iou_threshold=0.15
        )
        end_nms_time=time.time()

        preprocessing_time=end_preprocessing_time-start_preprocessing_time
        print(f"Preprocessing Time (s): {preprocessing_time:.2f}")

        inference_time=end_inference_time-start_inference_time
        print(f"Inference Time (s): {inference_time:.2f}")
        
        nms_time=end_nms_time-start_nms_time
        print(f"NMS Time (s): {nms_time:.2f}")

        counting_time = preprocessing_time + inference_time + nms_time
        print(f"Total Counting Time (s): {counting_time:.2f}")

        times={
            "preprocessing_time": preprocessing_time,
            "inference_time": inference_time,
            "nms_time": nms_time,
            "total_counting_time": counting_time
        }

        if save_final_boxes:
            self.save_labels(final_boxes, path_obj, output_dir=output_label_dir)
        return total_count, final_boxes, times

    def nms_count(self, results, coordinates, iou_threshold=0.5):
        all_global_boxes = []
        all_scores = []

        # Coordinate Transformation: Patch-space to Global-space
        for i, res in enumerate(results):
            if len(res.boxes) == 0:
                continue
            # Get the (x_start, y_start) for specific tile
            x_offset, y_offset, _, _ = coordinates[i]

            # Extract boxes in [x1, y1, x2, y2] format
            boxes = res.boxes.xyxy.clone().detach()
            scores = res.boxes.conf.clone().detach()

            # Add the offsets to the local coordinates
            boxes[:, [0, 2]] += x_offset  # Add x_start to x1 and x2
            boxes[:, [1, 3]] += y_offset  # Add y_start to y1 and y2

            all_global_boxes.append(boxes)
            all_scores.append(scores)

        # Combine all detections into single tensors
        if len(all_global_boxes) == 0 or all_global_boxes[0].shape[0] == 0:
            return 0  # No olives detected in any tile

        combined_boxes = torch.cat(all_global_boxes)
        combined_scores = torch.cat(all_scores)

        # 4. Apply Global Non-Maximum Suppression
        # This removes duplicates where tiles overlapped
        keep_indices = nms(combined_boxes, combined_scores, iou_threshold)

        final_boxes = combined_boxes[keep_indices]
        final_scores = combined_scores[keep_indices]
        
        # Combine boxes and scores: [x1, y1, x2, y2, score]
        final_data = torch.cat([final_boxes, final_scores.unsqueeze(1)], dim=1)
        total_olives = len(final_data)

        print(f"Final Count: {total_olives}")
        return total_olives, final_data

    def save_labels(self, final_boxes, img_path_obj: Path, output_dir: str = "labels", class_id: int = 0):
        """
        Saves final_boxes to a .txt file in YOLO format using pathlib.
        """
        # Create output directory if it doesn't exist
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Get image dimensions (OpenCV still needs string path)
        img = cv2.imread(str(img_path_obj))
        img_h, img_w = img.shape[:2]

        # 2. Match the image name exactly (.stem gets filename without extension)
        txt_file = out_path / f"{img_path_obj.stem}.txt"

        with txt_file.open("w") as f:
            for item in final_boxes:
                # Handle both 4-element (box only) and 5-element (box + score) cases
                item_list = item.tolist()
                if len(item_list) == 5:
                    x1, y1, x2, y2, conf = item_list
                else:
                    x1, y1, x2, y2 = item_list
                    conf = 1.0 # Default if missing
                
                # Conversion to YOLO center-based format
                w = x2 - x1
                h = y2 - y1
                x_center = x1 + (w / 2)
                y_center = y1 + (h / 2)

                # Normalization
                f.write(f"{class_id} {x_center/img_w:.6f} {y_center/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f} {conf:.6f}\n")
        
        print(f"Saved: {txt_file}")

    def count_folder(self, folder_path: str, conf: float = 0.25, slice_size: int = 640,overlap_ratio: float = 0.5, save_final_boxes: bool = True, output_label_dir: str = "labels"):
        """
        Counts olives for every image in a folder and saves the results.
        """
        folder = Path(folder_path)
        # Define common image extensions
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        
        # Use rglob to find all images recursively in subfolders
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(folder.rglob(ext)))
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        results_summary = {}
        times_summary = {}

        for img_path in image_paths:
            print(f"Processing: {img_path.name}...")
            try:
                # We call your existing 'count' method
                total_count, _, times = self.count(
                    img_path=str(img_path),
                    conf=conf,
                    slice_size=slice_size,
                    save_final_boxes=save_final_boxes,
                    overlap_ratio=overlap_ratio,
                    output_label_dir=output_label_dir
                )
                results_summary[img_path.name] ={"number of olive":total_count,
                                                    "times":times}
                times_summary[img_path.name] = times
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")

        return results_summary,times_summary