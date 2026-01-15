from ultralytics import YOLO
from src.utils.slice_utils import slice_img
import numpy as np
from torchvision.ops import nms
import torch


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
                slice_size=640,
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
                slice_size=640,
                overlap_ratio=overlap_ratio,
            )
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return None

        # result[7].show()
        results = self.model.predict(tiles, conf=conf, verbose=True)

        # Apply NMS and get final count
        total_count, final_boxes = self.nms_count(
            results, coordinates, iou_threshold=0.15
        )
        return total_count, final_boxes

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
        total_olives = len(final_boxes)

        print(f"Final Count: {total_olives}")
        return total_olives, final_boxes
