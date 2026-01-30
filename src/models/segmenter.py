from ultralytics import YOLO
from src.utils.slice_detection_utils import slice_img
import numpy as np
from torchvision.ops import nms
import torch
from pathlib import Path
import cv2
import time


class OliveSegmenter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """
        Loads the YOLO model from a checkpoint file.
        """
        print(f"Loading model from {model_path}")
        model = YOLO(model_path)
        return model

    def segment(
        self,
        img: list = None,
        img_path: str = None,
        conf: float = 0.25,
        overlap_ratio: float = 0.2,
        slice_size: int = 640,
        save_final_labels: bool = False,
        output_label_dir: str = "labels",
        iou_threshold: float = 0.5,
    ):
        """
        Segment olives inside the image using YOLO model.
        Args:
            img (list): image in numpy array format.
            img_path (str): Path to the image file.
            conf (float): Confidence threshold for detections.
        Returns:
            count (int): Total count.
            final_data (list): List of dicts with box, score, mask.
            times (dict): Timing info.
        """
        path_obj = Path(img_path)
        if img is None and img_path is None:
            raise ValueError("Either img or img_path must be provided.")
        
        try:
            start_preprocessing_time = time.time()
            tiles, coordinates = slice_img(
                img=img,
                img_path=img_path,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            end_preprocessing_time = time.time()
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return None

        # Predict with retina_masks=True for better mask quality if supported, or just default
        start_inference_time = time.time()
        # Ensure we ask for masks
        results = self.model.predict(tiles, conf=conf, verbose=True, retina_masks=True)
        end_inference_time = time.time()

        start_nms_time = time.time()
        # Apply NMS and get final results with masks
        total_count, final_data = self.nms_segment(
            results, coordinates, iou_threshold=iou_threshold
        )
        end_nms_time = time.time()

        preprocessing_time = end_preprocessing_time - start_preprocessing_time
        print(f"Preprocessing Time (s): {preprocessing_time:.2f}")

        inference_time = end_inference_time - start_inference_time
        print(f"Inference Time (s): {inference_time:.2f}")
        
        nms_time = end_nms_time - start_nms_time
        print(f"NMS Time (s): {nms_time:.2f}")

        counting_time = preprocessing_time + inference_time + nms_time
        print(f"Total Segmentation Time (s): {counting_time:.2f}")

        times = {
            "preprocessing_time": preprocessing_time,
            "inference_time": inference_time,
            "nms_time": nms_time,
            "total_counting_time": counting_time
        }

        if save_final_labels:
            self.save_labels(final_data, path_obj, output_dir=output_label_dir)
            
        return total_count, final_data, times

    def nms_segment(self, results, coordinates, iou_threshold=0.5):
        all_global_boxes = []
        all_scores = []
        all_masks = [] # List of list of polygon coordinates (normalized or absolute? Let's use absolute pixel coords first)

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
            boxes[:, [0, 2]] += x_offset
            boxes[:, [1, 3]] += y_offset

            # Extract masks
            if res.masks is not None:
                # xy is a list of polygons (each polygon is np array of points)
                # We need to shift them too.
                # res.masks.xy returns a list of arrays, one per object
                current_masks = res.masks.xy
                shifted_masks = []
                for poly in current_masks:
                    # poly is (N, 2)
                    poly_shifted = poly.copy()
                    poly_shifted[:, 0] += x_offset
                    poly_shifted[:, 1] += y_offset
                    shifted_masks.append(poly_shifted)
            else:
                # Should not happen if seg model is used, but handle gracefully
                shifted_masks = [None] * len(boxes)

            all_global_boxes.append(boxes)
            all_scores.append(scores)
            all_masks.extend(shifted_masks)

        # Combine all detections into single tensors
        if len(all_global_boxes) == 0 or all_global_boxes[0].shape[0] == 0:
            return 0, []

        combined_boxes = torch.cat(all_global_boxes)
        combined_scores = torch.cat(all_scores)
        
        # Apply Global Non-Maximum Suppression
        keep_indices = nms(combined_boxes, combined_scores, iou_threshold)

        final_boxes = combined_boxes[keep_indices]
        final_scores = combined_scores[keep_indices]
        
        # Filter masks using the same indices
        # all_masks is a flat list aligning with combined_boxes
        # keep_indices is a tensor of indices
        keep_indices_list = keep_indices.tolist()
        final_masks = [all_masks[i] for i in keep_indices_list]

        # Structure final data
        final_data = []
        for i in range(len(final_boxes)):
            item = {
                "box": final_boxes[i],    # Tensor (4,)
                "score": final_scores[i], # Tensor ()
                "mask": final_masks[i]    # Numpy array (N, 2)
            }
            final_data.append(item)

        total_olives = len(final_data)
        print(f"Final Count: {total_olives}")
        
        return total_olives, final_data

    def save_labels(self, final_data, img_path_obj: Path, output_dir: str = "labels", class_id: int = 0):
        """
        Saves final data to a .txt file in YOLO segmentation format.
        <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path_obj))
        if img is None:
            print(f"Could not load image {img_path_obj} for dimension check.")
            return
        img_h, img_w = img.shape[:2]

        txt_file = out_path / f"{img_path_obj.stem}.txt"

        with txt_file.open("w") as f:
            for item in final_data:
                mask = item["mask"]
                if mask is None:
                    continue # specific to segmentation
                
                # Normalize mask coordinates
                # mask is (N, 2)
                normalized_mask = mask.copy()
                normalized_mask[:, 0] /= img_w
                normalized_mask[:, 1] /= img_h
                
                # Flatten to list
                coords = normalized_mask.flatten().tolist()
                
                # Format: class_id x1 y1 x2 y2 ...
                line = [str(class_id)] + [f"{c:.6f}" for c in coords]
                f.write(" ".join(line) + "\n")
        
        print(f"Saved: {txt_file}")

    def segment_folder(self, folder_path: str, conf: float = 0.25, slice_size: int = 640, overlap_ratio: float = 0.5, save_final_labels: bool = True, output_label_dir: str = "labels", iou_threshold: float = 0.5):
        """
        Segments olives for every image in a folder.
        """
        folder = Path(folder_path)
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(folder.rglob(ext)))
        
        print(f"Found {len(image_paths)} images in {folder_path}")
        
        results_summary = {}
        times_summary = {}

        for img_path in image_paths:
            print(f"Processing: {img_path.name}...")
            try:
                total_count, _, times = self.segment(
                    img_path=str(img_path),
                    conf=conf,
                    slice_size=slice_size,
                    save_final_labels=save_final_labels,
                    overlap_ratio=overlap_ratio,
                    output_label_dir=output_label_dir,
                    iou_threshold=iou_threshold
                )
                results_summary[img_path.name] = {
                    "number of olive": total_count,
                    "times": times
                }
                times_summary[img_path.name] = times
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")
                import traceback
                traceback.print_exc()

        return results_summary, times_summary
