from ultralytics import YOLO
from src.utils.slice_utils import slice_img
import numpy as np
from torchvision.ops import nms
import torch
import cv2
import os

class OliveExtractor:
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

    def extract(
        self,
        img_path: str,
        output_dir: str,
        conf: float = 0.25,
        overlap_ratio: float = 0.2,
        iou_threshold: float = 0.15
    ):
        """
        Extracts olives from the image and saves them as transparent PNGs.
        
        Args:
            img_path (str): Path to the image file.
            output_dir (str): Directory to save extracted images.
            conf (float): Confidence threshold.
            overlap_ratio (float): Overlap ratio for slicing.
            iou_threshold (float): IoU threshold for NMS.
            
        Returns:
            int: Number of olives extracted.
        """
        
        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Slice Image
        print("Slicing image...")
        try:
            tiles, coordinates = slice_img(
                img=img,
                slice_size=640,
                overlap_ratio=overlap_ratio
            )
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return 0

        # Predict
        print(f"Running inference on {len(tiles)} tiles...")
        results = self.model.predict(tiles, conf=conf, verbose=False)
        
        # Aggregate Results
        all_boxes = []
        all_scores = []
        all_masks_xy = [] 

        for i, res in enumerate(results):
            if not res.boxes:
                continue
            
            # Get tile offset
            x_offset, y_offset, _, _ = coordinates[i]
            
            # Boxes
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            
            # Shift boxes
            boxes[:, [0, 2]] += x_offset
            boxes[:, [1, 3]] += y_offset
            
            all_boxes.append(torch.tensor(boxes))
            all_scores.append(torch.tensor(scores))
            
            # Masks
            if res.masks is not None:
                for polygon in res.masks.xy:
                    # Shift polygon
                    shifted_poly = polygon.copy()
                    shifted_poly[:, 0] += x_offset
                    shifted_poly[:, 1] += y_offset
                    all_masks_xy.append(shifted_poly)
            # If no masks, we can't do pixel extraction effectively for that detection
            
        if not all_boxes:
            print("No olives detected.")
            return 0

        # Concatenate
        combined_boxes = torch.cat(all_boxes)
        combined_scores = torch.cat(all_scores)
        
        # NMS
        print("Applying NMS...")
        keep_indices = nms(combined_boxes, combined_scores, iou_threshold)
        
        final_boxes = combined_boxes[keep_indices].numpy()
        final_scores = combined_scores[keep_indices].numpy()
        # Filter masks based on NMS indices
        # We need to be careful: all_masks_xy corresponds to the FLATTENED list of detections across all tiles
        # But wait, NMS takes the concatenated boxes.
        # We need to ensure we have a 1-to-1 mapping before NMS.
        
        # Re-gathering to ensure alignment:
        # Actually, let's collect them pair-wise into a list of objects first to be safe, 
        # because nms indices index into the tensor.
        
        # Let's flatten everything into lists first
        flat_boxes = []
        flat_scores = []
        flat_masks = []
        
        for i, res in enumerate(results):
            if not res.boxes:
                continue
            x_offset, y_offset, _, _ = coordinates[i]
            
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            
            if res.masks is not None:
                masks = res.masks.xy
            else:
                masks = [None] * len(boxes) # Should not happen in seg model
                
            for box, score, mask in zip(boxes, scores, masks):
                if mask is None: continue 
                
                # Shift box
                box[[0, 2]] += x_offset
                box[[1, 3]] += y_offset
                
                # Shift mask
                shifted_poly = mask.copy()
                shifted_poly[:, 0] += x_offset
                shifted_poly[:, 1] += y_offset
                
                flat_boxes.append(box)
                flat_scores.append(score)
                flat_masks.append(shifted_poly)
                
        if not flat_boxes:
             print("No detections with masks found.")
             return 0
             
        # Convert to tensors for NMS
        tensor_boxes = torch.tensor(np.array(flat_boxes))
        tensor_scores = torch.tensor(np.array(flat_scores))
        
        keep_indices = nms(tensor_boxes, tensor_scores, iou_threshold)
        
        final_boxes = tensor_boxes[keep_indices].numpy()
        final_scores = tensor_scores[keep_indices].numpy()
        final_masks = [flat_masks[i] for i in keep_indices] # select from list
        
        print(f"Found {len(final_boxes)} olives after NMS.")
        
        # Save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for idx, (box, polygon) in enumerate(zip(final_boxes, final_masks)):
            self.save_extracted_olive(img, polygon, final_scores[idx], idx, output_dir)
            
        print(f"Saved {len(final_boxes)} olive images to {output_dir}")
        return len(final_boxes)

    def save_extracted_olive(self, original_img, polygon, score, idx, output_dir):
        # Polygon to integer
        pts = polygon.astype(np.int32)
        
        # Bounding rect
        x, y, w, h = cv2.boundingRect(pts)
        
        # Clamp
        x = max(0, x)
        y = max(0, y)
        w = min(w, original_img.shape[1] - x)
        h = min(h, original_img.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return
            
        # Crop
        crop = original_img[y:y+h, x:x+w].copy()
        
        # Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts_local = pts - [x, y]
        cv2.fillPoly(mask, [pts_local], 255)
        
        # Merge
        b, g, r = cv2.split(crop)
        rgba = cv2.merge([b, g, r, mask])
        
        # Save
        filename = f"olive_{idx:04d}_score_{score:.2f}.png"
        cv2.imwrite(os.path.join(output_dir, filename), rgba)
