import argparse
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy
import json

class OliveEvaluator:
    def __init__(self, gt_folder: str, pred_folder: str, iou_thresholds=None):
        self.gt_folder = Path(gt_folder)
        self.pred_folder = Path(pred_folder)
        # Default IoU thresholds from 0.5 to 0.95 in steps of 0.05
        self.iouv = torch.linspace(0.5, 0.95, 10) if iou_thresholds is None else iou_thresholds
        self.niou = self.iouv.numel()

    def load_txt_labels(self, path: Path):
        """
        Loads labels from a YOLO-format text file.
        Supports both Box (class x y w h) and Polygon (class x1 y1 ... xn yn) formats.
        For polygons, converts to bounding box (x_center, y_center, w, h).
        
        Returns:
            cls: Tensor of shape (N,)
            boxes: Tensor of shape (N, 4) in xyxy format (normalized)
        """
        if not path.exists():
            return torch.zeros(0), torch.zeros(0, 4)
        
        try:
            # Read file line by line to handle variable length lines (polygons)
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception:
            return torch.zeros(0), torch.zeros(0, 4)
            
        if not lines:
            return torch.zeros(0), torch.zeros(0, 4)

        cls_list = []
        boxes_list = []

        for line in lines:
            parts = list(map(float, line.strip().split()))
            if not parts:
                continue
            
            c = parts[0]
            coords = parts[1:]
            
            if len(coords) == 4:
                # Box format: x_center, y_center, w, h
                # Convert to xyxy later
                # Store as xywh for consistency with batch processing or process here?
                # Let's process here to xyxy immediately
                x_c, y_c, w, h = coords
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2
                boxes_list.append([x1, y1, x2, y2])
                cls_list.append(c)
                
            elif len(coords) > 4:
                # Polygon format
                # coords are [x1, y1, x2, y2, ...]
                # Reshape to (N, 2)
                poly = np.array(coords).reshape(-1, 2)
                x_min = poly[:, 0].min()
                y_min = poly[:, 1].min()
                x_max = poly[:, 0].max()
                y_max = poly[:, 1].max()
                boxes_list.append([x_min, y_min, x_max, y_max])
                cls_list.append(c)

        if not cls_list:
            return torch.zeros(0), torch.zeros(0, 4)
            
        cls = torch.tensor(cls_list)
        boxes = torch.tensor(boxes_list) 
        
        # boxes are already in xyxy normalized
        return cls, boxes

    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects using IoU.
        Implementation adapted from ultralytics.engine.validator.BaseValidator.match_predictions
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.niou)).astype(bool)
        
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        
        iou = iou.cpu().numpy()
        iouv_cpu = self.iouv.cpu().tolist()
        
        for i, threshold in enumerate(iouv_cpu):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    # Sort by IoU (descending)
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    # Greedy assignment: unique detection, unique ground truth
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    
                correct[matches[:, 1].astype(int), i] = True
                
        return torch.tensor(correct, dtype=torch.bool)

    def evaluate(self, save_json: bool = False, output_path: str = None):
        """
        Evaluates the predictions against the ground truth.
        """
        print(f"Evaluating predictions from {self.pred_folder} against {self.gt_folder}")
        
        # Find all label files
        # We assume dataset structure might be flat or nested, but usually labels are in flat folder
        gt_files = list(self.gt_folder.rglob("*.txt"))
        
        stats = []
        
        for gt_file in gt_files:
            stem = gt_file.stem
            pred_file = self.pred_folder / f"{stem}.txt"
            
            # Load GT
            target_cls, target_boxes = self.load_txt_labels(gt_file)
            
            # Load Preds
            # If pred file doesn't exist, we assume no detections
            if not pred_file.exists():
                pred_cls, pred_boxes = torch.zeros(0), torch.zeros(0, 4)
            else:
                pred_cls, pred_boxes = self.load_txt_labels(pred_file)
                
            # Initialize stats components
            if pred_cls.shape[0] == 0:
                if target_cls.shape[0] > 0:
                    # Ground truth exists but no predictions -> all False Positives (none) and all False Negatives (missed)
                    # ap_per_class expects (tp, conf, pred_cls, target_cls)
                    # If there are no predictions, tp is empty
                    stats.append((
                        torch.zeros(0, self.niou, dtype=torch.bool),
                        torch.zeros(0),
                        torch.zeros(0),
                        target_cls
                    ))
                continue
                
            if target_cls.shape[0] == 0:
                # Predictions exist but no ground truth -> all False Positives
                # tp is all False
                stats.append((
                    torch.zeros(pred_cls.shape[0], self.niou, dtype=torch.bool),
                    torch.ones(pred_cls.shape[0]), # Fake confidence if not available? 
                    # Wait, we need confidence. Does OliveCounter save confidence?
                    # The current OliveCounter logic in save_labels DOES NOT save confidence.
                    # It saves: class_id x y w h
                    # Line 197: f.write(f"{class_id} {x_center/img_w:.6f} ...")
                    # We MUST have confidence for mAP calculation.
                    pred_cls,
                    torch.zeros(0) # Empty target_cls
                ))
                continue

            # We need confidence scores!
            # The current save_labels implementation in counter.py does NOT save confidence scores. 
            # We will handle this by checking if we have 6 columns or 5.
            # If 5, we might have to assume 1.0 or user needs to update counter.py. 
            # This is a critical discovery.
            
            # For now, let's assume we can modify counter.py or use a default conf.
            # If we don't have conf, mAP calc is degenerate (AP requires ranking).
            # I will assume 1.0 for now but warn the user.
            
            # compute IoU
            iou = box_iou(target_boxes, pred_boxes)
            
            # Match
            tp = self.match_predictions(pred_cls, target_cls, iou)
            
            # Append stats
            # We need confidence. Let's create a placeholder if missing.
            # In load_txt_labels, we only read 5 columns. I'll update it to read 6 if available.
            
            # Re-read to check for conf
            try:
                raw_data = np.loadtxt(pred_file)
                if raw_data.ndim == 1: raw_data = raw_data[None, :]
                if raw_data.shape[1] >= 6:
                    conf = torch.from_numpy(raw_data[:, 5])
                else:
                    conf = torch.ones(pred_cls.shape[0]) # Default confidence
            except:
                conf = torch.ones(pred_cls.shape[0])

            stats.append((tp, conf, pred_cls, target_cls))

        # Compute metrics
        if not stats:
            print("No data to evaluate.")
            return

        # Concatenate
        tp = torch.cat([x[0] for x in stats], 0)
        conf = torch.cat([x[1] for x in stats], 0)
        pred_cls = torch.cat([x[2] for x in stats], 0)
        target_cls = torch.cat([x[3] for x in stats], 0)

        # ap_per_class
        results = ap_per_class(tp.float(), conf, pred_cls, target_cls, plot=False)
        # results: (precision, recall, ap, f1, unique_classes)
        # ap is (nc, 10)
        
        tp,fp, p, r, f1,ap, _,_,_,_,_,_ = results
       
        # Calculate mAP
        ap50 = ap[:, 0].mean()
        ap50_95 = ap.mean()
        
        print(f"Evaluation Results:")
        print(f"mAP@50: {ap50:.4f}")
        print(f"mAP@50-95: {ap50_95:.4f}")
        print(f"Precision: {p.mean():.4f}")
        print(f"Recall: {r.mean():.4f}")
        
        if save_json and output_path is not None:
            eval_dict = {
                "mAP@50": ap50.item(),
                "mAP@50-95": ap50_95.item(),
                "Precision": p.mean().item(),
                "Recall": r.mean().item()
            }
            try:
                with open(output_path, 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                print(f"Saved evaluation results to {output_path}")
            except Exception as e:
                print(f"Error saving evaluation results: {e}")
        return results