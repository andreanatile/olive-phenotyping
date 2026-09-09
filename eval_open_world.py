import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Try to import transformers for Grounding DINO
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes should be in [xmin, ymin, xmax, ymax] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def read_yolo_labels(label_path, img_width, img_height, target_class_id=None):
    """
    Reads YOLO format labels and converts them to [xmin, ymin, xmax, ymax].
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if target_class_id is not None and class_id != target_class_id:
                    continue
                
                # YOLO format: normalized [x_center, y_center, width, height]
                x_c, y_c, w, h = map(float, parts[1:5])
                
                xmin = (x_c - w / 2) * img_width
                ymin = (y_c - h / 2) * img_height
                xmax = (x_c + w / 2) * img_width
                ymax = (y_c + h / 2) * img_height
                
                boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Matches predicted boxes to ground truth boxes to calculate TP, FP, FN.
    Greedy matching based on highest IoU.
    """
    tp = 0
    fp = 0
    fn = 0
    
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
        
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
        
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, p_box in enumerate(pred_boxes):
        for j, g_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(p_box, g_box)
            
    # Greedy matching
    matched_gt = set()
    matched_pred = set()
    
    # Flatten and sort IoUs in descending order
    # Store indices (pred_idx, gt_idx, iou)
    iou_list = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                iou_list.append((i, j, iou_matrix[i, j]))
                
    iou_list.sort(key=lambda x: x[2], reverse=True)
    
    for pred_idx, gt_idx, iou in iou_list:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            tp += 1
            
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


class YOLOWorldEvaluator:
    def __init__(self, model_name='yolov8s-world.pt', classes=None, conf=0.25):
        print(f"Loading YOLO-World model: {model_name}")
        self.model = YOLO(model_name)
        if classes:
            self.model.set_classes(classes)
        self.conf = conf
        
    def predict(self, img_path):
        results = self.model.predict(img_path, conf=self.conf, verbose=False)
        boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        return boxes


class GroundingDINOEvaluator:
    def __init__(self, model_name="IDEA-Research/grounding-dino-base", text_prompt="olive.", box_threshold=0.3, text_threshold=0.3):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package is required for Grounding DINO. Install it via 'pip install transformers'")
            
        print(f"Loading Grounding DINO model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
    def predict(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )
        
        boxes = []
        if len(results) > 0:
            boxes = results[0]["boxes"].cpu().numpy().tolist()
        return boxes


def run_evaluation(images_dir, labels_dir, model_type='yolo_world', text_prompt="olive", iou_threshold=0.5, conf=0.25):
    """
    Runs evaluation on a dataset using a specified model.
    """
    if model_type == 'yolo_world':
        evaluator = YOLOWorldEvaluator(classes=[text_prompt], conf=conf)
    elif model_type == 'grounding_dino':
        # Grounding DINO needs a trailing period for best results
        prompt = text_prompt if text_prompt.endswith('.') else f"{text_prompt}."
        evaluator = GroundingDINOEvaluator(text_prompt=prompt, box_threshold=conf)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                  glob.glob(os.path.join(images_dir, "*.png"))
                  
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Starting evaluation...")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_path in tqdm(image_paths):
        # 1. Get image dimensions for label conversion
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 2. Get Ground Truth
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        label_path = os.path.join(labels_dir, f"{name}.txt")
        
        # Assuming we evaluate for class 0 (olive)
        gt_boxes = read_yolo_labels(label_path, w, h, target_class_id=0)
        
        # 3. Get Predictions
        pred_boxes = evaluator.predict(img_path)
        
        # 4. Calculate metrics
        tp, fp, fn = match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 5. Compute aggregate metrics
    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    
    print("\n" + "="*40)
    print(f"Evaluation Results for {model_type.upper()}")
    print("="*40)
    print(f"Prompt / Class   : {text_prompt}")
    print(f"IoU Threshold    : {iou_threshold}")
    print(f"Confidence/Score : {conf}")
    print("-" * 40)
    print(f"Total Images     : {len(image_paths)}")
    print(f"True Positives   : {total_tp}")
    print(f"False Positives  : {total_fp}")
    print(f"False Negatives  : {total_fn}")
    print("-" * 40)
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1-Score         : {f1_score:.4f}")
    print("="*40)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Open World Models (YOLO-World, Grounding DINO)")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--labels_dir", type=str, required=True, help="Path to folder containing YOLO format labels")
    parser.add_argument("--model", type=str, choices=['yolo_world', 'grounding_dino'], default='yolo_world', help="Model to use")
    parser.add_argument("--prompt", type=str, default="olive", help="Text prompt / class name to search for")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence / Box threshold")
    
    args = parser.parse_args()
    
    run_evaluation(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        model_type=args.model,
        text_prompt=args.prompt,
        iou_threshold=args.iou,
        conf=args.conf
    )
