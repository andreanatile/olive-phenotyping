import argparse
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.ops import nms
from ultralytics import YOLO

# Import slicing tools from the repo
from src.utils.slice_detection_utils import slice_img

def main(args):
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)

    print(f"Loading image from {args.img_path}...")
    img = cv2.imread(args.img_path)
    if img is None:
        raise ValueError(f"Could not read image at {args.img_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Slicing image...")
    tiles, coordinates = slice_img(
        img=img,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio
    )

    print(f"Running inference on {len(tiles)} tiles...")
    results = model.predict(tiles, conf=args.conf, verbose=False)

    # 1. Patching Strategy: Gather all boxes in global coordinates WITHOUT NMS
    all_global_boxes = []
    all_scores = []
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0:
            continue
        x_offset, y_offset, _, _ = coordinates[i]
        boxes = res.boxes.xyxy.clone().detach()
        scores = res.boxes.conf.clone().detach()

        # Add offsets to shift from patch space to global space
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset

        all_global_boxes.append(boxes)
        all_scores.append(scores)

    if len(all_global_boxes) == 0:
        print("No olives detected in any tile.")
        return

    combined_boxes = torch.cat(all_global_boxes)
    combined_scores = torch.cat(all_scores)

    print(f"Total boxes before NMS (Patching Strategy): {len(combined_boxes)}")

    # 2. Reconstruction Strategy: Apply NMS
    keep_indices = nms(combined_boxes, combined_scores, args.iou_threshold)
    final_boxes = combined_boxes[keep_indices]
    final_scores = combined_scores[keep_indices]

    print(f"Total boxes after NMS (Reconstruction Strategy): {len(final_boxes)}")

    # 3. Identify suppressed boxes
    suppressed_mask = torch.ones(len(combined_boxes), dtype=torch.bool)
    suppressed_mask[keep_indices] = False
    suppressed_boxes = combined_boxes[suppressed_mask]
    
    print(f"Total boxes suppressed by NMS: {len(suppressed_boxes)}")

    # Visualization
    def draw_boxes(image, boxes, color, thickness=2):
        drawn_img = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, thickness)
        return drawn_img
    
    # Combined Image: Green = Kept, Red = Suppressed
    combined_img = img_rgb.copy()
    combined_img = draw_boxes(combined_img, suppressed_boxes, (255, 0, 0), thickness=3) # RED for suppressed
    combined_img = draw_boxes(combined_img, final_boxes, (0, 255, 0), thickness=2)      # GREEN for kept

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: Patching Strategy
    patch_img = draw_boxes(img_rgb, combined_boxes, (255, 165, 0)) # Orange
    axs[0].imshow(patch_img)
    axs[0].set_title(f"Patching Strategy (No NMS)\n{len(combined_boxes)} Boxes", fontsize=16)
    axs[0].axis('off')

    # Plot 2: Reconstruction Strategy
    recon_img = draw_boxes(img_rgb, final_boxes, (0, 255, 0)) # Green
    axs[1].imshow(recon_img)
    axs[1].set_title(f"Reconstruction Strategy (With NMS)\n{len(final_boxes)} Boxes", fontsize=16)
    axs[1].axis('off')

    # Plot 3: Highlight Suppressed
    axs[2].imshow(combined_img)
    axs[2].set_title(f"Comparison\nGreen = Kept, Red = Suppressed ({len(suppressed_boxes)})", fontsize=16)
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Patching vs Reconstruction Strategy to highlight NMS suppression")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.15, help="IoU threshold for NMS")
    parser.add_argument("--slice_size", type=int, default=640, help="Size of each slice")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="Overlap ratio between slices")
    parser.add_argument("--output", type=str, default="nms_comparison.jpg", help="Path to save the output visualization")
    
    args = parser.parse_args()
    main(args)
