import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms, box_iou
from ultralytics import YOLO
from glob import glob

# Import slicing tools from the repo
from src.utils.slice_detection_utils import slice_img

def process_image(img_path, model, args):
    print(f"\nProcessing image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"  -> Could not read image, skipping.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tiles, coordinates = slice_img(
        img=img,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio
    )

    results = model.predict(tiles, conf=args.conf, verbose=False)

    # 1. Gather all global boxes
    all_global_boxes = []
    all_scores = []
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0:
            continue
        x_offset, y_offset, _, _ = coordinates[i]
        boxes = res.boxes.xyxy.clone().detach().cpu()
        scores = res.boxes.conf.clone().detach().cpu()

        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset

        all_global_boxes.append(boxes)
        all_scores.append(scores)

    if len(all_global_boxes) == 0:
        print("  -> No olives detected in any tile.")
        return

    combined_boxes = torch.cat(all_global_boxes)
    combined_scores = torch.cat(all_scores)

    # 2. Apply NMS (Reconstruction Strategy)
    keep_indices = nms(combined_boxes, combined_scores, args.iou_threshold)
    final_boxes = combined_boxes[keep_indices]

    # Organize outputs by image name
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img_out_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(img_out_dir, exist_ok=True)

    def draw_boxes(image, boxes, color, thickness=2):
        drawn_img = image.copy()
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, thickness)
        return drawn_img

    patches_plotted = 0
    
    for i, res in enumerate(results):
        tx1, ty1, tx2, ty2 = coordinates[i]
        
        p_boxes = res.boxes.xyxy.clone().detach().cpu() if len(res.boxes) > 0 else torch.empty((0, 4))
        
        r_boxes_list = []
        for gbox in final_boxes:
            gx1, gy1, gx2, gy2 = gbox.tolist()
            ix1, iy1 = max(gx1, tx1), max(gy1, ty1)
            ix2, iy2 = min(gx2, tx2), min(gy2, ty2)
            
            if ix1 < ix2 and iy1 < iy2:
                lx1, ly1 = ix1 - tx1, iy1 - ty1
                lx2, ly2 = ix2 - tx1, iy2 - ty1
                r_boxes_list.append([lx1, ly1, lx2, ly2])
                
        r_boxes = torch.tensor(r_boxes_list).cpu() if r_boxes_list else torch.empty((0, 4)).cpu()
        
        # Skip completely empty patches (unless requested otherwise)
        if len(p_boxes) == 0 and len(r_boxes) == 0 and not args.save_empty:
            continue
            
        if len(p_boxes) > 0 and len(r_boxes) > 0:
            ious = box_iou(p_boxes, r_boxes)
            max_ious, _ = ious.max(dim=1)
            kept_mask = max_ious > 0.1 
            kept_p_boxes = p_boxes[kept_mask]
            suppressed_p_boxes = p_boxes[~kept_mask]
        else:
            kept_p_boxes = torch.empty((0, 4))
            suppressed_p_boxes = p_boxes
            
        supp_count = len(suppressed_p_boxes)
        
        if args.only_suppressed and supp_count == 0:
            continue # Skip patches with no suppression
            
        # Plot and save
        tile_img = cv2.cvtColor(tiles[i], cv2.COLOR_BGR2RGB)
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Image: {base_name} | Patch #{i} | {supp_count} Suppressed Olives", fontsize=20)
        
        # Left
        img_patch = draw_boxes(tile_img, p_boxes, (255, 165, 0), thickness=2)
        axs[0].imshow(img_patch)
        axs[0].set_title(f"Patch Strategy ({len(p_boxes)} boxes)", fontsize=14)
        axs[0].axis('off')
        
        # Center
        img_recon = draw_boxes(tile_img, r_boxes, (0, 255, 0), thickness=2)
        axs[1].imshow(img_recon)
        axs[1].set_title(f"Reconstruction Strategy ({len(r_boxes)} boxes)", fontsize=14)
        axs[1].axis('off')
        
        # Right
        img_compare = tile_img.copy()
        img_compare = draw_boxes(img_compare, kept_p_boxes, (0, 255, 0), thickness=2)
        img_compare = draw_boxes(img_compare, suppressed_p_boxes, (255, 0, 0), thickness=3)
        axs[2].imshow(img_compare)
        axs[2].set_title(f"Highlight ({supp_count} suppressed in Red)", fontsize=14)
        axs[2].axis('off')

        plt.tight_layout()
        out_path = os.path.join(img_out_dir, f"patch_{i:03d}.jpg")
        plt.savefig(out_path, dpi=150)
        plt.close(fig) # Prevent memory leaks
        patches_plotted += 1

    print(f"  -> Saved {patches_plotted} patch comparisons in {img_out_dir}")


def main(args):
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_paths.extend(glob(os.path.join(args.folder_path, ext)))
        
    if not image_paths:
        print(f"No images found in {args.folder_path}")
        return
        
    print(f"Found {len(image_paths)} images. Starting processing...")
    for img_path in image_paths:
        process_image(img_path, model, args)
        
    print(f"\nAll done! Visualizations are organized by image in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Strategies per Patch for an Entire Folder")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.15, help="IoU threshold for NMS")
    parser.add_argument("--slice_size", type=int, default=640, help="Size of each slice")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="Overlap ratio between slices")
    parser.add_argument("--output_dir", type=str, default="patch_comparisons_folder", help="Directory to save the outputs")
    parser.add_argument("--only_suppressed", action="store_true", help="Only plot patches that have >0 suppressed olives")
    parser.add_argument("--save_empty", action="store_true", help="Save plots even for completely empty patches (no olives)")
    
    args = parser.parse_args()
    main(args)
