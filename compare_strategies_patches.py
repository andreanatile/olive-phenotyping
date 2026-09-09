import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms, box_iou
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

    # 1. Gather all global boxes
    all_global_boxes = []
    all_scores = []
    
    for i, res in enumerate(results):
        if len(res.boxes) == 0:
            continue
        x_offset, y_offset, _, _ = coordinates[i]
        boxes = res.boxes.xyxy.clone().detach().cpu()
        scores = res.boxes.conf.clone().detach().cpu()

        # Translate to global space
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset

        all_global_boxes.append(boxes)
        all_scores.append(scores)

    if len(all_global_boxes) == 0:
        print("No olives detected in any tile.")
        return

    combined_boxes = torch.cat(all_global_boxes)
    combined_scores = torch.cat(all_scores)

    # 2. Apply NMS (Reconstruction Strategy)
    keep_indices = nms(combined_boxes, combined_scores, args.iou_threshold)
    final_boxes = combined_boxes[keep_indices]

    print("Analyzing patch-level differences...")
    
    patch_data = []
    
    for i, res in enumerate(results):
        tx1, ty1, tx2, ty2 = coordinates[i]
        
        # A. Patch original boxes (local to the patch)
        p_boxes = res.boxes.xyxy.clone().detach().cpu() if len(res.boxes) > 0 else torch.empty((0, 4))
        
        # B. Reconstructed boxes cropped to this tile (local to the patch)
        r_boxes_list = []
        for gbox in final_boxes:
            gx1, gy1, gx2, gy2 = gbox.tolist()
            
            # Intersect global box with the tile bounding box
            ix1, iy1 = max(gx1, tx1), max(gy1, ty1)
            ix2, iy2 = min(gx2, tx2), min(gy2, ty2)
            
            # If valid intersection area
            if ix1 < ix2 and iy1 < iy2:
                # Convert back to local patch coordinates
                lx1, ly1 = ix1 - tx1, iy1 - ty1
                lx2, ly2 = ix2 - tx1, iy2 - ty1
                r_boxes_list.append([lx1, ly1, lx2, ly2])
                
        r_boxes = torch.tensor(r_boxes_list).cpu() if r_boxes_list else torch.empty((0, 4)).cpu()
        
        # C. Find which original patch boxes were suppressed by global NMS
        if len(p_boxes) > 0 and len(r_boxes) > 0:
            # Calculate IoU between the patch's raw boxes and the final reconstructed boxes present in this tile
            ious = box_iou(p_boxes, r_boxes)
            max_ious, _ = ious.max(dim=1)
            
            # If the patch box has a high IoU with a reconstructed box, it was KEPT/MERGED
            kept_mask = max_ious > 0.1 
            kept_p_boxes = p_boxes[kept_mask]
            suppressed_p_boxes = p_boxes[~kept_mask]
        else:
            kept_p_boxes = torch.empty((0, 4))
            suppressed_p_boxes = p_boxes
            
        suppression_count = len(suppressed_p_boxes)
        patch_data.append((i, suppression_count, p_boxes, r_boxes, kept_p_boxes, suppressed_p_boxes))

    # Sort patches by the number of suppressed boxes (highest first)
    patch_data.sort(key=lambda x: x[1], reverse=True)
    
    top_3_patches = patch_data[:3]
    
    # --- Visualization ---
    def draw_boxes(image, boxes, color, thickness=2):
        drawn_img = image.copy()
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, thickness)
        return drawn_img

    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle("Top 3 Patches: Highlighting Clustered Olives Suppressed by NMS", fontsize=22, y=0.98)
    
    for row_idx, data in enumerate(top_3_patches):
        patch_idx, supp_count, p_boxes, r_boxes, kept_p_boxes, suppressed_p_boxes = data
        
        tile_img = cv2.cvtColor(tiles[patch_idx], cv2.COLOR_BGR2RGB)
        
        # Plot Left: Patch Strategy
        img_patch = draw_boxes(tile_img, p_boxes, (255, 165, 0), thickness=2) # Orange
        axs[row_idx, 0].imshow(img_patch)
        axs[row_idx, 0].set_title(f"Patch #{patch_idx} Strategy\n({len(p_boxes)} localized predictions)", fontsize=14)
        axs[row_idx, 0].axis('off')
        
        # Plot Center: Reconstruction Strategy
        img_recon = draw_boxes(tile_img, r_boxes, (0, 255, 0), thickness=2) # Green
        axs[row_idx, 1].imshow(img_recon)
        axs[row_idx, 1].set_title(f"Reconstruction Strategy\n({len(r_boxes)} final boxes in this area)", fontsize=14)
        axs[row_idx, 1].axis('off')
        
        # Plot Right: Highlight Suppressed
        img_compare = tile_img.copy()
        img_compare = draw_boxes(img_compare, kept_p_boxes, (0, 255, 0), thickness=2)      # Green for kept
        img_compare = draw_boxes(img_compare, suppressed_p_boxes, (255, 0, 0), thickness=3) # Red for suppressed
        
        axs[row_idx, 2].imshow(img_compare)
        axs[row_idx, 2].set_title(f"Missing Olives Highlight\n({len(suppressed_p_boxes)} boxes suppressed in Red)", fontsize=14)
        axs[row_idx, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.output, dpi=300)
    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Strategies per Patch")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.15, help="IoU threshold for NMS")
    parser.add_argument("--slice_size", type=int, default=640, help="Size of each slice")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="Overlap ratio between slices")
    parser.add_argument("--output", type=str, default="patch_nms_comparison.jpg", help="Path to save the output visualization")
    
    args = parser.parse_args()
    main(args)
