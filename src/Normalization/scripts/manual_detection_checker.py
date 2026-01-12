import cv2
import numpy as np
import os
import glob
import json
from Normalization.scripts.Normalizer import Normalization

# --- CONFIG ---
N_PATCHES = 24
INPUT_DIR = "/home/girobat/Olive/Corrected/foto 11.09.24 olivo università/not_detected"
OUTPUT_JSON = "/home/girobat/Olive/Corrected/foto 11.09.24 olivo università/not_detected/colorchecker_rgb.json"
DISPLAY_SCALE = 0.5

json_data = {}


import cv2
import numpy as np
import os

# Constants
DISPLAY_SIZE = (800, 800)  # Fixed window size for easier math
N_PATCHES = 24


def select_patches(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot read {image_path}")
        return None

    h, w = img.shape[:2]
    # View state: [x, y, w, h] of the original image we are looking at
    view = [0, 0, w, h]

    patches = []  # Stores actual RGB values
    completed_rects = []  # Stores [(x1, y1), (x2, y2)] in ORIGINAL image coords

    drawing = False
    start_pt = None  # Current drawing start point (original coords)
    cur_pt = None  # Current mouse position (original coords)

    # For panning
    panning = False
    pan_start = None

    def get_orig_coords(nx, ny):
        """Convert normalized window coordinates (0-1) to original image coordinates."""
        ox = view[0] + nx * view[2]
        oy = view[1] + ny * view[3]
        return int(ox), int(oy)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, cur_pt, view, panning, pan_start, patches, completed_rects

        # Calculate normalized coordinates (0.0 to 1.0) within the window
        nx, ny = x / DISPLAY_SIZE[0], y / DISPLAY_SIZE[1]
        ox, oy = get_orig_coords(nx, ny)

        # LEFT CLICK: Draw Bounding Boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE:
            cur_pt = (ox, oy)
            if panning:
                dx = int((nx - pan_start[0]) * view[2])
                dy = int((ny - pan_start[1]) * view[3])
                view[0] = max(0, min(w - view[2], view[0] - dx))
                view[1] = max(0, min(h - view[3], view[1] - dy))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Save the box in original coordinates
            completed_rects.append((start_pt, (ox, oy)))
            # Extract ROI and calculate mean color
            x1, x2 = sorted([start_pt[0], ox])
            y1, y2 = sorted([start_pt[1], oy])
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                mean_color = roi.mean(axis=(0, 1))
                patches.append(mean_color[::-1].tolist())  # BGR to RGB
                print(f"Patch {len(patches)} recorded.")

        # RIGHT CLICK: Panning
        elif event == cv2.EVENT_RBUTTONDOWN:
            panning = True
            pan_start = (nx, ny)
        elif event == cv2.EVENT_RBUTTONUP:
            panning = False

        # MOUSE WHEEL: Zooming
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 0.9 if flags > 0 else 1.1
            new_vw = min(w, max(50, view[2] * zoom_factor))
            new_vh = min(h, max(50, view[3] * zoom_factor))

            # Adjust view to keep mouse position centered during zoom
            view[0] = max(0, min(w - new_vw, view[0] + (view[2] - new_vw) * nx))
            view[1] = max(0, min(h - new_vh, view[1] + (view[3] - new_vh) * ny))
            view[2], view[3] = new_vw, new_vh

    cv2.namedWindow("Select patches")
    cv2.setMouseCallback("Select patches", on_mouse)

    print(f"\n🖼️ Instructions:")
    print("- Left Click & Drag: Draw BBox")
    print("- Mouse Wheel: Zoom | Right Click & Drag: Pan")
    print("- 'u': Undo last | 'r': Reset all | 's': Save & Exit")

    while True:
        # 1. Crop the original image based on current view
        vx, vy, vw, vh = map(int, view)
        display_img = img[vy : vy + vh, vx : vx + vw].copy()

        # 2. Draw existing rectangles
        # Need to convert original coords to current view's relative coords
        for rect in completed_rects:
            p1 = (
                int((rect[0][0] - vx) * (DISPLAY_SIZE[0] / vw)),
                int((rect[0][1] - vy) * (DISPLAY_SIZE[1] / vh)),
            )
            p2 = (
                int((rect[1][0] - vx) * (DISPLAY_SIZE[0] / vw)),
                int((rect[1][1] - vy) * (DISPLAY_SIZE[1] / vh)),
            )
            cv2.rectangle(display_img, p1, p2, (0, 255, 0), 2)

        # 3. Draw active rectangle
        if drawing and start_pt and cur_pt:
            p1 = (
                int((start_pt[0] - vx) * (DISPLAY_SIZE[0] / vw)),
                int((start_pt[1] - vy) * (DISPLAY_SIZE[1] / vh)),
            )
            p2 = (
                int((cur_pt[0] - vx) * (DISPLAY_SIZE[0] / vw)),
                int((cur_pt[1] - vy) * (DISPLAY_SIZE[1] / vh)),
            )
            cv2.rectangle(display_img, p1, p2, (0, 0, 255), 1)

        # 4. Show resized image
        final_render = cv2.resize(display_img, DISPLAY_SIZE)
        cv2.imshow("Select patches", final_render)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        elif key == ord("u"):  # Undo
            if patches:
                patches.pop()
                completed_rects.pop()
                print("Last patch removed.")
        elif key == ord("r"):  # Reset
            patches = []
            completed_rects = []
            print("All patches cleared.")
        elif key == ord("q"):
            return None

    cv2.destroyAllWindows()
    return np.array(patches)


# --- MAIN LOOP ---


def extract_patches_from_not_detected(not_detected_dir_path, output_json_path):
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

    for img_path in image_paths:
        result = select_patches(img_path)
        if result is not None:
            json_data[os.path.basename(img_path)] = result.tolist()
            print(f"✅ Saved RGB values for {os.path.basename(img_path)}")
        else:
            print(f"⚠️ Skipped {os.path.basename(img_path)}")
        break

    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"\n💾 All results saved in {OUTPUT_JSON}")
