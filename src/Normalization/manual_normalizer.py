#!/usr/bin/env python3
import argparse
import cv2
import imageio
import numpy as np
import json
import colour
from pathlib import Path

# DA CANCELLLARE
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]

REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
    "sRGB",
    REFERENCE_COLOUR_CHECKER.illuminant,
)


class ManualSwatchesNormalizer:
    """Interactively select ColorChecker patches and apply manual normalization."""

    DEFAULT_N_PATCHES = 24
    DEFAULT_DISPLAY_SCALE = 0.5

    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
        "ColorChecker24 - After November 2014"
    ]

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        "sRGB",
        REFERENCE_COLOUR_CHECKER.illuminant,
    )

    def __init__(
        self,
        not_detected_dir_path,
        corrected_dir_path,
        swatches_saved_json_path,
        n_patches=DEFAULT_N_PATCHES,
        display_scale=DEFAULT_DISPLAY_SCALE,
    ):
        # Convert strings to Path objects
        self.input_dir = Path(not_detected_dir_path)
        self.output_dir = Path(corrected_dir_path)
        self.output_json = Path(swatches_saved_json_path)

        self.n_patches = n_patches
        self.display_scale = display_scale
        self.json_data = {}

    def select_patches(self, image_path):
        """
        Open an image with a locked zoom feature.
        Press 'z' to zoom into the current mouse position and lock the view.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️ Cannot read {image_path}")
            return None

        orig_img = img.copy()
        h, w = orig_img.shape[:2]

        # --- CONFIGURATION ---
        MAX_WIN_W, MAX_WIN_H = 1280, 720
        fit_scale = min(MAX_WIN_W / w, MAX_WIN_H / h, self.display_scale)

        # State management
        patches = []
        completed_rects_orig = []
        drawing = False
        start_point_orig = None
        mouse_orig = (w // 2, h // 2)  # Tracking mouse for zoom target

        # Zoom state
        is_zoomed = False
        locked_zoom_center = (w // 2, h // 2)
        zoom_factor = 4.0

        def get_display_frame():
            """Generates the frame based on whether zoom is locked or off."""
            if not is_zoomed:
                display = cv2.resize(
                    orig_img,
                    (0, 0),
                    fx=fit_scale,
                    fy=fit_scale,
                    interpolation=cv2.INTER_AREA,
                )
                curr_scale = fit_scale
                offset = (0, 0)
            else:
                # Use the center that was LOCKED when 'z' was pressed
                zw, zh = int(w / zoom_factor), int(h / zoom_factor)
                x1 = max(0, min(locked_zoom_center[0] - zw // 2, w - zw))
                y1 = max(0, min(locked_zoom_center[1] - zh // 2, h - zh))
                crop = orig_img[y1 : y1 + zh, x1 : x1 + zw].copy()
                display = cv2.resize(
                    crop,
                    (int(w * fit_scale), int(h * fit_scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
                curr_scale = (w * fit_scale) / zw
                offset = (x1, y1)

            # Draw completed rectangles
            for r in completed_rects_orig:
                p1 = (
                    int((r[0][0] - offset[0]) * curr_scale),
                    int((r[0][1] - offset[1]) * curr_scale),
                )
                p2 = (
                    int((r[1][0] - offset[0]) * curr_scale),
                    int((r[1][1] - offset[1]) * curr_scale),
                )
                cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

            return display, curr_scale, offset

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, start_point_orig, patches, mouse_orig

            _, curr_scale, offset = get_display_frame()
            # Always track where the mouse is in "Original Image" pixels
            mouse_orig = (
                int(x / curr_scale + offset[0]),
                int(y / curr_scale + offset[1]),
            )

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point_orig = mouse_orig

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    display_img, _, _ = get_display_frame()
                    p1_disp = (
                        int((start_point_orig[0] - offset[0]) * curr_scale),
                        int((start_point_orig[1] - offset[1]) * curr_scale),
                    )
                    # Live drawing of the rectangle
                    cv2.rectangle(display_img, p1_disp, (x, y), (0, 255, 0), 1)
                    cv2.imshow("Select patches", display_img)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, x2 = sorted([start_point_orig[0], mouse_orig[0]])
                y1, y2 = sorted([start_point_orig[1], mouse_orig[1]])

                roi = orig_img[y1:y2, x1:x2]
                if roi.size > 0:
                    mean_color = roi.mean(axis=(0, 1))
                    patches.append(mean_color[::-1].tolist())
                    completed_rects_orig.append(((x1, y1), (x2, y2)))

        window_name = "Select patches"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, on_mouse)

        print(f"🖼️ {image_path.name}")
        print("INSTRUCTIONS:")
        print("1. Hover mouse over a patch.")
        print("2. Press 'z' to LOCK zoom on that spot.")
        print("3. Draw your rectangle. Press 'z' again to zoom out.")

        while True:
            display_img, _, _ = get_display_frame()

            # UI Overlay: Show a crosshair if zoomed for better precision
            if is_zoomed and not drawing:
                # Just a visual guide
                h_disp, w_disp = display_img.shape[:2]
                cv2.line(
                    display_img,
                    (0, h_disp // 2),
                    (w_disp, h_disp // 2),
                    (255, 255, 255),
                    1,
                )
                cv2.line(
                    display_img,
                    (w_disp // 2, 0),
                    (w_disp // 2, h_disp),
                    (255, 255, 255),
                    1,
                )

            cv2.imshow(window_name, display_img)
            cv2.setWindowTitle(
                window_name,
                f"Patches: {len(patches)}/{self.n_patches} | Zoom: {'LOCKED' if is_zoomed else 'OFF'}",
            )

            key = cv2.waitKey(10) & 0xFF
            if key == ord("z"):
                is_zoomed = not is_zoomed
                if is_zoomed:
                    # Capture the current mouse position as the anchor
                    locked_zoom_center = mouse_orig
            elif key == ord("u") and patches:
                patches.pop()
                completed_rects_orig.pop()
            elif key == ord("s") or len(patches) == self.n_patches:
                break
            elif key == ord("q"):
                patches = []
                break

        cv2.destroyAllWindows()
        return np.array(patches) if len(patches) > 0 else None

    def manual_detection_swatches(self, save_swatches=True):
        """Manually select swatches and save into JSON."""
        # Pathlib globbing
        images_path = sorted(self.input_dir.glob("*.jpg"))

        for path in images_path:
            patches = self.select_patches(path)
            if patches is None:
                print(f"⚠️ Skipped {path.name}")
                continue

            patches = np.array(patches) / 255.0
            self.json_data[path.name] = patches.tolist()
            print(f"✅ Saved RGB values for {path.name}")

        if save_swatches:
            # Ensure JSON parent directory exists
            self.output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_json, "w") as f:
                json.dump(self.json_data, f, indent=4)
            print(f"\n💾 All swatches saved in {self.output_json}")


if __name__ == "__main__":
    # Example usage with local paths
    manual_normalizer = ManualSwatchesNormalizer(
        not_detected_dir_path="data/Normalized/not_detected",
        corrected_dir_path="data/Normalized/corrected",
        swatches_saved_json_path="data/Normalized/manual_swatches.json",
        n_patches=24,
        display_scale=0.5,
    )

    manual_normalizer.manual_detection_swatches(save_swatches=True)
