#!/usr/bin/env python3
import os
import glob
import shutil
import argparse
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
import numpy as np
import cv2

# --- CONSTANTS ---
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    "ColorChecker24 - After November 2014"
]

REFERENCE_SWATCHES = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
    "sRGB",
    REFERENCE_COLOUR_CHECKER.illuminant,
)


class ColorNormalizer:
    def __init__(self, root_dir, output_dir, method="Cheung 2004", degree=1):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.method = method
        self.degree = degree

        self.corrected_folder = os.path.join(self.output_dir, "corrected")
        self.not_detected_folder = os.path.join(self.output_dir, "not_detected")
        os.makedirs(self.corrected_folder, exist_ok=True)
        os.makedirs(self.not_detected_folder, exist_ok=True)

    def Normalization(self, img, detected):
        """
        Apply colour correction to an image using detected colour checker data.

        Parameters
        ----------
        img : ndarray
            Original input image (linear RGB, decoded with colour.cctf_decoding).
        detected : list
            Output of `detect_colour_checkers_segmentation(img, additional_data=True)`.
            Should contain at least one detected colour checker.
        method : str, optional
            Colour correction method, one of:
            ["Cheung 2004", "Finlayson 2015", "Vandermonde"].
        degree : int, optional
            Polynomial degree for the correction model.

        Returns
        -------
        image_corrected : ndarray or None
            The colour-corrected image, or None if correction failed.
        """

        try:
            # --- Validate detection result ---
            if not detected or len(detected) == 0:
                print("⚠️ No colour checker data provided to Normalization().")
                return None

            # Each detected checker is a dictionary; get the first one
            detected_swatches, swatch_masks, colour_checker_image, quadrilateral = (
                detected[0].values
            )

            # --- Apply colour correction ---
            image_corrected = colour.colour_correction(
                img,
                detected_swatches,
                REFERENCE_SWATCHES,  # from your global reference (e.g., ColorChecker24)
                method=self.method,
                degree=self.degree,
            )

            return image_corrected

        except Exception as e:
            print(f"❌ Error during colour correction: {e}")
            return None

    def _copy_to_not_detected(self, src_path):
        """Copy failed image to the not_detected folder."""
        dest_path = os.path.join(self.not_detected_folder, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)

    def process_image(self, img_path):
        """
        Process a single image:
        - Load and linearize it.
        - Detect colour checker.
        - Apply correction.
        - Save corrected image, or copy to not_detected if it fails.
        """
        filename = os.path.basename(img_path)
        try:
            # Load image and decode from sRGB
            img = colour.cctf_decoding(colour.io.read_image(img_path))

            # Detect colour checker
            detected = detect_colour_checkers_segmentation(img, additional_data=True)
            if not detected:
                print(f"⚠️ No checker detected: {filename}")
                self._copy_to_not_detected(img_path)
                return False

            # Apply correction
            img_corrected = self.Normalization(img, detected)

            if img_corrected is None:
                print(f"⚠️ Correction failed: {filename}")
                self._copy_to_not_detected(img_path)
                return False

            # Save corrected image
            out_path = os.path.join(self.out_folder, filename)

            # Clip values to [0, 1]
            img_srgb = np.clip(colour.cctf_encoding(img_corrected), 0, 1)

            # Save
            img_bgr = cv2.cvtColor((img_srgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            out_path = os.path.join(self.output_dir, filename)
            # Save image
            cv2.imwrite(out_path, img_bgr)
            print(f"✅ Saved corrected: {out_path}")
            return True

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            self._copy_to_not_detected(img_path, self.not_detected_folder)
            return False

    def run(self):
        print(f"Processing root: {self.root_dir}")
        print(f"Output will be saved to: {self.output_dir}")
        print(f"Using method: {self.method} with degree: {self.degree}")

        total, failed = 0, 0

        # Walk through all subdirectories and process .jpg images
        for dirpath, _, _ in os.walk(self.root_dir):
            images_paths = glob.glob(os.path.join(dirpath, "*.jpg"))
            for img_path in images_paths:
                success = self.process_image(img_path)
                total += 1
                if not success:
                    failed += 1

        print("\n✅ Processing completed.")
        print(f"   Total images processed: {total}")
        print(f"   Failed or not detected: {failed}")


# -------------------------
# --- Main script logic ---
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize colour checker images recursively while preserving folder structure."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/girobat/Olive/foto olivo del  07.08.24",
        help="Root directory containing the folders with images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/girobat/Olive/Corrected",
        help="Output directory where corrected images will be saved.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Cheung 2004",
        help="Method to use for Color correction.",
    )
    parser.add_argument("--degree", type=int, default=1, help="Degree of polynomial.")

    args = parser.parse_args()
    root_dir = args.root_dir
    output_dir = args.output_dir
    method = args.method
    degree = args.degree

    normalizer = ColorNormalizer(
        root_dir=root_dir,
        output_dir=output_dir,
        method=method,
        degree=degree,
    )
    normalizer.run()
