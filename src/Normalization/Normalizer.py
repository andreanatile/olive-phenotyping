#!/usr/bin/env python3
from fileinput import filename
import json
import shutil
import argparse
from pathlib import Path
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
    def __init__(
        self,
        root_dir,
        output_dir,
        method="Cheung 2004",
        degree=1,
        preload_swatches_path=None,
    ):
        # Convert strings to Path objects
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.method = method
        self.degree = degree
        self.preload_swatches_path = preload_swatches_path
        # Path objects allow easy joining with the / operator
        self.corrected_folder = self.output_dir / "corrected"
        self.not_detected_folder = self.output_dir / "not_detected"

        # parents=True creates the directory and its parents; exist_ok avoids errors if it exists
        self.corrected_folder.mkdir(parents=True, exist_ok=True)
        self.not_detected_folder.mkdir(parents=True, exist_ok=True)

    def Normalization(self, img, detected):
        try:
            if not detected or len(detected) == 0:
                print("⚠️ No colour checker data provided.")
                return None

            # Unpack the detection data
            detected_swatches = detected[0].swatch_colours

            image_corrected = colour.colour_correction(
                img,
                detected_swatches,
                REFERENCE_SWATCHES,
                method=self.method,
                degree=self.degree,
            )
            return image_corrected

        except Exception as e:
            print(f"❌ Error during colour correction: {e}")
            return None

    def _copy_to_not_detected(self, src_path):
        """Copy failed image to the not_detected folder using Path."""
        dest_path = self.not_detected_folder / src_path.name
        shutil.copy(src_path, dest_path)

    def process_image(self, img_path):
        """Processes a single image Path object."""
        filename = img_path.name
        self.one_not_detected = False
        try:
            # colour.io.read_image works fine with Path objects
            img = colour.cctf_decoding(colour.io.read_image(str(img_path)))

            # detect the colour checker inside the image
            detected = detect_colour_checkers_segmentation(img, additional_data=True)

            # Handle no detection
            if not detected:
                print(f"⚠️ No checker detected: {filename}")
                self._copy_to_not_detected(img_path)
                return False

            # Apply normalization
            img_corrected = self.Normalization(img, detected)

            # Handle correction failure
            if img_corrected is None:
                print(f"⚠️ Correction failed: {filename}")
                self._copy_to_not_detected(img_path)
                return False

            # Prepare for saving
            self.save_corrected_image(img_corrected, filename)
            return True

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            self._copy_to_not_detected(img_path)
            self.one_not_detected = True
            return False

    def save_corrected_image(self, img_corrected, filename):
        img_srgb = np.clip(colour.cctf_encoding(img_corrected), 0, 1)
        img_bgr = cv2.cvtColor((img_srgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Save to the 'corrected' subfolder
        out_path = self.corrected_folder / filename

        # cv2.imwrite usually needs a string, so we convert the Path
        cv2.imwrite(str(out_path), img_bgr)
        print(f"✅ Saved corrected: {out_path}")

    def run(self):
        """
        Process the entire root directory for images.
        """
        print(f"Processing root: {self.root_dir}")
        print(f"Output: {self.output_dir}")

        total, failed = 0, 0

        # rglob handles recursive searching easily
        # This finds all .jpg and .JPG files
        for img_path in self.root_dir.rglob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg"]:
                success = self.process_image(img_path)
                total += 1
                if not success:
                    failed += 1

        print(f"\n✅ Completed. Processed: {total}, Failed: {failed}")
        if self.preload_swatches_path is not None and failed > 0:
            print("Preloading swatches for not detected images...")
            self.correct_preload_swatches(self.preload_swatches_path)

    def correct_preload_swatches(self, preload_swatches_path=None):
        """
        For images in the not_detected folder, preload swatches from a given path
        and re-attempt normalization.
        """
        preload_path = Path(preload_swatches_path)
        if not preload_path.exists():
            print(f"❌ Preload swatches path does not exist: {preload_path}")
            return

        with open(preload_path) as f:
            not_detected_swatches = json.load(f)

        total, failed = 0, 0

        for img_path in self.not_detected_folder.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg"]:
                total += 1
                try:
                    # Handle missing swatches
                    if not_detected_swatches.get(img_path.name) is None:
                        print(f"⚠️ No preloaded swatches for: {img_path.name}")
                        failed += 1
                        continue

                    # colour.io.read_image works fine with Path objects
                    img = colour.cctf_decoding(colour.io.read_image(str(img_path)))

                    # Apply normalization
                    img_corrected = self.Normalization(
                        img, [not_detected_swatches[img_path.name]]
                    )

                    # Handle correction failure
                    if img_corrected is None:
                        print(f"⚠️ Correction failed: {filename}")
                        failed += 1
                        continue

                    # Prepare for saving
                    self.save_corrected_image(img_corrected, filename)
                except Exception as e:
                    print(
                        f"❌ Error processing {filename} with preloaded swatches: {e}"
                    )
                    failed += 1


def Normalization_args(parser):
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Root directory of images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results."
    )
    parser.add_argument(
        "--method", type=str, default="Cheung 2004", help="Correction method."
    )
    parser.add_argument(
        "--degree", type=int, default=1, help="Degree of polynomial correction."
    )
    parser.add_argument(
        "--preload_swatches_filepath",
        type=str,
        default=None,
        help="Path for json file to preload color swatches for not detected images.",
    )
    return parser

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize colour checker images.")
    parser = Normalization_args(parser)

    args = parser.parse_args()

    normalizer = ColorNormalizer(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        method=args.method,
        degree=args.degree,
    )
    normalizer.run()
