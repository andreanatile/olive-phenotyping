import argparse
from src.Normalization.Normalizer import ColorNormalizer, Normalization_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize colour checker images.")
    parser = Normalization_args(parser)

    args = parser.parse_args()

    normalizer = ColorNormalizer(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        method=args.method,
        degree=args.degree,
        preload_swatches_path=args.preload_swatches_filepath,
    )
    normalizer.run()

"""
python3 normalizing.py --root_dir "/mnt/c/Datasets/Olive/foto_originali" --output_dir "/mnt/c/Datasets/Olive/normalized" --method "Cheung 2004" --degree 3
"""