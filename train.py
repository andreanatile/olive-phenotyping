import argparse
from src.models.utils import parse_cfg
from src.models.yolo11 import add_yolo_parser



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run closed-world object detection model training.")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        'config_path', type=str, help='Path to the dataset configuration YAML.'
    )
    parent_parser.add_argument(
        '--output-name', type=str, default=None,
        help='The name of the output checkpoint file. By default, uses the model name along with the current timestamp.'
    )

    subparsers = parser.add_subparsers(dest='model_type', required=True, help='Select the model to run.')
    add_yolo_parser(subparsers, parent_parser, train=True)
    args = parser.parse_args()

    config = parse_cfg(args.config_path)
    args.__dict__.update(config)
    
    detector, class_map = args.load_func(args)
    detector.train(args)

    print("Training completed.")
