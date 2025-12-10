import os
from ultralytics import YOLO
import json
from src.models.base_model import BaseModel
from src.models.constants import DETECTION, YOLO_MODELS


def add_yolo_parser(subparsers, parent_parser, train=False):
    yolo_parser = subparsers.add_parser(
        "yolo11", help="Use Ultralytics YOLO-World model.", parents=[parent_parser]
    )
    # --- Required Arguments ---
    yolo_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLO model configuration or weights (e.g., 'yolov11n.pt', 'yolov11m.yaml').",
    )
    yolo_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    yolo_parser.add_argument(
        "--patience", type=int, default=10, help="Level of patience for early stopping."
    )
    yolo_parser.set_defaults(load_func=Yolo11Detector.load_detector)
    if train:
        yolo_parser.add_argument(
            "--yaml-path",
            type=str,
            help="Path to the Ultralytics dataset configuration YAML.",
        )


class Yolo11Detector(BaseModel):
    """
    Detector class for the YOLO-World model from the Ultralytics library.
    """

    def __init__(self, model_id):
        self.model_type = DETECTION
        super().__init__(model_id)

    def load_model(self, model_id):
        """
        Loads the YOLOWorld model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        model = YOLO(model_id)
        return model, None

    def predict(self, images, class_map, **kwargs):
        """
        Performs inference on a batch of images using the Ultralytics YOLO-World model.
        The model's classes should be set once before calling this method.
        """
        score_threshold = kwargs.get("score_threshold")

        if not score_threshold:
            print("Argument score_threshold not specified. Using default value (0.05)")
            score_threshold = 0.05

        results_batch = self.model.predict(images, conf=score_threshold, verbose=False)

        all_processed_results = []
        for result in results_batch:
            processed_for_image = []
            # Get the mapping from class index to class name (prompt) for this result
            names = result.names

            for box in result.boxes:
                class_id_tensor = box.cls
                # Ensure a class was detected for the bounding box
                if class_id_tensor.numel() == 0:
                    continue

                class_id = int(class_id_tensor[0])
                label = names[class_id]
                score = float(box.conf[0])

                # The .xyxy attribute provides box coordinates in [xmin, ymin, xmax, ymax] format
                bounding_box = box.xyxy[0].tolist()

                # Ensure the detected label is one of the prompts we care about
                if label in class_map:
                    processed_for_image.append(
                        {
                            "score": score,
                            "label": label,
                            "box": bounding_box,
                            "class_index": class_map[label],  # This will always be 0
                        }
                    )
            all_processed_results.append(processed_for_image)

        return all_processed_results

    def train(self, args):
        """
        Trains the YOLO-World model using the Ultralytics library's built-in training method.
        Additional training parameters can be passed via kwargs.
        """

        metrics_object = self.model.train(
            data=args.yaml_path,
            batch=args.batch_size,
            imgsz=args.image_size,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
        )

        # Ensure a metrics object was returned (training completed successfully)
        if metrics_object:
            # Check if the returned object has the 'results_dict' property
            if hasattr(metrics_object, "results_dict"):
                # This is the recommended way to get all metrics (mAP50, mAP50-95, loss components, etc.)
                serializable_metrics = metrics_object.results_dict

            # Fallback for older versions or slightly different structure
            elif hasattr(metrics_object, "keys") and isinstance(
                metrics_object.keys, (list, tuple)
            ):
                # If it behaves like a dictionary/tuple of results, access the internal dictionary
                serializable_metrics = (
                    metrics_object.results_dict
                    if hasattr(metrics_object, "results_dict")
                    else dict(zip(metrics_object.keys, metrics_object.values))
                )
            else:
                # If all else fails, print the string representation and return
                print(
                    "Warning: Could not convert DetMetrics to dictionary. Printing object representation instead."
                )
                print(metrics_object)
                return

            # --- 3. Save to JSON File ---
            save_directory = getattr(metrics_object, "save_dir", None)
            if save_directory and serializable_metrics:
                output_path = os.path.join(save_directory, "training_metrics.json")
                try:
                    with open(output_path, "w") as json_file:
                        json.dump(serializable_metrics, json_file, indent=4)
                    print(f"Training metrics saved to {output_path}")
                except Exception as e:
                    print(f"Error saving training metrics: {e}")

            else:
                print("Warning: Save directory not found. Metrics not saved to file.")

    @staticmethod
    def load_detector(args):
        print("Loading yolo detector...")
        model_id = YOLO_MODELS[args.model]
        class_map = {label: idx for idx, label in enumerate(args.class_names)}
        detector = Yolo11Detector(model_id)

        return detector, class_map
