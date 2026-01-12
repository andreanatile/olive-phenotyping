from ultralytics import YOLO
from src.utils.slice_utils import slice_img


class OliveCounter:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """
        Loads the YOLOWorld model from a checkpoint file.
        The processor is integrated into the model object in this library.
        """
        print(f"Loading model from {model_path}")
        model = YOLO(model_path)
        return model

    def count(
        self,
        img: list = None,
        img_path: str = None,
        conf: int = 0.25,
        overlap_ratio: float = 0.2,
    ):
        """
        Count the number of olives inside the image using YOLO model.
        Args:
            img (list): image in numpy array format.
            img_path (str): Path to the image file.
            conf (int): Confidence threshold for detections.
        Returns:
            counts (list): List of counts of detected olives for each image.
        """
        if img is None and img_path is None:
            raise ValueError("Either img or img_path must be provided.")
        try:
            tiles, coordinates = slice_img(
                img=img,
                img_path=img_path,
                slice_size=640,
                overlap_ratio=overlap_ratio,
            )
        except Exception as e:
            print(f"Error in slicing image: {e}")
            return None

        results = self.model.predict(tiles, conf=conf, verbose=True)
        return results
