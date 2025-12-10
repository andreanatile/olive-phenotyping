from abc import ABC, abstractmethod
import torch
from pathlib import Path
from PIL import Image
from src.models import utils
from tqdm import tqdm
from src.models.constants import DETECTION, SEGMENTATION


class BaseModel(ABC):
    """
    Abstract base class for a zero-shot object detector.
    """

    def __init__(self, model_id):
        # self.metrics = BboxMetrics if self.model_type == DETECTION else SegmentationMetrics
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.processor = self.load_model(model_id)
        print("Model and processor loaded.")

    @abstractmethod
    def predict(self, images, class_map, **kwargs):
        """Performs inference on a batch of images."""
        pass

    @abstractmethod
    def train(self, **kwargs):
        """Trains the model. To be implemented in subclasses."""
        pass

    def model_identifier(self):
        return self.model_id.split("/")[-1].replace(".pt", "")
