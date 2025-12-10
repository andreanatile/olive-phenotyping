from base_model import BaseModel
from constants import DETECTION, RT_DETR_MODELS
from transformers import RTDetrV2ForObjectDetection, RTDetrV2ImageProcessor
import torch


class RTDetrDetector(BaseModel):
    def __init__(self, model_id):
        self.model_type = DETECTION
        super().__init__(model_id)

    def load_model(self, model_id):
        processor = RTDetrV2ImageProcessor.from_pretrained(model_id)
        model = RTDetrV2ForObjectDetection.from_pretrained(model_id)
        return model, processor

    @staticmethod
    def load_detector(args):
        print("Loading RT-DETR detector...")
        model_id = RT_DETR_MODELS[args.model]
        class_map = {label: idx for idx, label in enumerate(args.class_names)}
        detector = RTDetrDetector(model_id)

        return detector, class_map

    def predict(self, images, class_map, **kwargs):
        score_threshold = kwargs.get("score_threshold")

        if not score_threshold:
            print("Argument score_threshold not specified. Using default value (0.05)")
            score_threshold = 0.05

        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)

        results = []
        for i in range(len(images)):
            image_results = []
            target_sizes = torch.tensor([images[i].size[::-1]])
            results_per_image = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]

            for score, label, box in zip(
                results_per_image["scores"],
                results_per_image["labels"],
                results_per_image["boxes"],
            ):
                box = box.tolist()
                image_results.append(
                    {
                        "box": [round(b, 2) for b in box],
                        "label": self.model.config.id2label[label.item()],
                        "score": round(score.item(), 3),
                    }
                )
            results.append(image_results)
        return results

    def train(self, args):
        metrics_object = self.model.train(
            data=args.yaml_path,
            batch=args.batch_size,
            imgsz=args.image_size,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
        )
