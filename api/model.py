"""Model loading and inference for serving.

Loads the checkpoint once at startup. Provides a single predict() function
that takes a PIL Image and returns a dict with the top prediction and all
class probabilities.
"""
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.train import build_model
from src.transforms import build_eval_transforms


class ClassifierService:
    """Wraps a trained model for single-image inference."""

    def __init__(self, config_path: str, checkpoint_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.class_names = self.cfg["data"]["classes"]
        self.device = torch.device("cpu")  # serving on CPU — see note

        self.model = build_model(self.cfg["model"]).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform = build_eval_transforms(
            image_size=self.cfg["training"]["image_size"],
            mean=self.cfg["normalize"]["mean"],
            std=self.cfg["normalize"]["std"],
        )

        self.checkpoint_epoch = ckpt["epoch"]
        self.checkpoint_val_f1 = ckpt["val_metrics"]["macro_f1"]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Run inference on one image. Returns top prediction + all probabilities."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_idx = int(probs.argmax())
        return {
            "predicted_class": self.class_names[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probabilities": {
                name: float(p) for name, p in zip(self.class_names, probs)
            },
        }