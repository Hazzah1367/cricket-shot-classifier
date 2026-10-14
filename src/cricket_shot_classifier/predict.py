"""Predict three closed-set shot probabilities from a local image."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from . import CLASSES
from .data import make_transform
from .model import choose_device, load_checkpoint


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = choose_device(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.transform = make_transform(self.checkpoint["config"]["image_size"])

    def predict(self, image):
        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0).to(self.device)
            probabilities = self.model(x).softmax(1)[0].cpu().tolist()
        return {
            "prediction": CLASSES[max(range(3), key=probabilities.__getitem__)],
            "probabilities": dict(zip(CLASSES, probabilities, strict=True)),
            "note": "Uncalibrated closed-set scores; other shots/non-cricket inputs are unsupported.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resnet18/best.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint, args.device)
    with Image.open(args.image) as image:
        print(json.dumps(predictor.predict(image), indent=2))


if __name__ == "__main__":
    main()
