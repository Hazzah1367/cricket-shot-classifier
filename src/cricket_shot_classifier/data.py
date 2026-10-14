"""Shared training and inference preprocessing; retains bat and full stance."""

import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from . import CLASSES
from .prepare import read_rows, safe_path


class Letterbox:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.pad(
            image,
            (self.size, self.size),
            Image.Resampling.BILINEAR,
            color=(124, 116, 104),
            centering=(0.5, 0.5),
        )


def make_transform(size=224, training=False):
    steps = [Letterbox(size)]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5, fill=(124, 116, 104)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


class ShotDataset(Dataset):
    def __init__(self, manifest, root, transform):
        self.rows = read_rows(manifest)
        if not self.rows:
            raise ValueError("Empty dataset")
        self.root, self.transform = root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_path(self.root, row["path"])) as image:
            x = self.transform(image)
        return x, CLASSES.index(row["label"]), index


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def validate_splits(splits, root):
    """Fail closed if a file changed or known source/group/hash leakage was introduced."""
    import hashlib

    seen = {field: {} for field in ["group_id", "source_group", "sha256", "path"]}
    photographer_days = {}
    for split in ["train", "val", "test"]:
        rows = read_rows(splits / f"{split}.csv")
        if {r["label"] for r in rows} != set(CLASSES):
            raise ValueError(f"{split} must contain all three classes")
        paths = set()
        for row in rows:
            if row["review_status"] != "accepted" or row["path"] in paths:
                raise ValueError("Unreviewed or repeated image in split manifest")
            paths.add(row["path"])
            if row.get("artist") and row.get("date"):
                day = (row["artist"], row["date"][:10])
                if day in photographer_days and photographer_days[day] != split:
                    raise ValueError(f"Photographer/day leakage: {day}")
                photographer_days[day] = split
            digest = hashlib.sha256(safe_path(root, row["path"]).read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Image changed: {row['path']}")
            for field, registry in seen.items():
                value = row[field]
                if not value or (value in registry and registry[value] != split):
                    raise ValueError(f"Leakage/empty {field}: {value}")
                registry[value] = split
