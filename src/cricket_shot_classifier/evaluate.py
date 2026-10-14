"""Held-out evaluation with per-class metrics, group-bootstrap intervals and error records."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from . import CLASSES
from .data import ShotDataset, make_transform, validate_splits
from .model import choose_device, load_checkpoint
from .prepare import read_rows
from .train import fingerprint


def group_intervals(y, pred, groups, repetitions=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    scores = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        scores.append(
            [
                accuracy_score(y[indices], pred[indices]),
                f1_score(
                    y[indices], pred[indices], labels=[0, 1, 2], average="macro", zero_division=0
                ),
            ]
        )
    bounds = np.percentile(scores, [2.5, 97.5], axis=0)
    return {metric: bounds[:, i].tolist() for i, metric in enumerate(["accuracy", "macro_f1"])}


def evaluate(checkpoint_path, splits, root, output, split="test"):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Evaluation output already exists; keep the original held-out report."
        )
    validate_splits(splits, root)
    device = choose_device("auto")
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["split_fingerprint"] != fingerprint(splits):
        raise ValueError("Checkpoint and split manifests differ")
    dataset = ShotDataset(
        splits / f"{split}.csv", root, make_transform(checkpoint["config"]["image_size"])
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    ys, probabilities = [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            probabilities.extend(model(x.to(device)).softmax(1).cpu().tolist())
            ys.extend(y.tolist())
    probabilities = np.asarray(probabilities)
    y = np.asarray(ys)
    pred = probabilities.argmax(1)
    groups = np.array([r["group_id"] for r in dataset.rows])
    matrix = confusion_matrix(y, pred, labels=[0, 1, 2])
    train_labels = [CLASSES.index(r["label"]) for r in read_rows(splits / "train.csv")]
    majority = int(np.bincount(train_labels, minlength=3).argmax())
    baseline = np.full(len(y), majority)
    metrics = {
        "split": split,
        "n_images": len(y),
        "n_groups": len(set(groups)),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "classification_report": classification_report(
            y, pred, labels=[0, 1, 2], target_names=CLASSES, zero_division=0, output_dict=True
        ),
        "confusion_matrix": matrix.tolist(),
        "class_order": CLASSES,
        "group_bootstrap_95_percent": group_intervals(y, pred, groups),
        "majority_baseline": {
            "class": CLASSES[majority],
            "accuracy": accuracy_score(y, baseline),
            "macro_f1": f1_score(y, baseline, labels=[0, 1, 2], average="macro", zero_division=0),
        },
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_stage": checkpoint["stage"],
        "split_fingerprint": checkpoint["split_fingerprint"],
        "limitations": [
            "Small sample and few independent source groups.",
            "Assistant visual review; labels need cricket-expert verification.",
            "Source groups are conservative inferred match/photographer groups.",
            "Mostly amateur match photos; broadcast/video generalisation is untested.",
            "Scores are uncalibrated. No rejection class or out-of-distribution detection.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    records = []
    for i, row in enumerate(dataset.rows):
        records.append(
            {
                "path": row["path"],
                "source_url": row["source_url"],
                "group_id": row["group_id"],
                "true_label": CLASSES[y[i]],
                "predicted_label": CLASSES[pred[i]],
                "correct": bool(y[i] == pred[i]),
                **{f"p_{c}": float(probabilities[i, j]) for j, c in enumerate(CLASSES)},
            }
        )
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted",
        ylabel="True",
        title=f"{split.title()} confusion matrix (image counts)",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    failures = [r for r in records if not r["correct"]]
    sections = [
        '<!doctype html><meta charset="utf-8"><title>Misclassifications</title>',
        "<h1>Held-out errors</h1><p>Review stance, camera angle, occlusion and background.</p>",
    ]
    # Include copied thumbnails so the report is portable; attribution remains beside each error.
    from PIL import Image, ImageOps

    for i, row in enumerate(failures):
        with Image.open(root / row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((400, 400))
            image.save(output / f"error_{i:03d}.jpg")
        sections.append(
            f'<figure><img src="error_{i:03d}.jpg"><figcaption>'
            f"True: {row['true_label']}; predicted: {row['predicted_label']}; "
            f'<a href="{html.escape(row["source_url"], quote=True)}">Source and licence</a>'
            "</figcaption></figure>"
        )
    (output / "errors.html").write_text("\n".join(sections))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resnet18/best.pt"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/test"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.splits, args.root, args.output, args.split)


if __name__ == "__main__":
    main()
