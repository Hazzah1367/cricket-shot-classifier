"""Train the head, then fine-tune the last backbone block; select on validation macro-F1."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import CLASSES
from .data import ShotDataset, make_transform, seed_worker, validate_splits
from .model import (
    build_model,
    choose_device,
    extract_features,
    freeze_batchnorm_stats,
    head,
    set_stage,
)


def fingerprint(splits):
    digest = hashlib.sha256()
    for split in ["train", "val", "test"]:
        digest.update(split.encode())
        digest.update((splits / f"{split}.csv").read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


def configure(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batchnorm_stats(model)
    labels, predictions, total_loss = [], [], 0.0
    with torch.set_grad_enabled(training):
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, labels=list(range(3)), average="macro", zero_division=0
        ),
    }


def cache_features(model, loader, architecture, device, views, batch_size, generator):
    model.eval()
    features, targets = [], []
    # no_grad (not inference_mode): these tensors must be valid inputs to head backprop.
    with torch.no_grad():
        for view in range(views):
            for images, labels, _ in loader:
                features.append(extract_features(model, architecture, images.to(device)).cpu())
                targets.append(labels)
            print(f"Cached {views} view(s): completed {view + 1}", flush=True)
    x, y = torch.cat(features), torch.cat(targets)
    dataset = TensorDataset(x, y, torch.arange(len(y)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=views > 1, generator=generator)


def train(config, splits, root, output, pretrained=True):
    cfg = yaml.safe_load(Path(config).read_text())
    required = {
        "cache_head_views",
        "seed",
        "architecture",
        "image_size",
        "batch_size",
        "num_workers",
        "cpu_threads",
        "head_epochs",
        "finetune_epochs",
        "head_lr",
        "backbone_lr",
        "weight_decay",
        "label_smoothing",
        "dropout",
        "patience",
        "device",
    }
    if set(cfg) != required:
        raise ValueError(f"Config missing/unknown keys: {set(cfg) ^ required}")
    if (
        min(cfg[k] for k in ["head_epochs", "batch_size", "cpu_threads", "image_size", "patience"])
        < 1
    ):
        raise ValueError("Epochs, dimensions, batch size, threads and patience must be positive")
    if cfg["cache_head_views"] < 0 or cfg["finetune_epochs"] < 0:
        raise ValueError("Cache views and fine-tune epochs cannot be negative")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Run directory is nonempty; choose a new run name.")
    validate_splits(splits, root)
    output.mkdir(parents=True, exist_ok=True)
    configure(cfg["seed"], cfg["cpu_threads"])
    device = choose_device(cfg["device"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders = {}
    for split in ["train", "val"]:
        dataset = ShotDataset(
            splits / f"{split}.csv",
            root,
            make_transform(cfg["image_size"], training=split == "train"),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=split == "train",
            num_workers=cfg["num_workers"],
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
    model = build_model(cfg["architecture"], pretrained=pretrained, dropout=cfg["dropout"]).to(
        device
    )
    # Start with ordinary CE + mild smoothing; add class weights only if train support warrants it.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    best, history, total_epoch = (-1.0, float("-inf")), [], 0
    started = time.perf_counter()
    cached = None
    if cfg["cache_head_views"]:
        cached = {}
        for split in ["train", "val"]:
            views = cfg["cache_head_views"] if split == "train" else 1
            cached[split] = cache_features(
                model,
                loaders[split],
                cfg["architecture"],
                device,
                views,
                cfg["batch_size"],
                generator,
            )
    for stage, epochs in [("head", cfg["head_epochs"]), ("finetune", cfg["finetune_epochs"])]:
        if not epochs:
            continue
        if stage == "finetune":
            checkpoint = torch.load(output / "best.pt", weights_only=True, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
        set_stage(model, cfg["architecture"], stage)
        head_ids = {id(p) for p in head(model, cfg["architecture"]).parameters()}
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad and id(p) in head_ids],
                "lr": cfg["head_lr"],
            }
        ]
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        if backbone:
            groups.append({"params": backbone, "lr": cfg["backbone_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
        stale = 0
        for _ in range(epochs):
            total_epoch += 1
            active_model = head(model, cfg["architecture"]) if stage == "head" and cached else model
            active_loaders = cached if stage == "head" and cached else loaders
            training = run_epoch(
                active_model, active_loaders["train"], criterion, device, optimizer
            )
            validation = run_epoch(active_model, active_loaders["val"], criterion, device)
            record = {"epoch": total_epoch, "stage": stage, "train": training, "val": validation}
            history.append(record)
            print(json.dumps(record), flush=True)
            rank = (validation["macro_f1"], -validation["loss"])
            if rank > best:
                best, stale = rank, 0
                checkpoint = {
                    "schema_version": 1,
                    "classes": CLASSES,
                    "config": cfg,
                    "pretrained": pretrained,
                    "epoch": total_epoch,
                    "stage": stage,
                    "validation": validation,
                    "split_fingerprint": fingerprint(splits),
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                torch.save(checkpoint, output / "best.pt")
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    summary = {
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "best_validation_macro_f1": best[0],
        "config": cfg,
        "split_fingerprint": fingerprint(splits),
        "pretrained": pretrained,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "git_commit": commit,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2))
    print(f"Best checkpoint: {output / 'best.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/resnet18"))
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Tests only: random initialization"
    )
    args = parser.parse_args()
    train(args.config, args.splits, args.root, args.output, not args.no_pretrained)


if __name__ == "__main__":
    main()
