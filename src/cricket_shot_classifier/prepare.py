"""Validate reviewed images, merge source/duplicate groups and create fixed split manifests."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from . import CLASSES


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path


def merge_groups(rows, hashes, threshold=6):
    """Transitive union, including cross-label duplicates and mirrored copies."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            distance = min(a - b for a in hashes[i] for b in hashes[j])
            identical = rows[i]["sha256"] == rows[j]["sha256"]
            same_day = (
                rows[i].get("artist")
                and rows[i].get("date")
                and rows[i]["artist"] == rows[j].get("artist")
                and rows[i]["date"][:10] == rows[j].get("date", "")[:10]
            )
            if (
                same_day
                or rows[i]["source_group"] == rows[j]["source_group"]
                or identical
                or distance <= threshold
            ):
                union(i, j)
            if identical or distance <= threshold:
                pairs.append(
                    {
                        "left": rows[j]["path"],
                        "right": rows[i]["path"],
                        "phash_distance": distance,
                        "cross_label": rows[i]["label"] != rows[j]["label"],
                    }
                )
    groups = [f"group_{find(i):04d}" for i in range(len(rows))]
    return groups, pairs


def split_rows(rows, seed=42):
    """Five group folds: one test, one validation, three training. Never inspect test scores."""
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    for label in CLASSES:
        count = len({r["group_id"] for r in rows if r["label"] == label})
        if count < 5:
            raise ValueError(f"{label} has only {count} independent groups; need >=5.")
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            np.zeros(len(rows)), y, groups
        )
    )
    test, val = set(folds[0][1]), set(folds[1][1])
    result = {s: [] for s in ["train", "val", "test"]}
    for i, row in enumerate(rows):
        split = "test" if i in test else "val" if i in val else "train"
        result[split].append({**row, "split": split})
    for split, items in result.items():
        if {r["label"] for r in items} != set(CLASSES):
            raise ValueError(f"{split} lacks a class: collect more independent groups.")
    return result


def prepare(manifest, root, output, seed, threshold):
    if output.exists() and any(output.glob("*.csv")):
        raise FileExistsError(
            "Split manifests already exist. Choose a new versioned output directory."
        )
    rows = [r for r in read_rows(manifest) if r["review_status"] == "accepted"]
    if not rows:
        raise ValueError("No reviewed/accepted images. Review candidate contact sheets first.")
    hashes = []
    for row in rows:
        if row["label"] not in CLASSES or not row["source_group"].strip():
            raise ValueError(f"Invalid label/source group: {row}")
        path = safe_path(root, row["path"])
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            hashes.append((imagehash.phash(image), imagehash.phash(ImageOps.mirror(image))))
    groups, pairs = merge_groups(rows, hashes, threshold)
    for row, group in zip(rows, groups, strict=True):
        row["group_id"] = group
    output.mkdir(parents=True, exist_ok=True)
    (output / "duplicate_pairs.json").write_text(json.dumps(pairs, indent=2))
    if any(p["cross_label"] for p in pairs):
        raise ValueError(
            "Cross-label near-duplicates found; resolve duplicate_pairs.json before splitting."
        )
    result = split_rows(rows, seed)
    for split, items in result.items():
        write_rows(output / f"{split}.csv", items)
    audit = {
        "seed": seed,
        "phash_threshold": threshold,
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "counts": {s: dict(Counter(r["label"] for r in items)) for s, items in result.items()},
        "groups": {s: len({r["group_id"] for r in items}) for s, items in result.items()},
        "near_duplicate_pairs": len(pairs),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/reviewed.csv"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-threshold", type=int, default=6)
    args = parser.parse_args()
    prepare(args.manifest, args.root, args.output, args.seed, args.phash_threshold)


if __name__ == "__main__":
    main()
