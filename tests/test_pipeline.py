"""Small synthetic fixtures validate mechanics only, never model quality."""

import hashlib

import imagehash
import pytest
import torch
from PIL import Image

from cricket_shot_classifier import CLASSES
from cricket_shot_classifier.collect import source_group
from cricket_shot_classifier.data import Letterbox, make_transform, validate_splits
from cricket_shot_classifier.model import (
    build_model,
    extract_features,
    head,
    load_checkpoint,
    set_stage,
)
from cricket_shot_classifier.predict import Predictor
from cricket_shot_classifier.prepare import merge_groups, safe_path, split_rows, write_rows


def test_preprocessing_preserves_geometry():
    image = Image.new("RGB", (400, 200), "red")
    result = Letterbox(224)(image)
    assert result.size == (224, 224)
    assert result.getpixel((0, 112)) == (255, 0, 0)
    assert result.getpixel((0, 0)) == (124, 116, 104)
    tensor = make_transform()(image)
    assert tensor.shape == (3, 224, 224) and torch.isfinite(tensor).all()


def test_source_match_grouping():
    first = source_group("File:Team A v. Team B 001.jpg", {})
    assert first == source_group("File:Team A v. Team B 017.jpg", {})
    assert first != source_group("File:Team A v. Team C 001.jpg", {})


def test_transitive_duplicate_and_source_group_union():
    rows = [
        {"path": str(i), "source_group": g, "sha256": str(i), "label": "cover_drive"}
        for i, g in enumerate(["a", "a", "b", "c"])
    ]
    # A and B share a source; B and C share a visual near-duplicate.
    hashes = [
        (imagehash.hex_to_hash(h), imagehash.hex_to_hash(h))
        for h in ["0" * 16, "f" * 16, "f" * 16, "0f" * 8]
    ]
    groups, pairs = merge_groups(rows, hashes, 0)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert len(pairs) == 1


def test_group_splits_are_disjoint_and_repeatable():
    rows = [
        {"label": c, "group_id": f"{c}_{i}", "path": f"{c}_{i}_{j}"}
        for c in CLASSES
        for i in range(10)
        for j in range(2)
    ]
    a, b = split_rows(rows), split_rows(rows)
    assert a == b
    group_sets = [{r["group_id"] for r in a[s]} for s in ["train", "val", "test"]]
    assert not (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    )
    with pytest.raises(ValueError, match="independent groups"):
        split_rows([r for r in rows if r["group_id"].endswith("_0")])


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../private.jpg")


@pytest.mark.parametrize("architecture", ["resnet18", "mobilenet_v3_small"])
def test_model_stages_and_checkpoint_roundtrip(tmp_path, architecture):
    torch.set_num_threads(2)
    model = build_model(architecture, pretrained=False)
    set_stage(model, architecture, "head")
    assert all(p.requires_grad for p in head(model, architecture).parameters())
    assert any(not p.requires_grad for p in model.parameters())
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    set_stage(model, architecture, "finetune")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > before
    model.eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "classes": CLASSES,
            "config": {"architecture": architecture, "dropout": 0.2, "image_size": 224},
            "state_dict": model.state_dict(),
        },
        path,
    )
    restored, _ = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(model(x), restored(x))
        assert torch.allclose(
            model(x), head(model, architecture)(extract_features(model, architecture, x))
        )
    output = Predictor(path).predict(Image.new("RGB", (320, 240)))
    assert set(output["probabilities"]) == set(CLASSES)
    assert sum(output["probabilities"].values()) == pytest.approx(1.0)


def test_split_guard_detects_changed_images_and_leakage(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    for split in ["train", "val", "test"]:
        rows = []
        for i, c in enumerate(CLASSES):
            path = tmp_path / f"{split}_{c}.jpg"
            Image.new("RGB", (20, 20), (i * 60, 0, 0)).save(path)
            # Deliberately make hashes unique by varying each source's content.
            with path.open("ab") as f:
                f.write(split.encode())
            rows.append(
                {
                    "path": path.name,
                    "label": c,
                    "review_status": "accepted",
                    "group_id": f"{split}_{c}",
                    "source_group": f"{split}_{c}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        write_rows(splits / f"{split}.csv", rows)
    validate_splits(splits, tmp_path)
    (tmp_path / "train_cover_drive.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image changed"):
        validate_splits(splits, tmp_path)


def test_photographer_day_unifies_alternative_fixture_names():
    rows = [
        {
            "path": "a",
            "source_group": "fixture_a",
            "sha256": "a",
            "label": "pull_shot",
            "artist": "Photographer",
            "date": "2025-06-01 14:15:00",
        },
        {
            "path": "b",
            "source_group": "alternative_name",
            "sha256": "b",
            "label": "cover_drive",
            "artist": "Photographer",
            "date": "2025-06-01 16:20:00",
        },
    ]
    hashes = [
        (imagehash.hex_to_hash("0" * 16), imagehash.hex_to_hash("0" * 16)),
        (imagehash.hex_to_hash("f" * 16), imagehash.hex_to_hash("f" * 16)),
    ]
    groups, pairs = merge_groups(rows, hashes, 0)
    assert groups[0] == groups[1] and not pairs


def test_cross_label_duplicates_are_flagged():
    rows = [
        {"path": "a", "source_group": "a", "sha256": "same", "label": "cover_drive"},
        {"path": "b", "source_group": "b", "sha256": "same", "label": "sweep_shot"},
    ]
    hashes = [(imagehash.hex_to_hash("0" * 16), imagehash.hex_to_hash("0" * 16))] * 2
    groups, pairs = merge_groups(rows, hashes, 0)
    assert groups[0] == groups[1]
    assert pairs[0]["cross_label"]
