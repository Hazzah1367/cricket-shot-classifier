"""Small ImageNet backbones, staged freezing and portable checkpoint loading."""

import torch
from torch import nn
from torchvision import models

from . import CLASSES


def build_model(architecture="resnet18", pretrained=True, dropout=0.2):
    if architecture == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, len(CLASSES)))
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.classifier[-2].p = dropout
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def head(model, architecture):
    return model.fc if architecture == "resnet18" else model.classifier


def set_stage(model, architecture, stage):
    for p in model.parameters():
        p.requires_grad = False
    for p in head(model, architecture).parameters():
        p.requires_grad = True
    if stage == "finetune":
        last = model.layer4 if architecture == "resnet18" else model.features[-3:]
        for p in last.parameters():
            p.requires_grad = True
    elif stage != "head":
        raise ValueError(stage)


def freeze_batchnorm_stats(model):
    # Tiny datasets/batches should not overwrite pretrained running mean/variance.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def choose_device(request="auto"):
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != CLASSES or checkpoint.get("schema_version") != 1:
        raise ValueError("Incompatible checkpoint class mapping/schema")
    cfg = checkpoint["config"]
    model = build_model(cfg["architecture"], pretrained=False, dropout=cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def extract_features(model, architecture, images):
    """Return the frozen classifier input; equivalent to the full forward in eval mode."""
    if architecture == "resnet18":
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
    elif architecture == "mobilenet_v3_small":
        x = model.features(images)
    else:
        raise ValueError(architecture)
    return torch.flatten(model.avgpool(x), 1)
