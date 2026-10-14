# Model card

## Task and architecture
Input: one RGB image containing a cricket batter playing one of the three supported shots.
Output: softmax scores in the fixed order `cover_drive`, `pull_shot`, `sweep_shot`.
This is a closed-set model: it will assign one of these classes to an unsupported shot or
non-cricket input. Scores are uncalibrated and must not be interpreted as reliable certainty.

ResNet-18 uses ImageNet-1K V1 weights and a new dropout + linear classification head.
Train the head first, then unfreeze only the final backbone stage. Frozen batch-normalisation
statistics avoid drifting on small batches. AdamW, modest weight decay, label smoothing,
gradient clipping and early stopping on validation macro-F1 regularise the small dataset.
The best validation checkpoint is retained across both stages. Test data does not select
architecture, epochs, thresholds or hyperparameters.

## Why these models
ResNet-18 is a transparent, stable transfer-learning baseline with about 11.7M ImageNet
parameters and 1.81 GFLOPs at 224px. MobileNetV3-Small is much lighter (about 2.54M parameters,
0.06 GFLOPs) and is included as a CPU-cost comparison. FLOPs alone do not guarantee latency;
actual measured runtimes and validation metrics decide which is more useful here.
A large vision transformer is not justified for this small MVP. Pose estimation could help
later but adds dependency and keypoint failure modes; image quality should be addressed first.

## Preprocessing
EXIF orientation and RGB conversion; aspect-ratio preserving padding to 224×224; ImageNet
mean/std normalisation. This intentionally differs from the weights' default centre crop
so the bat and feet are retained. Shared transform code prevents training/serving drift.
Training-only horizontal flip, ±5-degree rotation and mild colour jitter. Shot class does
not depend on handedness, so flipping is valid here. Avoid vertical flips, severe crops,
large rotations or augmentations that alter the defining pose.

## Reproducibility and evaluation
See the README and `reports/` for actual runs. CPU seeds, deterministic backend flags,
config snapshots, dependency versions, split fingerprints and checkpoint metadata are recorded.
Exact bitwise results across PyTorch versions/devices are not promised.
Report accuracy, macro-F1, per-class precision/recall/F1/support and confusion matrix.
Group-bootstrap intervals reflect source clustering; intervals remain unstable when there
are few groups and do not cover label error or domain shift. Compare against the train-set
majority-class baseline. Inspect high-confidence errors and source-specific failures.

## Scope
This prototype is for learning and experimentation. It is not validated for automated
coaching decisions, biomechanical analysis, broadcast tagging at scale or commercial accuracy
claims. Expand the independent dataset and conduct external evaluation before those uses.

Official model references:
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html

## CPU feature cache

With `cache_head_views: 2`, the frozen backbone processes two randomly augmented training
views per image once, plus the deterministic validation images. Head training then uses
these cached features. This saves repeated backbone computation and uses well under a
megabyte of feature memory for this dataset. It reduces augmentation diversity compared
with sampling a new image transformation each head epoch. Set `cache_head_views: 0` to
disable caching. The three-epoch fine-tuning stage still processes original images with
fresh training augmentation. Cache only training and validation features; test images are
not used in fitting. A short initial streaming run was stopped because of observed CPU
cost; it was not evaluated on test data or used as the final checkpoint.
