# Cricket Shot Classifier 🏏

An implemented PyTorch MVP for **cover drive**, **pull shot** and **sweep shot** from one
batting photograph. Includes genuine Wikimedia photos with individual credits, reviewed
manifests, source-grouped splits, ResNet-18 transfer learning, a MobileNetV3-Small comparison,
evaluation reports, CLI inference and a Streamlit upload interface.

**Experimental:** the initial corpus is small and biased towards amateur match photography.
Labels receive an assistant visual review, not an independent expert review. No single-image
model can reliably classify an arbitrary moment of every shot. See the dataset and model cards.

## Current dataset

334 real candidates were inspected; **138 accepted images (46 per class)** are used for training.
Splits are **84 train / 27 validation / 27 test**, grouped by inferred fixture, photographer/day
and detected duplicates. All candidate images and rejection decisions are retained for review.
137 accepted photos share one photographer: generalisation outside this collection is untested.
See [review notes](docs/REVIEW_NOTES.md) and the saved reports for the actual model outcomes.

## Measured outcome

Selected ResNet-18: **66.7% test accuracy (18/27), macro-F1 0.658**. Validation macro-F1 was
0.688 versus MobileNet's 0.625. Pull recall was only 4/9. These are small held-out collection
results, not evidence of broad cricket-image performance. Full metrics, intervals, comparison,
error inspection and tested/pending status are in [results](reports/RESULTS.md).

## Quick start (Python 3.11–3.13; tested on Python 3.12)

Run all commands from this folder. A CPU is enough; no paid service or GPU is required.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Windows/Linux CPU builds avoid a large CUDA download:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` in place of `python`
without changing your execution policy.

On macOS install torch/torchvision from normal PyPI instead of the CPU index; device `auto`
uses Apple MPS where available. Linux/Windows CPU training uses four threads and 224px inputs.
The first training run downloads official ImageNet weights (about 45 MB for ResNet-18).

The downloadable bundle includes trained checkpoints. After installing dependencies, you can
run inference or the app immediately. To retrain, choose a fresh output directory; existing
run directories are protected against accidental overwrites.

```bash
python -m cricket_shot_classifier.train --config configs/resnet18.yaml --output artifacts/resnet18
python -m cricket_shot_classifier.evaluate --checkpoint artifacts/resnet18/best.pt --output reports/my_test
python -m cricket_shot_classifier.predict path/to/batting.jpg --checkpoint artifacts/resnet18/best.pt
python -m streamlit run app.py
```

If using the supplied trained checkpoint, extract it to `artifacts/resnet18/best.pt` and skip
training. `CRICKET_CHECKPOINT` can point the app at another checkpoint. Uploading an image
runs locally; this project does not send uploads to an external prediction API.
Output probabilities are uncalibrated closed-set scores. Other shots and non-cricket images
are unsupported, and the model has no automatic rejection mechanism.

## Structure

| Path | Purpose |
|---|---|
| `src/cricket_shot_classifier/` | Collection, split audit, transforms, model, train, evaluate, predict |
| `configs/` | Small, explicit training configurations |
| `data/images/` | 138 accepted resized photos, subject to their own licences |
| `data/rejected_images/` | 196 excluded candidates retained for review |
| `data/candidates.csv` | Original source metadata and pending category labels |
| `data/reviewed.csv` | Accepted/rejected label review decisions |
| `data/splits/v1/` | Immutable train/val/test manifests, hashes and grouping audit |
| `data/ATTRIBUTION.md` | Per-photo photographer and licence credits |
| `reports/` | Run results, confusion matrix, predictions and error review |
| `artifacts/` | Local model checkpoints; ignored by Git |
| `tests/`, `.github/workflows/` | Unit/contract tests and CPU CI |
| `app.py` | Streamlit upload interface |
| `docs/` | Dataset/model cards and collection review details |

## Dataset workflow

The supplied dataset is ready to train using fixed manifests. To collect a **new version**:

```bash
python -m cricket_shot_classifier.collect --output data/local/new --limit-per-class 100
python -m cricket_shot_classifier.gallery --manifest data/local/new/candidates.csv --root data/local --output reports/new_review
```

The collector deliberately rejects unknown/unsupported licence records. It never downloads
Getty/news photographs from generic search or substitutes synthetic training images.
Review contact sheets plus full images, set `review_status` to `accepted` or `rejected`,
record notes, and verify `source_group`. Save a reviewed CSV, then:

```bash
python -m cricket_shot_classifier.prepare --manifest data/local/new/reviewed.csv --root data/local --output data/splits/v2
```

For this nested example the manifest paths are relative to `data/local`; train/evaluate v2
with `--root data/local --splits data/splits/v2`. For the supplied v1, paths are relative to
project root and no overrides are needed. Use a new run/output directory for each experiment.

Five stratified group folds reserve one fold each for validation and test. Match/source
names and SHA-256/perceptual duplicates are connected **before** splitting. Known cross-label
near-duplicates must be resolved before splitting. Approximate 60/20/20 fractions vary by
source size. All augmentations run on training images only. For video data, explicitly group
all frames from the same source video, preferably the whole match. More frames do not create
more independent examples.

Do not count the pre-augmented Kaggle "drives" collection as a cover-drive dataset: that label
also includes straight/off drives. The project excludes it for that reason and its leakage risk.

## Modelling decisions

- Start with ResNet-18 ImageNet transfer learning; train a dropout + linear head, then the last
  residual stage. Keep batch-normalisation running statistics fixed on this small corpus.
- AdamW, mild label smoothing and weight decay; early stopping selects validation macro-F1.
- Preserve the whole frame using aspect-ratio padding rather than cropping bat/feet away.
  Use ImageNet normalisation, modest colour jitter, ±5° rotations and handedness-safe flipping.
- MobileNetV3-Small is a cheaper comparison; a larger transformer or pose pipeline is not
  justified until the independent dataset improves.
- Configs, seeds, environment versions, split fingerprints and checkpoint class mappings
  are saved. Image integrity and source-group disjointness are checked before each run.

For the comparison, train MobileNet against the **same** manifests, inspect validation scores
and runtime, then evaluate the selected model once on the fixed test set:

```bash
python -m cricket_shot_classifier.train --config configs/mobilenet_v3_small.yaml --output artifacts/mobilenet
```

Reports include accuracy, macro-F1, per-class precision/recall/F1, support, confusion matrix,
per-image scores, the training-majority baseline, source-group bootstrap intervals and a
portable misclassification gallery. Repeatedly tuning against test results invalidates
that test as an untouched holdout; expand the dataset and reserve a fresh external test set.

## Tests and reproducibility

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests use synthetic fixtures solely to check software behaviour; their predictions are not
reported as evidence of cricket accuracy. Tested direct dependency versions are pinned in `constraints.txt` and recorded in
`reports/environment.txt`; install the compatible CPU torch build first. Different hardware
or library versions can change numerical results. Checkpoint loading uses `weights_only=True`
and verifies the schema and fixed class mapping.

## GitHub

Intended repository: `Hazzah1367/cricket-shot-classifier`. The connected integration can write
files but does not expose repository creation. If the repository has not been created yet,
create an empty repository at https://github.com/new (private is a sensible MVP default),
then grant the GitHub connection access. Alternatively, with GitHub CLI installed and signed in:

```bash
bash scripts/publish_github.sh
```

The script verifies the account, creates a private repository if missing and pushes the source,
photos, manifests and reports. The v1 candidate corpus is about 100 MiB with small individual JPEGs, acceptable for this
fixed MVP; move growing datasets to versioned object storage or Git LFS. Model binaries
remain local/downloadable; do not commit them
into ordinary Git history. On Windows use Git Bash for this helper or create/push through
GitHub Desktop. Never put credentials, personal access tokens or API keys in the repository.

## Licences and next improvements

Original code is MIT. Photos retain individual CC BY/CC BY-SA/public-domain terms, listed in
`data/ATTRIBUTION.md` and CSV metadata; resized derivatives keep those terms. Official pretrained
weights retain upstream conditions. The credits and modifications must accompany photo copies.

Prioritise independent photographers/matches, expert label verification and a distinct
broadcast/phone test corpus. Inspect ambiguous poses and background shortcuts before adding
architectural complexity. Consider foreground crops/pose features or short-video models only
when evidence shows that a single frame lacks sufficient information.

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
