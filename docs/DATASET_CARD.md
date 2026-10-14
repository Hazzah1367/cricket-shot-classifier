# Dataset card

## Intended task
Single-image, closed-set classification of cover drive, pull shot and conventional sweep.
A photograph must show the batter's bat, arms, stance and relevant follow-through.
A single frame cannot reliably identify every shot: ambiguous frames should be excluded.
Reverse sweeps, slog sweeps that cannot be distinguished confidently, straight/off drives,
hooks and other shots are outside this initial label vocabulary.

## Sources and rights
Real photographs come from Wikimedia Commons' three shot-specific categories:
- https://commons.wikimedia.org/wiki/Category:Cricket_cover_drive_shots
- https://commons.wikimedia.org/wiki/Category:Cricket_pull_shots
- https://commons.wikimedia.org/wiki/Category:Cricket_sweep_shots

`data/candidates.csv` records each file's original page, original download URL, photographer,
licence and licence URL. These are per-file rights, not a new licence granted by this project.
The collector accepts CC BY, CC BY-SA, CC0 or public-domain records and skips other/unknown rights.
Photos are oriented, converted to RGB, resized to at most 1024 pixels on the longest edge,
and re-encoded as JPEG. Resized photos and other derived versions retain the source's terms,
including CC BY-SA where applicable. See `data/ATTRIBUTION.md` for readable credits.
The MIT licence covers original code and documentation only. Do not remove the photo credits.

## Label review
Commons category membership supplies candidate labels, not guaranteed ground truth.
`data/reviewed.csv` preserves accepted/rejected decisions and notes. Initial review is by the
assistant using numbered contact sheets and selected full images, not a qualified independent
cricket coach. A cricket expert should verify the labels before treating this as a benchmark.
Do not use model predictions to decide test labels or exclusions.

## Grouping and duplicates
`source_group` conservatively groups match names; photo suffixes are removed. Photographer/date
metadata is also used where match names are unavailable. Groups require manual review: names
may vary, and repeat fixtures can share names. Before splitting, SHA-256 and 64-bit perceptual
hashes (including horizontal mirrors) connect likely duplicates into transitive groups.
Photos by the same photographer on the same day are also connected, conservatively handling
alternative fixture names. Every connected group stays in one partition. Perceptual hashing misses some crops/edits;
contact-sheet review is still required. No video frames are used in the supplied corpus.
For added video frames, explicitly set every frame from a source video to the same source group;
use match-level grouping when multiple videos originate from the same match.

Five stratified group folds give approximately 60% train / 20% validation / 20% test.
Actual fractions depend on group sizes. Splits are fixed with seed 42 and persisted as CSVs.
Training refuses changed images or overlapping source/group/hash identifiers across splits.
All augmentation happens online on the training partition only.

## Bias and limits
- Many photographs are amateur cricket in Essex, often in white kit, from one photographer.
- Match/background/photographer cues may correlate with shot class.
- Camera viewpoint, bat occlusion, image scale and handedness affect recognisability.
- Player identities can recur across fixtures; no player-level holdout was constructed.
- The unit of independence is the inferred source group, not necessarily every photograph.
- Generalisation to broadcast footage, phones, indoor nets, women/junior cricket or other
  geographies is not established by the initial evaluation.
- Add genuinely new matches/photographers before increasing frames or augmentations.
