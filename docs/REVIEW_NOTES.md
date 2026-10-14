# Initial dataset review

334 genuine Wikimedia candidate photos were inspected through numbered contact sheets.
138 were accepted: 46 cover drives, 46 pulls and 46 sweeps. Labels combine the source category
with visible batting pose. This is an assistant review, not independent cricket-expert ground truth.
The target of roughly 50–100 images per class was approached without retaining unclear frames.

Examples of collection problems:
- `commons_70575172` appears among pull candidates but visibly shows a kneeling sweep; its pull
  candidate was rejected and its sweep candidate retained.
- `commons_70854389` appears among drive and sweep candidates; the kneeling pose supports sweep,
  so its drive candidate was rejected.
- `commons_72721729` is another kneeling sweep also present in pull candidates; only sweep retained.
- Wide shots with a tiny batter, obscured bat/arms and upright preparatory backlifts were excluded.
- Redundant burst frames were limited by visual review; all retained photos sharing a source or
  photographer/day are kept in one split even if their perceptual hashes differ substantially.

`data/reviewed.csv` retains every decision. `data/candidates.csv` retains the original labels
and rights records. All candidate photos are included for transparent review: accepted photos are in `data/images/`
and rejected candidates in `data/rejected_images/`. Only accepted
rows in `data/splits/v1/` are used by the model. Near-duplicate hashing is a heuristic, not proof
that unknown duplicates do not exist. Metadata-based grouping was made conservative to reduce
risk from alternative match names.

The v1 split contains 84 training images (28 per class), 27 validation (9 per class) and 27 test
(9 per class), with 33, 8 and 9 connected source groups respectively. There are zero flagged
near-duplicate pairs among the accepted images at perceptual-hash Hamming threshold 6.

137 accepted photos are by Acabashi; one is by Harrias. Holding out inferred match/source groups
reduces fixture leakage but does not establish cross-photographer or broadcast generalisation.
Many follow-throughs remain inherently ambiguous from one frame. Further expert verification
and a deliberately independent external test collection are the highest-priority improvements.
