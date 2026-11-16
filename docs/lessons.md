# Lessons from the reference library

The defaults in `scripts/fc/config.py` were measured on one family library of a few tens of thousands of photos
with young siblings, using the owner's blind answers. Your library will differ; the
review loops exist so you can check. This is what went wrong or surprised us, and what the code does about it
now. The threshold numbers are the ones in the code; sample sizes are left out on purpose.

## Recognition

| What happened | What the code does now |
|---|---|
| At cosine distance 0.37 clustering merged two siblings into one huge cluster; 0.31–0.36 kept them apart. | Clustering uses a fixed 0.34 and only relaxes the density (25 → 10 → 5 → 3 neighbours). Leftover clusters never write. |
| Synology called many of the younger sibling's faces by the older sibling's name. Age-free galleries could not separate them. | Children are compared within age bins from `born`; nobody is scored before birth or after death. |
| Babies of similar age were confused with each other below 12 months. | `baby_floor` 0.65 for faces of children under `baby_months` (12). |
| Calibration on owner labels alone chose floor 0.42 and margins near 0: the labelled set under-represents strangers and look-alike babies. | Reviewed defaults (floor 0.50, margin 0.08, sibling margin 0.12) are a lower bound; calibration can only raise them. |
| In a blind round on the edges, every judged acceptance between 0.50 and 0.67 was right; below 0.50 the first errors appeared. | Default floor 0.50. |
| Synology tags with a best score below 0.35 were nearly always someone else for children, but often right for adults. | They become blind cards (`edges build --low-score`), never an automatic unlink. |
| Top-k 3/10 and within-class whitening looked better only because they lowered the floor; equal at a matched false-accept rate. | Kept top-5 mean. Experiments use a frozen 30%-of-days holdout, used once. |
| A wider age window (±6 months) beat ±2 on the holdout by about one point of recall, with no person worse. | `bin_overlap_months` is configurable; 2 by default, try 6 with a holdout check. |
| Same-day photos inflate accuracy: the same event is in gallery and test. | Calibration and cross-validation are separated by capture day. |

## Two circles

Asking the owner to label everyone does not scale; dozens of outer people would need thousands of cards. Using
Synology's tags as galleries brings Synology's mistakes in. So the outer circle uses tags filtered for
self-consistency and distance from the close circle, a stricter floor, and can never take a face from a
close-circle person. On the reference library this left close-circle recall and errors unchanged, and no
close-circle face was lost to the outer circle. In the owner's blind outer check (cards with four shuffled
names each) every judged pick from 0.55 up was right and the first error appeared between 0.50 and 0.55, so the
outer floor stays at 0.55.

Taking every Synology-named person into the outer circle (`from_synology: true`) is convenient and wide: the
curator then plans edits and merges for people you never looked at. It is off by default, and outer-circle
changes are never written by an automatic run.

## Planning and applying

| What happened | What the code does now |
|---|---|
| Most merges of one run went stale: the face-edit cap filled with one-by-one reassignments, so the merges' group cleaning was capped and the merge safety check refused them. | Merges are chosen first and their cleaning is reserved in the face-edit budget. `requeue --run ID` queues held-back merges with their cleaning. |
| `run --from plan` ran on into `apply`. | `--from` always stops after `plan`. |
| Synology answered reads with intermittent DSM error 801 while its indexer was busy. | Reads retry with backoff and report the index status; verification waits for the indexer to go idle. |
| A face, item or group can change between planning and applying (the owner, Synology's indexer). | Apply re-reads before every write and skips (`stale`) anything that changed; a merge also re-lists the people and checks that the group is still unnamed, the target still has the planned name and the faces are the planned ones. |
| A reassignment is two calls (unlink, then link). A failure between them left the face with nobody, and the change marked failed was invisible to undo. | `undo` is built from the write log and checks each face against the NAS; `repair --run ID` reconciles changes that stopped half-way. |
| A merge cannot be taken back by a tool: the group is gone. | Merges are never part of a plain apply: each is listed and they are applied on their own, after their own yes. Large ones are flagged and never written by an automatic run. |
| An apply of several thousand edits takes one to two hours. Started from a chat window, it stops if the window closes. | Start long runs in a terminal. Apply resumes where it stopped: run the same command again. |
| Rating and tag edits rewrite the original file bytes for JPEG (and PNG, CR2 per Synology); face edits are not proven to. | Face Curator writes only face/person links. Warn before bulk rating or tag edits if other tools hash originals. |
| Rotated photos: Synology's face boxes are fractions of the upright image. | Detection runs on the upright XL preview; verified on EXIF orientations 1, 6 and 8. |
| Many "named" Synology person ids were tiny unnamed groups, not people. | Only people with a non-empty name count as named; `min_faces` keeps one-off names out of the outer circle. |
| "Clean audit" once meant five right answers, and a staged run that applied nothing counted towards auto mode. | Auto mode needs every condition in docs/design.md section 3, with floors in code, and is demoted by any mismatch or card error. |

## Working with an AI assistant

- Labels are the owner's. The assistant builds cards and reads results; it never answers cards or judges faces.
- Agent sandboxes may block keyring access; then the owner runs the [NAS] commands (`run`, `apply --yes`,
  `undo --yes`, `repair`) in a terminal, and the assistant runs local stages and analyses.
- Preview every apply (`apply --run ID` without `--yes`) and read the per-circle counts before writing. The yes
  belongs to that preview; a re-plan needs a new one.
- When a limit is in the way (a cap, a held threshold, auto mode not earned), the assistant reports the numbers
  and the owner decides. The assistant does not edit limits to make a run pass.
