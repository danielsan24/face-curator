# Ledger

SQLite (WAL) at `<data_dir>/ledger.sqlite` (`data_dir` from the config; default
`~/Library/Application Support/face-curator`); schema version 1, in `scripts/fc/db.py`. Each run exports every
table except embeddings to `<data_dir>/exports/YYYY-MM/*.jsonl.gz` and keeps the last three
`<backup_dir>/ledger-YYYY-MM-DD.sqlite`. The ledger, its backups and exports are readable by the current user
only. They hold names, birth dates and who is in which photo: keep them off synced or shared folders.

| Table | Holds | Lifetime |
|---|---|---|
| `meta` | schema version; `auto_demoted` (when and why auto mode was demoted) | small |
| `items` | Synology item id, file name, capture time, size, type, person-id signature, local image path | updated by sync |
| `syno_people`, `syno_faces` | Synology's current people and faces (face id, person id, box) | replaced per item on sync |
| `faces` | detected faces (box as fractions of the upright image, landmarks), matched Synology face | grows |
| `embeddings` | float16 vectors per face and model | grows |
| `labels` | owner truth: person key, `not:<k>`, `unknown`, `not_face`, `syno:<group>`; latest wins; a label taken back is kept with source `retracted` | append-only |
| `decisions` | per run: person, score, runner-up, age bin, status, reason, challenger | per run |
| `clusters` | per run: residual cluster and pass | per run |
| `changes` | per run: planned edits and their outcome (`planned`, `capped`, `held`, `done`, `stale`, `failed`, `unverified`); kinds `label-fix`, `unlink`, `reassign`, `assign`, `add`, `group-unlink`, `merge`, and `restore` in undo runs | per run |
| `writes` | every NAS call: method, parameters, prior state, result. The row is written with `ok=0` before the call and set to `ok=1` after it; `undo` and `repair` read this table | append-only |
| `cards` | review cards and answers; `payload` keeps the batch of a one-off audit and any replaced answer (`previous`) | append-only |
| `bumps` | threshold raises per person and age bin after a card error | until cleared (`bumps --clear … --yes`, after a backup) |
| `calibration`, `runs` | thresholds and statistics; run stages, mode, report | per run |

Run ids are `YYYY-MM`; the undo of a run is `undo-YYYY-MM`. Photo identity is the Synology item id. Face edits
are not proven to rewrite the original files; rating and tag edits are.

## Importing an existing face cache (optional)

`import-face-cache --dir DIR` reads buffalo_l faces that were already computed, so the first run does not embed
those photos again. `DIR` has the layout of the photo-picker skill's library folder (the local mirror that
`synophotos.py sync` writes and photo-picker scores):

| File under `DIR` | One JSON object per line |
|---|---|
| `cache/ingest_index.jsonl` | `photo_id` (`t<item id>` for the Shared Space), `syno_id` (the Synology item id), `file` (the local image, relative to `DIR`), `taken_at` (ISO capture time) |
| `cache/faces/faces.jsonl` | `photo_id`, and `faces`: a list of `{"bbox": [x1, y1, x2, y2], "det": score, "emb": "<base64 of 512 float16 values>"}` with the box as fractions of the upright image and the embedding from `buffalo_l` (`w600k_r50.onnx`) |

Only `t<item id>` photos are imported (the item id links them to Synology directly); rows with other ids are
counted and skipped. The import is idempotent. Imported faces are marked `face-cache` in `faces.source`. Run
`selftest` afterwards: it re-embeds a sample from the local images and checks that the imported embeddings
match this skill's own pipeline.
