# Face Curator: design

Face Curator keeps people recognition in a Synology Photos library accurate. It decides on your Mac who each
face is, writes the corrections to Synology Photos, and keeps every decision and write in a local ledger.
Synology's person database is treated as a projection of that ledger, never as the truth.

## 1. Two circles of people

| | Close circle (`close_circle`) | Outer circle (`outer_circle`) |
|---|---|---|
| Who | the people you label yourself: household, close family | other people you list (or, with `from_synology`, everyone named in Synology) |
| Gallery | your labels only | Synology's own tags, filtered, plus any labels of yours |
| Children | per age bin, from `born` | `born` optional |
| Score needed | `default_floor` (0.50), calibrated, can only rise | + `outer_floor_bump` (0.55) |
| Review cards, studies, blind checks | yes | only the outer check (`edges build --outer`) |
| Priority under the caps | first | after the close circle |
| Can take a face from the other circle | yes | **never** |
| Written by an automatic run | yes, within the ceilings | **never**: you apply them after the preview |

An outer gallery uses a Synology-tagged face only if it agrees with the person's other tags (top-5 mean ≥
`outer_consistency`), sits at least `outer_close_gap` further from every close-circle gallery, and the person has
at least `outer_min` such faces (up to `outer_max`, spread across capture years). With `from_synology: true`
(off by default), every Synology-named person with `min_faces` tagged faces joins the outer circle after each
sync, except names in `exclude` or already configured: the curator then plans edits and merges for everyone
Synology knows, so turn it on only after reading a few plans.

## 2. Deciding who a face is

- **Embeddings:** SCRFD detector + ArcFace `buffalo_l` (512-d), computed locally from a cached XL preview.
- **Galleries:** per person; for children per age bin (0–6 m, 6–12 m, 12–24 m, 2–4, 4–6, 6–8, 8+ y), with
  `bin_overlap_months` shared between neighbouring bins so a birthday does not split a gallery.
- **Score:** mean of the top-5 cosine similarities to the gallery valid on the photo's date. Nobody is scored
  before their birth or after their death.
- **Accept only if all hold:** score ≥ floor (higher for babies under `baby_months`, faces without a date, outer
  people); margin over the runner-up ≥ `default_margin` (≥ `default_sibling_margin` against a sibling); every
  sibling who could be in the photo has a usable gallery at that age; the challenger model, if set, agrees.
  Otherwise the face is left alone (abstain).
- **Calibration:** each run fits floor and margins on your labels, event-separated (same-day exemplars are
  removed), to 99.5% precision plus a safety bump. It can only make the reviewed defaults stricter.
- **Leftovers:** faces nobody wins are clustered (fixed cosine distance 0.34, min neighbours 25 → 10 → 5 → 3).
  Clusters only feed reports and cards, never writes.

## 3. From decisions to Synology edits

Change kinds, applied in this order: `label-fix`/`unlink` (Synology contradicts your label), `reassign`/`assign`
(a face moves to the accepted person), `add` (missed faces; off by default), `group-unlink` (clean an unnamed
Synology group before merging it), `merge` (an unnamed group whose faces are ≥ 95% accepted for one person). An undo run has one more kind,
`restore` (put a face back as the write log recorded it).

- **Caps per run:** `face_edits` and `merges`, with maxima in code that the config cannot exceed (10,000 and
  50). Merges are chosen first (close circle, then largest), and their group cleaning is reserved inside the
  face-edit budget, so a merge is never planned without room for its cleaning. Everything over the caps is kept
  as `capped` for the next run (`requeue` after an apply). A merge of more than `caps.merge_auto_max_faces`
  (200, which the config cannot raise) faces is listed separately (`large_merges`).
- **Names.** People are matched by their exact Synology name. A configured name that two Synology people carry
  is reported and nothing is planned for it.
- **Preview, then yes.** `apply --run ID` prints the plan and writes nothing. `apply --run ID --yes` writes the
  face edits only. Merges are applied on their own, `apply --run ID --kinds merge --yes`, after a preview that
  lists every one (group id, face count, target name): a merge cannot be undone by `undo`.
- **Apply** re-reads each photo before writing and skips (`stale`) anything that changed since the plan. Right
  before each merge it re-lists the people from the NAS and merges only if the group is still unnamed, the
  target still exists under the planned name, the group's cleaning is done and the group holds exactly the
  planned faces. It stops at the fifth failed change. `apply --yes`, `undo --yes`, `repair --yes` and `run` take
  the same lock, so only one of them writes at a time.
- **The write log.** The intent of every NAS call is stored in `writes` before the call is sent, and its result
  after; unlinks and merges also go to synology-photos' journal with the names read from the NAS. `undo` is
  built from that log, not from the status of the changes: it puts back every face the run wrote to, including
  a reassignment that failed between its unlink and its link and a write whose answer never arrived; each face
  is checked against the NAS first and left alone if someone changed it since. Merges are listed for a manual
  split. `repair --run ID` re-reads the NAS for every change that is not done but may have been written, and
  reconciles the ledger (and, with `--yes`, finishes a link that stopped after its unlink).
- **Staged until earned.** The rule, word for word as in `SKILL.md` and the README:

**Auto mode** (the scheduled run writes to the NAS with nobody looking at a preview) is used for a run only
when all of this holds; otherwise the run is staged and only plans:

1. `mode.policy: auto` is set in the config (the default is `staged`);
2. auto mode is not demoted;
3. at least 3 earlier runs are done and had their face edits applied (written to the NAS, none left planned);
4. the latest one-off audit (`audit --run ID`, 600 cards by default) has no open card, at least 200 judged
   answers and no error;
5. each of the last 2 done runs has at least 20 judged cards and no error;
6. this run is calibrated, with a lower 95% bound of at least 0.99.

A verify mismatch after any apply, or a card that finds a wrong accepted decision, demotes auto mode in the
ledger; only a new one-off audit, completed without an error, re-earns it. An automatic run writes at most 500
face edits and 5 merges, never an outer-circle change and never a merge of more than
`caps.merge_auto_max_faces` (200) faces; everything else stays planned for the owner. The config can make these
numbers stricter, not looser (floors in code: 3, 2, 200, 20 and 0.98; ceilings: 500 and 5). `--mode` can only
force `staged`. `status` says whether auto mode is earned and, if not, why.

## 4. Your part: labels and blind checks

| Loop | What you do | What it feeds |
|---|---|---|
| Galleries | label 20–50 faces per close person and age bin | the close-circle galleries |
| Monthly cards | about 20 cards per run | precision checks, threshold bumps per person and bin; 20 judged and no error in each of the last 2 runs is a condition of auto mode |
| Audit | `audit --run ID`: 600 cards once (about 45 minutes); at least 200 judged answers and no error | one of the conditions of auto mode (section 3) |
| `compare` | blind "who is this?" where Synology and the curator differ | which system is right |
| `study` | stratified blind study on confusions, missed tags, and a factorial on the hardest sibling pair | population-weighted error rates |
| `edges` | blind cards on the noisy edges (floor bands, close pairs, thin child bins) | the thresholds |
| `edges --outer` | 80 blind 4-name cards on outer-circle picks | the outer floor |

Cards are shuffled and blind: neither system's answer is shown. Labels are yours alone; the assistant never
answers cards.

## 5. Run stages

`sync` (list people and items, read faces; a rolling 1/12 of the library is re-read every run) → `embed` →
`assign` → `cluster` → `plan` → `apply` → `cards` → `close`. `sync`, `embed` and `apply` use the NAS.
`run --id ID --from assign|cluster|plan` re-runs local stages and always stops after `plan`.

## 6. Safety

- Nothing deletes, moves or copies a photo. The synology-photos transport refuses those methods; `delete_face`
  only unlinks a person from a face.
- Face Curator calls synology-photos' client in-process, so that skill's own dry run and `--yes` are not in the
  path. The gates are the ones in this document: preview and `--yes`, merges on their own, the re-checks before
  each write, the earned auto mode with its ceilings.
- Credentials live only in the system keyring (macOS Keychain), read by the synology-photos client.
- Images stay on the Mac. The review page listens on 127.0.0.1, asks every request for a random per-session
  token (in the URL it opens; a header for saves, a cookie for images), refuses other Host and Origin values
  (DNS rebinding, cross-site requests), accepts only JSON bodies of at most 1 MB, serves nothing cacheable, and
  builds the page without turning any name into HTML (plus a Content-Security-Policy without inline script).
  A card that has an answer is changed only by an explicit replace, and the earlier answer is kept.
- Files are created for the current user only (folders 0700, files 0600): config, ledger, backups, exports,
  reports, previews, crops. Keep `data_dir` and `backup_dir` off synced or shared folders: embeddings and crops
  are biometric data.
- Model files are loaded only if their sha256 matches the registration, and the two `buffalo_l` files must
  match the hashes pinned in `scripts/fc/embed.py`.
- Run ids are `YYYY-MM`, optionally with one lowercase letter for a second run in the month (`2026-11b`), and `undo-…` for an undo run: nothing else reaches a file name or a notification.
- The ledger is backed up every run (last three) and exported as JSONL.
