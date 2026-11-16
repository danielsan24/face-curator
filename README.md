# face-curator (Claude Code skill)

A Claude Code skill and CLI that keeps people recognition in a Synology Photos library accurate. It decides on
your Mac who each face is, using galleries you confirm, and writes the corrections (face links, unlinks and
merges of unnamed groups) back to Synology Photos through the
[synology-photos skill](https://github.com/danielsan24/synology-photos). Every decision and write is kept in a
local ledger, so runs can be audited and undone.

- **Two circles.** The *close circle* is the few people you label yourself, with birth dates for children
  (age-aware galleries keep siblings apart). The *outer circle* is other people Synology already knows, learned
  from Synology's own tags with stricter rules.
- **Supervised.** Small blind review rounds measure precision and set the thresholds. Nothing is written
  without a preview and your `--yes`, unless you turn on auto mode and the runs have earned it.
- **It never deletes, moves or copies a photo**, and images never leave the Mac.

> Not affiliated with or endorsed by Synology. The Synology Photos web API is undocumented: every call used here
> was confirmed from captures of the web app (see synology-photos) and may change with a DSM update. Use at
> your own risk, on a library you have backed up.

## Risks

- **It writes person labels to your NAS**: it links faces to people, unlinks them, and merges unnamed groups
  into named people. Wrong labels show up in Synology Photos for everyone who uses the library.
- **A merge cannot be undone by `undo`.** `undo` puts back face links and unlinks from the write log; a merged
  group can only be split again by hand in Synology Photos (the journal lists its faces). This is why merges are
  shown one by one and applied on their own (`apply --kinds merge --yes`).
- **Face embeddings and face crops are biometric data.** They, the cached previews, the ledger (names, birth
  dates, who is in which photo), its backups, exports and reports stay in `data_dir` on this Mac, in files only
  you can read. Keep `data_dir` and `backup_dir` out of synced or shared folders.
- **Auto mode writes unattended.** With `mode.policy: auto`, and only once it is earned (the exact rule is
  below), the weekly scheduled run writes to the NAS with nobody looking at a preview. The default is `staged`.
- Face Curator calls synology-photos' Python client directly, so that skill's own dry run and `--yes` are not
  in the path: the gates described here are the only ones.

## Contents

- `SKILL.md`: what Claude reads (rules, commands, flows)
- `scripts/facecurator.py`: the CLI (`uv run scripts/facecurator.py --help`), with `facecurator.py.lock`
  pinning its dependencies; `scripts/fc/`: the modules; `scripts/review/`: the local review page
- `scripts/test_facecurator.py`: offline tests (`scripts/requirements-test.txt` pins their dependencies)
- `scripts/runner.sh`, `launchd/face-curator.plist`: the optional weekly schedule
- `docs/design.md`: circles, scoring, change kinds, caps, review loops, safety
- `docs/lessons.md`: what went wrong on the reference library and how the code handles it now
- `references/config.example.yaml`, `references/ledger-schema.md`, `references/models.md`
- `CONTRIBUTING.md`, `CHANGELOG.md`

## Requirements

- macOS (system keyring, launchd, notifications), Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- A Synology NAS with DSM 7 and Synology Photos, and the companion
  [synology-photos skill](https://github.com/danielsan24/synology-photos), **version 0.1.0 or later**, set up
  with a DSM account **without 2FA** that can edit people in the space you curate (a read-only account can sync
  and plan but not apply). Face Curator uses that skill's Python internals, so update the two together; it
  refuses to run with an older or incompatible copy.
- InsightFace `buffalo_l` weights, downloaded by you (see "Models").
- Disk: about 300 KB per photo for the preview cache, plus the ledger (about 1 KB per face for its embedding).
- Time for the first run: the sync reads every tagged photo's faces at about 5 requests per second and
  downloads a preview of each photo (hours on a large library), then detection and embedding take about 1 hour
  per 40,000 photos. It runs on the CPU (onnxruntime's CPU provider; no GPU is used). Both stages resume.

## Install

```bash
git clone https://github.com/danielsan24/synology-photos ~/.claude/skills/synology-photos   # if not there yet
git clone https://github.com/danielsan24/face-curator.git ~/.claude/skills/face-curator
```

Set up synology-photos first (`references/setup.md` in that skill: DSM account, keyring item, config) and check
it with `uv run ~/.claude/skills/synology-photos/scripts/synophotos.py login-test`.

### Models

Face Curator uses two files of InsightFace's `buffalo_l` pack: the detector `det_10g.onnx` (about 17 MB) and
the recogniser `w600k_r50.onnx` (about 174 MB). Download `buffalo_l.zip` (about 280 MB) from the InsightFace
model zoo (<https://github.com/deepinsight/insightface/tree/master/model_zoo>) and unzip it into
`~/.insightface/models/buffalo_l/`. `models add-buffalo` records both files and checks their sha256 against the
hashes pinned in `scripts/fc/embed.py`; a file with another hash is refused, and the hash is checked again
every time a model is loaded. Details and optional second models: `references/models.md`.

## Quick start

```bash
FC=~/.claude/skills/face-curator/scripts/facecurator.py
uv run $FC init                       # writes ~/.config/face-curator/config.yaml from the example
uv run $FC models add-buffalo         # --dir if the files are not in ~/.insightface/models/buffalo_l
uv run $FC run --until plan           # first sync, embedding and plan; writes nothing to the NAS
uv run $FC people --top 30 --yaml     # paste your close circle into the config, add birth dates
uv run $FC review                     # label 20-50 faces per close person (per age bin for children)
uv run $FC run --id ID --from assign  # re-plan with your labels (ID is the run's id, YYYY-MM; `status` shows it)
uv run $FC apply --run ID             # preview; read it, then the same line with --yes
uv run $FC apply --run ID --kinds merge   # the merges, each listed; then the same line with --yes
uv run $FC run --id ID                # finish the run: review cards, export, backup, report
```

In Claude Code you can simply ask, for example "set up face-curator for my Synology Photos library: me, my
partner, our two kids born 2016-03 and 2018-09".

## The monthly routine

1. **Run.** `run` (or the weekly schedule, which starts one run per month) syncs, embeds new photos, assigns,
   plans, makes about 20 review cards, exports, backs up and writes a report. In staged mode it writes nothing
   to the NAS.
2. **Review cards.** `review` opens the page; answer the Cards tab (about 20 a month).
3. **Preview.** `apply --run ID` prints what would change: counts per kind, the first face edits, every merge.
4. **Yes.** If the preview is what you want: `apply --run ID --yes` writes the face edits. Then look at each
   merge (`apply --run ID --kinds merge`) and write them with `apply --run ID --kinds merge --yes`.
5. **Close.** If the run was stopped before its end (`run --until plan`), `run --id ID` again takes it through
   `cards` and `close`, and the run becomes `done`. A scheduled run is already done at step 1.

Always pass `--id ID` (or `--run ID`) for a run that is not this month's: without it the commands mean the
current month.

### The review page

`review` serves a page on this Mac only (`http://127.0.0.1:8765/?t=…`; `--port N` or `review.port` changes the
port) and opens it in the browser; `--no-open` only prints the URL. The `t=` part is a random token made for this
session: the page, its images and every save need it, so other websites open in your browser cannot read or
change anything. Do not share the URL. Keep the command running while you label; Ctrl-C in its terminal stops
it. If the page says "Not saved", the answer was not recorded (the server stopped, or a run was writing to the
ledger): try again.

| Tab | What you do | Keys |
|---|---|---|
| Galleries | click a cell (person and age), then mark the faces that are **not** that person (click: red) or not a face (shift-click: amber); the rest count as that person | Enter saves the batch |
| Cards | "who is this?" between the curator's pick (A) and the runner-up (B), with reference faces of both | A, B, N neither, U can't tell, X not a face, S skip, Z undo |
| Audit | the same cards, for the one-off audit | the same |
| Compare | blind checks (`compare`, `study`, `edges`): the full list of names, neither system's answer shown | number keys, O someone else, U, X, S, Z |

A: the curator was right; the face becomes a confirmed example. B, N or X: the curator was wrong; the face gets
your label (the other person, neither of them, or not a face), that person's threshold for that age is raised,
their changes at that age are held back, and auto mode is demoted. U: recorded as "can't tell", counted neither
way. S: nothing recorded. A card that already has an answer is not overwritten: the page asks you to press the
key again to replace it, and the earlier answer is kept in the ledger.

## Auto mode

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

`install-schedule` writes the launchd agent to `launchd/generated/<label>.plist` inside the skill folder and
prints the lines that install or remove it (one try per week, Sunday 06:00; logs in
`~/Library/Logs/face-curator/`). With `mode.policy: staged` the scheduled run only reads and plans. The
scheduled run needs AC power and, with `runner.require_lan: true` (the default), a NAS that answers on the LAN:
that needs `lan_host` set in the synology-photos config, otherwise every try ends with "NAS not reachable on the
LAN".

## Other commands

`status` (runs, gallery coverage, cards waiting, whether auto mode is earned), `progress [--run ID]` (one line
with an ETA while a run is syncing or embedding), `report --run ID`, `undo --run ID` (preview; `--yes` writes),
`repair --run ID` (after a failure or a stop in the middle of an apply: re-reads the NAS and reconciles the
ledger), `requeue --run ID`, `audit --run ID`, `bumps`, `selftest [--n 50]` (re-embeds stored faces from local
images and checks that the embeddings still match: run it after changing a model file or onnxruntime),
`compare` / `study` / `edges` (blind checks), `import-face-cache --dir DIR` (optional: reuse buffalo_l faces a
photo-picker library folder already holds), `--version`. `SKILL.md` has the table.

Environment: `FACECURATOR_CONFIG` (path of the config file; default `~/.config/face-curator/config.yaml`),
`FACECURATOR_DEBUG=1` (print the traceback of a failed stage, with secrets masked).

## Tests

```bash
cd ~/.claude/skills/face-curator/scripts
uv run --no-project --with-requirements requirements-test.txt pytest -q test_facecurator.py
```

Offline. The fake NAS tests need synology-photos installed at `~/.claude/skills/synology-photos` (or
`SYNOPHOTOS_PATH` pointing at its `scripts/synophotos.py`); without it they are skipped.

## Licence

The code is MIT, see `LICENSE`. The InsightFace `buffalo_l` weights are not part of this repository and are not
covered by that licence: InsightFace publishes its pretrained models for non-commercial research use only
(<https://github.com/deepinsight/insightface#license>). Whether your use fits is for you to check.

Personal data stays out of git: the config, ledger, images, models and reports live outside the repository.
