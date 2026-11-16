# Contributing

Face Curator writes person labels into somebody's photo library. Changes are judged on what they can do to that
library first.

## Tests

Offline: synthetic embeddings and a fake NAS behind the real synology-photos transport guard. Nothing in the
tests logs in anywhere or reads a real ledger.

```bash
cd scripts
uv run --no-project --with-requirements requirements-test.txt pytest -q test_facecurator.py
```

The fake NAS tests need the synology-photos skill (0.1.0 or later) at `~/.claude/skills/synology-photos`, or
`SYNOPHOTOS_PATH` pointing at its `scripts/synophotos.py`; without it they are skipped. Every safety rule has a
test (the review page's token, host and origin checks; the auto-mode gate; the merge re-check; undo and repair;
file modes). A change to one of them changes its test too, on purpose, and needs a reviewer.

## Rules for changes

- **Never widen the synology-photos allow-list from here.** This skill calls that skill's client in-process and
  sends only the three write calls it already uses (`delete_face`, `add_face`, `merge`) and the reads. A new
  NAS call is added in synology-photos first, from a capture of the web app, following its `CONTRIBUTING.md`.
  Do not copy its client, patch its guards at run time, or call the NAS in any other way.
- **Gates only get stricter without a discussion.** The floors and ceilings in `scripts/fc/config.py`
  (`HARD_CAPS`, `MODE_FLOORS`, `AUTO_CEILING`), the checks before each write in `scripts/fc/changes.py` and the
  request checks in `scripts/fc/review.py` are not loosened in a pull request that does something else.
- **Threshold changes need measurements.** A change to a default in `scripts/fc/config.py` (floors, margins,
  clustering distance, outer-circle filters) comes with the numbers behind it: the blind check or holdout used,
  its sample size, the result before and after. Describe the library in general terms only (docs/lessons.md
  shows the style).
- **Captures and fixtures must be synthetic.** No real names, birth dates, photo ids, file names, face crops,
  embeddings or ledger rows, in tests, docs, examples or issues. Use made-up names (Ada, Bea, Alice Example)
  and made-up dates.
- **synology-photos is a moving part.** This skill uses its Python internals (`SynoClient` and its `ok`, `pages`,
  `fetch`, `login`; `apis`, `js`, `journal`, `group_faces`, `redact`, `tcp_ok`, `load_config`, `CONFIRMED_WRITES`). When one of them changes there, update
  `SYNOPHOTOS_MIN` and the name check in `scripts/fc/library.py` here in the same change.

## Before a commit

Commits run gitleaks and a 2 MB file limit as pre-commit hooks. Install them once:

```bash
uvx pre-commit install
```

After changing the dependencies in the header of `scripts/facecurator.py`, run
`uv lock --script scripts/facecurator.py`, bring `scripts/requirements-test.txt` in line with the new lock, and
commit both.
