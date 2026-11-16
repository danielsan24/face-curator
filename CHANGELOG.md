# Changelog

## 0.1.0 (2026-11-16)

First public version. A supervised, age-aware face pipeline for a Synology Photos library: two circles of people,
galleries the owner confirms, calibration that can only get stricter, blind checks, a local review page, a ledger
of every decision and write. Writes to the NAS are staged behind a preview and `--yes`; merges are applied on
their own after being re-checked against the live names; auto mode has to be earned and is demoted by any
mismatch or audit error; `undo` is built from the write log and `repair` reconciles half-written changes. The
review page needs a per-session token and refuses foreign hosts and origins. Needs synology-photos 0.1.0 or
later.
