"""The monthly run: resumable stages under one NAS login.

Stages: sync -> embed -> assign -> cluster -> plan -> apply -> cards -> close. A crash leaves the run
'failed' at its stage; the next try resumes there. A scheduled try exits quietly unless this month's run is not
done, the Mac is on AC power and the NAS answers on the LAN.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import subprocess
import traceback
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable

from fc import assign, cards, changes, cluster, library, report
from fc import db as D
from fc.config import Config

STAGES = ["sync", "embed", "assign", "cluster", "plan", "apply", "cards", "close"]
NAS_STAGES = {"sync", "embed", "apply"}
REPLAN_STAGES = ("assign", "cluster", "plan")


RUN_ID = re.compile(r"\d{4}-(0[1-9]|1[0-2])[a-z]?")
UNDO_ID = re.compile(r"(undo-){1,2}\d{4}-(0[1-9]|1[0-2])[a-z]?")


def check_run_id(run: str, undo: bool = False) -> str:
    """Run ids are YYYY-MM, optionally with one lowercase letter for a second run in a month (2026-11b); an undo
    run: undo-YYYY-MM. They end up in file names and in notifications, so
    nothing else is accepted, anywhere a run id comes from outside."""
    if not isinstance(run, str) or not (RUN_ID.fullmatch(run) or (undo and UNDO_ID.fullmatch(run))):
        raise SystemExit(f"run id must look like 2026-11{' or undo-2026-11' if undo else ''}, not {str(run)[:40]!r}")
    return run


def month_id(d: date | None = None) -> str:
    d = d or date.today()
    return f"{d.year:04d}-{d.month:02d}"


def run_no(run: str) -> int:
    y, m = run[:7].split("-")
    return int(y) * 12 + int(m)


@contextmanager
def lock(data_dir: Path):
    """One writer at a time: `run`, `apply --yes`, `undo --yes` and `repair --yes` all take this lock."""
    f = open(os.open(data_dir / "run.lock", os.O_WRONLY | os.O_CREAT, 0o600), "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another face-curator run holds the lock")
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def on_ac_power() -> bool:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True
    return "AC Power" in out or "Battery Power" not in out


def conditions(cfg: Config, sp) -> tuple[bool, list[str]]:
    R = cfg["runner"]
    why = []
    if R["require_ac"] and not on_ac_power():
        why.append("on battery")
    if R["require_lan"]:
        scfg = sp.load_config()
        if not scfg.get("lan_host") or not sp.tcp_ok(scfg["lan_host"], int(scfg["nas_port"])):
            why.append("NAS not reachable on the LAN")
    return not why, why


def auto_gate(db, cfg: Config, run: str) -> list[str]:
    """Why this run may NOT write to the NAS on its own; an empty list means auto mode is earned. All of:

    1. `mode.policy: auto` is set in the config (the default is staged);
    2. auto mode is not demoted (a verify mismatch or an audit error demotes it, in the ledger, until a one-off
       audit created afterwards is completed without an error);
    3. at least `staged_runs_before_auto` (>= 3) earlier runs are done and had their face edits applied;
    4. the latest one-off audit (`audit --run ID`) has no open card, at least `audit_min_judged` (>= 200) judged
       answers and no error;
    5. each of the last `clean_audits_for_auto` (>= 2) done runs has at least `min_judged_per_run` (>= 20) judged
       cards and no error;
    6. this run's calibration is `calibrated` with a lower 95% bound of at least `min_lower95` (>= 0.98).

    The numbers in brackets are floors fixed in code (config.MODE_FLOORS)."""
    M = cfg.mode
    why: list[str] = []
    if M["policy"] != "auto":
        return ["mode.policy is not auto in the config"]
    dem = cards.demotion(db)
    aud = cards.audit_state(db)
    if dem and not (aud and aud["batch"] > (dem.get("at") or "") and not aud["open"] and not aud["errors"]
                    and aud["judged"] >= M["audit_min_judged"]):
        why.append(f"auto mode was demoted on {dem.get('at')} ({dem.get('reason')}): a new one-off audit, completed "
                   "without an error, re-earns it")
    done = [r[0] for r in db.execute("SELECT id FROM runs WHERE status='done' AND mode IN ('staged','auto') AND id < ? "
                                     "ORDER BY id", (run,))]
    # "Applied" = the owner applied the run's face edits: something was written and no face edit is left planned.
    left = "SELECT 1 FROM changes WHERE run=? AND status='planned' AND kind!='merge' LIMIT 1"
    applied = [r for r in done if db.execute("SELECT 1 FROM writes WHERE run=? AND ok=1 LIMIT 1", (r,)).fetchone()
               and not db.execute(left, (r,)).fetchone()]
    if len(applied) < M["staged_runs_before_auto"]:
        why.append(f"{len(applied)} earlier run(s) done with their face edits applied; {M['staged_runs_before_auto']} needed")
    if not aud:
        why.append("no one-off audit yet (`audit --run ID`, then the Audit tab of `review`)")
    elif aud["open"]:
        why.append(f"the one-off audit has {aud['open']} open card(s)")
    elif aud["errors"]:
        why.append(f"the one-off audit found {aud['errors']} error(s): a new audit is needed")
    elif aud["judged"] < M["audit_min_judged"]:
        why.append(f"the one-off audit has {aud['judged']} judged answers; {M['audit_min_judged']} needed")
    last = done[-M["clean_audits_for_auto"]:]
    if len(last) < M["clean_audits_for_auto"]:
        why.append(f"{len(last)} earlier run(s) done; {M['clean_audits_for_auto']} with clean cards needed")
    for prev in last:
        a = cards.audit_result(db, prev)
        if a["errors"]:
            why.append(f"run {prev}: {a['errors']} card error(s)")
        elif a["judged"] < M["min_judged_per_run"]:
            why.append(f"run {prev}: {a['judged']} judged cards; {M['min_judged_per_run']} needed")
    cal = db.execute("SELECT status, lower95 FROM calibration WHERE run=? AND model=?",
                     (run, cfg["models"]["reference"])).fetchone()
    if not cal or cal["status"] != "calibrated":
        why.append("this run is not calibrated")
    elif cal["lower95"] is None or not (cal["lower95"] >= M["min_lower95"]):
        why.append(f"calibration lower 95% bound {cal['lower95']} is below {M['min_lower95']}")
    return why


def choose_mode(db, cfg: Config, run: str, forced: str | None = None) -> str:
    """staged unless every condition of `auto_gate` holds. `forced` can only force staged: asking for auto
    never skips the gate."""
    if forced == "staged":
        return "staged"
    return "staged" if auto_gate(db, cfg, run) else "auto"


def notify(text: str) -> None:
    """A macOS notification. The text is passed as an argument, never spliced into the AppleScript source."""
    try:
        subprocess.run(["osascript", "-e", "on run argv", "-e",
                        'display notification (item 1 of argv) with title "Face Curator"', "-e", "end run", "--",
                        str(text)[:200]], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def after_apply(db, cfg: Config, nas, run: str, res: dict, log=print) -> dict:
    """Wait for the indexer, verify a sample of the writes, record both; any mismatch demotes auto mode."""
    idle = nas.wait_idle(cfg["runner"]["index_wait_minutes"])
    ver = {**verify_sample(db, nas, run), "index_idle": idle}
    if ver["mismatch"]:
        cards.demote(db, f"run {run}: {ver['mismatch']} of {ver['checked']} verified writes did not match the NAS", run=run)
        db.commit()
        log(f"verify: {ver['mismatch']} mismatch(es) in {ver['checked']} checked writes: auto mode is demoted to "
            "staged until a new one-off audit is completed without an error")
    D.set_run(db, run, stats={"apply": res, "verify": ver})
    return ver


def verify_sample(db, nas: library.Nas, run: str, n: int = 20) -> dict:
    rows = db.execute("SELECT * FROM changes WHERE run=? AND status='done' AND kind != 'merge'", (run,)).fetchall()
    sample = random.Random(run).sample(rows, min(n, len(rows)))
    bad = 0
    for c in sample:
        live = nas.faces(c["item"])
        box = json.loads(c["box"])
        b = (box["top_left"]["x"], box["top_left"]["y"], box["bottom_right"]["x"], box["bottom_right"]["y"])
        hit = [f for f in live if library.iou(library.syno_box(f["face_bounding_box"]), b) >= 0.5]
        want = None if c["kind"] in ("unlink", "group-unlink") else c["to_person"]
        if not any((f.get("person_id") or None) == want for f in hit) and not (want is None and not hit):
            bad += 1
    return {"checked": len(sample), "mismatch": bad}


def execute(cfg: Config, run: str | None = None, scheduled: bool = False, until: str | None = None,
            mode: str | None = None, nas_factory: Callable | None = None, engine=None, log=print,
            from_stage: str | None = None, sp=None) -> str | None:
    run = check_run_id(run or month_id())
    if mode not in (None, "staged", "auto"):
        raise SystemExit("mode must be staged (auto is earned, never forced)")
    dbp = D.ledger_path(cfg.data_dir)
    with lock(cfg.data_dir):
        db = D.connect(dbp)
        row = D.run_row(db, run)
        if row and row["status"] == "done":
            log(f"run {run} already done")
            return None
        if nas_factory is None:
            sp = library.load_synophotos(cfg.path_of("synology_photos"))
            if scheduled:
                ok, why = conditions(cfg, sp)
                if not ok:
                    log(f"run {run}: not today ({', '.join(why)}); next try next week")
                    return None
            nas_factory = lambda: library.Nas(cfg, sp)  # noqa: E731
        # `stage` holds the last *completed* stage, so a crash or a kill resumes at the stage after it.
        start = STAGES.index(row["stage"]) + 1 if row and row["stage"] in STAGES else 0
        if row and row["status"] == "failed" and "current" not in D.run_stats(db, run) and row["stage"] in STAGES:
            start = STAGES.index(row["stage"])   # rows written by the first version marked the stage in progress
        if from_stage:
            # Re-plan with new labels or thresholds: only the local stages, and only before anything was applied.
            if from_stage not in REPLAN_STAGES:
                raise SystemExit(f"--from accepts {', '.join(REPLAN_STAGES)}")
            done = row["stage"] if row else None
            if done not in STAGES or STAGES.index(done) < STAGES.index(from_stage) - 1 or \
                    STAGES.index(done) >= STAGES.index("apply") or \
                    db.execute("SELECT COUNT(*) FROM changes WHERE run=? AND status NOT IN ('planned', 'capped')",
                               (run,)).fetchone()[0]:
                raise SystemExit(f"run {run}: --from {from_stage} needs the stages before it done and nothing applied")
            start = STAGES.index(from_stage)
            if until is None or STAGES.index(until) > STAGES.index("plan"):
                until = "plan"                   # a re-plan is local only: never runs on into apply
        D.set_run(db, run, status="running")
        log(f"run {run}: stages {STAGES[start:]}")
        stop = STAGES.index(until) + 1 if until else len(STAGES)
        todo = STAGES[start:stop]
        redact = sp.redact if sp else str
        try:
            for stage in todo:
                D.set_run(db, run, status="running", stats={"current": stage})
                if stage in NAS_STAGES:
                    # A fresh login per NAS stage (single attempt each): hours of local work between stages
                    # would otherwise outlive DSM's idle session timeout.
                    with nas_factory() as nas:
                        _stage(db, cfg, run, stage, nas, mode, engine, log)
                else:
                    _stage(db, cfg, run, stage, None, mode, engine, log)
                D.set_run(db, run, stage=stage, stats={"current": None})
            if todo and todo[-1] == "close":
                D.set_run(db, run, status="done", finished=D.now())
        except SystemExit as e:
            _error(db, run, redact(str(e)), log)
            raise
        except Exception as e:  # noqa: BLE001
            _error(db, run, redact(f"{type(e).__name__}: {e}"), log)
            if os.environ.get("FACECURATOR_DEBUG"):
                log(redact(traceback.format_exc()))
            notify(f"run {run} stopped at a stage error; it resumes next week")
            return run
        return run


def _synced_since(db, item: int, since: str) -> bool:
    r = db.execute("SELECT faces_synced FROM items WHERE id=?", (item,)).fetchone()
    return bool(r and r[0] and r[0] >= since)


def _error(db, run: str, msg: str, log) -> None:
    st = D.run_stats(db, run)
    D.set_run(db, run, status="failed", stats={"errors": (st.get("errors") or []) + [f"{D.now()} {msg}"]})
    log(f"run {run} failed: {msg}")


def _stage(db, cfg: Config, run: str, stage: str, nas, mode: str | None, engine, log) -> None:
    if stage == "sync":
        listed = D.run_stats(db, run).get("sync_listed")
        if listed:                                   # resuming: people and items were listed earlier in this run
            log(f"sync: people and items already listed at {listed['at']}, resuming the face reads")
        else:
            n_people = library.sync_people(db, nas)
            it = library.sync_items(db, nas, log=log)
            listed = {"people": n_people, **it, "at": D.now()}
            D.set_run(db, run, stats={"sync_listed": listed})
        # Faces already read in this run are not read again (rolling slice included).
        started = D.run_row(db, run)["started"]
        todo = [i for i in library.items_needing_faces(db, run_no(run), cfg["sync"]["rolling_fraction"])
                if not _synced_since(db, i, started)]
        log(f"sync faces: {len(todo)} items")
        library.sync_faces(db, nas, todo, log=log)
        n_read = db.execute("SELECT COUNT(*) FROM items WHERE faces_synced >= ?", (started,)).fetchone()[0]
        D.set_run(db, run, stats={"sync": {**{k: v for k, v in listed.items() if k != 'at'}, "faces_read": n_read}})
        # outer_circle.from_synology: people named since the config was loaded (first run included) join now.
        from fc.config import synology_counts
        added = cfg.add_synology_outer(synology_counts(D.ledger_path(cfg.data_dir)))
        if added:
            log(f"outer circle from Synology: +{len(added)} people")
    elif stage == "embed":
        from fc.embed import Engine
        eng = engine or Engine(cfg.models_dir, cfg.active_models)
        items = library.items_needing_embedding(db, eng.models)
        log(f"embed: {len(items)} items")
        D.set_run(db, run, stats={"embed": library.embed_items(db, cfg, eng, items, run, nas, log=log)})
    elif stage == "assign":
        D.set_run(db, run, stats={"assign": assign.run_assignment(db, cfg, run, log=log)})
    elif stage == "cluster":
        D.set_run(db, run, stats={"cluster": cluster.run_clustering(db, cfg, run, log=log)})
    elif stage == "plan":
        D.set_run(db, run, stats={"plan": changes.plan(db, cfg, run, log=log)})
    elif stage == "apply":
        why = ["--mode staged"] if mode == "staged" else auto_gate(db, cfg, run)
        m = "staged" if why else "auto"
        D.set_run(db, run, mode=m, stats={"mode_why": why})
        if m != "auto":
            log("apply: staged, nothing written" + (f" ({'; '.join(why)})" if cfg.mode["policy"] == "auto" else ""))
            return
        _, held = assign.load_bumps(db)
        res = changes.apply(db, cfg, nas, run, held_people=held, log=log, auto=True)
        after_apply(db, cfg, nas, run, res, log=log)
    elif stage == "cards":
        D.set_run(db, run, stats={"cards": cards.monthly(db, cfg, run, log=log)})
    elif stage == "close":
        if not D.integrity_ok(db):
            raise RuntimeError("ledger integrity check failed")
        D.export_jsonl(db, cfg.data_dir / "exports" / run)
        b = D.backup(db, cfg.backup_dir)
        p = report.write(db, cfg, run)
        waiting = db.execute("SELECT COUNT(*) FROM cards WHERE answer IS NULL").fetchone()[0]
        log(f"close: report {p}, backup {b}")
        notify(f"run {run} done, {waiting} cards waiting")
