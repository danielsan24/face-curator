#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pyyaml>=6", "pillow>=10", "onnxruntime>=1.20",
#   "opencv-python-headless>=4.10", "requests>=2.31", "urllib3>=2", "keyring>=25", "dnspython>=2.6"]
# ///
"""Face Curator CLI: supervised, age-aware face recognition for Synology Photos with a durable ledger.

NAS access goes through the synology-photos skill (one login per command). Commands that log in to the NAS are
marked [NAS]. Nothing is written to the NAS without `--yes` on a preview the owner has seen, except by a
scheduled run in auto mode, which has to be earned (see `status` and docs/design.md).

First time: `init`, edit the config (close circle with birth dates, outer circle), `models add-buffalo`,
`run --until plan` [NAS], `people --yaml` to pick the close circle, then `review` to label galleries.

Environment: FACECURATOR_CONFIG (path of the config file), FACECURATOR_DEBUG=1 (tracebacks of stage errors).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fc import __version__, assign, cards, changes, library, report  # noqa: E402
from fc import db as D  # noqa: E402
from fc import run as R  # noqa: E402
from fc.config import CONFIG_PATH, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _db(cfg):
    return D.connect(D.ledger_path(cfg.data_dir))


def _open_dirs(cfg) -> list[Path]:
    """Folders of ours that other users of this Mac can read (made before files were private by default)."""
    out = []
    for p in (cfg.path_of("data_dir"), CONFIG_PATH.parent):
        try:
            if p.is_dir() and p.stat().st_mode & 0o077:
                out.append(p)
        except OSError:
            pass
    return out


def _warn_open(cfg) -> None:
    for p in _open_dirs(cfg):
        print(f"note: {p} is readable by other users of this Mac; to make it private: chmod -R go-rwx {_sh(p)}")


def _sh(p) -> str:
    import shlex
    return shlex.quote(str(p))


def cmd_init(args):
    if not CONFIG_PATH.is_file():
        D.private_dir(CONFIG_PATH.parent)
        D.atomic_write(CONFIG_PATH, (REPO / "references" / "config.example.yaml").read_text())
        print(f"wrote {CONFIG_PATH} from the example: set your close circle (with birth dates) and outer circle")
    cfg = load_config()
    db = _db(cfg)
    print(f"ledger: {D.ledger_path(cfg.data_dir)} ({db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]} faces)")
    print(f"people: {len(cfg.close_people)} close, {len(cfg.outer_people)} outer"
          f"{' (+ Synology-named people after the first sync)' if cfg['outer_circle'].get('from_synology') else ''}")
    _warn_open(cfg)
    if not cfg.close_people:
        print("next: `models add-buffalo`, then `run --until plan` [NAS] for the first sync, then `people --yaml`")


def cmd_import(args):
    cfg = load_config()
    root = args.dir or cfg.raw.get("face_cache_dir")
    if not root:
        raise SystemExit("import-face-cache: give --dir DIR (a photo-picker library folder), or set face_cache_dir "
                         "in the config. This step is optional.")
    library.import_face_cache(_db(cfg), Path(root).expanduser())


def cmd_models(args):
    from fc.embed import DETECTOR, MODELS, Registry

    cfg = load_config()
    reg = Registry(cfg.models_dir)
    if args.action == "list":
        for name, spec in MODELS.items():
            e = reg.lock.get(spec.file)
            print(f"{name:28s} {'registered' if e else 'missing':10s} {spec.note}")
        print(f"{'detector (SCRFD 10G)':28s} {'registered' if reg.lock.get(DETECTOR) else 'missing'}")
        return
    if args.action == "add-buffalo":
        base = Path(args.dir or "~/.insightface/models/buffalo_l").expanduser()
        for rel, f in ((DETECTOR, "det_10g.onnx"), (MODELS["buffalo_l"].file, "w600k_r50.onnx")):
            e = reg.register(rel, base / f)        # refused unless the sha256 is the pinned one (fc/embed.py)
            print(f"registered {rel}: sha256 {e['sha256'][:16]}… ({e['size']} bytes), matches the pinned hash")
        return
    if args.action == "add":
        if args.name not in MODELS or not args.file:
            raise SystemExit(f"usage: models add NAME --file PATH; NAME in {sorted(MODELS)}")
        e = reg.register(MODELS[args.name].file, Path(args.file))
        print(f"registered {args.name}: sha256 {e['sha256'][:16]}… ({e['size']} bytes)")


def cmd_selftest(args):
    """Re-embed stored faces from local images; cosine with the stored embedding must be ~1 (pipeline unchanged)."""
    import numpy as np

    from fc.embed import Engine, load_image

    cfg = load_config()
    db = _db(cfg)
    eng = Engine(cfg.models_dir, ["buffalo_l"])
    rows = db.execute("SELECT DISTINCT f.item FROM faces f JOIN embeddings e ON e.face=f.id AND e.model='buffalo_l' "
                      "ORDER BY RANDOM() LIMIT ?", (args.n * 4,)).fetchall()
    rows = [r for r in rows if library.image_path(db, cfg, r[0], None)][:args.n]
    sims = []
    for (item,) in rows:
        p = library.image_path(db, cfg, item, None)
        img = load_image(p, cfg["detect"]["max_side"]) if p else None
        if img is None:
            continue
        dets = eng.faces(img)
        known = [(r[0], (r[1], r[2], r[3], r[4])) for r in db.execute("SELECT id,x1,y1,x2,y2 FROM faces WHERE item=?", (item,))]
        m = library.match_boxes([(i, d["box"]) for i, d in enumerate(dets)], known, min_iou=0.5)
        for i, (fid, _) in m.items():
            v = eng.rec("buffalo_l").embed(img, [dets[i]["kps"]])[0]
            ref = D.unpack(db.execute("SELECT vec FROM embeddings WHERE face=? AND model='buffalo_l'", (fid,)).fetchone()[0])
            sims.append(float(v @ (ref / np.linalg.norm(ref))))
    if not sims:
        raise SystemExit("selftest: no comparable faces (no local images yet: run the embed stage first)")
    s = np.array(sims)
    print(f"selftest: {len(s)} faces from {len(rows)} photos; cosine min {s.min():.4f} median {np.median(s):.4f}")
    if np.median(s) < 0.98:
        raise SystemExit("selftest FAILED: new embeddings are not comparable with the stored ones")
    print("selftest ok: new buffalo_l embeddings match the stored ones")


def cmd_people(args):
    """Top Synology-named people by face count in the ledger."""
    cfg = load_config()
    db = _db(cfg)
    rows = db.execute("SELECT p.name, COUNT(*) n FROM syno_faces s JOIN syno_people p ON p.id=s.person_id "
                      "WHERE p.name != '' GROUP BY p.name ORDER BY n DESC LIMIT ?", (args.top,)).fetchall()
    if not rows:
        raise SystemExit("no Synology people yet: run the first sync (`run --until plan`) [NAS]")
    if args.yaml:
        from fc.config import slug
        print("# paste the people you will label yourself under close_circle; add born: YYYY-MM-DD for children")
        print("close_circle:")
        for name, n in rows:
            p = cfg.by_name(name)
            mark = f"   # now {p.tier}" if p and p.tier else ""
            # the name comes from the NAS: always a quoted string, so it cannot change the meaning of the YAML
            print(f"  {slug(name) + ':':22s} {{name: {json.dumps(name, ensure_ascii=False)}}}   # {n} faces{mark}")
        return
    print("most represented (faces in the ledger):")
    for name, n in rows:
        p = cfg.by_name(name)
        print(f"  {n:7d}  {json.dumps(name, ensure_ascii=False)}{f'  [{p.tier}]' if p and p.tier else ''}")


def cmd_run(args):
    cfg = load_config()
    R.execute(cfg, R.check_run_id(args.id) if args.id else None, scheduled=args.scheduled, until=args.until,
              mode=args.mode, from_stage=args.from_stage)


def cmd_status(args):
    cfg = load_config()
    db = _db(cfg)
    last = None
    for r in db.execute("SELECT id, mode, stage, status, started, finished FROM runs ORDER BY id DESC LIMIT 6"):
        last = last or (r["id"] if R.RUN_ID.fullmatch(r["id"]) else None)
        print(f"run {r['id']}: {r['status']} at {r['stage']} ({r['mode'] or '-'}) {r['started']} -> {r['finished'] or '-'}")
    labels = D.latest_labels(db)
    gal = assign.build_galleries(db, cfg, cfg["models"]["reference"], labels)
    print("gallery coverage:")
    for k, row in assign.gallery_coverage(cfg, gal).items():
        print(f"  {cfg.people[k].name}: " + ", ".join(f"{b} {n}" for b, n in row.items()))
    w = db.execute("SELECT kind, COUNT(*) FROM cards WHERE answer IS NULL GROUP BY kind").fetchall()
    print("cards waiting: " + (", ".join(f"{k} {n}" for k, n in w) or "none"))
    print(f"faces {db.execute('SELECT COUNT(*) FROM faces').fetchone()[0]}, labels {len(labels)}, "
          f"synology faces {db.execute('SELECT COUNT(*) FROM syno_faces').fetchone()[0]}")
    why = R.auto_gate(db, cfg, last or R.month_id())
    print("mode: " + ("auto is earned: the next scheduled run writes to the NAS unattended" if not why
                      else "staged (every run only plans; you apply). Auto mode is not earned: " + "; ".join(why)))
    dup = changes.duplicate_names(db, cfg)
    if dup:
        print("names used by more than one Synology person (nothing is planned for them): "
              + "; ".join(f"{n} {v}" for n, v in dup.items()))
    _warn_open(cfg)


def cmd_progress(args):
    """One line of progress and ETA for a run (reads the ledger only; never touches the NAS)."""
    from datetime import datetime

    cfg = load_config()
    db = D.connect(D.ledger_path(cfg.data_dir), readonly=True)
    rid = R.check_run_id(args.run or R.month_id())
    r = D.run_row(db, rid)
    if r is None:
        raise SystemExit(f"no run {rid}")
    st = json.loads(r["stats"] or "{}")
    now = datetime.now()
    cur = st.get("current") or "-"
    out = [f"{now:%H:%M} run {rid}: status {r['status']}, last completed stage {r['stage'] or 'none'}, now in {cur}"]
    if st.get("errors"):
        out.append(f"errors: {st['errors'][-1]}")
    read = db.execute("SELECT faces_synced FROM items WHERE faces_synced >= ? ORDER BY faces_synced", (r["started"],)).fetchall()
    if cur == "sync" and read:
        todo = [i for i in library.items_needing_faces(db, R.run_no(rid), cfg["sync"]["rolling_fraction"])
                if not R._synced_since(db, i, r["started"])]
        t0, t1 = datetime.fromisoformat(read[0][0]), datetime.fromisoformat(read[-1][0])
        rate = (len(read) - 1) / max((t1 - t0).total_seconds(), 1)
        line = f"face reads: {len(read):,} done, {len(todo):,} left, {rate:.1f}/s"
        if rate:
            eta = len(todo) / rate / 60
            line += f", ETA ~{eta:.0f} min ({datetime.fromtimestamp(now.timestamp() + eta * 60):%H:%M})"
        out.append(line + "; embedding follows")
    if cur == "embed" or r["stage"] in ("sync",):
        done = db.execute("SELECT COUNT(*) FROM items WHERE detected = ?", (rid,)).fetchone()[0]
        left = db.execute("SELECT COUNT(*) FROM items WHERE last_seen IS NOT NULL AND (type IS NULL OR type != 'video') "
                          "AND detected IS NULL").fetchone()[0]
        out.append(f"detection: {done:,} photos done, {left:,} left")
    print(" | ".join(out))


def cmd_review(args):
    from fc.review import serve

    cfg = load_config()
    serve(cfg, D.ledger_path(cfg.data_dir), args.port, not args.no_open)


def _need_run(db, run: str) -> None:
    if D.run_row(db, run) is None:
        raise SystemExit(f"no run {run} in the ledger (`status` lists the runs)")


def cmd_apply(args):
    cfg = load_config()
    run = R.check_run_id(args.run, undo=True)
    db = _db(cfg)
    _need_run(db, run)
    kinds = set(args.kinds) if args.kinds else set(changes.FACE_KINDS)
    if "merge" in kinds and kinds != {"merge"}:
        raise SystemExit(f"merges are applied on their own, after you looked at each group: "
                         f"`apply --run {run} --kinds merge` (preview), then add --yes")
    rows = db.execute("SELECT kind, COUNT(*) FROM changes WHERE run=? AND status='planned' GROUP BY kind", (run,)).fetchall()
    print(f"plan for run {run}: " + (", ".join(f"{k} {n}" for k, n in rows) or "nothing planned"))
    n_sel = sum(n for k, n in rows if k in kinds)
    edits = db.execute("SELECT * FROM changes WHERE run=? AND status='planned' AND kind != 'merge' ORDER BY seq LIMIT 10",
                       (run,)).fetchall()
    n_edits = sum(n for k, n in rows if k != "merge")
    for c in edits:
        print(f"  {c['seq']:5d} {c['kind']:12s} item {c['item']} face {c['syno_face_id']} {c['from_person']} -> "
              f"{c['to_person']} ({c['person']}, score {c['score']})")
    if n_edits > len(edits):
        print(f"  ... and {n_edits - len(edits)} more face edits (the report has the counts per circle and kind)")
    merges = changes.merge_rows(db, cfg, run)
    if merges:
        print(f"merges ({len(merges)}; every one is listed; a merge cannot be undone by `undo`, only split by hand):")
        for m in merges:
            print(f"  group {m['group']}: {m['faces']} faces -> {json.dumps(m['target'], ensure_ascii=False)} "
                  f"(person {m['target_id']})" + ("  LARGE" if m["large"] else "")
                  + (f"  [{m['cleaning']} cleaning edit(s) to apply first]" if m["cleaning"] else ""))
    if not args.yes:
        what = "the merges above" if kinds == {"merge"} else "the face edits"
        print(f"dry run: nothing was written. Add --yes to write {what} to Synology Photos (photos are never deleted "
              "or moved).")
        if merges and kinds != {"merge"}:
            print(f"merges are never part of a plain `apply --yes`: after the face edits, look at each group above and "
                  f"run `apply --run {run} --kinds merge --yes`.")
        return
    if not n_sel:
        print("nothing to write for these kinds")
        return
    with R.lock(cfg.data_dir):
        _, held = assign.load_bumps(db)
        with library.Nas(cfg) as nas:
            res = changes.apply(db, cfg, nas, run, kinds=kinds, held_people=held)
            R.after_apply(db, cfg, nas, run, res)
    print(f"report: {report.write(db, cfg, run)}")
    left = changes.merge_rows(db, cfg, run)
    if left and kinds != {"merge"}:
        print(f"{len(left)} merge(s) wait for their own yes: `apply --run {run} --kinds merge` shows them")


def cmd_requeue(args):
    cfg = load_config()
    run = R.check_run_id(args.run)
    changes.requeue_merges(_db(cfg), cfg, run)
    print(f"preview: `facecurator apply --run {run}`; write: add --yes (merges: --kinds merge)")


def cmd_audit(args):
    cfg = load_config()
    run = R.check_run_id(args.run)
    print(f"audit cards created: {cards.audit(_db(cfg), cfg, run, args.n)}; answer them with `review` (Audit tab)")


def cmd_undo(args):
    cfg = load_config()
    run = R.check_run_id(args.run, undo=True)
    new = R.check_run_id(f"undo-{run}", undo=True)
    db = _db(cfg)
    _need_run(db, run)
    res = changes.plan_undo(db, run, new)
    print(f"undo of {run} planned as run {new}: {res['changes']} face(s) to put back" +
          (f" ({res['uncertain']} with a write whose outcome is unknown: each is checked against the NAS first)"
           if res["uncertain"] else "") +
          (f"; merges need manual splitting in Photos (the journal has their faces): groups {res['merges_manual']}"
           if res["merges_manual"] else ""))
    if not args.yes:
        print(f"dry run: `facecurator undo --run {run} --yes` applies it")
        return
    with R.lock(cfg.data_dir):
        with library.Nas(cfg) as nas:
            out = changes.apply(db, cfg, nas, new)
            R.after_apply(db, cfg, nas, new, out)
    print(f"report: {report.write(db, cfg, new)}")


def cmd_repair(args):
    """[NAS] Reconcile changes that may have been written but are not marked done (a failure or a stop mid-write)."""
    cfg = load_config()
    run = R.check_run_id(args.run, undo=True)
    db = _db(cfg)
    _need_run(db, run)
    if not args.yes:
        with library.Nas(cfg) as nas:
            changes.repair(db, cfg, nas, run, write=False)
        print(f"dry run (the NAS was only read): `facecurator repair --run {run} --yes` updates the ledger and "
              "finishes half-written links")
        return
    with R.lock(cfg.data_dir):
        with library.Nas(cfg) as nas:
            changes.repair(db, cfg, nas, run, write=True)
    print(f"report: {report.write(db, cfg, run)}")


def cmd_report(args):
    cfg = load_config()
    run = R.check_run_id(args.run, undo=True)
    db = _db(cfg)
    _need_run(db, run)
    print(report.write(db, cfg, run))


def cmd_bumps(args):
    cfg = load_config()
    db = _db(cfg)
    if args.clear:
        row = db.execute("SELECT * FROM bumps WHERE person=? AND bin=?", args.clear).fetchone()
        if row is None:
            raise SystemExit(f"no raised threshold for {args.clear[0]} {args.clear[1]}")
        print(f"{row['person']} {row['bin']}: +{row['amount']} {'(staged)' if row['staged'] else ''} {row['reason']}")
        if not args.yes:
            print("dry run: this hold exists because the owner found a wrong accepted decision. Clearing it lowers "
                  "the threshold again and lets these faces be written. Add --yes to clear it (the ledger is backed "
                  "up first).")
            return
        print(f"ledger backed up: {D.backup(db, cfg.backup_dir)}")
        db.execute("DELETE FROM bumps WHERE person=? AND bin=?", args.clear)
        db.commit()
    for b in db.execute("SELECT * FROM bumps"):
        print(f"{b['person']} {b['bin']}: +{b['amount']} {'(staged)' if b['staged'] else ''} {b['reason']}")


def cmd_compare(args):
    from fc import compare

    cfg = load_config()
    db = _db(cfg)
    if args.action == "build":
        compare.build(db, cfg, n_coverage=args.coverage, n_agree=args.agree, n_disagree=args.disagree,
                      has_image=lambda item: library.image_path(db, cfg, item, None) is not None)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator compare report`")
    else:
        print(json.dumps(compare.report(db, cfg, args.run), indent=1))


def cmd_study(args):
    from fc import study

    cfg = load_config()
    db = _db(cfg)
    if args.action == "build":
        study.build(db, cfg, has_image=lambda item: library.image_path(db, cfg, item, None) is not None,
                    min_synced=args.min_synced, pair=tuple(args.pair) if args.pair else None)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator study report`")
    else:
        print(json.dumps(study.report(db, cfg, args.run), indent=1))


def cmd_edges(args):
    from fc import edges

    cfg = load_config()
    db = _db(cfg)
    has = lambda item: library.image_path(db, cfg, item, None) is not None  # noqa: E731
    if args.action == "build" and args.outer:
        edges.build_outer_check(db, cfg, has_image=has)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator edges report`")
    elif args.action == "build" and args.low_score is not None:
        edges.build_low_score(db, cfg, below=args.low_score, has_image=has)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator edges report`")
    elif args.action == "build" and args.synology_check:
        edges.build_synology_check(db, cfg, args.synology_check, n=args.n, has_image=has)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator edges report`")
    elif args.action == "build":
        edges.build(db, cfg, has_image=has)
        print("answer them blind: `facecurator review` → Compare tab; then `facecurator edges report`")
    else:
        print(json.dumps(edges.report(db, cfg, args.run), indent=1))


def cmd_install_schedule(args):
    """Generate the launchd agent. Nothing is installed here: the owner runs the printed lines."""
    import getpass
    import re
    from xml.sax.saxutils import escape

    label = "local." + re.sub(r"[^A-Za-z0-9_-]", "_", getpass.getuser()) + ".face-curator"
    plist = REPO / "launchd" / "face-curator.plist"
    text = (plist.read_text().replace("__RUNNER__", escape(str(HERE / "runner.sh")))
            .replace("__HOME__", escape(str(Path.home()))).replace("__LABEL__", escape(label)))
    dest = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    out = D.atomic_write(REPO / "launchd" / "generated" / f"{label}.plist", text)
    print(f"generated {out}")
    try:
        cfg = load_config()
    except SystemExit:
        cfg = None                 # no config yet: the plist does not depend on it
    if cfg is None:
        print("no config yet (`init`): the scheduled run will do nothing until there is one.")
    elif cfg.mode["policy"] == "auto":
        print("mode.policy is auto: once auto mode is earned (`status`), the scheduled run WRITES to the NAS unattended.")
    else:
        print("mode.policy is staged: the scheduled run reads from the NAS and plans; it writes nothing.")
    print(f"To install (owner, in Terminal.app):\n  mkdir -p ~/Library/Logs/face-curator && chmod 700 ~/Library/Logs/face-curator\n"
          f"  cp {_sh(out)} {_sh(dest)}\n  launchctl bootstrap gui/$(id -u) {_sh(dest)}\n"
          f"To remove:\n  launchctl bootout gui/$(id -u) {_sh(dest)}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="facecurator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"facecurator {__version__}")
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("init", help="write the config from the example (if missing) and create the ledger").set_defaults(f=cmd_init)
    ic = s.add_parser("import-face-cache", help="optional: import an existing buffalo_l face cache (the layout of a "
                                                "photo-picker library folder; see references/ledger-schema.md)")
    ic.add_argument("--dir", help="the folder that holds cache/ingest_index.jsonl and cache/faces/faces.jsonl "
                                  "(default: face_cache_dir in the config)")
    ic.set_defaults(f=cmd_import)
    m = s.add_parser("models", help="list | add-buffalo [--dir] | add NAME --file PATH")
    m.add_argument("action", choices=["list", "add-buffalo", "add"])
    m.add_argument("name", nargs="?")
    m.add_argument("--file")
    m.add_argument("--dir")
    m.set_defaults(f=cmd_models)
    t = s.add_parser("selftest", help="re-embed local images and compare with the stored buffalo_l embeddings")
    t.add_argument("--n", type=int, default=50)
    t.set_defaults(f=cmd_selftest)
    pp = s.add_parser("people", help="most represented Synology-named people (to choose the close circle)")
    pp.add_argument("--top", type=int, default=12)
    pp.add_argument("--yaml", action="store_true", help="print them as a close_circle block to paste into the config")
    pp.set_defaults(f=cmd_people)
    r = s.add_parser("run", help="[NAS] the monthly pipeline (resumes an unfinished run)")
    r.add_argument("--id", help="run id, default the current month YYYY-MM")
    r.add_argument("--scheduled", action="store_true", help="check AC power and LAN first; exit quietly if not")
    r.add_argument("--until", choices=R.STAGES, help="stop after this stage")
    r.add_argument("--from", dest="from_stage", choices=R.REPLAN_STAGES,
                   help="re-run from a local stage (new labels or thresholds) and stop after plan; refused once anything is applied")
    r.add_argument("--mode", choices=["staged"], help="keep this run staged whatever the config says (auto mode "
                                                      "cannot be forced: it is earned, see `status`)")
    r.set_defaults(f=cmd_run)
    s.add_parser("status", help="runs, gallery coverage, cards waiting, whether auto mode is earned").set_defaults(f=cmd_status)
    pg = s.add_parser("progress", help="one line of progress and ETA for a run (local)")
    pg.add_argument("--run", help="run id, default the current month")
    pg.set_defaults(f=cmd_progress)
    v = s.add_parser("review", help="open the local review page (galleries, cards, audit, blind checks)")
    v.add_argument("--port", type=int, help="default review.port in the config (8765)")
    v.add_argument("--no-open", action="store_true", help="only print the URL (it carries this session's token)")
    v.set_defaults(f=cmd_review)
    a = s.add_parser("apply", help="[NAS with --yes] apply a staged run's face edits; merges on their own with --kinds merge")
    a.add_argument("--run", required=True)
    a.add_argument("--kinds", nargs="+", choices=list(changes.KIND_ORDER),
                   help="default: every face edit, no merge. `--kinds merge` applies the merges, alone")
    a.add_argument("--yes", action="store_true")
    a.set_defaults(f=cmd_apply)
    rr = s.add_parser("repair", help="[NAS] re-read the live state of changes that failed or stopped mid-write and "
                                     "reconcile the ledger (dry run; --yes applies)")
    rr.add_argument("--run", required=True)
    rr.add_argument("--yes", action="store_true")
    rr.set_defaults(f=cmd_repair)
    rq = s.add_parser("requeue", help="after an apply: queue merges held by the caps, with their group cleaning (local)")
    rq.add_argument("--run", required=True)
    rq.set_defaults(f=cmd_requeue)
    au = s.add_parser("audit", help="create the one-off audit cards (default 600) for a run")
    au.add_argument("--run", required=True)
    au.add_argument("--n", type=int, default=600)
    au.set_defaults(f=cmd_audit)
    u = s.add_parser("undo", help="[NAS with --yes] put back every face a run wrote to (from the write log); merges "
                                  "are listed for a manual split")
    u.add_argument("--run", required=True)
    u.add_argument("--yes", action="store_true")
    u.set_defaults(f=cmd_undo)
    rp = s.add_parser("report", help="rewrite a run's report")
    rp.add_argument("--run", required=True)
    rp.set_defaults(f=cmd_report)
    b = s.add_parser("bumps", help="list raised thresholds; --clear PERSON BIN (dry run; --yes backs up, then clears)")
    b.add_argument("--clear", nargs=2, metavar=("PERSON", "BIN"))
    b.add_argument("--yes", action="store_true")
    b.set_defaults(f=cmd_bumps)
    cp = s.add_parser("compare", help="blind head-to-head with Synology: build | report")
    cp.add_argument("action", choices=["build", "report"])
    cp.add_argument("--coverage", type=int, default=100, help="faces Synology names nobody on (sample)")
    cp.add_argument("--agree", type=int, default=40, help="control faces where both agree (sample)")
    cp.add_argument("--disagree", type=int, help="cap on disagreements (default: all)")
    cp.add_argument("--run", help="report: comparison set id (default: latest)")
    cp.set_defaults(f=cmd_compare)
    sd = s.add_parser("study", help="blind study (confusions, missed tags, factorial): build | report")
    sd.add_argument("action", choices=["build", "report"])
    sd.add_argument("--pair", nargs=2, metavar="KEY", help="build: the factorial's pair (default: hardest sibling pair)")
    sd.add_argument("--run")
    sd.add_argument("--min-synced", type=float, default=0.95, help="share of photos whose Synology faces were read")
    sd.set_defaults(f=cmd_study)
    eg = s.add_parser("edges", help="blind round on the noisy edges (floor bands, close pairs, thin galleries): build | report")
    eg.add_argument("action", choices=["build", "report"])
    eg.add_argument("--run")
    eg.add_argument("--synology-check", nargs="+", metavar="PERSON",
                    help="build: faces Synology names PERSON that the latest run doubts (n per person)")
    eg.add_argument("--n", type=int, default=20)
    eg.add_argument("--outer", action="store_true",
                    help="build: blind check of outer-circle picks (80 cards in 3 score bands, 4 shuffled names each)")
    eg.add_argument("--low-score", type=float, nargs="?", const=0.35, metavar="BELOW",
                    help="build: every Synology tag on a gallery person whose best score is below BELOW (0.35)")
    eg.set_defaults(f=cmd_edges)

    s.add_parser("install-schedule", help="generate the launchd agent (weekly try, Sunday 06:00) into launchd/generated/ "
                                          "and print the lines that install it").set_defaults(f=cmd_install_schedule)
    return p


def main(argv=None) -> None:
    os.umask(0o077)           # everything this tool writes is personal: readable by this user only
    args = parser().parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
