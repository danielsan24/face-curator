"""Change sets: decisions -> Synology edits, in the owner's order, then a guarded apply.

Order: 1 point fixes (reassign / assign a face to the right gallery person; owner labels first, then by score),
2 group cleaning (unlink every face of an unnamed group that is not accepted for its majority person),
3 merges of cleaned groups into that person. Caps bound face edits and merges per run; the code fixes maxima
the config cannot exceed (`config.HARD_CAPS`).

Apply re-reads each photo's faces right before writing and skips the change if Synology no longer matches the
plan; a merge also re-lists the people and checks both names and the group's faces. Every write is recorded in
`writes` (the intent before the call, the result after) and in synology-photos' journal. `undo` is built from
that write log, so it also covers changes that failed half-way; `repair` reconciles such changes with the live
state. No call here deletes, moves or copies a photo: `delete_face` only unlinks a person from a face.

synology-photos is called in-process: its own dry run and `--yes` are not in this path. The gates here (preview,
`--yes`, the earned auto mode, the re-checks before each write) are the only ones.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from typing import Any

from fc import db as D
from fc.config import Config
from fc.library import Nas, box_dict, iou, store_syno_faces, syno_box

# "unlink" comes only from owner labels (not this person / not a face), so it ranks with the owner's label fixes.
# "restore" exists only in undo runs: it puts a face back to the person it had before a run's writes.
KIND_ORDER = {"label-fix": 0, "unlink": 0, "reassign": 1, "assign": 1, "restore": 1, "add": 2, "group-unlink": 4,
              "merge": 5}
FACE_KINDS = [k for k in KIND_ORDER if k != "merge"]
UNLINK_KINDS = ("unlink", "group-unlink")
MAX_FAILURES = 5              # apply stops at the fifth failed change


def duplicate_names(db: sqlite3.Connection, cfg: Config) -> dict[str, list[int]]:
    """Configured names that more than one Synology person id carries: {name: [ids]}."""
    ids: dict[str, list[int]] = defaultdict(list)
    for r in db.execute("SELECT id, name FROM syno_people WHERE name != '' ORDER BY id"):
        ids[r[1]].append(r[0])
    wanted = {p.name for p in cfg.people.values()}
    return {n: v for n, v in ids.items() if len(v) > 1 and n in wanted}


def syno_ids(db: sqlite3.Connection, cfg: Config) -> dict[str, int]:
    """person key -> Synology person id, by exact name. Missing names are reported, never created. A name that
    two Synology people carry maps to nobody: no change is planned for it until the owner fixes it in Photos."""
    dup = duplicate_names(db, cfg)
    by_name = {r[1]: r[0] for r in db.execute("SELECT id, name FROM syno_people WHERE name != ''") if r[1] not in dup}
    return {k: by_name[p.name] for k, p in cfg.people.items() if p.name in by_name}


def plan(db: sqlite3.Connection, cfg: Config, run: str, log=print) -> dict:
    sid = syno_ids(db, cfg)
    dup = duplicate_names(db, cfg)
    if dup:
        log("plan: no changes are planned for " + "; ".join(f"{n!r} (Synology person ids {v})" for n, v in dup.items())
            + ": the name is used by more than one person in Synology Photos. Merge or rename them there first.")
    missing = [p.name for p in cfg.gallery_people if p.key not in sid and p.name not in dup]
    rev = {v: k for k, v in sid.items()}
    Ch = cfg["changes"]
    labels = D.latest_labels(db)
    rows = db.execute(
        "SELECT d.face, d.person, d.score, d.status, d.reason, f.item, f.x1, f.y1, f.x2, f.y2, f.det, f.syno_face_id, "
        "s.person_id AS sp, s.x1 AS sx1, s.y1 AS sy1, s.x2 AS sx2, s.y2 AS sy2 FROM decisions d JOIN faces f ON f.id=d.face "
        "LEFT JOIN syno_faces s ON s.face_id=f.syno_face_id WHERE d.run=?", (run,)).fetchall()
    out: list[dict] = []
    group_faces: dict[int, list[sqlite3.Row]] = defaultdict(list)
    unnamed = {r[0] for r in db.execute("SELECT id FROM syno_people WHERE name = ''")}
    for r in rows:
        if r["sp"] in unnamed:
            group_faces[r["sp"]].append(r)
    # 2/3 first decide which unnamed groups will be merged: their faces are handled by the merge, not one by one.
    merges: dict[int, str] = {}
    for g, faces in group_faces.items():
        all_syno = db.execute("SELECT COUNT(*) FROM syno_faces WHERE person_id=?", (g,)).fetchone()[0]
        if all_syno < Ch["group_min_faces"] or len(faces) < all_syno:   # every face must be embedded and decided
            continue
        acc = Counter(_target(r, labels) for r in faces)
        top, n_top = acc.most_common(1)[0]
        if top is None or top not in sid:
            continue
        open_cards = db.execute("SELECT COUNT(*) FROM cards WHERE answer IS NULL AND face IN (%s)"
                                % ",".join("?" * len(faces)), [r["face"] for r in faces]).fetchone()[0]
        if n_top / len(faces) >= Ch["group_merge_min_accepted"] and not open_cards:
            merges[g] = top
    close = {p.key for p in cfg.close_people}
    protected = 0
    for r in rows:
        tgt = _target(r, labels)
        lab = labels.get(r["face"])
        sp = r["sp"]
        in_merge = sp in merges
        if in_merge and tgt == merges[sp]:
            continue                                   # rides along with the merge
        if in_merge:                                   # 2: clean the group before the merge
            if tgt and tgt in sid:
                out.append(_ch("reassign", r, sp, sid[tgt], tgt, group=sp))
            else:
                out.append(_ch("group-unlink", r, sp, None, None, group=sp))
            continue
        if tgt and tgt in sid:
            to = sid[tgt]
            if r["syno_face_id"] is not None:
                if sp == to:
                    continue
                if not lab and cfg.people[tgt].outer and rev.get(sp) in close:
                    protected += 1             # the outer circle never takes a face from the close circle
                    continue
                out.append(_ch("label-fix" if lab else ("reassign" if sp else "assign"), r, sp, to, tgt))
            elif lab or (Ch["add_missed_faces"] and (r["det"] or 0) >= Ch["min_det_add"]
                         and (r["y2"] - r["y1"]) >= Ch["min_height_add"]):
                out.append(_ch("add", r, None, to, tgt))
        elif lab and r["syno_face_id"] is not None and sp:
            # Owner said: not this person / not a face / unknown -> unlink the wrong person.
            wrong = rev.get(sp)
            if lab == "not_face" or (lab.startswith("not:") and wrong in lab[4:].split(",")) or \
                    (lab in cfg.people and lab != wrong and lab not in sid):
                out.append(_ch("unlink", r, sp, None, None))
    for g, person in merges.items():
        out.append({"kind": "merge", "item": None, "face": None, "syno_face_id": None, "from_person": g,
                    "to_person": sid[person], "box": None, "group_id": g, "person": person, "score": None,
                    "detail": json.dumps({"faces": len(group_faces[g]), "target_name": cfg.people[person].name})})
    tier = lambda c: 0 if c["person"] is None or not cfg.people[c["person"]].outer else 1   # noqa: E731
    out.sort(key=lambda c: (KIND_ORDER[c["kind"]], tier(c), -(c["score"] or 0)))
    db.execute("DELETE FROM changes WHERE run=? AND status IN ('planned', 'capped')", (run,))   # a re-plan replaces both
    caps = cfg.caps                                  # the config's caps, never above config.HARD_CAPS
    # Merges are chosen first and their cleaning edits reserved inside the face-edit cap: a merge whose cleaning
    # does not fit would only go stale at apply time (most merges of a run were once lost to a cap filled by
    # one-by-one edits).
    cleaning = Counter(c["group_id"] for c in out if c["kind"] != "merge" and c["group_id"] is not None)
    chosen, reserved = _choose_merges([(c["group_id"], json.loads(c["detail"])["faces"], cleaning[c["group_id"]], tier(c))
                                       for c in out if c["kind"] == "merge"], caps)
    budget = caps["face_edits"] - reserved
    edits = 0
    counts: Counter = Counter()
    held: Counter = Counter()
    by_tier: dict[str, Counter] = {"close": Counter(), "outer": Counter()}
    for seq, c in enumerate(out):
        status = "planned"
        if c["kind"] == "merge" or c["group_id"] is not None:      # a merge and its cleaning go (or wait) together
            if c["group_id"] not in chosen:
                status = "capped"
        else:
            edits += 1
            if edits > budget:
                status = "capped"
        counts[c["kind"]] += status == "planned"
        held[c["kind"]] += status != "planned"
        by_tier[("close", "outer")[tier(c)]][c["kind"] + ("" if status == "planned" else " (capped)")] += 1
        db.execute("INSERT INTO changes(run, seq, kind, item, face, syno_face_id, from_person, to_person, box, group_id, "
                   "person, score, status, detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (run, seq, c["kind"], c["item"], c["face"], c["syno_face_id"], c["from_person"], c["to_person"],
                    c["box"], c["group_id"], c["person"], c["score"], status, c.get("detail")))
    db.commit()
    large = [{"group": g, "faces": len(group_faces[g]), "target": cfg.people[person].name}
             for g, person in merges.items() if len(group_faces[g]) > caps["merge_auto_max_faces"]]
    res = {"planned": dict(counts), "capped": dict(held), "groups_to_merge": len(merges), "missing_people": missing,
           "by_tier": {k: dict(v) for k, v in by_tier.items()}, "outer_kept_off_close": protected,
           "duplicate_names": dup, "large_merges": large}
    log(f"plan: {res}")
    return res


def _choose_merges(cands: list[tuple], caps: dict) -> tuple[set[int], int]:
    """cands: (group, faces, cleaning edits[, tier 0 close / 1 outer]). Close circle first, then the largest groups,
    within the merge cap, their cleaning within the edit cap."""
    chosen: set[int] = set()
    reserved = 0
    for g, _faces, n, *_ in sorted(cands, key=lambda t: (t[3] if len(t) > 3 else 0, -t[1])):
        if len(chosen) < caps["merges"] and reserved + n <= caps["face_edits"]:
            chosen.add(g)
            reserved += n
    return chosen, reserved


def requeue_merges(db: sqlite3.Connection, cfg: Config, run: str, log=print) -> dict:
    """After an apply: put merges back in the queue that were capped, or went stale only because their cleaning was
    capped, together with that cleaning. Local only; `apply` re-checks every face and group against the NAS."""
    if db.execute("SELECT COUNT(*) FROM changes WHERE run=? AND status='planned'", (run,)).fetchone()[0]:
        raise SystemExit(f"run {run}: planned changes are still waiting; apply them first")
    merges = db.execute("SELECT id, group_id, status, person FROM changes WHERE run=? AND kind='merge' AND (status='capped' OR "
                        "(status='stale' AND detail LIKE '%cleaning change(s) not done'))", (run,)).fetchall()
    cands, prev = [], set()
    for m in merges:
        g = m["group_id"]
        st = Counter(r[0] for r in db.execute("SELECT status FROM changes WHERE run=? AND group_id=? AND kind!='merge' "
                                               "AND status!='done'", (run, g)))
        if set(st) - {"capped"}:          # a cleaning change failed or went stale: needs a fresh plan, not a requeue
            continue
        faces = db.execute("SELECT COUNT(*) FROM syno_faces WHERE person_id=?", (g,)).fetchone()[0]
        p = cfg.people.get(m["person"])
        cands.append((g, faces, st["capped"], 1 if p is not None and p.outer else 0))
        if m["status"] == "stale":
            prev.add(g)
    # groups already tried once go first, then the largest
    chosen, reserved = _choose_prev_first(cands, prev, cfg.caps)
    for m in merges:
        if m["group_id"] in chosen:
            faces = next(c[1] for c in cands if c[0] == m["group_id"])
            p = cfg.people.get(m["person"])
            db.execute("UPDATE changes SET status='planned', detail=? WHERE id=?",
                       (json.dumps({"faces": faces, **({"target_name": p.name} if p else {})}), m["id"]))
            db.execute("UPDATE changes SET status='planned' WHERE run=? AND group_id=? AND kind!='merge' AND status='capped'",
                       (run, m["group_id"]))
    db.commit()
    res = {"merges": len(chosen), "cleaning": reserved, "left": len(cands) - len(chosen)}
    log(f"requeue {run}: {res}")
    return res


def _choose_prev_first(cands: list[tuple[int, int, int]], prev: set[int], caps: dict) -> tuple[set[int], int]:
    first, rest = [c for c in cands if c[0] in prev], [c for c in cands if c[0] not in prev]
    chosen, reserved = _choose_merges(first, caps)
    more, extra = _choose_merges(rest, {"merges": caps["merges"] - len(chosen), "face_edits": caps["face_edits"] - reserved})
    return chosen | more, reserved + extra


def _target(r: sqlite3.Row, labels: dict[int, str]) -> str | None:
    lab = labels.get(r["face"])
    if lab is not None:
        return lab if not (lab.startswith("not:") or lab in ("unknown", "not_face") or lab.startswith("syno:")) else None
    return r["person"] if r["status"] == "accepted" else None


def _ch(kind: str, r: sqlite3.Row, frm: int | None, to: int | None, person: str | None, group: int | None = None) -> dict:
    if r["syno_face_id"] is not None and r["sx1"] is not None:
        box = box_dict(r["sx1"], r["sy1"], r["sx2"], r["sy2"])      # Synology's own box for its own face
    else:
        box = box_dict(r["x1"], r["y1"], r["x2"], r["y2"])          # our box, for a face Synology missed
    return {"kind": kind, "item": r["item"], "face": r["face"], "syno_face_id": r["syno_face_id"], "from_person": frm,
            "to_person": to, "box": json.dumps(box), "group_id": group, "person": person, "score": r["score"],
            "detail": None}


# --- apply -----------------------------------------------------------------------------------------------
def _write(db: sqlite3.Connection, nas: Nas, change: int, run: str, method: str, prior: Any, **params: Any) -> Any:
    api = nas.a["person"]
    wid = db.execute("INSERT INTO writes(change, run, at, api, method, params, prior, ok) VALUES(?,?,?,?,?,?,?,0)",
                     (change, run, D.now(), api, method, json.dumps(params), json.dumps(prior))).lastrowid
    db.commit()                                        # the intent is durable before the call is sent
    version = {"delete_face": 1, "add_face": 3, "merge": 2, "set": 1}[method]
    res = nas.c.ok(api, method, version, **params)
    db.execute("UPDATE writes SET ok=1, result=? WHERE id=?", (json.dumps(res)[:2000], wid))
    db.commit()
    return res


def _live_face(faces: list[dict], face_id: int) -> dict | None:
    return next((f for f in faces if f.get("face_id") == face_id), None)


def _box_tuple(box: dict) -> tuple[float, float, float, float]:
    return (box["top_left"]["x"], box["top_left"]["y"], box["bottom_right"]["x"], box["bottom_right"]["y"])


def _find_face(faces: list[dict], box: dict, face_ids: tuple | list = ()) -> dict | None:
    """The live face a change is about: by Synology face id while it exists, else the face that overlaps the box
    best (IoU >= 0.5): a face that was unlinked and linked again has a new id."""
    for fid in face_ids:
        f = _live_face(faces, fid) if fid is not None else None
        if f is not None:
            return f
    b = _box_tuple(box)
    best = max(faces, key=lambda f: iou(syno_box(f["face_bounding_box"]), b), default=None)
    return best if best is not None and iou(syno_box(best["face_bounding_box"]), b) >= 0.5 else None


def _detail(c: sqlite3.Row) -> dict:
    try:
        v = json.loads(c["detail"] or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def merge_rows(db: sqlite3.Connection, cfg: Config, run: str, status: str = "planned") -> list[dict]:
    """Every merge of a run with that status: group id, faces, target name, and whether it is a large one."""
    out = []
    for c in db.execute("SELECT * FROM changes WHERE run=? AND kind='merge' AND status=? ORDER BY seq", (run, status)):
        d = _detail(c)
        name = d.get("target_name") or (db.execute("SELECT name FROM syno_people WHERE id=?", (c["to_person"],)).fetchone()
                                        or [""])[0]
        faces = d.get("faces")
        out.append({"id": c["id"], "group": c["group_id"], "faces": faces, "target": name, "target_id": c["to_person"],
                    "large": bool(faces and faces > cfg.caps["merge_auto_max_faces"]),
                    "cleaning": db.execute("SELECT COUNT(*) FROM changes WHERE run=? AND group_id=? AND kind != 'merge' "
                                           "AND status != 'done'", (run, c["group_id"])).fetchone()[0]})
    return out


def auto_selection(db: sqlite3.Connection, cfg: Config, run: str, rows: list[sqlite3.Row]) -> tuple[set[int], dict]:
    """Which planned changes an automatic run may write. Hard limits, independent of the caps: at most
    `auto_max_face_edits` face edits and `auto_max_merges` merges per run (config.AUTO_CEILING), no change for an
    outer-circle person, no merge above `caps.merge_auto_max_faces` faces. A merge and its cleaning go together
    or not at all. Everything else stays planned (staged) for the owner. Returns (change ids, {reason: count})."""
    M, caps = cfg.mode, cfg.caps
    used = dict(db.execute("SELECT CASE WHEN c.kind='merge' THEN 'm' ELSE 'e' END, COUNT(DISTINCT w.change) FROM writes w "
                           "JOIN changes c ON c.id=w.change WHERE w.run=? GROUP BY 1", (run,)).fetchall())
    budget = M["auto_max_face_edits"] - used.get("e", 0)       # a resumed run does not get a second allowance
    merges_left = M["auto_max_merges"] - used.get("m", 0)

    def outer(person: str | None) -> bool:
        return person is not None and person in cfg.people and cfg.people[person].outer

    allowed: set[int] = set()
    why: Counter = Counter()
    groups: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for c in rows:
        if c["group_id"] is not None:
            groups[c["group_id"]].append(c)
    for g, cs in groups.items():
        merge = next((c for c in cs if c["kind"] == "merge"), None)
        cleaning = [c for c in cs if c["kind"] != "merge"]
        faces = _detail(merge).get("faces") if merge is not None else None
        if merge is None:
            why["cleaning without its merge"] += len(cs)
        elif outer(merge["person"]):
            why["outer circle"] += len(cs)
        elif not faces or faces > caps["merge_auto_max_faces"]:
            why["large merge"] += len(cs)
        elif merges_left <= 0:
            why["auto ceiling: merges"] += len(cs)
        elif len(cleaning) > budget:
            why["auto ceiling: face edits"] += len(cs)
        else:
            allowed |= {c["id"] for c in cs}
            budget -= len(cleaning)
            merges_left -= 1
    for c in rows:
        if c["group_id"] is not None:
            continue
        if outer(c["person"]):
            why["outer circle"] += 1
        elif budget <= 0:
            why["auto ceiling: face edits"] += 1
        else:
            allowed.add(c["id"])
            budget -= 1
    return allowed, dict(why)


def apply(db: sqlite3.Connection, cfg: Config, nas: Nas, run: str, kinds: set[str] | None = None,
          held_people: set[tuple[str, str]] | None = None, log=print, auto: bool = False) -> dict:
    """Apply planned changes of `run`. held_people: (person, bin) pairs kept staged after an audit error.
    auto: an unattended run; only `auto_selection` is written, the rest stays planned for the owner."""
    sp = nas.sp
    rows = db.execute("SELECT * FROM changes WHERE run=? AND status='planned' ORDER BY seq", (run,)).fetchall()
    bins = {r[0]: r[1] for r in db.execute("SELECT face, bin FROM decisions WHERE run=?", (run,))}
    done: Counter = Counter()
    allowed, why = auto_selection(db, cfg, run, rows) if auto else (None, {})
    for c in rows:
        if kinds and c["kind"] not in kinds:
            continue
        if allowed is not None and c["id"] not in allowed:
            done["left_staged"] += 1
            continue
        if held_people and (c["person"], bins.get(c["face"])) in held_people:
            _status(db, c["id"], "held", "person/bin staged after an audit error")
            done["held"] += 1
            continue
        try:
            st = _apply_one(db, nas, sp, run, c)
        except Exception as e:  # noqa: BLE001 - one failed change must not stop the batch silently
            st = "failed"
            _status(db, c["id"], st, sp.redact(str(e))[:500])
            log(f"  change {c['id']} {c['kind']} failed: {sp.redact(str(e))[:200]}")
        done[st] += 1
        if done["failed"] >= MAX_FAILURES:
            log(f"  {MAX_FAILURES} failures: stopping apply")
            break
    db.commit()
    res: dict = dict(done)
    if why:
        res["left_staged_why"] = why
    log(f"apply: {res}")
    return res


def _status(db: sqlite3.Connection, cid: int, st: str, detail: str | None = None) -> None:
    db.execute("UPDATE changes SET status=?, detail=COALESCE(?, detail) WHERE id=?", (st, detail, cid))
    db.commit()


def _apply_one(db: sqlite3.Connection, nas: Nas, sp: Any, run: str, c: sqlite3.Row) -> str:
    kind = c["kind"]
    if kind == "merge":
        return _apply_merge(db, nas, sp, run, c)
    if kind == "restore":
        return _apply_restore(db, nas, sp, run, c)
    item = c["item"]
    faces = nas.faces(item)
    box = json.loads(c["box"])
    if kind in ("reassign", "assign", "label-fix", "unlink", "group-unlink"):
        f = _live_face(faces, c["syno_face_id"])
        if f is None or (f.get("person_id") or None) != c["from_person"]:
            store_syno_faces(db, item, faces)
            _status(db, c["id"], "stale", f"live person {f.get('person_id') if f else 'face gone'}")
            return "stale"
        if c["from_person"]:
            sp.journal("unlink_face", space=nas.space, person_id=c["from_person"], item=item, face_id=f["face_id"],
                       box=f["face_bounding_box"], reason=f"face-curator {run} {kind}")
            _write(db, nas, c["id"], run, "delete_face", {"person_id": c["from_person"], "face": f},
                   person_id=c["from_person"], face_id=[f["face_id"]])
        if kind not in UNLINK_KINDS:
            _write(db, nas, c["id"], run, "add_face", {"person_id": c["from_person"], "face": f}, id_item=item,
                   face=[{"face_bounding_box": f["face_bounding_box"], "face_id_temp": f"{item}-0",
                          "person_id": c["to_person"]}])
    elif kind == "add":
        nb = _box_tuple(box)
        over = [f for f in faces if iou(nb, syno_box(f["face_bounding_box"])) >= 0.3]
        if any(f.get("person_id") for f in over):
            store_syno_faces(db, item, faces)
            _status(db, c["id"], "stale", "Synology now has a named face there")
            return "stale"
        if over:   # an unlinked face is already there (e.g. undo of an unlink): link that one, with its own box
            box = over[0]["face_bounding_box"]
        _write(db, nas, c["id"], run, "add_face", {"face": over[0] if over else None}, id_item=item,
               face=[{"face_bounding_box": box, "face_id_temp": f"{item}-0", "person_id": c["to_person"]}])
    after = nas.faces(item)
    store_syno_faces(db, item, after)
    if kind not in UNLINK_KINDS:
        new = [f for f in after if f.get("person_id") == c["to_person"]
               and iou(syno_box(f["face_bounding_box"]), _box_tuple(box)) >= 0.5]
        if not new:
            _status(db, c["id"], "unverified", "write accepted but the face is not visible on re-read")
            return "unverified"
        _status(db, c["id"], "done", json.dumps({"new_face_id": new[0]["face_id"]}))
    else:
        _status(db, c["id"], "done")
    return "done"


def _apply_restore(db: sqlite3.Connection, nas: Nas, sp: Any, run: str, c: sqlite3.Row) -> str:
    """Undo of one change: put the face back to the person it had before (`to_person`, None = nobody). Written
    only if the face is still in a state the undone run's own writes can have left (`ours`: person ids, 0 =
    unlinked); already back = done without a write; anything else was changed by someone since = stale."""
    d, item, box, want = _detail(c), c["item"], json.loads(c["box"]), c["to_person"]
    ours = {x or None for x in d.get("ours") or []}
    faces = nas.faces(item)
    f = _find_face(faces, box, d.get("face_ids") or ())
    cur = (f.get("person_id") or None) if f is not None else None
    if cur == want:
        store_syno_faces(db, item, faces)
        _status(db, c["id"], "done", json.dumps({**d, "note": "already in the prior state: nothing written"}))
        return "done"
    if cur not in ours:
        store_syno_faces(db, item, faces)
        _status(db, c["id"], "stale", json.dumps({**d, "note": f"live person {cur}: changed since the run"}))
        return "stale"
    if cur:
        sp.journal("unlink_face", space=nas.space, person_id=cur, item=item, face_id=f["face_id"],
                   box=f["face_bounding_box"], reason=f"face-curator {run} restore")
        _write(db, nas, c["id"], run, "delete_face", {"person_id": cur, "face": f}, person_id=cur, face_id=[f["face_id"]])
    if want:
        fb = f["face_bounding_box"] if f is not None else box
        _write(db, nas, c["id"], run, "add_face", {"person_id": cur, "face": f}, id_item=item,
               face=[{"face_bounding_box": fb, "face_id_temp": f"{item}-0", "person_id": want}])
    after = nas.faces(item)
    store_syno_faces(db, item, after)
    g = _find_face(after, box)
    now = (g.get("person_id") or None) if g is not None else None
    if now != want:
        _status(db, c["id"], "unverified", json.dumps({**d, "note": "write accepted but not visible on re-read"}))
        return "unverified"
    _status(db, c["id"], "done", json.dumps({**d, **({"new_face_id": g["face_id"]} if g is not None and want else {})}))
    return "done"


def _apply_merge(db: sqlite3.Connection, nas: Nas, sp: Any, run: str, c: sqlite3.Row) -> str:
    """Merge only if, read from the NAS right now: the group is still unnamed, the target still exists under the
    planned name, and the group holds exactly the faces the plan expected to ride along (it was cleaned)."""
    g, target, d = c["group_id"], c["to_person"], _detail(c)
    expected = {r[0] for r in db.execute(
        "SELECT f.syno_face_id FROM decisions d JOIN faces f ON f.id=d.face JOIN syno_faces s ON s.face_id=f.syno_face_id "
        "WHERE d.run=? AND s.person_id=? ", (run, g))}
    cleaned = {r[0] for r in db.execute("SELECT syno_face_id FROM changes WHERE run=? AND group_id=? AND kind != 'merge' "
                                        "AND status='done'", (run, g))}
    pending = db.execute("SELECT COUNT(*) FROM changes WHERE run=? AND group_id=? AND kind != 'merge' AND status != 'done'",
                         (run, g)).fetchone()[0]
    if pending:
        _status(db, c["id"], "stale", f"{pending} cleaning change(s) not done")
        return "stale"
    planned_name = d.get("target_name")
    if planned_name is None:                       # a plan made before names were recorded: the name synced then
        planned_name = (db.execute("SELECT name FROM syno_people WHERE id=?", (target,)).fetchone() or [None])[0]
    people = {p["id"]: (p.get("name") or "") for p in nas.people()}        # live, right before the merge
    if g not in people:
        return _merge_stale(db, c, d, "the group no longer exists in Synology Photos")
    if people[g] != "":
        return _merge_stale(db, c, d, f"the group was named {people[g]!r} after the plan")
    if target not in people:
        return _merge_stale(db, c, d, "the target person no longer exists in Synology Photos")
    if not planned_name or people[target] != planned_name:
        return _merge_stale(db, c, d, f"the target is now named {people[target]!r}, the plan said {planned_name!r}")
    live = sp.group_faces(nas.c, nas.space, g)
    live_ids = {r["face_id"] for r in live}
    if live_ids != (expected - cleaned):
        _status(db, c["id"], "stale", f"group changed: {len(live_ids)} live faces, {len(expected - cleaned)} expected")
        return "stale"
    faces = [{k: x[k] for k in ("item", "face_id", "box")} for x in live]
    sp.journal("merge", space=nas.space, merged_id=g, merged_name=people[g], target_id=target,
               target_name=people[target], faces=faces, reason=f"face-curator {run} merge")
    _write(db, nas, c["id"], run, "merge", {"group": g, "group_name": people[g], "target_name": people[target],
                                           "faces": faces},
           target_id=target, merged_id=[g], name=sp.js(people[target]))
    for it in {x["item"] for x in live}:
        store_syno_faces(db, it, nas.faces(it))
    _status(db, c["id"], "done")
    return "done"


def _merge_stale(db: sqlite3.Connection, c: sqlite3.Row, d: dict, why: str) -> str:
    _status(db, c["id"], "stale", json.dumps({**d, "stale": why}))
    return "stale"


# --- undo ------------------------------------------------------------------------------------------------
def plan_undo(db: sqlite3.Connection, run: str, new_run: str) -> dict:
    """A new run whose changes put back every face `run` wrote to. Built from the write log (`writes`), not from
    the status of the changes, so it also covers a change that failed between its unlink and its link, one that
    was written but not verified, and a write whose answer never arrived (the process was stopped): each
    `restore` change is checked against the live state when it is applied. Merges are listed for manual
    handling (their faces are in the journal); they cannot be split automatically. Planning again replaces the
    restores that are still waiting and keeps those already applied."""
    D.set_run(db, new_run, mode="undo", stage="planned")
    already = {}
    for u in db.execute("SELECT id, status, detail FROM changes WHERE run=?", (new_run,)).fetchall():
        d = _detail(u)
        of = d.get("undo_of")
        if of is None and str(u["detail"] or "").startswith("undo of change "):
            of = int(str(u["detail"]).rsplit(" ", 1)[1])
        if u["status"] == "planned":
            db.execute("DELETE FROM changes WHERE id=?", (u["id"],))
        elif of is not None:
            already[of] = u["status"]
    writes: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for w in db.execute("SELECT * FROM writes WHERE run=? ORDER BY id", (run,)):
        writes[w["change"]].append(w)
    seq = db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM changes WHERE run=?", (new_run,)).fetchone()[0]
    n, manual, uncertain = 0, [], 0
    for c in db.execute("SELECT * FROM changes WHERE run=? ORDER BY seq DESC", (run,)).fetchall():
        ws = writes.get(c["id"])
        if not ws:
            continue                                   # nothing was sent for this change
        methods = [w["method"] for w in ws]
        if c["kind"] == "merge" or "merge" in methods:
            manual.append(c["group_id"])
            continue
        if already.get(c["id"]) == "done":
            continue
        try:
            prior = json.loads(ws[0]["prior"] or "null") or {}
        except ValueError:
            prior = {}
        prior = prior if isinstance(prior, dict) else {}
        before = (c["from_person"] or None) if c["kind"] != "restore" else (prior.get("person_id") or None)
        ours = sorted({0 for m in methods if m == "delete_face"} | {c["to_person"] or 0 for m in methods if m == "add_face"})
        face = prior.get("face") if isinstance(prior.get("face"), dict) else None
        box = json.dumps(face["face_bounding_box"]) if face and face.get("face_bounding_box") else c["box"]
        ids = [x for x in (_detail(c).get("new_face_id"), c["syno_face_id"], face.get("face_id") if face else None)
               if x is not None]
        unsure = any(not w["ok"] for w in ws)
        uncertain += unsure
        db.execute("INSERT INTO changes(run, seq, kind, item, face, syno_face_id, from_person, to_person, box, person, "
                   "status, detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (new_run, seq, "restore", c["item"], c["face"], None, None, before, box, None, "planned",
                    json.dumps({"undo_of": c["id"], "ours": ours, "face_ids": ids, "was": c["status"],
                                **({"uncertain": True} if unsure else {})})))
        seq += 1
        n += 1
    db.commit()
    return {"changes": n, "merges_manual": manual, "uncertain": uncertain}


# --- repair ----------------------------------------------------------------------------------------------
def repair(db: sqlite3.Connection, cfg: Config, nas: Nas, run: str, write: bool = False, log=print) -> dict:
    """Re-read from the NAS the live state of every change of `run` that is not done but may have been written
    (failed, unverified, or with a row in the write log), and reconcile the ledger with it:

    - the live state is what the change wanted: the change becomes `done`;
    - nothing was written (the face is as before): the change is `planned` again;
    - a reassign stopped after its unlink (the face has nobody): the link is finished (the only NAS write here);
    - anything else: `stale` (someone changed it since).

    write=False reads only and reports what it would do."""
    sp = nas.sp
    rows = db.execute("SELECT * FROM changes WHERE run=? AND status != 'done' AND (status IN ('failed', 'unverified') "
                      "OR id IN (SELECT change FROM writes WHERE run=?)) ORDER BY seq", (run, run)).fetchall()
    people = None
    res: Counter = Counter()
    for c in rows:
        note, new, finish = None, None, None
        if c["kind"] == "merge":
            people = people if people is not None else {p["id"]: (p.get("name") or "") for p in nas.people()}
            if c["group_id"] not in people:
                new, note = "done", "the group is gone: the merge happened"
            else:
                new, note = "planned", "the group still exists: the merge did not happen"
        else:
            d, box = _detail(c), json.loads(c["box"])
            faces = nas.faces(c["item"])
            f = _find_face(faces, box, [x for x in (d.get("new_face_id"), c["syno_face_id"], *(d.get("face_ids") or []))
                                        if x is not None])
            cur = (f.get("person_id") or None) if f is not None else None
            want = None if c["kind"] in UNLINK_KINDS else (c["to_person"] or None)
            before = {x or None for x in d.get("ours") or []} if c["kind"] == "restore" else {c["from_person"] or None}
            wrote = db.execute("SELECT COUNT(*) FROM writes WHERE change=?", (c["id"],)).fetchone()[0]
            if cur == want:
                new, note = "done", "the live state is what the change wanted"
            elif cur in before:
                new, note = "planned", "the face is as it was before the change: nothing to undo, it can be applied again"
            elif cur is None and want is not None and wrote:
                finish = f if f is not None else {"face_bounding_box": box}
                note = f"unlinked but not linked: finish the link to person {want}"
            else:
                new, note = "stale", f"live person {cur}: changed by someone else"
            if write:
                store_syno_faces(db, c["item"], faces)
        log(f"  change {c['id']} {c['kind']} ({c['status']}): {note}")
        if not write:
            res[f"would: {new or 'finish the link'}"] += 1
            continue
        if finish is not None:
            try:
                _write(db, nas, c["id"], run, "add_face", {"person_id": None, "face": finish, "repair": True},
                       id_item=c["item"], face=[{"face_bounding_box": finish["face_bounding_box"],
                                                 "face_id_temp": f"{c['item']}-0", "person_id": c["to_person"]}])
                after = nas.faces(c["item"])
                store_syno_faces(db, c["item"], after)
                g = _find_face(after, json.loads(c["box"]))
                new = "done" if g is not None and (g.get("person_id") or None) == c["to_person"] else "unverified"
            except Exception as e:  # noqa: BLE001
                new, note = "failed", sp.redact(str(e))[:300]
                log(f"  change {c['id']}: finishing failed: {note}")
        db.execute("UPDATE writes SET result=COALESCE(result, ?) WHERE change=? AND ok=0",
                   (json.dumps({"repair": D.now(), "found": note}), c["id"]))
        db.execute("UPDATE changes SET status=?, detail=? WHERE id=?",
                   (new, json.dumps({**_detail(c), "repair": note, "was": c["status"]}), c["id"]))
        db.commit()
        res[new] += 1
    out = dict(res)
    log(f"repair {run}: {out or 'nothing to reconcile'}")
    return out
