"""Review material: gallery bootstrap batches, the one-off audit, and the monthly cards.

Monthly cards (default 20): random audit of accepted decisions (stratified by person, one per day), the
smallest sibling margins among acceptances, challenger disagreements, and "bridge" faces (accepted for a gallery
person while sitting in a large Synology group dominated by someone else). Unanswered cards carry over; at most
`max_waiting` are open at once. Every answer becomes an owner label; a wrong accepted decision also bumps that
person-and-bin threshold and keeps it staged.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from fc import ages
from fc import db as D
from fc.config import Config

ANSWERS = {"a", "b", "neither", "unknown", "not_face", "skip"}
JUDGED = {"audit", "random", "sibling_margin", "disagreement", "bridge"}   # cards that test an accepted decision


def _one_per_day(rows: list[sqlite3.Row], n: int, rng: np.random.Generator, key=None) -> list[sqlite3.Row]:
    rows = list(rows)
    if key is None:
        rng.shuffle(rows)
    else:
        rows.sort(key=key)
    seen, out = set(), []
    for r in rows:
        k = (r["person"], (r["taken"] or "")[:10])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
        if len(out) >= n:
            break
    return out


def _decided(db: sqlite3.Connection, run: str, where: str, params=()) -> list[sqlite3.Row]:
    return db.execute(
        "SELECT d.*, i.taken, f.item, f.syno_face_id FROM decisions d JOIN faces f ON f.id=d.face "
        "JOIN items i ON i.id=f.item WHERE d.run=? AND " + where, (run, *params)).fetchall()


def _open_faces(db: sqlite3.Connection) -> set[int]:
    return {r[0] for r in db.execute("SELECT face FROM cards WHERE answer IS NULL")}


def _insert(db: sqlite3.Connection, run: str, kind: str, r: sqlite3.Row, alt: str | None, payload: dict | None = None) -> None:
    db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
               (run, kind, r["face"], r["person"], alt, r["bin"], json.dumps(payload or {}), D.now()))


def monthly(db: sqlite3.Connection, cfg: Config, run: str, seed: int | None = None, log=print) -> dict:
    C = cfg["cards"]
    waiting = db.execute("SELECT COUNT(*) FROM cards WHERE answer IS NULL AND kind NOT IN ('h2h','study','edge')").fetchone()[0]
    room = max(0, C["max_waiting"] - waiting)
    want = {"random": C["random"], "sibling_margin": C["sibling_margin"], "disagreement": C["disagreement"],
            "bridge": C["bridge"]}
    total = min(room, sum(want.values()))
    rng = np.random.default_rng(seed if seed is not None else abs(hash(run)) % 2**32)
    taken = _open_faces(db)
    made: Counter = Counter()

    def add(kind: str, rows: list[sqlite3.Row], alt_fn) -> None:
        for r in rows:
            if sum(made.values()) >= total or r["face"] in taken:
                continue
            _insert(db, run, kind, r, alt_fn(r))
            taken.add(r["face"])
            made[kind] += 1

    close = {p.key for p in cfg.close_people}
    acc = [r for r in _decided(db, run, "d.status='accepted'") if r["person"] in close]   # owner cards: close circle
    # stratified random: round-robin over people
    by_p: dict[str, list] = defaultdict(list)
    for r in _one_per_day(acc, len(acc), rng):
        by_p[r["person"]].append(r)
    rr, i = [], 0
    while len(rr) < want["random"] and any(by_p.values()):
        for p in sorted(by_p):
            if by_p[p] and len(rr) < want["random"]:
                rr.append(by_p[p].pop(0))
        i += 1
    add("random", rr, lambda r: r["runner"])
    sib = [r for r in acc if r["runner"] in cfg.siblings.get(r["person"], set())]
    add("sibling_margin", _one_per_day(sib, want["sibling_margin"] * 3, rng,
                                       key=lambda r: r["score"] - r["runner_score"])[:want["sibling_margin"]],
        lambda r: r["runner"])
    dis = [r for r in _decided(db, run, "d.status='vetoed'") if r["person"] in close]
    add("disagreement", _one_per_day(dis, want["disagreement"], rng),
        lambda r: (r["challenger"] or "").lstrip("?") or r["runner"])
    add("bridge", bridges(db, cfg, run, acc)[:want["bridge"]], lambda r: f"syno:{r['_group']}")
    # Fill any shortfall with more sibling-margin, then random cards.
    if sum(made.values()) < total:
        add("sibling_margin", _one_per_day(sib, total, rng, key=lambda r: r["score"] - r["runner_score"]),
            lambda r: r["runner"])
    if sum(made.values()) < total:
        add("random", _one_per_day(acc, total, rng), lambda r: r["runner"])
    db.commit()
    log(f"cards: {dict(made)} (waiting before: {waiting})")
    return dict(made)


def bridges(db: sqlite3.Connection, cfg: Config, run: str, acc: list[sqlite3.Row], min_group: int = 10) -> list:
    """Accepted faces whose Synology group (>= min_group faces) is mostly decided as someone else."""
    sp_of = {r[0]: r[1] for r in db.execute("SELECT face_id, person_id FROM syno_faces WHERE person_id IS NOT NULL")}
    dom: dict[int, Counter] = defaultdict(Counter)
    for r in _decided(db, run, "d.status IN ('accepted','labelled')"):
        g = sp_of.get(r["syno_face_id"])
        if g is not None:
            dom[g][r["person"]] += 1
    out = []
    for r in acc:
        g = sp_of.get(r["syno_face_id"])
        if g is None or sum(dom[g].values()) < min_group:
            continue
        top, n = dom[g].most_common(1)[0]
        if top != r["person"] and n / sum(dom[g].values()) >= 0.8:
            d = dict(r)
            d["_group"] = g
            out.append(d)
    out.sort(key=lambda d: d["score"])
    return out


def audit(db: sqlite3.Connection, cfg: Config, run: str, n: int = 600, seed: int = 11) -> int:
    """One-off audit: n random accepted decisions, one per person per day (bursts collapsed). The cards of one
    call form a batch; the auto-mode gate looks at the latest batch only (`audit_state`)."""
    rng = np.random.default_rng(seed)
    taken = _open_faces(db)
    rows = [r for r in _one_per_day(_decided(db, run, "d.status='accepted'"), n * 2, rng) if r["face"] not in taken][:n]
    batch = D.now()
    for r in rows:
        _insert(db, run, "audit", r, r["runner"], {"batch": batch})
    db.commit()
    return len(rows)


# --- gallery bootstrap -----------------------------------------------------------------------------------
def gallery_candidates(db: sqlite3.Connection, cfg: Config, person: str, bin_i: int | None, n: int,
                       model: str, has_image) -> list[dict]:
    """Diverse unlabelled candidates for one person (and age bin): faces Synology attaches to that person, or
    faces the latest run scored highest for them; farthest-point sampling from the existing exemplars, at most
    one face per day per batch, sizeable faces only."""
    p = cfg.people[person]
    edges = cfg["age_bins_months"]
    labels = D.latest_labels(db)
    sid = db.execute("SELECT id FROM syno_people WHERE name=?", (p.name,)).fetchone()
    last = db.execute("SELECT id FROM runs ORDER BY started DESC LIMIT 1").fetchone()
    q = ("SELECT f.id, f.item, f.y1, f.y2, f.det, i.taken FROM faces f JOIN items i ON i.id=f.item "
         "LEFT JOIN syno_faces s ON s.face_id=f.syno_face_id WHERE (s.person_id=? ")
    params: list = [sid[0] if sid else -1]
    if last:
        q += "OR f.id IN (SELECT face FROM decisions WHERE run=? AND person=? AND score >= 0.5)"
        params += [last[0], person]
    q += ") AND (f.y2 - f.y1) >= 0.06 AND COALESCE(f.det, 1) >= 0.7"
    rows = [r for r in db.execute(q, params) if r["id"] not in labels]
    if p.child and bin_i is not None:
        rows = [r for r in rows if ages.bin_index(ages.months(p.born, (r["taken"] or "")[:10] or None), edges) == bin_i]
    rows = [r for r in rows if p.allowed_on((r["taken"] or "")[:10] or None) and has_image(r["item"])]
    if not rows:
        return []
    ids, vecs = D.vectors(db, model, [r["id"] for r in rows])
    if len(ids) == 0:
        return []
    meta = {r["id"]: r for r in rows}
    ex_ids = [f for f, v in labels.items() if v == person]
    _, ex = D.vectors(db, model, ex_ids)
    if len(ex):
        dmin = 1 - (vecs @ ex.T).max(1)
    else:
        dmin = np.full(len(ids), 2.0)
        dmin[0] = 3.0
    day_of = np.array([(meta[int(x)]["taken"] or "")[:10] for x in ids], dtype=object)
    n_days = len(set(day_of))
    alive = np.ones(len(ids), dtype=bool)
    chosen: list[int] = []
    days: set[str] = set()
    while len(chosen) < n and alive.any():
        i = int(np.argmax(np.where(alive, dmin, -np.inf)))
        alive[i] = False
        if day_of[i] in days and len(days) < n_days:
            continue                       # prefer a new capture day while one remains
        chosen.append(i)
        days.add(day_of[i])
        dmin = np.minimum(dmin, 1 - vecs @ vecs[i])
    out = []
    for i in chosen:
        r = meta[int(ids[i])]
        m = ages.months(p.born, (r["taken"] or "")[:10] or None) if p.child else None
        out.append({"face": int(ids[i]), "item": r["item"], "day": (r["taken"] or "")[:10],
                    "age": None if m is None else round(m / 12, 1)})
    return out


# --- answers ---------------------------------------------------------------------------------------------
class AlreadyAnswered(ValueError):
    """The card has an answer; it is changed only when the page asks for that on purpose (`replace`)."""


def _payload(c: sqlite3.Row) -> dict:
    try:
        v = json.loads(c["payload"] or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def keep_previous(db: sqlite3.Connection, c: sqlite3.Row) -> None:
    """Record the answer that is about to be replaced in the card's payload (`previous`: oldest first)."""
    pl = _payload(c)
    pl.setdefault("previous", []).append({"answer": c["answer"], "answered": c["answered"], "replaced": D.now()})
    db.execute("UPDATE cards SET payload=? WHERE id=?", (json.dumps(pl), c["id"]))


def retract(db: sqlite3.Connection, cfg: Config, c: sqlite3.Row) -> bool:
    """Take back the answer of card `c`: its label is marked retracted, the threshold bump and the auto-mode
    demotion it caused are lifted, and the card is open again. Does not commit."""
    if c["answer"] is None:
        return False
    last = db.execute("SELECT id FROM labels WHERE face=? AND via=? AND source='owner' ORDER BY id DESC LIMIT 1",
                      (c["face"], f"card:{c['kind']}")).fetchone()
    if last and c["answer"] != "skip":
        db.execute("UPDATE labels SET source='retracted' WHERE id=?", (last[0],))
    b = db.execute("SELECT amount, reason FROM bumps WHERE person=? AND bin=?", (c["person"], c["bin"])).fetchone()
    if b and (b["reason"] or "").startswith(f"card {c['id']} "):
        amt = b["amount"] - cfg["assign"]["error_bump"]
        if amt <= 1e-9:
            db.execute("DELETE FROM bumps WHERE person=? AND bin=?", (c["person"], c["bin"]))
        else:
            db.execute("UPDATE bumps SET amount=? WHERE person=? AND bin=?", (round(amt, 3), c["person"], c["bin"]))
    d = demotion(db)
    if d and d.get("card") == c["id"]:
        D.set_meta(db, DEMOTED, json.dumps(d["before"]) if d.get("before") else None)
    db.execute("UPDATE cards SET answer=NULL, answered=NULL WHERE id=?", (c["id"],))
    return True


def answer(db: sqlite3.Connection, cfg: Config, card_id: int, ans: str, replace: bool = False) -> dict:
    if ans not in ANSWERS:
        raise ValueError(f"answer must be one of {sorted(ANSWERS)}")
    if isinstance(card_id, bool) or not isinstance(card_id, int):
        raise ValueError("card id must be an integer")
    c = db.execute("SELECT * FROM cards WHERE id=? AND kind NOT IN ('h2h','study','edge')", (card_id,)).fetchone()
    if c is None:
        raise ValueError("no such card")
    previous = None
    if c["answer"] is not None:
        if replace is not True:
            raise AlreadyAnswered(f"card {card_id} already has the answer {c['answer']!r}; it is changed only on purpose")
        previous = c["answer"]
        keep_previous(db, c)
        retract(db, cfg, c)
    db.execute("UPDATE cards SET answer=?, answered=? WHERE id=?", (ans, D.now(), card_id))
    if ans == "skip":
        db.commit()
        return {"label": None, "error": False, "previous": previous}
    a, b = c["person"], c["alt"]
    value = {"a": a, "b": b, "unknown": "unknown", "not_face": "not_face",
             "neither": "not:" + ",".join(x for x in (a, b) if x and not x.startswith("syno:"))}[ans]
    D.add_label(db, c["face"], value, via=f"card:{c['kind']}", run=c["run"])
    error = c["kind"] in JUDGED and ans != "a" and ans != "unknown"
    if error and a:
        A = cfg["assign"]
        old = db.execute("SELECT amount FROM bumps WHERE person=? AND bin=?", (a, c["bin"])).fetchone()
        db.execute("INSERT OR REPLACE INTO bumps VALUES(?,?,?,?,?,?)",
                   (a, c["bin"], round((old[0] if old else 0.0) + A["error_bump"], 3), 1, D.now(),
                    f"card {card_id} ({c['kind']}) answered {ans}"))
    if error:
        demote(db, f"card {card_id} ({c['kind']}) found a wrong accepted decision", card=card_id, run=c["run"])
    db.commit()
    return {"label": value, "error": error, "previous": previous}


def audit_result(db: sqlite3.Connection, run: str) -> dict:
    rows = db.execute("SELECT kind, answer FROM cards WHERE run=? AND answer IS NOT NULL AND answer NOT IN ('skip','unknown')",
                      (run,)).fetchall()
    judged = [r for r in rows if r["kind"] in JUDGED]
    errors = sum(1 for r in judged if r["answer"] != "a")
    return {"judged": len(judged), "errors": errors}


# --- auto mode: demotion and the one-off audit --------------------------------------------------------------
DEMOTED = "auto_demoted"      # meta key: {"at", "reason", "card", "run", "before"}; absent = not demoted


def demotion(db: sqlite3.Connection) -> dict | None:
    v = D.get_meta(db, DEMOTED)
    try:
        return json.loads(v) if v else None
    except ValueError:
        return {"at": "", "reason": "unreadable demotion record"}


def demote(db: sqlite3.Connection, reason: str, card: int | None = None, run: str | None = None) -> None:
    """Auto mode is off from now on, whatever the config says, until a one-off audit created after this moment
    is completed without an error (`audit_state`). Kept in the ledger (`meta`). Does not commit."""
    D.set_meta(db, DEMOTED, json.dumps({"at": D.now(), "reason": reason, "card": card, "run": run,
                                        "before": demotion(db)}))


def audit_state(db: sqlite3.Connection) -> dict:
    """The latest one-off audit (cards of kind `audit` from one `audit` call): {batch, cards, open, judged,
    errors}. `batch` is the time it was created; {} when there is none."""
    groups: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for c in db.execute("SELECT * FROM cards WHERE kind='audit'"):
        groups[_payload(c).get("batch") or f"run:{c['run']}"].append(c)
    if not groups:
        return {}
    stamp = {k: (k if not k.startswith("run:") else min((c["created"] or "") for c in v)) for k, v in groups.items()}
    key = max(groups, key=lambda k: stamp[k])
    rows = groups[key]
    judged = [c for c in rows if c["answer"] is not None and c["answer"] not in ("skip", "unknown")]
    return {"batch": stamp[key], "cards": len(rows), "open": sum(1 for c in rows if c["answer"] is None),
            "judged": len(judged), "errors": sum(1 for c in judged if c["answer"] != "a")}
