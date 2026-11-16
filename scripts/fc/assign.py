"""Supervised, age-aware open-set assignment with abstention.

Two tiers: close-circle galleries are built from owner labels only; outer-circle galleries from owner labels
plus filtered Synology tags (outer_faces) and need a higher floor. A face is scored against each gallery person allowed on its capture
date (birth / death exclusion): score = mean of the top-5 cosine similarities to that person's exemplars, where a
child's exemplars come from the face's age bin widened by `bin_overlap_months`. A face is accepted for the best
person when the score clears the floor, the gap to the runner-up clears the margin (a larger margin when the
runner-up is a sibling), every allowed sibling has a usable gallery at that age, and the challenger model (if
any) does not pick someone else. Everything else abstains.

Calibration is event-separated: each labelled face is scored with exemplars from its own capture day removed.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from fc import ages
from fc import db as D
from fc.config import Config, Person

TOPK = 5
MIN_VALID = 3          # exemplars needed in the window for a score at all


@dataclass
class Gallery:
    person: Person
    faces: np.ndarray            # face ids
    vecs: np.ndarray             # (k, d)
    months: np.ndarray           # exemplar age in months (nan for adults / undated)
    days: np.ndarray             # capture day strings (object)


@dataclass
class FaceSet:
    ids: np.ndarray
    vecs: np.ndarray
    days: np.ndarray             # 'YYYY-MM-DD' or ''


@dataclass
class Thresholds:
    floor: float
    margin: float
    sibling_margin: float
    status: str = "default"      # default | calibrated
    stats: dict = field(default_factory=dict)


def face_days(db: sqlite3.Connection, ids: np.ndarray) -> np.ndarray:
    m = {r[0]: (r[1] or "")[:10] for r in db.execute("SELECT f.id, i.taken FROM faces f JOIN items i ON i.id=f.item")}
    return np.array([m.get(int(i), "") for i in ids], dtype=object)


def load_faceset(db: sqlite3.Connection, model: str, faces=None) -> FaceSet:
    ids, vecs = D.vectors(db, model, faces)
    return FaceSet(ids, vecs, face_days(db, ids))


def build_galleries(db: sqlite3.Connection, cfg: Config, model: str, labels: dict[int, str] | None = None
                    ) -> dict[str, Gallery]:
    labels = D.latest_labels(db) if labels is None else labels
    out: dict[str, Gallery] = {}
    for p in cfg.gallery_people:
        fids = [f for f, v in labels.items() if v == p.key]
        ids, vecs = D.vectors(db, model, fids)
        out[p.key] = _gallery(db, p, ids, vecs)
    outer = cfg.outer_people
    if outer:
        close = [g.vecs for g in out.values() if not g.person.outer and len(g.vecs)]
        for p in outer:
            fids = [int(f) for f in out[p.key].faces] + outer_faces(db, cfg, model, p, labels, close)
            ids, vecs = D.vectors(db, model, fids)
            out[p.key] = _gallery(db, p, ids, vecs)
    return out


def _gallery(db: sqlite3.Connection, p: Person, ids: np.ndarray, vecs: np.ndarray) -> Gallery:
    days = face_days(db, ids)
    mon = np.array([ages.months(p.born, d) if (p.born and d) else np.nan for d in days], dtype=np.float64)
    return Gallery(p, ids, vecs, mon, days)


def outer_faces(db: sqlite3.Connection, cfg: Config, model: str, p: Person, labels: dict[int, str],
                close_vecs: list[np.ndarray]) -> list[int]:
    """Outer circle: Synology-tagged faces of p that agree with each other and are not closer to a close-circle gallery.
    Faces the owner labelled (anything) never come from Synology. Spread across capture years."""
    A = cfg["assign"]
    cand = [r[0] for r in db.execute(
        "SELECT f.id FROM faces f JOIN syno_faces s ON s.face_id=f.syno_face_id JOIN syno_people sp ON sp.id=s.person_id "
        "WHERE sp.name=?", (p.name,)) if r[0] not in labels]
    ids, X = D.vectors(db, model, cand)
    if len(ids) < A["outer_min"]:
        return []
    s = X @ X.T
    np.fill_diagonal(s, -1.0)
    k = min(TOPK, len(ids) - 1)
    cons = -np.sort(-s, axis=1)[:, :k].mean(1)
    near = np.full(len(ids), -1.0)
    for g in close_vecs:
        near = np.maximum(near, -np.sort(-(X @ g.T), axis=1)[:, :min(TOPK, len(g))].mean(1))
    ok = (cons >= A["outer_consistency"]) & (near < cons - A["outer_close_gap"])
    days = face_days(db, ids)
    years: dict[str, list[int]] = {}
    for i in np.argsort(-cons):
        if ok[i]:
            years.setdefault((days[i] or "nodate")[:4], []).append(int(ids[i]))
    out: list[int] = []
    while len(out) < A["outer_max"] and any(years.values()):
        for y in sorted(years):
            if years[y] and len(out) < A["outer_max"]:
                out.append(years[y].pop(0))
    return out if len(out) >= A["outer_min"] else []


def gallery_coverage(cfg: Config, gals: dict[str, Gallery]) -> dict[str, dict[str, int]]:
    """person -> bin label -> exemplars inside the bin proper (adults: 'all')."""
    edges = cfg["age_bins_months"]
    out: dict[str, dict[str, int]] = {}
    for k, g in gals.items():
        if not g.person.child:
            out[k] = {"all": int(len(g.faces))}
            continue
        row = {}
        for i in range(len(edges)):
            b = np.array([ages.bin_index(m, edges) == i for m in g.months]) if len(g.months) else np.zeros(0, bool)
            row[ages.bin_label(i, edges)] = int(b.sum())
        out[k] = row
    return out


def _bins_for(person: Person, days: np.ndarray, edges: list[int]) -> np.ndarray:
    return np.array([-1 if (b := ages.bin_index(ages.months(person.born, d) if d else None, edges)) is None else b
                     for d in days], dtype=np.int64)


def score(fs: FaceSet, gals: dict[str, Gallery], cfg: Config, exclude_same_day: bool = False
          ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (person keys, scores (n, P) with -inf = not scorable, valid exemplar counts (n, P))."""
    edges, ov = cfg["age_bins_months"], cfg["bin_overlap_months"]
    keys = list(gals)
    n = len(fs.ids)
    S = np.full((n, len(keys)), -np.inf, dtype=np.float32)
    C = np.zeros((n, len(keys)), dtype=np.int32)
    for j, k in enumerate(keys):
        g = gals[k]
        allowed = np.array([g.person.allowed_on(d or None) for d in fs.days], dtype=bool)
        if len(g.faces) == 0 or not allowed.any():
            continue
        sims = fs.vecs @ g.vecs.T                                   # (n, k)
        mask = np.ones_like(sims, dtype=bool)
        if g.person.child:
            fb = _bins_for(g.person, fs.days, edges)
            lo = np.full(n, -np.inf)
            hi = np.full(n, np.inf)
            for b in range(len(edges)):
                w = ages.window(b, edges, ov)
                lo[fb == b], hi[fb == b] = w
            gm = np.nan_to_num(g.months, nan=-1e9)
            mask &= (gm[None, :] >= lo[:, None]) & (gm[None, :] < hi[:, None])
        if exclude_same_day:
            mask &= g.days[None, :] != fs.days[:, None]
        mask &= fs.ids[:, None] != g.faces[None, :]
        sims = np.where(mask, sims, -np.inf)
        cnt = mask.sum(1)
        kk = min(TOPK, sims.shape[1])
        top = -np.sort(-sims, axis=1)[:, :kk]
        valid = np.isfinite(top)
        mean = np.where(valid.sum(1) > 0, np.where(valid, top, 0).sum(1) / np.maximum(valid.sum(1), 1), -np.inf)
        ok = allowed & (cnt >= MIN_VALID)
        S[:, j] = np.where(ok, mean, -np.inf)
        C[:, j] = np.where(allowed, cnt, -1)   # -1: not allowed on that date
    return keys, S, C


@dataclass
class Decision:
    face: int
    person: str | None
    score: float
    runner: str | None
    runner_score: float
    bin: str
    status: str            # accepted | abstained
    reason: str


def decide(fs: FaceSet, keys: list[str], S: np.ndarray, C: np.ndarray, cfg: Config, th: Thresholds,
           bumps: dict[tuple[str, str], float] | None = None) -> list[Decision]:
    bumps = bumps or {}
    A = cfg["assign"]
    edges = cfg["age_bins_months"]
    min_active = cfg["gallery"]["min_active"]
    out = []
    for i in range(len(fs.ids)):
        row = S[i]
        order = np.argsort(-row)
        b1 = order[0] if len(order) else None
        s1 = float(row[b1]) if b1 is not None else -math.inf
        b2 = order[1] if len(order) > 1 else None
        s2 = float(row[b2]) if b2 is not None and np.isfinite(row[b2]) else -1.0
        day = fs.days[i] or None
        if b1 is None or not np.isfinite(s1):
            out.append(Decision(int(fs.ids[i]), None, -1.0, None, s2, "-", "abstained", "no-gallery"))
            continue
        k1 = keys[b1]
        p1 = cfg.people[k1]
        binl = ages.bin_label(ages.bin_index(ages.months(p1.born, day), edges), edges) if p1.child else "all"
        runner = keys[b2] if b2 is not None and np.isfinite(row[b2]) else None
        floor = th.floor
        if p1.child and day and (m := ages.months(p1.born, day)) is not None and m < A["baby_months"]:
            floor = max(floor, A["baby_floor"])
        floor += bumps.get((k1, binl), 0.0) + (A["no_date_floor_bump"] if not day else 0.0) + \
            (A["outer_floor_bump"] if p1.outer else 0.0)
        sibs = cfg.siblings.get(k1, set())
        margin = th.sibling_margin if runner in sibs else th.margin
        reason = ""
        # Every sibling allowed on that date must have a usable gallery there, else the sibling cannot compete.
        for s in sibs:
            if s in keys:
                js = keys.index(s)
                if C[i, js] >= 0 and C[i, js] < min_active:
                    reason = f"thin-sibling-gallery:{s}"
        if not reason and C[i, b1] < min_active:
            reason = "thin-gallery"
        if not reason and s1 < floor:
            reason = "below-floor"
        if not reason and s1 - s2 < margin:
            reason = "margin" if runner not in sibs else "sibling-margin"
        out.append(Decision(int(fs.ids[i]), k1, s1, runner, s2, binl, "abstained" if reason else "accepted",
                            reason or "ok"))
    return out


# --- calibration -----------------------------------------------------------------------------------------
def _truth_ok(pred: str, truth: str) -> bool:
    if truth.startswith("not:"):
        return pred not in truth[4:].split(",")
    return pred == truth


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    if k == n:
        return 0.05 ** (1.0 / n)      # exact one-sided 95% bound with zero errors
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def impostors(db: sqlite3.Connection, cfg: Config, n: int, seed: int = 7) -> list[int]:
    """Faces Synology attaches to named people who are not gallery people: used only as negatives (they can only
    make thresholds stricter)."""
    gallery_names = {p.name for p in cfg.gallery_people}
    rows = db.execute("SELECT f.id FROM faces f JOIN syno_faces s ON s.face_id=f.syno_face_id "
                      "JOIN syno_people p ON p.id=s.person_id WHERE p.name != '' ").fetchall()
    names = {r[0]: r[1] for r in db.execute("SELECT f.id, p.name FROM faces f JOIN syno_faces s ON s.face_id=f.syno_face_id "
                                            "JOIN syno_people p ON p.id=s.person_id")}
    ids = [r[0] for r in rows if names.get(r[0]) not in gallery_names]
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.choice(ids, size=min(n, len(ids)), replace=False)] if ids else []


def calibrate(db: sqlite3.Connection, cfg: Config, model: str, gals: dict[str, Gallery],
              labels: dict[int, str]) -> Thresholds:
    A = cfg["assign"]
    gal_keys = set(gals)
    truth: dict[int, str] = {}
    for f, v in labels.items():
        if v in gal_keys or v.startswith("not:"):
            truth[f] = v
        elif v in ("unknown", "not_face") or v.startswith("syno:") or v in cfg.people:
            truth[f] = "none"
    for f in impostors(db, cfg, A["impostor_sample"]):
        truth.setdefault(f, "none")
    default = Thresholds(A["default_floor"], A["default_margin"], A["default_sibling_margin"], "default")
    if not truth:
        return default
    fs = load_faceset(db, model, truth)
    keys, S, C = score(fs, gals, cfg, exclude_same_day=True)
    if S.shape[1] == 0:
        return default
    order = np.argsort(-S, axis=1)
    s1 = np.take_along_axis(S, order[:, :1], 1)[:, 0]
    s2 = np.take_along_axis(S, order[:, 1:2], 1)[:, 0] if S.shape[1] > 1 else np.full(len(s1), -1.0)
    s2 = np.where(np.isfinite(s2), s2, -1.0)
    pred = np.array([keys[j] for j in order[:, 0]], dtype=object)
    runner = np.array([keys[j] if S.shape[1] > 1 else "" for j in (order[:, 1] if S.shape[1] > 1 else order[:, 0])],
                      dtype=object)
    finite = np.isfinite(s1)
    tr = np.array([truth[int(f)] for f in fs.ids], dtype=object)
    correct = np.array([_truth_ok(p, t) if t != "none" else False for p, t in zip(pred, tr)])
    is_sib = np.array([r in cfg.siblings.get(p, set()) for p, r in zip(pred, runner)])
    n_pos = int(sum(1 for t in tr if t in gal_keys))
    gap = s1 - s2
    best = None
    for floor in np.arange(0.40, 0.905, 0.01):
        base = finite & (s1 >= floor)
        for margin in np.arange(0.0, 0.205, 0.01):
            for sm in np.arange(margin, 0.255, 0.01):
                acc = base & np.where(is_sib, gap >= sm, gap >= margin)
                na = int(acc.sum())
                if na < A["min_calibration_accepts"]:
                    continue
                err = int((acc & ~correct).sum())
                if err / na > 1 - A["precision_target"]:
                    continue
                rec = int((acc & correct).sum())
                if best is None or rec > best[0]:
                    best = (rec, float(floor), float(margin), float(sm), na, err)
    if best is None:
        default.stats = {"n_pos": n_pos, "n_neg": len(tr) - n_pos, "reason": "not enough labelled faces"}
        return default
    rec, floor, margin, sm, na, err = best
    # Guard: the labelled set under-represents strangers and look-alike babies (on its own it once chose floor
    # 0.42 and zero margins), so calibration can only make the reviewed thresholds stricter.
    raw = (round(floor + A["safety_bump"], 3), round(margin, 3), round(sm, 3))
    th = Thresholds(max(raw[0], A["min_floor"]), max(raw[1], A["default_margin"]),
                    max(raw[2], A["default_sibling_margin"]), "calibrated")
    th.stats = {"clamped": raw != (th.floor, th.margin, th.sibling_margin), "raw": list(raw),
"n_pos": n_pos, "n_neg": int(len(tr) - n_pos), "accepted": na, "errors": err,
                "precision": round((na - err) / na, 5), "recall": round(rec / max(n_pos, 1), 4),
                "lower95": round(wilson_lower(na - err, na), 5),
                "raw_floor": round(floor, 3)}
    return th


def save_calibration(db: sqlite3.Connection, run: str, model: str, th: Thresholds) -> None:
    s = th.stats
    db.execute("INSERT OR REPLACE INTO calibration VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (run, model, th.status, th.floor, th.margin, th.sibling_margin, s.get("n_pos"), s.get("n_neg"),
                s.get("accepted"), s.get("errors"), s.get("precision"), s.get("recall"), s.get("lower95"), D.now()))
    db.commit()


def load_bumps(db: sqlite3.Connection) -> tuple[dict[tuple[str, str], float], set[tuple[str, str]]]:
    rows = db.execute("SELECT person, bin, amount, staged FROM bumps").fetchall()
    return {(r[0], r[1]): r[2] for r in rows}, {(r[0], r[1]) for r in rows if r[3]}


def run_assignment(db: sqlite3.Connection, cfg: Config, run: str, log=print) -> dict:
    """Calibrate per model, decide every embedded face with the reference model, veto with the challenger."""
    labels = D.latest_labels(db)
    ref = cfg["models"]["reference"]
    chal = cfg["models"].get("challenger")
    bumps, _ = load_bumps(db)
    gal = build_galleries(db, cfg, ref, labels)
    th = calibrate(db, cfg, ref, gal, labels)
    save_calibration(db, run, ref, th)
    fs = load_faceset(db, ref)
    keys, S, C = score(fs, gal, cfg)
    decs = decide(fs, keys, S, C, cfg, th, bumps)
    chal_best: dict[int, str | None] = {}
    if chal:
        galc = build_galleries(db, cfg, chal, labels)
        thc = calibrate(db, cfg, chal, galc, labels)
        save_calibration(db, run, chal, thc)
        fsc = load_faceset(db, chal)
        kc, Sc, Cc = score(fsc, galc, cfg)
        for d in decide(fsc, kc, Sc, Cc, cfg, thc, bumps):
            chal_best[d.face] = d.person if d.status == "accepted" else (f"?{d.person}" if d.person else None)
    db.execute("DELETE FROM decisions WHERE run=?", (run,))
    counts: dict[str, int] = {}
    for d in decs:
        status, reason = d.status, d.reason
        lab = labels.get(d.face)
        cb = chal_best.get(d.face)
        if lab is not None:
            status, reason = "labelled", lab
        elif status == "accepted" and chal:
            if cb is None or cb.lstrip("?") != d.person:
                status, reason = "vetoed", f"challenger:{cb}"
        counts[status] = counts.get(status, 0) + 1
        db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?)",
                   (run, d.face, d.person, round(d.score, 4), d.runner, round(d.runner_score, 4), d.bin, status,
                    reason, cb))
    db.commit()
    out = {"thresholds": {"floor": th.floor, "margin": th.margin, "sibling_margin": th.sibling_margin,
                          "status": th.status, **th.stats}, "decisions": counts,
           "coverage": gallery_coverage(cfg, gal)}
    log(f"assign: {json.dumps(out['decisions'])} thresholds {th.status} floor {th.floor} margin {th.margin} "
        f"sibling {th.sibling_margin}")
    return out
