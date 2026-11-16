"""Blind head-to-head: is the curator better than Synology's own recognition?

Only faces where the two systems differ can show which is more accurate, so the comparison set is:
- disagree: Synology names one gallery person, the curator accepts another (every such face);
- coverage: Synology names nobody on the photo, the curator accepts someone (random sample);
- agree:    both name the same person (random sample, a control for the owner's own answer rate).

The owner answers "who is this?" from the full list of gallery people without seeing either system's answer;
the card order is shuffled across the three groups. Answers become owner labels (they also seed galleries) but
never raise thresholds. Until the owner has labels, the curator's galleries here are built from Synology's own
single-person photos (30 faces per child per age bin, 50 per adult): the test then compares methods on the same
evidence. The report says which galleries were used.

Synology's view comes from the ledger (`syno_faces`), so a NAS sync must have run.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from fc import ages, assign
from fc import db as D
from fc.config import Config

KIND = "h2h"
BLIND_KINDS = ("h2h", "study", "edge")   # answered on the blind Compare tab
MIN_H, MIN_DET = 0.06, 0.7


def _sizeable(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return db.execute("SELECT f.id, f.item, f.syno_face_id, i.taken FROM faces f JOIN items i ON i.id=f.item "
                      "WHERE (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?", (MIN_H, MIN_DET)).fetchall()


def synology_view(db: sqlite3.Connection, cfg: Config) -> tuple[dict[int, str | None], str]:
    """face -> gallery person key, 'other' (a named non-gallery person) or None (Synology names nobody there)."""
    key_of = {p.name: p.key for p in cfg.close_people}
    rows = _sizeable(db)
    if not db.execute("SELECT COUNT(*) FROM syno_faces").fetchone()[0]:
        raise SystemExit("no Synology view yet: run a sync first (`run --until sync`)")
    names = {r[0]: r[1] for r in db.execute("SELECT s.face_id, p.name FROM syno_faces s "
                                            "LEFT JOIN syno_people p ON p.id=s.person_id")}
    out = {}
    for r in rows:
        if r["syno_face_id"] is None:
            out[r["id"]] = None                    # Synology has no face there
            continue
        n = names.get(r["syno_face_id"]) or ""
        out[r["id"]] = key_of.get(n, "other" if n else None)
    return out, "ledger (synced Synology faces)"


def _galleries(db, cfg: Config, syn: dict[int, str | None], days: dict[int, str], seed: int) -> tuple[dict, dict, str]:
    owner = {f: v for f, v in D.latest_labels(db).items() if v in {p.key for p in cfg.close_people}}
    if sum(Counter(owner.values()).values()) >= 8 * len(cfg.close_people):
        return assign.build_galleries(db, cfg, cfg["models"]["reference"], owner), owner, "owner labels"
    rng = np.random.default_rng(seed)
    edges = cfg["age_bins_months"]
    pick: dict[int, str] = {}
    for p in cfg.close_people:
        mine = [f for f, v in syn.items() if v == p.key]
        if not mine:
            continue
        if p.child:
            cells = defaultdict(list)
            for f in mine:
                cells[ages.bin_index(ages.months(p.born, days.get(f)), edges)].append(f)
            chosen = [f for b, li in cells.items() if b is not None for f in rng.choice(li, min(30, len(li)), replace=False)]
        else:
            chosen = list(rng.choice(mine, min(50, len(mine)), replace=False))
        pick.update({int(f): p.key for f in chosen})
    return (assign.build_galleries(db, cfg, cfg["models"]["reference"], pick), pick,
            "Synology single-person photos (no owner labels yet)")


def build(db: sqlite3.Connection, cfg: Config, n_coverage: int = 100, n_agree: int = 40, n_disagree: int | None = None,
          seed: int = 5, has_image=lambda item: True, log=print) -> dict:
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("a comparison set is already open; answer it (review → Compare) or report it first")
    syn, source = synology_view(db, cfg)
    rows = {r["id"]: r for r in _sizeable(db)}
    days = {f: (r["taken"] or "")[:10] for f, r in rows.items()}
    gals, gal_faces, gal_src = _galleries(db, cfg, syn, days, seed)
    test = [f for f in syn if f not in gal_faces and f not in D.latest_labels(db) and has_image(rows[f]["item"])]
    fs = assign.load_faceset(db, cfg["models"]["reference"], test)
    keys, S, C = assign.score(fs, gals, cfg)
    th = assign.Thresholds(cfg["assign"]["default_floor"], cfg["assign"]["default_margin"],
                           cfg["assign"]["default_sibling_margin"], "default")
    cal = db.execute("SELECT floor, margin, sibling_margin FROM calibration WHERE status='calibrated' AND model=? "
                     "ORDER BY at DESC LIMIT 1", (cfg["models"]["reference"],)).fetchone()
    if cal and gal_src == "owner labels":
        th = assign.Thresholds(cal[0], cal[1], cal[2], "calibrated")
    decs = assign.decide(fs, keys, S, C, cfg, th)
    groups: dict[str, list] = {"disagree": [], "coverage": [], "agree": []}
    for d in decs:
        if d.status != "accepted":
            continue
        s = syn[d.face]
        if s is None:
            groups["coverage"].append(d)
        elif s == d.person:
            groups["agree"].append(d)
        elif s != "other":
            groups["disagree"].append(d)
    rng = np.random.default_rng(seed)

    def one_per_day(ds: list, n: int | None) -> list:
        ds = list(ds)
        rng.shuffle(ds)
        seen, out = set(), []
        for d in ds:
            k = (d.person, days.get(d.face, ""))
            if k in seen:
                continue
            seen.add(k)
            out.append(d)
            if n is not None and len(out) >= n:
                break
        return out

    chosen = ([("disagree", d) for d in one_per_day(groups["disagree"], n_disagree)]
              + [("coverage", d) for d in one_per_day(groups["coverage"], n_coverage)]
              + [("agree", d) for d in one_per_day(groups["agree"], n_agree)])
    order = rng.permutation(len(chosen))
    run = f"h2h-{D.now()[:10]}"
    for i in order:
        g, d = chosen[i]
        db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                   (run, KIND, d.face, d.person, syn[d.face], d.bin,
                    json.dumps({"group": g, "score": round(d.score, 4), "runner": d.runner,
                                "runner_score": round(d.runner_score, 4), "galleries": gal_src, "synology_source": source}),
                    D.now()))
    db.commit()
    out = {"run": run, "available": {k: len(v) for k, v in groups.items()},
           "cards": dict(Counter(g for g, _ in chosen)), "galleries": gal_src, "synology_source": source,
           "thresholds": {"floor": th.floor, "margin": th.margin, "sibling_margin": th.sibling_margin, "status": th.status},
           "disagreement_pairs": dict(Counter(f"synology {syn[d.face]} / curator {d.person}"
                                              for d in groups["disagree"]).most_common())}
    log(json.dumps(out, indent=1))
    return out


def retract(db: sqlite3.Connection, c: sqlite3.Row) -> bool:
    """Take back the answer of a blind card: its label is marked retracted, the card is open again."""
    if c["answer"] is None:
        return False
    last = db.execute("SELECT id FROM labels WHERE face=? AND via=? AND source='owner' ORDER BY id DESC LIMIT 1",
                      (c["face"], f"card:{c['kind']}")).fetchone()
    if last and c["answer"] != "skip":
        db.execute("UPDATE labels SET source='retracted' WHERE id=?", (last[0],))
    db.execute("UPDATE cards SET answer=NULL, answered=NULL WHERE id=?", (c["id"],))
    return True


def answer(db: sqlite3.Connection, cfg: Config, card_id: int, ans: str, replace: bool = False) -> dict:
    """ans: a gallery person key, 'other' (someone not in the list), 'unknown', 'not_face' or 'skip'. An
    answered card is changed only with `replace` (the earlier answer is kept in the card's payload)."""
    from fc import cards

    if isinstance(card_id, bool) or not isinstance(card_id, int):
        raise ValueError("card id must be an integer")
    c = db.execute("SELECT * FROM cards WHERE id=? AND kind IN (?,?,?)", (card_id, *BLIND_KINDS)).fetchone()
    if c is None:
        raise ValueError("no such comparison card")
    # close circle always; outer-check cards add their own four outer names ("other" = none of all shown)
    gk = [p.key for p in cfg.close_people] + card_options(c)
    if not isinstance(ans, str) or ans not in set(gk) | {"other", "unknown", "not_face", "skip"}:
        raise ValueError("answer must be a person shown on the card, other, unknown, not_face or skip")
    if c["answer"] is not None:
        if replace is not True:
            raise cards.AlreadyAnswered(f"card {card_id} already has an answer; it is changed only on purpose")
        cards.keep_previous(db, c)
        retract(db, c)
    db.execute("UPDATE cards SET answer=?, answered=? WHERE id=?", (ans, D.now(), card_id))
    if ans != "skip":
        value = "not:" + ",".join(gk) if ans == "other" else ans
        D.add_label(db, c["face"], value, via=f"card:{c['kind']}", run=c["run"])
    db.commit()
    return {"saved": True}


def card_options(c) -> list[str]:
    try:
        return list(json.loads(c["payload"] or "{}").get("options") or [])
    except (TypeError, ValueError):
        return []


def _lower(k: int, n: int) -> float:
    return round(assign.wilson_lower(k, n), 4) if n else 0.0


def report(db: sqlite3.Connection, cfg: Config, run: str | None = None) -> dict:
    run = run or (db.execute(f"SELECT run FROM cards WHERE kind='{KIND}' ORDER BY id DESC LIMIT 1").fetchone() or [None])[0]
    if run is None:
        raise SystemExit("no comparison set yet: `facecurator compare build`")
    rows = db.execute("SELECT * FROM cards WHERE kind=? AND run=?", (KIND, run)).fetchall()
    by: dict[str, Counter] = defaultdict(Counter)
    wrong_pairs: Counter = Counter()
    for c in rows:
        g = json.loads(c["payload"])["group"]
        if c["answer"] is None:
            by[g]["open"] += 1
            continue
        if c["answer"] in ("skip", "unknown"):
            by[g]["no answer"] += 1
            continue
        truth = c["answer"]
        by[g]["judged"] += 1
        by[g]["curator right"] += c["person"] == truth
        by[g]["synology right"] += c["alt"] == truth
        by[g]["both wrong"] += c["person"] != truth and c["alt"] != truth
        if c["person"] != truth:
            wrong_pairs[f"curator said {c['person']}, owner {truth}"] += 1
    out = {"run": run, "groups": {}, "curator_errors": dict(wrong_pairs.most_common(10))}
    for g, k in by.items():
        n = k["judged"]
        out["groups"][g] = {**dict(k), "curator_precision": round(k["curator right"] / n, 4) if n else None,
                            "curator_lower95": _lower(k["curator right"], n),
                            "synology_precision": round(k["synology right"] / n, 4) if n and g != "coverage" else None}
    d = out["groups"].get("disagree", {})
    cov = out["groups"].get("coverage", {})
    verdict = []
    if d.get("judged"):
        verdict.append(f"disagreements: curator right {d['curator right']}, Synology right {d['synology right']}, "
                       f"both wrong {d['both wrong']} of {d['judged']}")
    if cov.get("judged"):
        verdict.append(f"new names where Synology had none: precision {cov['curator_precision']:.1%} "
                       f"(95% lower bound {cov['curator_lower95']:.1%}) on {cov['judged']}")
    out["verdict"] = verdict
    if d.get("judged") and cov.get("judged"):
        better = d["curator right"] > d["synology right"] and cov["curator_lower95"] >= 0.95
        out["conclusion"] = ("curator better on this evidence" if better else
                             "no evidence the curator is better yet")
    return out
