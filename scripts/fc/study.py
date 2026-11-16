"""Blind study: a stratified review that measures Synology and the curator on named confusions and missed tags,
plus a 2^(5-1) Resolution V factorial in the uncertain region of the hardest sibling pair.

Strata (question-driven):
- confusion A->B: faces Synology names A where the curator's top person is B (disagree), where the curator agrees
  but B is a close second (near, gap < 0.10), or a clear second (agree sample). The pairs are every sibling and
  look-alike pair from the config plus the most frequent close-circle disagreements in the library;
- missed X: faces Synology names nobody on, where the curator's top person is X, in three confidence bands;
- control: faces where Synology and an accepted curator decision agree.
Factorial: faces whose curator top two are the study pair (`study build --pair A B`, default: the sibling pair of
children with the most disagreements), score >= 0.55, cells by score, gap, face height, crowd and the younger
child's age (< 24 months); the fifth factor is set by E = ABCD (half fraction, Resolution V: main effects and all two-factor
interactions are estimable, aliased only with three-factor interactions).

Every card stores its stratum, the stratum's population size N_h and the cell levels, so the report can weight each
answer back to the library (Horvitz-Thompson). Cards are shuffled and answered blind (review -> Compare tab).
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from itertools import product

import numpy as np

from fc import ages, assign, compare
from fc import db as D
from fc.config import Config

KIND = "study"

# Directed confusions: caps for the disagree / near / agree strata. Configured pairs get CONF_PAIR; the most
# frequent other disagreements (TOP_CONFUSIONS directed pairs) get CONF_OTHER.
CONF_PAIR = (40, 8, 15)
CONF_OTHER = (10, 3, 6)
TOP_CONFUSIONS = 8
BIG_N = 4                                                      # people with most Synology faces get more cards
MISS_BANDS = (("confident", 0.67, None), ("borderline", 0.60, 0.67), ("weak", 0.50, 0.60))
MISS_ALLOC = {True: (15, 15, 10), False: (12, 10, 8)}          # big four / the others
CONTROL_N = 100
FACTORS = ("score", "gap", "height", "crowd", "age")           # E (age of the younger child) = A*B*C*D
CELL_N = 50


def _galleries(db: sqlite3.Connection, cfg: Config, syn: dict[int, str | None], days: dict[int, str], seed: int):
    """Owner labels first; each person / age bin is topped up to 30 (adults 50) from Synology's single-face photos."""
    owner = {f: v for f, v in D.latest_labels(db).items() if v in cfg.people and cfg.people[v].tier == "close"}
    rng = np.random.default_rng(seed)
    edges = cfg["age_bins_months"]
    pick: dict[int, str] = dict(owner)
    # A Synology face may top up a gallery only if the owner's own exemplars do not pick someone else for it:
    # otherwise Synology's mistakes would become exemplars and could never be tested.
    model = cfg["models"]["reference"]
    og = assign.build_galleries(db, cfg, model, owner)
    cand = [f for f, v in syn.items() if v in og]
    veto: set[int] = set()
    if cand and owner:
        fs = assign.load_faceset(db, model, cand)
        keys, S, _ = assign.score(fs, og, cfg)
        best = np.argmax(S, 1)
        for f, j, row in zip(fs.ids.tolist(), best, S):
            own = syn[int(f)]
            # Veto only when the owner's gallery for that very person can score the face and someone else wins:
            # people the owner has not labelled yet cannot veto anything.
            if own in keys and np.isfinite(row[keys.index(own)]) and keys[j] != own:
                veto.add(int(f))
    for p in cfg.close_people:
        def cell(f):
            return ages.bin_index(ages.months(p.born, days.get(f)), edges) if p.child else "all"
        have = Counter(cell(f) for f, v in owner.items() if v == p.key)
        pool = defaultdict(list)
        for f, v in syn.items():
            if v == p.key and f not in pick and f not in veto:
                pool[cell(f)].append(f)
        for c, li in pool.items():
            if c is None:
                continue
            need = max(0, (30 if p.child else 50) - have[c])
            pick.update({int(f): p.key for f in rng.choice(li, min(need, len(li)), replace=False)})
    src = Counter("owner" if f in owner else "synology" for f in pick)
    src["synology faces kept out (owner exemplars disagree)"] = len(veto)
    return assign.build_galleries(db, cfg, cfg["models"]["reference"], pick), pick, dict(src)


def _sample(rng, idx: list, n: int, day_person) -> list:
    """Up to n, one per (person, day) first so bursts count once; all of them when the pool is smaller."""
    idx = list(idx)
    rng.shuffle(idx)
    seen, first, rest = set(), [], []
    for i in idx:
        k = day_person(i)
        (rest if k in seen else first).append(i)
        seen.add(k)
    return (first + rest)[:n]


def confusions(cfg: Config, sv: np.ndarray, top: np.ndarray) -> dict[tuple[str, str], tuple[int, int, int]]:
    """Directed pairs to test: configured siblings and look-alikes, plus the most frequent disagreements."""
    out: dict[tuple[str, str], tuple[int, int, int]] = {}
    for groups in (cfg.siblings, cfg.lookalikes):
        for a, others in groups.items():
            for b in others:
                if cfg.people[a].tier == "close" and cfg.people[b].tier == "close":
                    out[(a, b)] = CONF_PAIR
    close = {p.key for p in cfg.close_people}
    seen = Counter((a, b) for a, b in zip(sv, top) if a in close and b in close and a != b)
    for pair, _ in seen.most_common():
        if len([k for k in out if out[k] == CONF_OTHER]) >= TOP_CONFUSIONS:
            break
        out.setdefault(pair, CONF_OTHER)
    return out


def study_pair(cfg: Config, sv: np.ndarray, top: np.ndarray, pair: tuple[str, str] | None) -> tuple[str, str] | None:
    """The factorial's pair: given, else the sibling pair of children Synology and the curator disagree on most."""
    if pair:
        for k in pair:
            if k not in cfg.people or cfg.people[k].tier != "close":
                raise SystemExit(f"study --pair: {k!r} is not a close-circle person")
        return tuple(pair)
    cands = {tuple(sorted((a, b))) for a, bs in cfg.siblings.items() for b in bs
             if cfg.people[a].child and cfg.people[b].child}
    if not cands:
        return None
    n = Counter(tuple(sorted((a, b))) for a, b in zip(sv, top) if a is not None and a != b)
    return max(sorted(cands), key=lambda c: n[c])


def build(db: sqlite3.Connection, cfg: Config, seed: int = 17, has_image=lambda item: True,
          min_synced: float = 0.95, pair: tuple[str, str] | None = None, log=print) -> dict:
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("a study is already open; answer it or report it first")
    # Only photos where Synology names somebody need their Synology faces read; on the others Synology names
    # nobody, so every face there is correctly "unnamed" even when it was detected after the sync.
    need = db.execute("SELECT COUNT(*) FROM items WHERE last_seen IS NOT NULL AND type != 'video' "
                      "AND COALESCE(persons_sig, '') != ''").fetchone()[0]
    synced = db.execute("SELECT COUNT(*) FROM items WHERE last_seen IS NOT NULL AND type != 'video' "
                        "AND COALESCE(persons_sig, '') != '' AND faces_synced IS NOT NULL").fetchone()[0]
    if not need or synced / need < min_synced:
        raise SystemExit(f"Synology faces read for {synced} of {need} photos: wait for the sync to finish")
    syn, source = compare.synology_view(db, cfg)
    rows = {r["id"]: r for r in db.execute(
        "SELECT f.id, f.item, f.y1, f.y2, i.taken FROM faces f JOIN items i ON i.id=f.item "
        "WHERE (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?", (compare.MIN_H, compare.MIN_DET))}
    per_item = Counter(r["item"] for r in rows.values())
    days = {f: (r["taken"] or "")[:10] for f, r in rows.items()}
    gals, gal_faces, gal_src = _galleries(db, cfg, syn, days, seed)
    labelled = D.latest_labels(db)
    test = [f for f in syn if f in rows and f not in gal_faces and f not in labelled and has_image(rows[f]["item"])]
    fs = assign.load_faceset(db, cfg["models"]["reference"], test)
    keys, S, C = assign.score(fs, gals, cfg)
    th = assign.Thresholds(cfg["assign"]["default_floor"], cfg["assign"]["default_margin"],
                           cfg["assign"]["default_sibling_margin"])
    decs = assign.decide(fs, keys, S, C, cfg, th)
    order = np.argsort(-S, 1)
    n = len(fs.ids)
    face = fs.ids.astype(int)
    top = np.array([keys[j] for j in order[:, 0]], dtype=object) if n else np.array([], dtype=object)
    run_ = np.array([keys[j] for j in order[:, 1]], dtype=object) if n else np.array([], dtype=object)
    s1 = S[np.arange(n), order[:, 0]] if n else np.zeros(0)
    s2 = np.where(np.isfinite(S[np.arange(n), order[:, 1]]), S[np.arange(n), order[:, 1]], -1.0) if n else np.zeros(0)
    gap = s1 - s2
    acc = np.array([d.status == "accepted" for d in decs])
    sv = np.array([syn[f] for f in face], dtype=object)
    rng = np.random.default_rng(seed)
    person_day = lambda i: (top[i], days.get(int(face[i]), ""))   # noqa: E731
    used: set[int] = set()
    cards: list[dict] = []
    pools: dict[str, int] = {}

    def take(stratum: str, mask: np.ndarray, k: int, extra: dict | None = None) -> None:
        idx = [i for i in np.nonzero(mask & np.isfinite(s1))[0] if int(face[i]) not in used]
        pools[stratum] = len(idx)
        for i in _sample(rng, idx, k, person_day):
            used.add(int(face[i]))
            cards.append({"i": int(i), "stratum": stratum, "N": len(idx), **(extra or {})})

    for (a, b), (kd, kn, ka) in confusions(cfg, sv, top).items():
        base = sv == a
        take(f"confusion:{a}>{b}:disagree", base & (top == b), kd)
        take(f"confusion:{a}>{b}:near", base & (top == a) & (run_ == b) & (gap < 0.10), kn)
        if ka:
            take(f"confusion:{a}>{b}:agree", base & (top == a) & (run_ == b) & (gap >= 0.10), ka)
    syn_n = Counter(v for v in sv if v is not None)
    big = {k for k, _ in syn_n.most_common(BIG_N)}
    for p in cfg.close_people:
        alloc = MISS_ALLOC[p.key in big]
        none_ = np.array([v is None for v in sv]) & (top == p.key)
        bands = {"confident": none_ & acc,                                   # the curator would tag it
                 "borderline": none_ & ~acc & (s1 >= 0.60),                  # close: score or gap short
                 "weak": none_ & ~acc & (s1 >= 0.50) & (s1 < 0.60)}
        for (band, _, _), k in zip(MISS_BANDS, alloc):
            take(f"missed:{p.key}:{band}", bands[band], k)
    take("control:agree", acc & np.array([v == t for v, t in zip(sv, top)]), CONTROL_N)
    # factorial
    height = np.array([rows[int(f)]["y2"] - rows[int(f)]["y1"] for f in face]) if n else np.zeros(0)
    crowd = np.array([per_item[rows[int(f)]["item"]] for f in face]) if n else np.zeros(0)
    sp = study_pair(cfg, sv, top, pair)
    kids = [cfg.people[k] for k in sp or () if cfg.people[k].child]
    young = max(kids, key=lambda p: p.born) if kids else None
    young_m = np.array([ages.months(young.born, days.get(int(f))) if young and days.get(int(f)) else np.nan
                        for f in face])
    region = (np.array([{t, r} == set(sp) for t, r in zip(top, run_)]) if sp else np.zeros(n, bool)) & (s1 >= 0.55)
    # Continuous factors are cut at their median inside the region, so the cells are balanced (fixed cuts left
    # half the cells empty: with age galleries the sibling gap is rarely below 0.12). The cuts are reported.
    def med(x):
        return float(np.median(x[region])) if region.any() else 0.0
    cuts = {"score": med(s1), "gap": med(gap), "height": med(height)}
    lv = {"score": (s1 < cuts["score"], s1 >= cuts["score"]), "gap": (gap < cuts["gap"], gap >= cuts["gap"]),
          "height": (height < cuts["height"], height >= cuts["height"]), "crowd": (crowd == 1, crowd >= 2),
          "age": (young_m < 24, young_m >= 24) if young else (crowd < 0, crowd >= 0)}
    cells = []
    for a_, b_, c_, d_ in product((-1, 1), repeat=4):
        e_ = a_ * b_ * c_ * d_
        lev = dict(zip(FACTORS, (a_, b_, c_, d_, e_)))
        m = region.copy()
        for fct, s in lev.items():
            m &= lv[fct][0 if s < 0 else 1]
        name = "factorial:" + ",".join(f"{k}{'+' if v > 0 else '-'}" for k, v in lev.items())
        before = len(cards)
        take(name, m, CELL_N, {"levels": lev, "cuts": {k: round(v, 3) for k, v in cuts.items()}})
        cells.append({"cell": name, "pool": pools[name], "cards": len(cards) - before})
    perm = rng.permutation(len(cards))
    run = f"study-{D.now()[:10]}"
    for j in perm:
        c = cards[j]
        i = c["i"]
        db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                   (run, KIND, int(face[i]), str(top[i]), sv[i], decs[i].bin,
                    json.dumps({"stratum": c["stratum"], "N": c["N"], "levels": c.get("levels"), "cuts": c.get("cuts"),
                                "score": round(float(s1[i]), 4), "runner": str(run_[i]),
                                "gap": round(float(gap[i]), 4), "accepted": bool(acc[i]),
                                "height": round(float(height[i]), 4), "crowd": int(crowd[i])}),
                    D.now()))
    db.commit()
    by = Counter(c["stratum"].split(":")[0] for c in cards)
    out = {"run": run, "cards": len(cards), "by_part": dict(by), "factorial_pair": list(sp) if sp else None, "galleries": gal_src, "synology_source": source,
           "factorial_cuts": {k: round(v, 3) for k, v in cuts.items()},
           "empty_strata": sorted(k for k, v in pools.items() if v == 0),
           "short_cells": [c for c in cells if c["cards"] < CELL_N]}
    log(json.dumps(out, indent=1))
    return out


# --- report ----------------------------------------------------------------------------------------------
def _ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - r), min(1.0, c + r))


def report(db: sqlite3.Connection, cfg: Config, run: str | None = None) -> dict:
    run = run or (db.execute(f"SELECT run FROM cards WHERE kind='{KIND}' ORDER BY id DESC LIMIT 1").fetchone() or [None])[0]
    if run is None:
        raise SystemExit("no study yet: `facecurator study build`")
    st: dict[str, dict] = defaultdict(lambda: {"N": 0, "n": 0, "open": 0, "syn_wrong": 0, "cur_wrong": 0,
                                                "truth": Counter(), "cells": []})
    for c in db.execute("SELECT * FROM cards WHERE kind=? AND run=?", (KIND, run)):
        p = json.loads(c["payload"])
        s = st[p["stratum"]]
        s["N"] = p["N"]
        if c["answer"] is None:
            s["open"] += 1
            continue
        if c["answer"] in ("skip", "unknown"):
            continue
        t = c["answer"]
        s["n"] += 1
        s["truth"][t] += 1
        s["syn_wrong"] += (c["alt"] is not None and c["alt"] != t)
        s["cur_wrong"] += c["person"] != t
        s["cells"].append((p.get("levels"), c["person"] != t))
    out: dict = {"run": run, "confusions": {}, "missed": {}, "control": {}, "factorial": {}}
    # confusions: share of Synology's "A" faces in these strata that are really B, weighted to the pool
    by_q: dict[str, list] = defaultdict(list)
    for k, s in st.items():
        part = k.split(":")
        if part[0] == "confusion":
            by_q[part[1]].append((part[2], s))
    for q, ss in sorted(by_q.items()):
        a, b = q.split(">")
        est_b = sum(s["N"] * s["truth"][b] / s["n"] for _, s in ss if s["n"])
        dis = dict(ss).get("disagree", {"n": 0, "syn_wrong": 0, "cur_wrong": 0})
        out["confusions"][q] = {
            "disagreements judged": dis["n"], "Synology wrong": dis["syn_wrong"], "curator wrong": dis["cur_wrong"],
            f"estimated faces named {a} that are really {b}": round(est_b),
            "strata": {name: {"pool": s["N"], "judged": s["n"], "truth": dict(s["truth"])} for name, s in ss}}
    for k, s in st.items():
        part = k.split(":")
        if part[0] == "missed":
            x, band = part[1], part[2]
            m = out["missed"].setdefault(x, {"estimated untagged faces": 0.0, "found by curator (accepted)": 0.0,
                                             "bands": {}})
            hit = s["truth"][x] / s["n"] if s["n"] else 0.0
            m["estimated untagged faces"] += s["N"] * hit
            if band == "confident":
                m["found by curator (accepted)"] += s["N"] * hit
                lo, hi = _ci(s["truth"][x], s["n"])
                m["curator precision (confident band)"] = f"{hit:.1%} ({lo:.0%}-{hi:.0%}, n={s['n']})"
            m["bands"][band] = {"pool": s["N"], "judged": s["n"], f"really {x}": s["truth"][x]}
    for x, m in out["missed"].items():
        tot = m["estimated untagged faces"]
        m["estimated untagged faces"] = round(tot)
        m["share the curator would tag"] = f"{m['found by curator (accepted)'] / tot:.0%}" if tot else "-"
        m["found by curator (accepted)"] = round(m["found by curator (accepted)"])
    c = st.get("control:agree")
    if c and c["n"]:
        lo, hi = _ci(c["n"] - c["cur_wrong"], c["n"])
        out["control"] = {"judged": c["n"], "both wrong": c["cur_wrong"], "precision 95% interval": f"{lo:.1%}-{hi:.1%}"}
    out["factorial"] = factorial_effects([x for k, s in st.items() if k.startswith("factorial:") for x in s["cells"]])
    return out


def factorial_effects(obs: list[tuple[dict, bool]]) -> dict:
    """Effects on the logit of the curator's error rate: main effects and two-factor interactions from the 16 cells
    (Resolution V). Cell rates use a 0.5 continuity correction; SE from the binomial variance of each logit."""
    cells: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for lev, wrong in obs:
        if lev is None:
            continue
        old_age = next((v for kk, v in lev.items() if kk.endswith("_age")), None)          # studies built by the first version
        k = tuple(lev.get(f, old_age) if f == "age" else lev[f] for f in FACTORS)
        cells[k][0] += int(wrong)
        cells[k][1] += 1
    if len(cells) < 2:
        return {"cells answered": len(cells)}
    keys = sorted(cells)
    L = np.array([math.log((cells[k][0] + 0.5) / (cells[k][1] - cells[k][0] + 0.5)) for k in keys])
    V = np.array([1 / (cells[k][0] + 0.5) + 1 / (cells[k][1] - cells[k][0] + 0.5) for k in keys])
    X = np.array(keys, dtype=float)
    out = {"cells answered": len(keys), "error rate by cell": {
        ",".join(f"{f}{'+' if v > 0 else '-'}" for f, v in zip(FACTORS, k)): f"{cells[k][0]}/{cells[k][1]}" for k in keys}}
    terms = {f: X[:, i] for i, f in enumerate(FACTORS)}
    for i in range(len(FACTORS)):
        for j in range(i + 1, len(FACTORS)):
            terms[f"{FACTORS[i]}x{FACTORS[j]}"] = X[:, i] * X[:, j]
    eff = {}
    m = len(keys)
    for name, col in terms.items():
        if np.all(col == col[0]):
            continue
        e = 2 * float(col @ L) / m
        se = 2 * math.sqrt(float(V.sum())) / m
        eff[name] = {"effect (logit)": round(e, 3), "se": round(se, 3), "clear": abs(e) > 2 * se}
    out["effects"] = dict(sorted(eff.items(), key=lambda kv: -abs(kv[1]["effect (logit)"])))
    return out
