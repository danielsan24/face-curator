"""Edge round: blind cards on the noisy edges, answered before the thresholds are set.

Galleries are the owner's labels only (what the next run will use). Every unlabelled face with a usable crop is
scored once; cards are drawn from three edges:

- floor:<band>:<who>   faces that clear the margin rules but score in a band between 0.42 and the 0.67 default.
                       <who> is what Synology says today: nobody (split child / adult), a named person outside the
                       galleries (the stranger risk), or another gallery person (a reassignment).
- margin:<pair>:<gap>  faces scoring >= 0.50 whose top two people are close: each sibling pair (config
                       `siblings`), look-alike pairs (config `lookalikes`), and any other pair, by gap band.
- gallery:<person>:<bin> thin child age bins (< 20 owner exemplars): candidates scored against the person's whole
                       gallery (no age window, since the window is what is empty); answers grow the gallery.

Each card stores its stratum and pool size N (for weighting) plus the scores; cards are shuffled and blind
(review -> Compare tab). `edges report` gives the curator's precision per stratum with 95% intervals.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict

import numpy as np

from fc import ages, assign, compare
from fc import db as D
from fc.config import Config
from fc.study import _ci, _sample

KIND = "edge"
FLOOR_BANDS = ((0.42, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.67))
FLOOR_ALLOC = {"nobody-child": 25, "nobody-adult": 15, "named-other": 15, "reassign": 10}
GAP_BANDS = {"sibling": ((0.0, 0.04), (0.04, 0.08), (0.08, 0.12)), "lookalike": ((0.0, 0.08), (0.08, 0.16)),
             "other": ((0.0, 0.08),)}
MARGIN_N = {"sibling-children": 25, "sibling": 10, "lookalike": 15, "other": 15}   # cards per gap band
THIN = 20            # owner exemplars in a child's age bin below which the bin is topped up
GALLERY_N = 20
GALLERY_MIN = 0.40   # whole-gallery score needed to be a candidate (unless Synology already names the person)


def _band(x: float, bands) -> str | None:
    for lo, hi in bands:
        if lo <= x < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def pair_kind(cfg: Config, a: str, b: str | None) -> str | None:
    """sibling:<a>-<b> (each sibling pair), lookalike (any look-alike pair), other; None without a runner-up."""
    if b is None:
        return None
    if b in cfg.siblings.get(a, set()):
        return "sibling:" + "-".join(sorted((a, b)))
    if b in cfg.lookalikes.get(a, set()):
        return "lookalike"
    return "other"


def build(db: sqlite3.Connection, cfg: Config, seed: int = 23, has_image=lambda item: True, log=print) -> dict:
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("an edge round is already open; answer it or report it first")
    model = cfg["models"]["reference"]
    edges_m = cfg["age_bins_months"]
    labels = D.latest_labels(db)
    owner = {f: v for f, v in labels.items() if v in cfg.people and cfg.people[v].tier == "close"}
    gals = assign.build_galleries(db, cfg, model, owner)
    syn, _ = compare.synology_view(db, cfg)
    busy = {r[0] for r in db.execute("SELECT face FROM cards WHERE answer IS NULL")}
    rows = {r["id"]: r for r in db.execute(
        "SELECT f.id, f.item, i.taken FROM faces f JOIN items i ON i.id=f.item "
        "WHERE (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?", (compare.MIN_H, compare.MIN_DET))}
    test = [f for f in rows if f not in labels and f not in busy and has_image(rows[f]["item"])]
    fs = assign.load_faceset(db, model, test)
    keys, S, C = assign.score(fs, gals, cfg)
    th = assign.Thresholds(cfg["assign"]["default_floor"], cfg["assign"]["default_margin"],
                           cfg["assign"]["default_sibling_margin"])
    # Margin rules at a floor of 0 tell which faces fail only on the floor.
    decs0 = assign.decide(fs, keys, S, C, cfg, assign.Thresholds(0.0, th.margin, th.sibling_margin))
    n = len(fs.ids)
    face = fs.ids.astype(int)
    order = np.argsort(-S, 1)
    s1 = S[np.arange(n), order[:, 0]]
    s2r = S[np.arange(n), order[:, 1]]
    s2 = np.where(np.isfinite(s2r), s2r, -1.0)
    top = np.array([keys[j] for j in order[:, 0]], dtype=object)
    run_ = np.array([keys[j] if np.isfinite(s2r[i]) else None for i, j in enumerate(order[:, 1])], dtype=object)
    gap = s1 - s2
    days = np.array([(rows[int(f)]["taken"] or "")[:10] for f in face], dtype=object)
    sv = np.array([syn.get(int(f)) for f in face], dtype=object)       # key | 'other' | None
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    cards: list[dict] = []
    pools: dict[str, int] = {}

    def take(stratum: str, idx, k: int, who=None) -> None:
        idx = [int(i) for i in idx if int(face[i]) not in used]
        pools[stratum] = len(idx)
        who = who if who is not None else top
        for i in _sample(rng, idx, k, lambda i: (who[i], days[i])):
            used.add(int(face[i]))
            cards.append({"i": i, "stratum": stratum, "N": len(idx), "person": str(who[i])})

    # 1. floor edge
    margin_ok = np.array([d.reason in ("ok", "below-floor") for d in decs0])
    child = np.array([bool(cfg.people[t].child) if t in cfg.people else False for t in top])
    who_syn = {"nobody-child": np.array([v is None for v in sv]) & child,
               "nobody-adult": np.array([v is None for v in sv]) & ~child,
               "named-other": np.array([v == "other" for v in sv]),
               "reassign": np.array([v is not None and v != "other" for v in sv]) & (sv != top)}
    for lo, hi in FLOOR_BANDS:
        inb = margin_ok & (s1 >= lo) & (s1 < hi)
        for w, k in FLOOR_ALLOC.items():
            take(f"floor:{lo:.2f}-{hi:.2f}:{w}", np.nonzero(inb & who_syn[w])[0], k)

    # 2. margin edge
    pk = np.array([pair_kind(cfg, a, b) for a, b in zip(top, run_)], dtype=object)
    strong = np.isfinite(s1) & (s1 >= 0.50)
    for kind in sorted({k for k in pk if k}):
        fam = kind.split(":")[0]
        n_cards = MARGIN_N["sibling-children" if fam == "sibling" and all(cfg.people[k].child for k in
                                                                         kind.split(":")[1].split("-")) else fam]
        for lo, hi in GAP_BANDS[fam]:
            m = strong & (pk == kind) & (gap >= lo) & (gap < hi)
            take(f"margin:{kind}:{lo:.2f}-{hi:.2f}", np.nonzero(m)[0], n_cards)

    # 3. thin child galleries: whole-gallery scores (no age window), candidates dated inside the thin bin
    cover = assign.gallery_coverage(cfg, gals)
    for p in cfg.close_people:
        if not p.child:
            continue
        g = gals[p.key]
        allowed = np.array([p.allowed_on(d or None) for d in days], dtype=bool)
        fb = np.array([-1 if (b := ages.bin_index(ages.months(p.born, d) if d else None, edges_m)) is None else b
                       for d in days])
        if len(g.faces) >= assign.MIN_VALID:
            sims = fs.vecs @ g.vecs.T
            kk = min(assign.TOPK, sims.shape[1])
            whole = -np.sort(-sims, 1)[:, :kk].mean(1)
        else:
            whole = np.full(n, -1.0)
        for b in range(len(edges_m)):
            lab = ages.bin_label(b, edges_m)
            if cover[p.key].get(lab, 0) >= THIN:
                continue
            m = allowed & (fb == b) & ((sv == p.key) | ((whole >= GALLERY_MIN) & ((top == p.key) | ~np.isfinite(s1))))
            if m.sum() < 3:
                continue
            take(f"gallery:{p.key}:{lab}", np.nonzero(m)[0], GALLERY_N, who=np.full(n, p.key, dtype=object))

    perm = rng.permutation(len(cards))
    run = f"edges-{D.now()[:10]}"
    for j in perm:
        c = cards[j]
        i = c["i"]
        db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                   (run, KIND, int(face[i]), c["person"], sv[i], None,
                    json.dumps({"stratum": c["stratum"], "N": c["N"], "score": round(float(s1[i]), 4),
                                "top": str(top[i]), "runner": run_[i], "gap": round(float(gap[i]), 4)}),
                    D.now()))
    db.commit()
    out = {"run": run, "cards": len(cards), "by_part": dict(Counter(c["stratum"].split(":")[0] for c in cards)),
           "gallery_bins": sorted({c["stratum"] for c in cards if c["stratum"].startswith("gallery:")}),
           "empty_strata": sorted(k for k, v in pools.items() if v == 0)}
    log(json.dumps(out, indent=1))
    return out


def report(db: sqlite3.Connection, cfg: Config, run: str | None = None) -> dict:
    run = run or (db.execute(f"SELECT run FROM cards WHERE kind='{KIND}' ORDER BY id DESC LIMIT 1").fetchone() or [None])[0]
    if run is None:
        raise SystemExit("no edge round yet: `facecurator edges build`")
    st: dict[str, dict] = defaultdict(lambda: {"N": 0, "n": 0, "open": 0, "right": 0, "syn_right": 0, "syn_n": 0,
                                                "truth": Counter()})
    for c in db.execute("SELECT * FROM cards WHERE kind=? AND run=?", (KIND, run)):
        p = json.loads(c["payload"])
        s = st[p["stratum"]]
        s["N"] = p["N"]
        if c["answer"] is None:
            s["open"] += 1
            continue
        if c["answer"] in ("skip", "unknown"):
            continue
        s["n"] += 1
        s["right"] += c["answer"] == c["person"]
        if c["alt"] not in (None, "other"):
            s["syn_n"] += 1
            s["syn_right"] += c["answer"] == c["alt"]
        s["truth"]["(the curator's pick)" if c["answer"] == c["person"] else c["answer"]] += 1
    out: dict = {"run": run, "open": sum(s["open"] for s in st.values()), "strata": {}}
    for k in sorted(st):
        s = st[k]
        lo, hi = _ci(s["right"], s["n"])
        out["strata"][k] = {"pool": s["N"], "judged": s["n"],
                            "curator right": f"{s['right']}/{s['n']}" + (f" ({lo:.0%}-{hi:.0%})" if s["n"] else ""),
                            "answers": dict(s["truth"])}
        if s["syn_n"]:
            out["strata"][k]["synology right"] = f"{s['syn_right']}/{s['syn_n']}"
    # pooled precision per floor band (weighted by pool size across the <who> strata)
    pooled: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for k, s in st.items():
        part = k.split(":")
        if part[0] == "floor" and s["n"]:
            b = pooled[part[1]]
            b[0] += s["N"] * s["right"] / s["n"]
            b[1] += s["N"]
            b[2] += s["right"]
            b[3] += s["n"]
    out["floor bands (pool-weighted precision)"] = {
        b: {"precision": f"{v[0] / v[1]:.1%}" if v[1] else "-", "right": f"{v[2]}/{v[3]}", "faces in band": int(v[1])}
        for b, v in sorted(pooled.items())}
    return out


# --- Synology check: faces Synology names X that the curator doubts ----------------
SYNO_PARTS = (("accepted-other", 8), ("top-other", 7), ("abstained-same", 5))


def build_synology_check(db: sqlite3.Connection, cfg: Config, people: list[str], n: int = 20, run: str | None = None,
                         seed: int = 29, has_image=lambda item: True, log=print) -> dict:
    """For each person X: up to n faces Synology names X where the latest run's decision doubts it. Parts, in
    order: the curator accepts someone else; its top is someone else but it abstains; its top is X but it
    abstains (low score or close runner-up, lowest scores first). Unused quota moves to the next part."""
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("an edge round is already open; answer it or report it first")
    run = run or db.execute("SELECT run FROM decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    syn, _ = compare.synology_view(db, cfg)
    labels = D.latest_labels(db)
    busy = {r[0] for r in db.execute("SELECT face FROM cards WHERE answer IS NULL")}
    rows = {r["face"]: r for r in db.execute(
        "SELECT d.*, f.item, i.taken FROM decisions d JOIN faces f ON f.id=d.face JOIN items i ON i.id=f.item "
        "WHERE d.run=? AND (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?", (run, compare.MIN_H, compare.MIN_DET))}
    rng = np.random.default_rng(seed)
    cards, pools = [], {}
    for x in people:
        if x not in cfg.people:
            raise SystemExit(f"unknown person {x}")
        mine = [r for f, r in rows.items() if syn.get(f) == x and f not in labels and f not in busy
                and r["status"] in ("accepted", "abstained") and has_image(r["item"])]
        parts = {"accepted-other": [r for r in mine if r["status"] == "accepted" and r["person"] != x],
                 "top-other": [r for r in mine if r["status"] == "abstained" and r["person"] not in (x, None)],
                 "abstained-same": sorted([r for r in mine if r["status"] == "abstained" and r["person"] == x],
                                          key=lambda r: r["score"])}
        left = n
        for i, (part, quota) in enumerate(SYNO_PARTS):
            pool = parts[part]
            pools[f"syno:{x}:{part}"] = len(pool)
            target = sum(q for _, q in SYNO_PARTS[:i + 1])          # cumulative quota; shortfalls carry forward
            k = left if i == len(SYNO_PARTS) - 1 else max(0, min(left, target - (n - left)))
            if part == "abstained-same":
                half = pool[:max(k * 3, k)]                      # the low-score end, still varied
                pick = _sample(rng, list(range(len(half))), k, lambda j: (half[j]["taken"] or "")[:10])
                chosen = [half[j] for j in pick]
            else:
                pick = _sample(rng, list(range(len(pool))), k, lambda j: (pool[j]["person"], (pool[j]["taken"] or "")[:10]))
                chosen = [pool[j] for j in pick]
            for r in chosen:
                cards.append((f"syno:{x}:{part}", len(pool), r))
            left -= len(chosen)
    rid = f"synocheck-{D.now()[:10]}"
    for j in rng.permutation(len(cards)):
        stratum, N, r = cards[j]
        x = stratum.split(":")[1]
        db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                   (rid, KIND, int(r["face"]), r["person"] or "", x, r["bin"],
                    json.dumps({"stratum": stratum, "N": N, "score": r["score"], "top": r["person"],
                                "runner": r["runner"], "gap": round((r["score"] or 0) - (r["runner_score"] or 0), 4),
                                "status": r["status"], "reason": r["reason"]}), D.now()))
    db.commit()
    out = {"run": rid, "from decisions of": run, "cards": len(cards),
           "by_stratum": dict(Counter(s for s, _, _ in cards)), "pools": pools}
    log(json.dumps(out, indent=1))
    return out


# --- low-score Synology tags ----------------------------------------------------
LOW_SCORE = 0.35     # reviewed default; see docs/lessons.md


def build_low_score(db: sqlite3.Connection, cfg: Config, below: float = LOW_SCORE, run: str | None = None,
                    has_image=lambda item: True, log=print) -> dict:
    """Every face Synology names a gallery person on where the latest run's best score is below `below`: one blind
    card each. A "someone else" answer becomes the owner's label, and the next plan unlinks Synology's tag."""
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("an edge round is already open; answer it or report it first")
    run = run or db.execute("SELECT run FROM decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    syn, _ = compare.synology_view(db, cfg)
    labels = D.latest_labels(db)
    busy = {r[0] for r in db.execute("SELECT face FROM cards WHERE answer IS NULL")}
    rows = [r for r in db.execute(
        "SELECT d.*, f.item FROM decisions d JOIN faces f ON f.id=d.face WHERE d.run=? AND d.status='abstained' "
        "AND d.score < ? AND (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?", (run, below, compare.MIN_H, compare.MIN_DET))
        if syn.get(r["face"]) not in (None, "other") and r["face"] not in labels and r["face"] not in busy
        and has_image(r["item"])]
    rng = np.random.default_rng(31)
    rid = f"lowscore-{D.now()[:10]}"
    N = Counter(syn[r["face"]] for r in rows)
    for j in rng.permutation(len(rows)):
        r = rows[j]
        x = syn[r["face"]]
        db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                   (rid, KIND, int(r["face"]), r["person"] or "", x, r["bin"],
                    json.dumps({"stratum": f"low:{x}", "N": N[x], "score": r["score"], "top": r["person"],
                                "runner": r["runner"], "reason": r["reason"]}), D.now()))
    db.commit()
    out = {"run": rid, "from decisions of": run, "below": below, "cards": len(rows), "by_person": dict(N.most_common())}
    log(json.dumps(out, indent=1))
    return out


# --- Outer-circle check: is a Synology-seeded gallery's pick right? -------------------
OUTER_BANDS = (("0.50-0.55", 0.50, 0.55, 20), ("0.55-0.62", 0.55, 0.62, 30), ("0.62+", 0.62, 9.0, 30))


def build_outer_check(db: sqlite3.Connection, cfg: Config, run: str | None = None, seed: int = 37,
                      has_image=lambda item: True, log=print) -> dict:
    """Blind cards on faces whose best match is an outer-circle person and where that would change Synology.
    Bands: 0.50-0.55 (accepted only without outer_floor_bump), 0.55-0.62, 0.62+. Each card offers four outer
    names in random order (the curator's pick, its outer runner-up, decoys) besides the close circle, so the pick
    stays hidden. Round-robin over people, one face per person and day."""
    if db.execute(f"SELECT COUNT(*) FROM cards WHERE kind='{KIND}' AND answer IS NULL").fetchone()[0]:
        raise SystemExit("an edge round is already open; answer it or report it first")
    run = run or db.execute("SELECT run FROM decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    outer = {p.key for p in cfg.outer_people}
    close = {p.key for p in cfg.close_people}
    key_of = {p.name: p.key for p in cfg.people.values()}
    labels = D.latest_labels(db)
    busy = {r[0] for r in db.execute("SELECT face FROM cards WHERE answer IS NULL")}
    margin = cfg["assign"]["default_margin"]
    rows = []
    for r in db.execute(
            "SELECT d.*, f.item, i.taken, sp.name AS syn FROM decisions d JOIN faces f ON f.id=d.face "
            "JOIN items i ON i.id=f.item LEFT JOIN syno_faces s ON s.face_id=f.syno_face_id "
            "LEFT JOIN syno_people sp ON sp.id=s.person_id WHERE d.run=? AND d.status IN ('accepted','abstained') "
            "AND d.score >= 0.50 AND d.score - d.runner_score >= ? AND (f.y2-f.y1)>=? AND COALESCE(f.det,1)>=?",
            (run, margin, compare.MIN_H, compare.MIN_DET)):
        cur = key_of.get(r["syn"] or "")
        if r["person"] in outer and cur != r["person"] and cur not in close and r["face"] not in labels \
                and r["face"] not in busy and has_image(r["item"]):
            rows.append((r, cur))
    rng = np.random.default_rng(seed)
    with_gallery = sorted({r["person"] for r, _ in rows})
    decoy_pool = sorted(outer)
    rid = f"outercheck-{D.now()[:10]}"
    out = {"run": rid, "from decisions of": run, "bands": {}}
    for band, lo, hi, quota in OUTER_BANDS:
        pool = [(r, c) for r, c in rows if lo <= r["score"] < hi]
        by_p: dict[str, list] = defaultdict(list)
        seen: set = set()
        for j in rng.permutation(len(pool)):
            r, c = pool[j]
            day = (r["person"], (r["taken"] or "")[:10])
            if day not in seen:
                seen.add(day)
                by_p[r["person"]].append((r, c))
        picked = []
        while len(picked) < quota and any(by_p.values()):
            for k in sorted(by_p, key=lambda _: rng.random()):
                if by_p[k] and len(picked) < quota:
                    picked.append(by_p[k].pop())
        for r, cur in picked:
            opts = [r["person"]] + ([r["runner"]] if r["runner"] in outer and r["runner"] != r["person"] else [])
            if cur in outer and cur not in opts:
                opts.append(cur)
            while len(opts) < 4:
                d = decoy_pool[int(rng.integers(len(decoy_pool)))]
                if d not in opts:
                    opts.append(d)
            rng.shuffle(opts)
            db.execute("INSERT INTO cards(run, kind, face, person, alt, bin, payload, created) VALUES(?,?,?,?,?,?,?,?)",
                       (rid, KIND, int(r["face"]), r["person"], cur or "other", r["bin"],
                        json.dumps({"stratum": f"outer:{band}", "N": len(pool), "score": r["score"],
                                    "runner": r["runner"], "options": opts}), D.now()))
        out["bands"][band] = {"pool": len(pool), "cards": len(picked), "people": len({r["person"] for r, _ in picked})}
    out["outer people with candidates"] = len(with_gallery)
    db.commit()
    log(json.dumps(out, indent=1))
    return out
