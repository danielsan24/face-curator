"""Offline tests: synthetic embeddings, a fake NAS behind the real synology-photos transport guard.

Run (versions pinned in requirements-test.txt; nothing here touches a NAS or a real ledger):

    cd scripts && uv run --no-project --with-requirements requirements-test.txt pytest -q test_facecurator.py

The fake NAS tests need the synology-photos skill (>= 0.1.0) at ~/.claude/skills/synology-photos, or
SYNOPHOTOS_PATH pointing at its scripts/synophotos.py; without it they are skipped. Every name, date and number
in the fixtures is made up.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fc import ages, assign, cards, changes, cluster, library  # noqa: E402
from fc import db as D  # noqa: E402
from fc import run as R  # noqa: E402
from fc.config import Config  # noqa: E402

SYNOPHOTOS = Path(os.environ.get("SYNOPHOTOS_PATH", "~/.claude/skills/synology-photos/scripts/synophotos.py")).expanduser()
RNG = np.random.default_rng(0)
DIM = 512


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_cfg(tmp: Path, **over) -> Config:
    raw = {
        "data_dir": str(tmp / "data"), "synology_photos": str(SYNOPHOTOS),
        "people": {"ada": {"name": "Ada", "born": "2016-03-09", "tier": "close"},
                   "bea": {"name": "Bea", "born": "2018-04-22", "tier": "close"},
                   "dad": {"name": "Dad", "tier": "close"},
                   "grandpa": {"name": "Grandpa", "died": "2020-02"}},
        "siblings": [["ada", "bea"]],
        "gallery": {"min_active": 5},
        "assign": {"min_calibration_accepts": 10, "impostor_sample": 50},
        "mode": {"policy": "auto"},
        "runner": {"require_ac": False, "require_lan": False, "index_wait_minutes": 0},
    }
    for k, v in over.items():
        raw[k] = v
    return Config(raw)


# --- synthetic library -----------------------------------------------------------------------------------
CENTERS = {k: unit(RNG.normal(size=DIM)) for k in ("ada", "bea", "dad", "other")}
CENTERS["bea"] = unit(CENTERS["ada"] * 0.55 + CENTERS["bea"] * 0.45)   # siblings look alike


def sample(k: str, n: int, noise: float = 0.028) -> np.ndarray:
    return unit(CENTERS[k][None, :] + RNG.normal(scale=noise, size=(n, DIM)))


SYNO_ID = {"ada": 101, "bea": 102, "dad": 103, "other": 104}


def build(db: sqlite3.Connection, spec: list[tuple[str, str, int | None]]) -> list[int]:
    """spec: (true person, capture day, Synology person id or None). One face per item. Returns face ids."""
    for name, pid in (("Ada", 101), ("Bea", 102), ("Dad", 103), ("Other", 104), ("", 900)):
        db.execute("INSERT OR REPLACE INTO syno_people VALUES(?,?,?,?)", (pid, name, 0, D.now()))
    fids = []
    base = db.execute("SELECT COALESCE(MAX(id), 999) + 1 FROM items").fetchone()[0] - 1000
    for n, (who, day, sp) in enumerate(spec, start=base):
        item = 1000 + n
        db.execute("INSERT INTO items(id, taken, type, last_seen, detected, faces_synced) VALUES(?,?,?,?,?,?)",
                   (item, day + "T12:00:00", "photo", D.now(), "test", D.now()))
        fid = db.execute("INSERT INTO faces(item,x1,y1,x2,y2,det,source,syno_face_id,iou) VALUES(?,?,?,?,?,?,?,?,?)",
                         (item, 0.4, 0.2, 0.5, 0.35, 0.9, "test", 50000 + n, 0.9)).lastrowid
        db.execute("INSERT INTO syno_faces VALUES(?,?,?,?,?,?,?,?)", (50000 + n, item, sp, 0.4, 0.2, 0.5, 0.35, D.now()))
        db.execute("INSERT INTO embeddings VALUES(?,?,?)", (fid, "buffalo_l", D.pack(sample(who, 1)[0])))
        fids.append(fid)
    db.commit()
    return fids


def days(start: str, n: int, step: int = 3) -> list[str]:
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i * step)).isoformat() for i in range(n)]


@pytest.fixture
def env(tmp_path):
    cfg = make_cfg(tmp_path)
    db = D.connect(D.ledger_path(cfg.data_dir))
    return cfg, db


def seed_library(db, labelled: int = 12):
    """Ada at ~2-4y, Bea at 0-6m/6-12m, Dad; owner labels on the first `labelled` of each."""
    spec = ([("ada", d, 101) for d in days("2018-11-21", 40)] + [("bea", d, 102) for d in days("2018-09-17", 40)]
            + [("dad", d, 103) for d in days("2018-11-18", 40)] + [("other", d, 104) for d in days("2018-12-18", 20)])
    fids = build(db, spec)
    who = [s[0] for s in spec]
    for k in ("ada", "bea", "dad"):
        for f in [f for f, w in zip(fids, who) if w == k][:labelled]:
            D.add_label(db, f, k, via="test")
    db.commit()
    return fids, who


# --- ages ------------------------------------------------------------------------------------------------
def test_age_bins():
    e = [0, 6, 12, 24, 48, 72, 96]
    from datetime import date
    born = date(2016, 3, 9)
    assert ages.bin_index(ages.months(born, "2016-03-16"), e) == 0
    assert ages.bin_index(ages.months(born, "2019-04-17"), e) == 3
    assert ages.bin_index(ages.months(born, "2015-11-17"), e) is None          # before birth
    assert ages.bin_label(6, e) == "8y+" and ages.bin_label(2, e) == "12-24m"
    assert ages.window(1, e, 2) == (4, 14)


# --- geometry / detector decode --------------------------------------------------------------------------
def test_umeyama_recovers_similarity():
    from fc.embed import ARCFACE_DST, umeyama
    th, s, t = 0.3, 1.7, np.array([12.0, -4.0])
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * s
    src = ARCFACE_DST.astype(np.float64)
    dst = src @ Rm.T + t
    M = umeyama(src, dst)
    assert np.allclose(M[:, :2], Rm, atol=1e-6) and np.allclose(M[:, 2], t, atol=1e-6)


def test_scrfd_decode_single_face():
    from fc.embed import decode
    S, outs = 640, []
    sizes = [(S // s) ** 2 * 2 for s in (8, 16, 32)]
    scores = [np.zeros((n, 1), np.float32) for n in sizes]
    bbox = [np.zeros((n, 4), np.float32) for n in sizes]
    kps = [np.zeros((n, 10), np.float32) for n in sizes]
    # anchor index for stride 16 at grid (x=10, y=5): row-major y*w + x, two anchors per cell
    w = S // 16
    a = (5 * w + 10) * 2
    scores[1][a] = 0.9
    scores[1][a + 1] = 0.85                       # duplicate anchor, removed by NMS
    bbox[1][a] = bbox[1][a + 1] = [1, 1, 1, 1]    # distances in stride units
    outs = scores + bbox + kps
    dets, k = decode(outs, S, det_scale=0.5, thresh=0.5)
    assert len(dets) == 1
    cx, cy = 10 * 16, 5 * 16
    assert np.allclose(dets[0, :4], np.array([cx - 16, cy - 16, cx + 16, cy + 16]) / 0.5)
    assert np.allclose(k[0][0], [cx / 0.5, cy / 0.5])


def test_match_boxes_one_to_one():
    local = [(1, (0.1, 0.1, 0.2, 0.2)), (2, (0.5, 0.5, 0.6, 0.6))]
    remote = [(10, (0.11, 0.1, 0.21, 0.2)), (20, (0.52, 0.5, 0.62, 0.61)), (30, (0.8, 0.8, 0.9, 0.9))]
    m = library.match_boxes(local, remote)
    assert m[1][0] == 10 and m[2][0] == 20 and len(m) == 2


# --- import ----------------------------------------------------------------------------------------------
def test_import_face_cache(tmp_path):
    import base64
    cfg = make_cfg(tmp_path)
    fp = tmp_path / "fp" / "cache" / "faces"
    fp.mkdir(parents=True)
    (tmp_path / "fp" / "cache" / "ingest_index.jsonl").write_text(
        json.dumps({"syno_id": "77", "photo_id": "t77", "file": "originals/a.jpg", "taken_at": "2022-01-01T10:00:00"}) + "\n"
        + json.dumps({"syno_id": "5", "photo_id": "abcdef0123456789", "file": "x"}) + "\n")
    emb = base64.b64encode(sample("ada", 1)[0].astype(np.float16).tobytes()).decode()
    (fp / "faces.jsonl").write_text(json.dumps({"photo_id": "t77", "faces": [{"bbox": [0.1, 0.1, 0.3, 0.3], "det": 0.9, "emb": emb}]}) + "\n")
    db = D.connect(D.ledger_path(cfg.data_dir))
    out = library.import_face_cache(db, tmp_path / "fp", log=lambda *_: None)
    assert out == {"items": 1, "faces": 1, "skipped_other_ids": 1}
    library.import_face_cache(db, tmp_path / "fp", log=lambda *_: None)                 # idempotent
    assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    assert db.execute("SELECT detected, taken FROM items WHERE id=77").fetchone()[:] == ("face-cache", "2022-01-01T10:00:00")
    with pytest.raises(SystemExit, match="cache/ingest_index.jsonl"):
        library.import_face_cache(db, tmp_path / "nothing-here", log=lambda *_: None)


# --- assignment ------------------------------------------------------------------------------------------
def test_assignment_accepts_and_respects_birth(env):
    cfg, db = env
    fids, who = seed_library(db)
    th = assign.Thresholds(0.5, 0.05, 0.08)
    labels = D.latest_labels(db)
    gal = assign.build_galleries(db, cfg, "buffalo_l", labels)
    fs = assign.load_faceset(db, "buffalo_l")
    keys, S, C = assign.score(fs, gal, cfg)
    decs = {d.face: d for d in assign.decide(fs, keys, S, C, cfg, th)}
    unl = [f for f, w in zip(fids, who) if f not in labels and w in ("ada", "bea", "dad")]
    acc = [decs[f] for f in unl if decs[f].status == "accepted"]
    assert len(acc) > 0.8 * len(unl)
    truth = dict(zip(fids, who))
    assert all(d.person == truth[d.face] for d in acc)
    # "other" faces are never accepted as anybody
    assert all(decs[f].status == "abstained" for f, w in zip(fids, who) if w == "other")
    # Bea cannot be scored before her birth
    fs2 = assign.FaceSet(fs.ids[:1], sample("bea", 1), np.array(["2017-11-17"], dtype=object))
    k2, S2, _ = assign.score(fs2, gal, cfg)
    assert not np.isfinite(S2[0, k2.index("bea")])


def test_thin_sibling_gallery_abstains(env):
    cfg, db = env
    fids, who = seed_library(db, labelled=12)
    # remove Bea's labels: an Ada-looking face dated when both children exist must abstain
    for f, w in zip(fids, who):
        if w == "bea":
            D.add_label(db, f, "unknown", via="test")
    db.commit()
    gal = assign.build_galleries(db, cfg, "buffalo_l")
    fs = assign.FaceSet(np.array([999999]), sample("ada", 1), np.array(["2019-01-15"], dtype=object))
    keys, S, C = assign.score(fs, gal, cfg)
    d = assign.decide(fs, keys, S, C, cfg, assign.Thresholds(0.3, 0.0, 0.0))[0]
    assert d.status == "abstained" and d.reason.startswith("thin-sibling-gallery")


def test_calibration_meets_target(env):
    cfg, db = env
    seed_library(db, labelled=30)
    th = assign.calibrate(db, cfg, "buffalo_l", assign.build_galleries(db, cfg, "buffalo_l"), D.latest_labels(db))
    assert th.status == "calibrated"
    assert th.stats["precision"] >= 0.995 and th.stats["accepted"] >= 10
    assert th.floor == max(round(th.stats["raw_floor"] + cfg["assign"]["safety_bump"], 3), cfg["assign"]["min_floor"])


def test_wilson_zero_errors():
    assert abs(assign.wilson_lower(598, 598) - 0.995) < 0.0001


# --- clustering ------------------------------------------------------------------------------------------
def test_cluster_schedule_keeps_strict_first():
    big = sample("dad", 40)
    small = sample("other", 4)
    noise = unit(RNG.normal(size=(5, DIM)))
    vecs = np.vstack([big, small, noise]).astype(np.float32)
    labels, passes = cluster.cluster(vecs, 0.4, [25, 10, 5, 3])
    assert len(set(labels[:40])) == 1 and labels[0] >= 0 and passes[0] == 0       # dense: first pass (25)
    assert len(set(labels[40:44])) == 1 and labels[40] != labels[0] and passes[40] == 3  # 4 faces: only at k=3
    assert (labels[44:] == -1).all()


# --- change sets -----------------------------------------------------------------------------------------
def _decide_all(db, cfg, run="2026-11"):
    D.set_run(db, run, stage="assign")
    return assign.run_assignment(db, cfg, run, log=lambda *_: None)


def test_plan_reassign_and_merge(env):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    # Synology mistakes: 3 Ada faces filed under Bea; an unnamed group of 10 Ada faces; one wrong face in it.
    ada_unl = [f for f, w in zip(fids, who) if w == "ada"][30:40]
    for f in ada_unl[:3]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    grp = build(db, [("ada", d, 900) for d in days("2018-12-18", 20, step=2)] + [("other", "2019-01-29", 900)])
    db.commit()
    _decide_all(db, cfg)
    res = changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    rows = db.execute("SELECT kind, face, group_id, from_person, to_person FROM changes WHERE run='2026-11' ORDER BY seq").fetchall()
    kinds = [r["kind"] for r in rows]
    re = [r for r in rows if r["kind"] == "reassign" and r["group_id"] is None]
    assert {r["face"] for r in re} >= set(ada_unl[:3]) and all(r["to_person"] == 101 for r in re)
    assert res["groups_to_merge"] == 1
    gu = [r for r in rows if r["kind"] == "group-unlink"]
    assert [r["face"] for r in gu] == [grp[-1]]                     # the stranger is unlinked before the merge
    assert kinds.index("group-unlink") < kinds.index("merge") and kinds[-1] == "merge"


def test_caps(env):
    cfg = make_cfg(Path(env[0].raw["data_dir"]).parent, caps={"face_edits": 2, "merges": 0})
    db = env[1]
    fids, who = seed_library(db, labelled=30)
    for f in [f for f, w in zip(fids, who) if w == "ada"][30:36]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    st = dict(db.execute("SELECT status, COUNT(*) FROM changes WHERE run='2026-11' GROUP BY status").fetchall())
    assert st["planned"] == 2 and st["capped"] >= 4


def test_caps_reserve_merge_cleaning_and_requeue(env):
    cfg = make_cfg(Path(env[0].raw["data_dir"]).parent, caps={"face_edits": 2, "merges": 1})
    db = env[1]
    fids, who = seed_library(db, labelled=30)
    for f in [f for f, w in zip(fids, who) if w == "ada"][30:36]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    grp = build(db, [("ada", d, 900) for d in days("2018-12-18", 20, step=2)] + [("other", "2019-01-29", 900)])
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    q = "SELECT kind, status, COUNT(*) FROM changes WHERE run='2026-11' GROUP BY kind, status"
    st = {(k, s): n for k, s, n in db.execute(q)}
    # the merge's one cleaning edit is reserved: merge + group-unlink planned, one 1:1 reassign left in the budget
    assert st[("merge", "planned")] == 1 and st[("group-unlink", "planned")] == 1
    assert st[("reassign", "planned")] == 1 and st[("reassign", "capped")] >= 5
    # what the first planner left behind: cleaning capped, merge stale at apply; requeue brings both back
    db.execute("UPDATE changes SET status='capped' WHERE kind='group-unlink'")
    db.execute("UPDATE changes SET status='stale', detail='1 cleaning change(s) not done' WHERE kind='merge'")
    db.execute("UPDATE changes SET status='done' WHERE status='planned'")
    db.commit()
    res = changes.requeue_merges(db, cfg, "2026-11", log=lambda *_: None)
    assert res == {"merges": 1, "cleaning": 1, "left": 0}
    st = {(k, s): n for k, s, n in db.execute(q)}
    assert st[("merge", "planned")] == 1 and st[("group-unlink", "planned")] == 1 and ("reassign", "planned") not in st
    assert json.loads(db.execute("SELECT detail FROM changes WHERE kind='merge'").fetchone()[0])["faces"] == len(grp)
    with pytest.raises(SystemExit):                                   # planned rows waiting: refuse
        changes.requeue_merges(db, cfg, "2026-11", log=lambda *_: None)


def test_outer_circle_gallery_from_synology_tags(env):
    base = env[0]
    people = {**base.raw["people"], "other": {"name": "Other", "tier": "outer"}}
    cfg = make_cfg(Path(base.raw["data_dir"]).parent, people=people)
    db = env[1]
    assert cfg.people["other"].outer and cfg.people["other"].gallery and cfg.people["ada"].tier == "close"
    assert [p.key for p in cfg.close_people] == ["ada", "bea", "dad"] and not cfg.people["grandpa"].gallery
    fids, who = seed_library(db, labelled=12)
    wrong = build(db, [("ada", d, 104) for d in days("2019-03-17", 3)])          # Synology tags 3 Ada faces "Other"
    unnamed = build(db, [("other", d, 900) for d in days("2019-04-17", 6)])       # Other's faces in an unnamed group
    gals = assign.build_galleries(db, cfg, "buffalo_l")
    seeded = {int(f) for f in gals["other"].faces}
    assert len(seeded) >= 8 and not seeded & set(wrong)                            # mis-tags stay out
    assert len(gals["ada"].faces) == 12                                         # owner galleries: labels only
    _decide_all(db, cfg)
    dec = {r["face"]: r for r in db.execute("SELECT * FROM decisions WHERE run='2026-11'")}
    assert all(dec[f]["status"] == "accepted" and dec[f]["person"] == "other" for f in unnamed)
    assert all(dec[f]["person"] == "ada" for f in wrong)                           # the mis-tags go back to Ada
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    to = {r["face"]: r["to_person"] for r in db.execute("SELECT face, to_person FROM changes WHERE run='2026-11'")}
    merged = db.execute("SELECT to_person FROM changes WHERE run='2026-11' AND kind='merge' AND group_id=900").fetchone()
    assert (merged and merged[0] == 104) or all(to.get(f) == 104 for f in unnamed)   # whole group merged, or 1:1


def test_outer_never_takes_from_close_and_close_goes_first(env):
    base = env[0]
    people = {**base.raw["people"], "other": {"name": "Other", "tier": "outer"}}
    cfg = make_cfg(Path(base.raw["data_dir"]).parent, people=people, caps={"face_edits": 1, "merges": 20})
    db = env[1]
    fids, who = seed_library(db, labelled=12)
    on_dad = build(db, [("other", d, 103) for d in days("2019-05-17", 2)])          # Other's faces tagged as Dad (close)
    fix = build(db, [("dad", d, 101) for d in days("2019-06-17", 1)])               # close -> close fix (Dad tagged Ada)
    no_one = build(db, [("other", d, None) for d in days("2019-07-18", 2)])         # outer: Synology face, no person
    db.execute("UPDATE syno_faces SET person_id=NULL WHERE item IN (SELECT item FROM faces WHERE id IN (?,?))", no_one)
    db.commit()
    _decide_all(db, cfg)
    dec = {r["face"]: r for r in db.execute("SELECT * FROM decisions WHERE run='2026-11'")}
    assert all(dec[f]["person"] == "other" and dec[f]["status"] == "accepted" for f in on_dad)
    res = changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    rows = {r["face"]: r for r in db.execute("SELECT * FROM changes WHERE run='2026-11'")}
    assert not set(on_dad) & set(rows) and res["outer_kept_off_close"] >= 2      # Dad keeps them; no write
    first = db.execute("SELECT person, status FROM changes WHERE run='2026-11' ORDER BY seq LIMIT 1").fetchone()
    assert first["status"] == "planned" and cfg.people[first["person"]].tier == "close"   # close first under the cap
    assert all(rows[f]["status"] == "capped" for f in no_one if f in rows)
    assert set(res["by_tier"]) == {"close", "outer"}
    # owner tools see only the close circle
    from fc import compare
    assert "other" not in [p.key for p in cfg.close_people]


def test_outer_check_cards(env):
    from fc import compare, edges
    base = env[0]
    people = {**base.raw["people"], "o1": {"name": "Other", "tier": "outer"},
              "x1": {"name": "X1", "tier": "outer"}, "x2": {"name": "X2", "tier": "outer"}, "x3": {"name": "X3", "tier": "outer"},
              "x4": {"name": "X4", "tier": "outer"}}
    cfg = make_cfg(Path(base.raw["data_dir"]).parent, people=people)
    db = env[1]
    seed_library(db, labelled=12)
    unnamed = build(db, [("other", d, 900) for d in days("2019-04-17", 6)])
    _decide_all(db, cfg)
    db.execute("UPDATE decisions SET score=0.60 WHERE face IN (%s)" % ",".join("?" * len(unnamed)), unnamed)
    db.commit()
    res = edges.build_outer_check(db, cfg, run="2026-11", log=lambda *_: None)
    cards = db.execute("SELECT * FROM cards WHERE run=?", (res["run"],)).fetchall()
    assert cards and {c["face"] for c in cards} <= set(unnamed) and res["bands"]["0.55-0.62"]["cards"] == len(cards)
    for c in cards:
        opts = compare.card_options(c)
        assert len(opts) == 4 and c["person"] in opts and all(cfg.people[k].outer for k in opts)
    from fc.review import App
    app = App(cfg, D.ledger_path(cfg.data_dir))
    shown = app.compare_cards()["cards"][0]
    assert set(shown) == {"id", "face", "day", "options"}                       # blind: no pick, no Synology name
    compare.answer(db, cfg, cards[0]["id"], cards[0]["person"])
    compare.answer(db, cfg, cards[1]["id"], "other")
    lab = D.latest_labels(db)
    assert lab[cards[0]["face"]] == "o1" and set(lab[cards[1]["face"]][4:].split(",")) >= set(compare.card_options(cards[1]))
    with pytest.raises(ValueError):
        compare.answer(db, cfg, cards[2]["id"], [k for k in ("x1", "x2", "x3", "x4", "o1")
                                                 if k not in compare.card_options(cards[2])][0])
    rep = edges.report(db, cfg, res["run"])
    assert rep["strata"]["outer:0.55-0.62"]["curator right"].startswith("1/2")


# --- fake NAS --------------------------------------------------------------------------------------------
class FakeClient:
    """In-memory Synology: faces per item. Every call goes through the real synophotos transport guard."""

    def __init__(self, sp, db):
        self.sp, self.calls = sp, []
        self.people = {r[0]: r[1] for r in db.execute("SELECT id, name FROM syno_people")}   # live names
        self.fail = None                               # callable(method, params) -> True to make that call fail
        self.faces = {}
        for r in db.execute("SELECT * FROM syno_faces"):
            self.faces.setdefault(r["item"], []).append(
                {"face_id": r["face_id"], "person_id": r["person_id"] or 0, "name": "",
                 "face_bounding_box": library.box_dict(r["x1"], r["y1"], r["x2"], r["y2"])})
        self.next_id = 900000

    def ok(self, api, method, version, **p):
        self.sp.SynoClient.check_allowed(api, method)
        if self.fail and self.fail(method, p):
            raise RuntimeError(f"NAS refused {method}")
        self.calls.append((method, p))
        if method == "list_face":
            return {"list": [dict(f) for f in self.faces.get(p["id_item"], [])]}
        if method == "delete_face":
            for fs in self.faces.values():
                for f in fs:
                    if f["face_id"] in p["face_id"] and f["person_id"] == p["person_id"]:
                        f["person_id"] = 0
            return {}
        if method == "add_face":
            spec = p["face"][0]
            fs = self.faces.setdefault(p["id_item"], [])
            b = library.syno_box(spec["face_bounding_box"])
            fs[:] = [f for f in fs if not (f["person_id"] == 0 and library.iou(library.syno_box(f["face_bounding_box"]), b) > 0.9)]
            self.next_id += 1
            fs.append({"face_id": self.next_id, "person_id": spec["person_id"], "name": "",
                       "face_bounding_box": spec["face_bounding_box"]})
            return {}
        if method == "merge":
            for fs in self.faces.values():
                for f in fs:
                    if f["person_id"] in p["merged_id"]:
                        f["person_id"] = p["target_id"]
            for g in p["merged_id"]:
                self.people.pop(g, None)
            return {"id": p["target_id"]}
        raise AssertionError(f"unexpected call {method}")

    def pages(self, api, method, version, limit=None, **p):
        if method == "list" and "person_id" in p:
            return iter([{"id": it} for it, fs in self.faces.items() if any(f["person_id"] == p["person_id"] for f in fs)])
        if method == "list" and api.endswith(".Person"):
            return iter([{"id": i, "name": n, "item_count": 0} for i, n in self.people.items()])
        return iter([])

    def index_status(self, api=None):
        return "idle"

    def logout(self):
        pass


@pytest.fixture
def sp(tmp_path, monkeypatch):
    if not SYNOPHOTOS.is_file():
        pytest.skip("synology-photos skill not found")
    mod = library.load_synophotos(SYNOPHOTOS)
    monkeypatch.setattr(mod, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    return mod


def test_apply_reassign_merge_and_guard(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    wrong = [f for f, w in zip(fids, who) if w == "ada"][30:33]
    for f in wrong:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    build(db, [("ada", d, 900) for d in days("2018-12-18", 20, step=2)] + [("other", "2019-01-29", 900)])
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    res = changes.apply(db, cfg, nas, "2026-11", log=lambda *_: None)
    assert res.get("failed", 0) == 0 and res["done"] >= 5
    methods = [m for m, _ in fake.calls if m != "list_face"]
    assert methods.count("merge") == 1 and "delete_face" in methods and "add_face" in methods
    # every wrong face is now Ada's, and the group was merged into Ada
    for f in wrong:
        item = db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0]
        assert [x["person_id"] for x in fake.faces[item]] == [101]
    assert not any(x["person_id"] == 900 for fs in fake.faces.values() for x in fs)
    assert db.execute("SELECT COUNT(*) FROM writes WHERE ok=1").fetchone()[0] == len(methods)
    lines = (Path(sp.JOURNAL_PATH)).read_text().splitlines()
    assert any(json.loads(x)["action"] == "merge" for x in lines)
    with pytest.raises(sp.Refused):
        sp.SynoClient.check_allowed("SYNO.FotoTeam.Browse.Item", "delete")


def test_apply_skips_stale(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    _decide_all(db, cfg)       # decisions use embeddings only; pick a face the curator accepts as Ada
    acc = {r[0] for r in db.execute("SELECT face FROM decisions WHERE status='accepted' AND person='ada'")}
    f = [f for f, w in zip(fids, who) if w == "ada" and f in acc][0]
    db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    fake = FakeClient(sp, db)
    item = db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0]
    fake.faces[item][0]["person_id"] = 103          # someone changed it in Photos since the sync
    res = changes.apply(db, cfg, library.Nas(cfg, sp, fake), "2026-11", log=lambda *_: None)
    assert res.get("stale") == 1
    assert not [m for m, _ in fake.calls if m != "list_face"]


def test_undo_restores(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    f = [f for f, w in zip(fids, who) if w == "ada"][31]
    db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    changes.apply(db, cfg, nas, "2026-11", log=lambda *_: None)
    item = db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0]
    assert fake.faces[item][0]["person_id"] == 101
    changes.plan_undo(db, "2026-11", "undo-2026-11")
    changes.apply(db, cfg, nas, "undo-2026-11", log=lambda *_: None)
    assert [x["person_id"] for x in fake.faces[item]] == [102]


# --- cards -----------------------------------------------------------------------------------------------
def test_cards_and_error_bump(env):
    cfg, db = env
    seed_library(db, labelled=12)
    _decide_all(db, cfg)
    made = cards.monthly(db, cfg, "2026-11", seed=1, log=lambda *_: None)
    assert 0 < sum(made.values()) <= 20
    c = db.execute("SELECT * FROM cards WHERE kind='random' LIMIT 1").fetchone()
    out = cards.answer(db, cfg, c["id"], "b")
    assert out["error"] and db.execute("SELECT amount, staged FROM bumps").fetchone()[:] == (0.03, 1)
    assert D.latest_labels(db)[c["face"]] == c["alt"]
    assert cards.audit_result(db, "2026-11") == {"judged": 1, "errors": 1}
    n = cards.audit(db, cfg, "2026-11", n=15)
    assert n == 15


def test_gallery_candidates_diverse(env):
    cfg, db = env
    fids, who = seed_library(db, labelled=5)
    cand = cards.gallery_candidates(db, cfg, "ada", 3, 8, "buffalo_l", lambda it: True)
    assert len(cand) == 8 and len({c["day"] for c in cand}) == 8
    assert not set(c["face"] for c in cand) & set(D.latest_labels(db))


# --- run -------------------------------------------------------------------------------------------------
class FakeNasCtx:
    def __init__(self, nas):
        self.nas = nas

    def __call__(self):
        return self

    def __enter__(self):
        return self.nas

    def __exit__(self, *a):
        pass


def _fake_nas(cfg, sp, db):
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    nas.people = lambda: [{"id": i, "name": n, "item_count": 0} for i, n in fake.people.items()]
    nas.folder_children = lambda fid: []
    nas.folder_items = lambda fid: []
    nas.faces = lambda item: fake.ok("SYNO.FotoTeam.Browse.Item", "list_face", 7, id_item=item)["list"]
    nas.__enter__ = lambda: nas
    return nas, fake


def earn_auto(db, run="2026-11", prev=("2026-08", "2026-09", "2026-10"), calibrated=True):
    """Everything `auto_gate` asks for: three done runs with writes, 20 clean cards each, a clean one-off audit of
    200, a calibration with a high lower bound."""
    for r in prev:
        D.set_run(db, r, status="done", mode="staged")
        db.execute("INSERT INTO writes(change, run, at, api, method, ok) VALUES(0,?,?,?,?,1)", (r, D.now(), "x", "add_face"))
        for k in range(20):
            db.execute("INSERT INTO cards(run, kind, face, answer, created) VALUES(?,?,?,?,?)", (r, "random", k, "a", D.now()))
    for k in range(200):
        db.execute("INSERT INTO cards(run, kind, face, answer, created, payload) VALUES(?,?,?,?,?,?)",
                   (prev[-1], "audit", k, "a", "2026-10-20T10:00:00", json.dumps({"batch": "2026-10-20T10:00:00"})))
    if calibrated:
        db.execute("INSERT OR REPLACE INTO calibration(run, model, status, lower95) VALUES(?,?,?,?)",
                   (run, "buffalo_l", "calibrated", 0.995))
    db.commit()


def test_run_is_staged_and_auto_cannot_be_forced(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    f = [f for f, w in zip(fids, who) if w == "ada"][31]
    db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    nas, fake = _fake_nas(cfg, sp, db)
    ctx = FakeNasCtx(nas)
    eng = SimpleNamespace(models=["buffalo_l"])
    R.execute(cfg, "2026-11", nas_factory=ctx, engine=eng, log=lambda *_: None)
    row = D.run_row(db, "2026-11")
    assert row["status"] == "done" and row["mode"] == "staged"            # policy auto, but nothing earned yet
    assert not [m for m, _ in fake.calls if m not in ("list_face",)]
    assert Path(row["report"]).is_file()
    assert (cfg.data_dir / "exports" / "2026-11" / "labels.jsonl.gz").is_file()
    assert "Auto mode was not used" in Path(row["report"]).read_text()
    # asking for auto does not skip the gate: the run stays staged and writes nothing
    R.execute(cfg, "2026-12", mode="auto", nas_factory=ctx, engine=eng, log=lambda *_: None)
    assert D.run_row(db, "2026-12")["mode"] == "staged"
    assert not [m for m, _ in fake.calls if m != "list_face"]
    assert D.run_stats(db, "2026-12")["mode_why"]
    with pytest.raises(SystemExit):
        R.execute(cfg, "2027-01", mode="yolo", nas_factory=ctx, engine=eng, log=lambda *_: None)
    import facecurator
    with pytest.raises(SystemExit):                                         # the flag can only force staged
        facecurator.parser().parse_args(["run", "--mode", "auto"])
    assert facecurator.parser().parse_args(["run", "--mode", "staged"]).mode == "staged"


def test_run_auto_once_earned_then_demoted_by_verify_mismatch(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    f = [f for f, w in zip(fids, who) if w == "ada"][31]
    db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    nas, fake = _fake_nas(cfg, sp, db)
    ctx, eng = FakeNasCtx(nas), SimpleNamespace(models=["buffalo_l"])
    R.execute(cfg, "2026-11", until="plan", nas_factory=ctx, engine=eng, log=lambda *_: None)
    earn_auto(db, "2026-11")
    assert R.auto_gate(db, cfg, "2026-11") == []
    R.execute(cfg, "2026-11", nas_factory=ctx, engine=eng, log=lambda *_: None)
    assert D.run_row(db, "2026-11")["mode"] == "auto" and "add_face" in [m for m, _ in fake.calls]
    assert D.run_stats(db, "2026-11")["verify"]["mismatch"] == 0 and cards.demotion(db) is None
    # a verify mismatch after writes demotes auto mode, in the ledger, and the report says so
    item = db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0]
    fake.faces[item][0]["person_id"] = 103
    lines = []
    R.after_apply(db, cfg, nas, "2026-11", {}, log=lines.append)
    assert cards.demotion(db)["run"] == "2026-11" and "demoted" in " ".join(lines)
    D.set_run(db, "2026-11", status="done")
    earn_auto(db, "2026-12", prev=("2026-09", "2026-10", "2026-11"))
    why = R.auto_gate(db, cfg, "2026-12")
    assert len(why) == 1 and "demoted" in why[0] and R.choose_mode(db, cfg, "2026-12") == "staged"
    from fc import report
    assert "Auto mode is demoted" in report.write(db, cfg, "2026-11").read_text()
    # only a new one-off audit, complete and clean, re-earns it
    for k in range(200):
        db.execute("INSERT INTO cards(run, kind, face, answer, created, payload) VALUES(?,?,?,?,?,?)",
                   ("2026-11", "audit", 5000 + k, "a" if k else None, "2999-01-01T00:00:00",
                    json.dumps({"batch": "2999-01-01T00:00:00"})))
    db.commit()
    assert any("demoted" in w for w in R.auto_gate(db, cfg, "2026-12"))        # one card still open
    db.execute("UPDATE cards SET answer='a' WHERE answer IS NULL AND kind='audit'")
    db.commit()
    assert R.auto_gate(db, cfg, "2026-12") == []


def test_auto_gate_needs_every_condition(env):
    cfg, db = env
    assert cfg.mode["policy"] == "auto"
    earn_auto(db)
    assert R.auto_gate(db, cfg, "2026-11") == [] and R.choose_mode(db, cfg, "2026-11") == "auto"
    assert R.choose_mode(db, cfg, "2026-11", "staged") == "staged"

    def broken(sql, *params, undo):
        db.execute(sql, params)
        db.commit()
        why = R.auto_gate(db, cfg, "2026-11")
        db.execute(undo, params) if "?" in undo else db.execute(undo)
        db.commit()
        assert R.auto_gate(db, cfg, "2026-11") == [], undo
        return " ".join(why)

    # 1 the policy must be set in the config; the default is staged
    assert R.auto_gate(db, Config({"data_dir": cfg.raw["data_dir"]}), "2026-11") == ["mode.policy is not auto in the config"]
    # 3 done runs count only if changes were written
    assert "2 earlier run(s) done with their face edits applied" in broken(
        "UPDATE writes SET ok=0 WHERE run='2026-08'", undo="UPDATE writes SET ok=1 WHERE run='2026-08'")
    # 4 the one-off audit: complete, at least 200 judged, no error
    assert "open card" in broken("UPDATE cards SET answer=NULL WHERE kind='audit' AND face=0",
                                 undo="UPDATE cards SET answer='a' WHERE kind='audit' AND face=0")
    assert "199 judged" in broken("UPDATE cards SET answer='unknown' WHERE kind='audit' AND face=0",
                                  undo="UPDATE cards SET answer='a' WHERE kind='audit' AND face=0")
    assert "found 1 error" in broken("UPDATE cards SET answer='b' WHERE kind='audit' AND face=0",
                                     undo="UPDATE cards SET answer='a' WHERE kind='audit' AND face=0")
    # 5 the last two runs: at least 20 judged cards each, no error (five clean cards are not an audit)
    assert "run 2026-09: 19 judged" in broken("UPDATE cards SET answer='skip' WHERE run='2026-09' AND face=0",
                                              undo="UPDATE cards SET answer='a' WHERE run='2026-09' AND face=0")
    assert "run 2026-10: 1 card error" in broken("UPDATE cards SET answer='neither' WHERE run='2026-10' AND kind='random' AND face=0",
                                                 undo="UPDATE cards SET answer='a' WHERE run='2026-10' AND kind='random' AND face=0")
    # 6 calibrated, with a lower 95% bound above the threshold (30 clean accepts give 0.886: not enough)
    assert "below 0.99" in broken("UPDATE calibration SET lower95=0.886", undo="UPDATE calibration SET lower95=0.995")
    assert "not calibrated" in broken("UPDATE calibration SET status='default'", undo="UPDATE calibration SET status='calibrated'")
    # the floors are in code: a config cannot lower them, and cannot raise the ceilings or the caps
    loose = make_cfg(Path(cfg.raw["data_dir"]).parent, mode={"policy": "auto", "staged_runs_before_auto": 0,
                     "clean_audits_for_auto": 0, "audit_min_judged": 1, "min_judged_per_run": 1, "min_lower95": 0.5,
                     "auto_max_face_edits": 10**6, "auto_max_merges": 10**6},
                     caps={"face_edits": 10**9, "merges": 10**9, "merge_auto_max_faces": 10**9})
    assert loose.mode == {"policy": "auto", "staged_runs_before_auto": 3, "clean_audits_for_auto": 2,
                          "audit_min_judged": 200, "min_judged_per_run": 20, "min_lower95": 0.98,
                          "auto_max_face_edits": 500, "auto_max_merges": 5}
    assert loose.caps == {"face_edits": 10000, "merges": 50, "merge_auto_max_faces": 200}
    # a NaN never passes for a number, and a face edit left planned means the run was not applied
    with pytest.raises(SystemExit, match="not a number"):
        make_cfg(Path(cfg.raw["data_dir"]).parent, mode={"policy": "auto", "min_lower95": float("nan")}).mode
    for bad in ({"mode": {"audit_min_judged": float("nan")}}, {"mode": {"auto_max_merges": float("inf")}},
                {"caps": {"merges": float("inf")}}, {"caps": {"face_edits": "many"}}):
        with pytest.raises(SystemExit, match="not a number"):
            c = make_cfg(Path(cfg.raw["data_dir"]).parent, **bad)
            c.mode, c.caps
    db.execute("INSERT INTO changes(run, seq, kind, status) VALUES ('2026-08', 999, 'assign', 'planned')")
    db.commit()
    assert "2 earlier run(s) done with their face edits applied" in " ".join(R.auto_gate(db, cfg, "2026-11"))
    db.execute("DELETE FROM changes WHERE run='2026-08' AND seq=999")
    db.commit()
    assert R.auto_gate(db, cfg, "2026-11") == []
    assert Config({}).mode["policy"] == "staged" and Config({}).caps["face_edits"] == 3000


def test_audit_error_demotes_and_undo_of_the_answer_lifts_it(env):
    cfg, db = env
    seed_library(db, labelled=12)
    _decide_all(db, cfg)
    assert cards.audit(db, cfg, "2026-11", n=10) == 10
    c = db.execute("SELECT * FROM cards WHERE kind='audit' LIMIT 1").fetchone()
    assert json.loads(c["payload"])["batch"] and cards.audit_state(db)["open"] == 10
    assert cards.answer(db, cfg, c["id"], "neither")["error"]
    d = cards.demotion(db)
    assert d["card"] == c["id"] and cards.audit_state(db)["errors"] == 1
    cards.retract(db, cfg, db.execute("SELECT * FROM cards WHERE id=?", (c["id"],)).fetchone())
    db.commit()
    assert cards.demotion(db) is None and cards.audit_state(db)["errors"] == 0


def test_auto_apply_has_hard_ceilings_and_leaves_outer_and_large_merges(env, sp):
    base, db = env
    people = {**base.raw["people"], "other": {"name": "Other", "tier": "outer"}}
    cfg = make_cfg(Path(base.raw["data_dir"]).parent, people=people,
                   mode={"policy": "auto", "auto_max_face_edits": 2, "auto_max_merges": 5},
                   caps={"face_edits": 3000, "merges": 20, "merge_auto_max_faces": 10})
    fids, who = seed_library(db, labelled=30)
    for f in [f for f, w in zip(fids, who) if w == "ada"][30:36]:           # six close-circle fixes
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    build(db, [("ada", d, 900) for d in days("2018-12-18", 20, step=2)])     # a 20-face group: larger than 10
    outer = build(db, [("other", d, None) for d in days("2019-07-18", 3)])   # outer-circle assigns
    db.execute("UPDATE syno_faces SET person_id=NULL WHERE item IN (SELECT item FROM faces WHERE id IN (?,?,?))", outer)
    db.commit()
    _decide_all(db, cfg)
    res = changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    assert res["large_merges"] == [{"group": 900, "faces": 20, "target": "Ada"}]
    kinds = dict(db.execute("SELECT kind, COUNT(*) FROM changes WHERE run='2026-11' AND status='planned' GROUP BY kind").fetchall())
    assert kinds["merge"] == 1 and kinds["reassign"] >= 6 and kinds["assign"] >= 1
    fake = FakeClient(sp, db)
    out = changes.apply(db, cfg, library.Nas(cfg, sp, fake), "2026-11", log=lambda *_: None, auto=True)
    assert out["done"] == 2 and out["left_staged"] == sum(kinds.values()) - 2
    assert set(out["left_staged_why"]) == {"auto ceiling: face edits", "large merge", "outer circle"}
    assert [m for m, _ in fake.calls if m != "list_face"].count("merge") == 0
    done = db.execute("SELECT person FROM changes WHERE run='2026-11' AND status='done'").fetchall()
    assert all(cfg.people[r[0]].tier == "close" for r in done)
    # a resumed automatic run gets no second allowance; by hand the owner can still apply the rest
    again = changes.apply(db, cfg, library.Nas(cfg, sp, fake), "2026-11", log=lambda *_: None, auto=True)
    assert again.get("done", 0) == 0
    assert changes.apply(db, cfg, library.Nas(cfg, sp, fake), "2026-11", kinds=set(changes.FACE_KINDS),
                         log=lambda *_: None)["done"] >= 5


def test_run_resumes_failed_stage(env, sp):
    cfg, db = env
    seed_library(db, labelled=30)
    nas, fake = _fake_nas(cfg, sp, db)
    boom = {"n": 0}
    orig = cluster.run_clustering

    def flaky(*a, **k):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("disk full")
        return orig(*a, **k)

    cluster.run_clustering = flaky
    try:
        R.execute(cfg, "2026-11", nas_factory=FakeNasCtx(nas), engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
        row = D.run_row(db, "2026-11")
        assert row["status"] == "failed" and row["stage"] == "assign"         # last completed stage
        assert D.run_stats(db, "2026-11")["current"] == "cluster"
        R.execute(cfg, "2026-11", nas_factory=FakeNasCtx(nas), engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
        assert D.run_row(db, "2026-11")["status"] == "done"
    finally:
        cluster.run_clustering = orig


# --- review server ---------------------------------------------------------------------------------------
import http.client  # noqa: E402


class Page:
    """The review server on a free port, and raw HTTP calls to it (headers exactly as given)."""

    def __init__(self, cfg, tmp_path):
        from PIL import Image

        from fc import review
        self.img = tmp_path / "p.jpg"
        Image.new("RGB", (400, 300), (120, 80, 60)).save(self.img)
        self.app = review.App(cfg, D.ledger_path(cfg.data_dir))
        self.app.write_timeout = 0.3
        self.srv = review.make_server(self.app, 0)
        self.port = self.srv.server_address[1]
        self.token = self.app.token
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def call(self, method, path, body=None, token=True, headers=None, raw=None):
        h = {"Host": f"127.0.0.1:{self.port}"}
        if token:
            h["X-FC-Token"] = self.token
            h["Cookie"] = f"fc_token_{self.port}={self.token}"
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        if data is not None and "Content-Length" not in h:
            h["Content-Length"] = str(len(data))
        for k, v in h.items():
            if v is not None:
                c.putheader(k, v)
        c.endheaders(data)
        r = c.getresponse()
        out = (r.status, dict(r.getheaders()), r.read())
        c.close()
        return out

    def json(self, method, path, body=None, **kw):
        st, _, data = self.call(method, path, body, **kw)
        return st, json.loads(data)

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def page(env, tmp_path):
    cfg, db = env
    pg = Page(cfg, tmp_path)
    pg.cfg, pg.db = cfg, db
    pg.fids, pg.who = seed_library(db, labelled=12)
    db.execute("UPDATE items SET local_path=?", (str(pg.img),))
    db.commit()
    _decide_all(db, cfg)
    cards.monthly(db, cfg, "2026-11", seed=2, log=lambda *_: None)
    yield pg
    pg.close()


def test_review_server(page):
    st, status = page.json("GET", "/api/status")
    assert st == 200 and {p["key"] for p in status["people"]} == {"ada", "bea", "dad"}
    st, hd, data = page.call("GET", "/crop/%d?s=120" % page.fids[0])
    assert st == 200 and data[:2] == b"\xff\xd8"
    assert page.call("GET", "/photo/%d" % page.fids[0])[2][:2] == b"\xff\xd8"
    st, tiles = page.json("GET", "/api/gallery?person=ada&bin=3")
    assert st == 200 and tiles
    st, out = page.json("POST", "/api/gallery", {"person": "ada", "yes": [tiles[0]["face"]], "no": [tiles[1]["face"]]})
    assert (st, out["saved"]) == (200, 2)
    st, deck = page.json("GET", "/api/cards")
    assert deck and deck[0]["a"]["name"]
    st, out = page.json("POST", "/api/card", {"id": deck[0]["id"], "answer": "b"})
    assert st == 200 and "label" in out
    assert page.json("POST", "/api/card/undo", {"id": deck[0]["id"]})[1]["undone"]
    c = sqlite3.connect(D.ledger_path(page.cfg.data_dir))
    assert c.execute("SELECT answer FROM cards WHERE id=?", (deck[0]["id"],)).fetchone()[0] is None
    assert c.execute("SELECT COUNT(*) FROM bumps").fetchone()[0] == 0


def test_review_refuses_requests_without_token_or_from_elsewhere(page):
    card = page.db.execute("SELECT id FROM cards WHERE answer IS NULL AND kind='random' LIMIT 1").fetchone()[0]
    body = {"id": card, "answer": "b"}
    own = f"http://127.0.0.1:{page.port}"

    def untouched():
        return page.db.execute("SELECT answer FROM cards WHERE id=?", (card,)).fetchone()[0] is None

    # no token: nothing is served, not the page, not an image, not a name
    for path in ("/", "/app.js", "/app.css", "/api/status", "/api/cards", "/api/compare", "/crop/%d" % page.fids[0],
                 "/photo/%d" % page.fids[0]):
        assert page.call("GET", path, token=False)[0] == 403, path
    assert page.call("POST", "/api/card", body, token=False)[0] == 403 and untouched()
    assert page.call("POST", "/api/card", body, token=False, headers={"X-FC-Token": "x" * 43})[0] == 403
    # the API wants the header: the cookie a browser attaches by itself is not enough (CSRF)
    assert page.call("POST", "/api/card", body, token=False,
                     headers={"Cookie": f"fc_token_{page.port}={page.token}"})[0] == 403 and untouched()
    # a "simple request" a foreign page can send: text/plain with a JSON body
    st, _, _ = page.call("POST", "/api/card", raw=json.dumps(body).encode(), headers={"Content-Type": "text/plain"})
    assert st == 415 and untouched()
    assert page.call("POST", "/api/card", raw=json.dumps(body).encode())[0] == 415           # no Content-Type at all
    # a foreign Origin, even with the token
    assert page.call("POST", "/api/card", body, headers={"Origin": "https://evil.example"})[0] == 403 and untouched()
    assert page.call("POST", "/api/card", body, headers={"Origin": "null"})[0] == 403
    assert page.call("GET", "/api/status", headers={"Origin": "https://evil.example"})[0] == 403
    # a foreign Host (DNS rebinding), even with the token
    for host in ("evil.example", f"evil.example:{page.port}", "127.0.0.1", f"127.0.0.1:{page.port + 1}", None):
        assert page.call("GET", "/api/status", headers={"Host": host})[0] == 403, host
        assert page.call("POST", "/api/card", body, headers={"Host": host})[0] == 403 and untouched()
    # with the token, from the page itself: accepted
    assert page.call("GET", "/api/status", headers={"Host": f"localhost:{page.port}"})[0] == 200
    assert page.call("POST", "/api/card", body, headers={"Origin": own})[0] == 200 and not untouched()
    # the page takes the token from the URL and hands the browser a cookie for the images
    st, hd, html = page.call("GET", f"/?t={page.token}", token=False)
    assert st == 200 and b"<script src=\"/app.js\">" in html
    assert hd["Set-Cookie"] == f"fc_token_{page.port}={page.token}; HttpOnly; SameSite=Strict; Path=/"
    assert page.call("GET", "/?t=wrong", token=False)[0] == 403
    assert page.call("GET", "/crop/%d?t=%s" % (page.fids[0], page.token), token=False)[0] == 200
    assert page.call("GET", "/crop/%d" % page.fids[0], token=False,
                     headers={"Cookie": f"other=1; fc_token_{page.port}={page.token}"})[0] == 200
    assert len(page.token) >= 40 and page.token != Page(page.cfg, page.img.parent).token     # random per session


def test_review_headers_and_names_are_never_html(page, tmp_path):
    evil = "<img src=x onerror=alert(1)>"
    people = {**page.cfg.raw["people"], "ada": {"name": evil, "born": "2016-03-09", "tier": "close"}}
    page.app.cfg = make_cfg(Path(page.cfg.raw["data_dir"]).parent, people=people)
    page.db.execute("UPDATE syno_people SET name=? WHERE id=101", (evil,))
    page.db.commit()
    st, hd, data = page.call("GET", "/api/status")
    assert st == 200 and evil in [p["name"] for p in json.loads(data)["people"]]     # data, as JSON, not as a page
    assert hd["Content-Type"] == "application/json"
    for path in ("/", "/app.js", "/app.css", "/api/status", "/api/cards", "/crop/%d" % page.fids[0],
                 "/photo/%d" % page.fids[0], "/nothing"):
        st, hd, _ = page.call("GET", path)
        assert hd["Cache-Control"] == "no-store" and hd["X-Content-Type-Options"] == "nosniff", path
        csp = hd["Content-Security-Policy"]
        assert "default-src 'self'" in csp and "script-src 'self'" in csp and "unsafe" not in csp, path
    # the page has no inline script or style, and the script never turns text into HTML
    html = (HERE / "review" / "index.html").read_text()
    js = (HERE / "review" / "app.js").read_text()
    assert "<script src=\"/app.js\"></script>" in html and html.count("<script") == 1 and "<style" not in html
    assert " on" not in "".join(ln for ln in html.splitlines() if "<" in ln and "=" in ln).replace(" content", "")
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert bad not in js, bad
    assert "textContent" in js or "createTextNode" in js


def test_review_card_is_not_overwritten_unless_replaced_on_purpose(page):
    card = page.db.execute("SELECT * FROM cards WHERE answer IS NULL AND kind='random' LIMIT 1").fetchone()
    assert page.json("POST", "/api/card", {"id": card["id"], "answer": "a"})[0] == 200
    st, out = page.json("POST", "/api/card", {"id": card["id"], "answer": "b"})
    assert st == 409 and out["answered"]
    assert page.json("POST", "/api/card", {"id": card["id"], "answer": "b", "replace": "yes"})[0] == 409   # only true
    row = page.db.execute("SELECT answer FROM cards WHERE id=?", (card["id"],)).fetchone()
    assert row["answer"] == "a" and not page.db.execute("SELECT COUNT(*) FROM bumps").fetchone()[0]
    st, out = page.json("POST", "/api/card", {"id": card["id"], "answer": "b", "replace": True})
    assert st == 200 and out["previous"] == "a" and out["error"]
    row = page.db.execute("SELECT answer, payload FROM cards WHERE id=?", (card["id"],)).fetchone()
    prev = json.loads(row["payload"])["previous"]
    assert row["answer"] == "b" and [x["answer"] for x in prev] == ["a"] and prev[0]["answered"]
    labs = page.db.execute("SELECT value, source FROM labels WHERE face=? AND via='card:random' ORDER BY id",
                           (card["face"],)).fetchall()
    assert [(x["value"], x["source"]) for x in labs] == [(card["person"], "retracted"), (card["alt"], "owner")]
    with pytest.raises(cards.AlreadyAnswered):
        cards.answer(page.db, page.cfg, card["id"], "a")
    # blind cards follow the same rule
    from fc import compare
    page.db.execute("INSERT INTO cards(run, kind, face, payload, created) VALUES('h2h-x', 'h2h', ?, '{}', 'now')", (page.fids[0],))
    page.db.commit()
    bid = page.db.execute("SELECT id FROM cards WHERE kind='h2h'").fetchone()[0]
    assert page.json("POST", "/api/compare", {"id": bid, "answer": "ada"})[0] == 200
    assert page.json("POST", "/api/compare", {"id": bid, "answer": "bea"})[0] == 409
    assert page.json("POST", "/api/compare", {"id": bid, "answer": "bea", "replace": True})[0] == 200
    assert D.latest_labels(page.db)[page.fids[0]] == "bea"
    assert page.json("POST", "/api/card", {"id": bid, "answer": "a"})[0] == 400              # not a blind card's route
    with pytest.raises(cards.AlreadyAnswered):
        compare.answer(page.db, page.cfg, bid, "ada")


def test_review_bad_bodies_are_refused(page):
    card = page.db.execute("SELECT id FROM cards WHERE answer IS NULL AND kind='random' LIMIT 1").fetchone()[0]
    js = {"Content-Type": "application/json"}
    assert page.call("POST", "/api/card", raw=b"{}", headers={**js, "Content-Length": str((1 << 20) + 1)})[0] == 413
    assert page.call("POST", "/api/card", raw=b"{}", headers={**js, "Content-Length": "-5"})[0] == 400
    assert page.call("POST", "/api/card", raw=b"{}", headers={**js, "Content-Length": "abc"})[0] == 400
    for bad in ({"id": None, "answer": "a"}, {"id": "1", "answer": "a"}, {"id": True, "answer": "a"}, {"answer": "a"},
                {"id": card}, {"id": card, "answer": ["a"]}, {"id": card, "answer": "maybe"}, {"id": 10**9, "answer": "a"},
                [1, 2], "x"):
        assert page.call("POST", "/api/card", bad)[0] == 400, bad
        assert page.call("POST", "/api/card/undo", bad)[0] in (200, 400), bad
    assert page.call("POST", "/api/card", raw=b"null", headers=js)[0] == 400
    assert page.call("POST", "/api/card", raw=b"{not json", headers=js)[0] == 400
    for bad in ({"person": "ada", "yes": "12"}, {"person": "ada", "yes": [None]}, {"person": "nobody", "yes": []},
                {"person": None}, {"person": "ada", "no": [10**9]}, {"person": "ada", "yes": list(range(501))}):
        assert page.call("POST", "/api/gallery", bad)[0] == 400, bad
    assert page.call("POST", "/api/compare", {"id": None, "answer": "ada"})[0] == 400
    assert page.call("POST", "/api/compare/undo", {"id": None})[0] == 400
    assert page.call("GET", "/crop/abc")[0] == 400 and page.call("GET", "/api/gallery")[0] == 400
    assert page.call("POST", "/api/nothing", {})[0] == 404 and page.call("GET", "/nothing")[0] == 404
    assert page.db.execute("SELECT answer FROM cards WHERE id=?", (card,)).fetchone()[0] is None
    assert not page.db.execute("SELECT COUNT(*) FROM labels WHERE via != 'test'").fetchone()[0]


def test_review_reads_while_a_run_writes_and_says_when_it_cannot_save(page):
    card = page.db.execute("SELECT id FROM cards WHERE answer IS NULL AND kind='random' LIMIT 1").fetchone()[0]
    writer = sqlite3.connect(D.ledger_path(page.cfg.data_dir))
    writer.execute("BEGIN IMMEDIATE")                       # a run in the middle of a write
    writer.execute("UPDATE runs SET stage=stage")
    try:
        import time
        t0 = time.monotonic()
        assert page.call("GET", "/api/status")[0] == 200
        assert page.call("GET", "/api/cards")[0] == 200
        assert page.call("GET", "/crop/%d?s=96" % page.fids[1])[0] == 200
        st, out = page.json("POST", "/api/card", {"id": card, "answer": "a"})
        assert st == 503 and "busy" in out["error"] and "nothing was saved" in out["error"]
        st, out = page.json("POST", "/api/gallery", {"person": "ada", "yes": [page.fids[0]]})
        assert st == 503
        assert time.monotonic() - t0 < 5
    finally:
        writer.rollback()
        writer.close()
    assert page.json("POST", "/api/card", {"id": card, "answer": "a"})[0] == 200
    js = (HERE / "review" / "app.js").read_text()
    assert "Not saved" in js and ".catch(" in js                 # a failed save is shown, never dropped


def test_connect_sets_up_the_schema_once_and_readonly_refuses_writes(tmp_path):
    path = tmp_path / "d" / "ledger.sqlite"
    D.connect(path).close()
    writer = sqlite3.connect(path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO meta VALUES('x', '1')")
    try:
        ro = D.connect(path, timeout=0.2, readonly=True)       # opening needs no write lock any more
        assert ro.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO meta VALUES('y', '1')")
        D.connect(path, timeout=0.2).close()
    finally:
        writer.rollback()
    assert D.get_meta(D.connect(path), "schema_version") == str(D.SCHEMA_VERSION)


def test_backup_and_export(env):
    cfg, db = env
    seed_library(db, labelled=3)
    p = D.backup(db, cfg.backup_dir)
    assert sqlite3.connect(p).execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 9
    assert D.integrity_ok(db)


# --- blind comparison ------------------------------------------------------------------------------------
def test_compare_build_answer_report(env, tmp_path):
    from fc import compare, review
    cfg, db = env
    fids, who = seed_library(db, labelled=12)          # owner galleries (8+ per gallery person)
    # Synology mistakes: 4 Ada faces named Bea (disagreements); 6 Ada faces with no Synology face (coverage)
    ada = [f for f, w in zip(fids, who) if w == "ada"]
    for f in ada[-4:]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    for f in ada[-10:-4]:
        db.execute("UPDATE faces SET syno_face_id=NULL WHERE id=?", (f,))
    db.execute("UPDATE faces SET y2 = y1 + 0.1")
    db.commit()
    out = compare.build(db, cfg, n_coverage=5, n_agree=5, log=lambda *_: None)
    assert out["galleries"] == "owner labels" and out["synology_source"].startswith("ledger")
    assert out["cards"]["disagree"] >= 1 and out["cards"]["coverage"] >= 1 and out["cards"]["agree"] == 5
    with pytest.raises(SystemExit):
        compare.build(db, cfg, log=lambda *_: None)                     # one open set at a time
    # the blind page never carries either system's answer, and the Cards tab never shows h2h cards
    app = review.App(cfg, D.ledger_path(cfg.data_dir))
    blind = app.compare_cards()
    assert blind["cards"] and all(set(c) == {"id", "face", "day"} for c in blind["cards"])
    assert not app.open_cards(None)
    truth = dict(zip(fids, who))
    for c in db.execute("SELECT id, face FROM cards WHERE kind='h2h'").fetchall():
        t = truth[c["face"]]
        compare.answer(db, cfg, c["id"], t if t in ("ada", "bea", "dad") else "other")
    rep = compare.report(db, cfg)
    d = rep["groups"]["disagree"]
    assert d["curator right"] == d["judged"] and d["synology right"] == 0
    assert rep["groups"]["coverage"]["curator_precision"] == 1.0
    assert rep["conclusion"].startswith("curator better") or rep["groups"]["coverage"]["curator_lower95"] < 0.95
    assert D.latest_labels(db)[ada[-1]] == "ada"                      # answers become owner labels
    assert not db.execute("SELECT COUNT(*) FROM bumps").fetchone()[0]   # and never raise thresholds


def test_resume_legacy_in_progress_row(env, sp):
    """A row written by the first version (stage = in progress, status failed, no 'current') retries that stage."""
    cfg, db = env
    seed_library(db, labelled=30)
    D.set_run(db, "2026-11", stage="cluster", status="failed")
    seen = []
    orig = cluster.run_clustering
    cluster.run_clustering = lambda *a, **k: seen.append(1) or orig(*a, **k)
    try:
        nas, _ = _fake_nas(cfg, sp, db)
        R.execute(cfg, "2026-11", nas_factory=FakeNasCtx(nas), engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
    finally:
        cluster.run_clustering = orig
    assert seen and D.run_row(db, "2026-11")["status"] == "done"


def test_reads_retry_on_dropped_connection_writes_do_not(env, sp):
    import requests
    cfg, db = env
    seed_library(db, labelled=0)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    nas.sleep = lambda s: None
    real_ok, fails = fake.ok, {"n": 0}

    def flaky(api, method, version, **p):
        if fails["n"] < 2:
            fails["n"] += 1
            raise requests.exceptions.ConnectionError("Remote end closed connection without response")
        return real_ok(api, method, version, **p)

    fake.ok = flaky
    item = db.execute("SELECT MIN(id) FROM items").fetchone()[0]
    assert nas.faces(item)                                     # two drops, then success
    fails["n"] = 0
    with pytest.raises(requests.exceptions.ConnectionError):   # a write is not retried
        changes._write(db, nas, 1, "r", "delete_face", None, person_id=101, face_id=[1])


def test_sync_resume_skips_listing_and_done_items(env, sp):
    cfg, db = env
    seed_library(db, labelled=0)
    nas, fake = _fake_nas(cfg, sp, db)
    calls = {"people": 0}
    orig_people = nas.people
    nas.people = lambda: calls.__setitem__("people", calls["people"] + 1) or orig_people()
    db.execute("UPDATE items SET faces_synced=NULL")
    db.commit()
    D.set_run(db, "2026-11", status="running")
    started = D.run_row(db, "2026-11")["started"]
    D.set_run(db, "2026-11", stats={"sync_listed": {"people": 5, "seen": 0, "new": 0, "changed": 0, "at": started}})
    first = db.execute("SELECT MIN(id) FROM items").fetchone()[0]
    db.execute("UPDATE items SET faces_synced=? WHERE id=?", (D.now(), first))
    db.commit()
    R._stage(db, cfg, "2026-11", "sync", nas, None, None, lambda *_: None)
    assert calls["people"] == 0                                # listing skipped on resume
    read = [p["id_item"] for m, p in fake.calls if m == "list_face"]
    assert first not in read and len(read) == db.execute("SELECT COUNT(*) FROM items").fetchone()[0] - 1


# --- Proposal B study ------------------------------------------------------------------------------------
def test_study_build_answer_report(env):
    from fc import review, study
    cfg, db = env
    fids, who = seed_library(db, labelled=30)          # full owner galleries: no Synology top-up
    ada = [f for f, w in zip(fids, who) if w == "ada"]
    for f in ada[-5:]:                                       # Synology calls 5 Ada faces "Bea"
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    for f in ada[-12:-5]:                                    # and has no face for 7 others
        db.execute("UPDATE faces SET syno_face_id=NULL WHERE id=?", (f,))
    db.execute("UPDATE faces SET y2 = y1 + 0.1")
    db.execute("UPDATE items SET persons_sig='101'")
    db.commit()
    db.execute("UPDATE items SET faces_synced=NULL WHERE id=(SELECT MIN(id) FROM items)")
    db.commit()
    with pytest.raises(SystemExit):
        study.build(db, cfg, min_synced=1.0, log=lambda *_: None)    # refuses before the sync is complete
    out = study.build(db, cfg, min_synced=0.9, log=lambda *_: None)
    assert out["cards"] > 0 and {"confusion", "missed"} <= set(out["by_part"])
    app = review.App(cfg, D.ledger_path(cfg.data_dir))
    assert all(set(c) == {"id", "face", "day"} for c in app.compare_cards()["cards"])     # still blind
    truth = dict(zip(fids, who))
    from fc import compare
    for c in db.execute("SELECT id, face FROM cards WHERE kind='study'").fetchall():
        t = truth[c["face"]]
        compare.answer(db, cfg, c["id"], t if t in ("ada", "bea", "dad") else "other")
    rep = study.report(db, cfg)
    q = rep["confusions"]["bea>ada"]                     # Synology says Bea, truth Ada
    assert q["disagreements judged"] >= 1 and q["Synology wrong"] == q["disagreements judged"] and q["curator wrong"] == 0
    assert rep["missed"]["ada"]["estimated untagged faces"] >= 1


def test_factorial_recovers_main_effect():
    from itertools import product as P
    from fc import study
    obs = []
    rng = np.random.default_rng(4)
    for a, b, c, d in P((-1, 1), repeat=4):
        lev = dict(zip(study.FACTORS, (a, b, c, d, a * b * c * d)))
        p = 0.30 if b < 0 else 0.03                         # only a narrow gap causes errors
        obs += [(lev, bool(rng.random() < p)) for _ in range(50)]
    eff = study.factorial_effects(obs)["effects"]
    top = next(iter(eff))
    assert top == "gap" and eff["gap"]["effect (logit)"] < 0 and eff["gap"]["clear"]
    assert not eff["height"]["clear"] and not eff["crowd"]["clear"]


def test_embed_skips_missing_preview(env, sp):
    import requests
    cfg, db = env
    db.execute("INSERT INTO items(id, type, last_seen, cache_key) VALUES(1, 'photo', 'x', 'k'), (2, 'photo', 'x', 'k')")
    db.commit()

    class Nas404:
        def xl(self, item, ck):
            if item == 1:
                raise requests.exceptions.HTTPError("404 Client Error: Not Found")
            raise RuntimeError("stop here")                  # second item: also failing, but streak < 20

    eng = SimpleNamespace(models=["buffalo_l"])
    out = library.embed_items(db, cfg, eng, [1, 2], "2026-11", nas=Nas404(), log=lambda *_: None)
    assert out["no_preview"] == 2
    assert db.execute("SELECT detected FROM items WHERE id=1").fetchone()[0] == "no-preview:2026-11"
    assert library.items_needing_embedding(db, ["buffalo_l"]) == []          # not retried next time

    class NasDown:
        def xl(self, item, ck):
            raise requests.exceptions.HTTPError("503")
    ids = list(range(10, 40))
    db.executemany("INSERT INTO items(id, type, last_seen) VALUES(?, 'photo', 'x')", [(i,) for i in ids])
    with pytest.raises(requests.exceptions.HTTPError):                      # 20 in a row stops the stage
        library.embed_items(db, cfg, eng, ids, "2026-11", nas=NasDown(), log=lambda *_: None)


# --- edge round ------------------------------------------------------------------------------------------
def test_edges_build_answer_report(env):
    from fc import compare, edges, review
    cfg, db = env
    fids, who = seed_library(db, labelled=10)
    db.execute("UPDATE faces SET y2 = y1 + 0.1")
    db.commit()
    out = edges.build(db, cfg, log=lambda *_: None)
    assert out["cards"] > 0
    with pytest.raises(SystemExit):
        edges.build(db, cfg, log=lambda *_: None)            # one open round at a time
    app = review.App(cfg, D.ledger_path(cfg.data_dir))
    shown = app.compare_cards()["cards"]
    assert all(set(c) == {"id", "face", "day"} for c in shown)                     # blind
    assert {c["id"] for c in shown} >= {r[0] for r in db.execute("SELECT id FROM cards WHERE kind='edge'")}
    truth = dict(zip(fids, who))
    gk = {p.key for p in cfg.gallery_people}
    for c in db.execute("SELECT id, face FROM cards WHERE kind='edge'").fetchall():
        t = truth[c["face"]]
        compare.answer(db, cfg, c["id"], t if t in gk else "other")
    rep = edges.report(db, cfg)
    assert rep["open"] == 0 and rep["strata"]
    assert sum(s["judged"] for s in rep["strata"].values()) == out["cards"]
    labs = D.latest_labels(db)
    assert all(c["face"] in labs for c in db.execute("SELECT face FROM cards WHERE kind='edge'"))


def test_calibration_never_loosens_reviewed_thresholds(env):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    labels = D.latest_labels(db)
    gals = assign.build_galleries(db, cfg, cfg["models"]["reference"], labels)
    th = assign.calibrate(db, cfg, cfg["models"]["reference"], gals, labels)
    A = cfg["assign"]
    assert th.floor >= A["min_floor"] and th.margin >= A["default_margin"]
    assert th.sibling_margin >= A["default_sibling_margin"]


def test_baby_floor(env):
    cfg, db = env
    keys = ["ada", "bea"]
    fs = assign.FaceSet(np.array([1, 2]), np.zeros((2, 4), np.float32), np.array(["2016-06-16", "2021-06-17"], object))
    S = np.array([[0.60, 0.30], [0.60, 0.30]], np.float32)
    C = np.full((2, 2), 30, np.int32)
    th = assign.Thresholds(0.50, 0.08, 0.12)
    d = assign.decide(fs, keys, S, C, cfg, th)
    assert (d[0].status, d[0].reason) == ("abstained", "below-floor")     # Ada at ~3 months: baby floor 0.65
    assert d[1].status == "accepted"                                         # Ada at 5: floor 0.50


def test_replan_from_local_stage(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    nas, fake = _fake_nas(cfg, sp, db)
    eng = SimpleNamespace(models=["buffalo_l"])
    R.execute(cfg, "2026-11", until="plan", nas_factory=FakeNasCtx(nas), engine=eng, log=lambda *_: None)
    assert D.run_row(db, "2026-11")["stage"] == "plan"

    def no_nas():
        raise AssertionError("re-planning must not log in")
    db.execute("UPDATE changes SET status='capped' WHERE run='2026-11' AND id=(SELECT MIN(id) FROM changes)")
    db.commit()
    R.execute(cfg, "2026-11", until="plan", from_stage="assign", nas_factory=no_nas, engine=eng, log=lambda *_: None)
    assert D.run_row(db, "2026-11")["stage"] == "plan"
    n1 = db.execute("SELECT COUNT(*) FROM changes WHERE run='2026-11'").fetchone()[0]
    R.execute(cfg, "2026-11", from_stage="plan", nas_factory=no_nas, engine=eng, log=lambda *_: None)
    assert db.execute("SELECT COUNT(*) FROM changes WHERE run='2026-11'").fetchone()[0] == n1     # no duplicates
    db.execute("INSERT INTO changes(run, seq, kind, status) VALUES('2026-11', 999, 'add', 'done')")
    db.commit()
    with pytest.raises(SystemExit):
        R.execute(cfg, "2026-11", until="plan", from_stage="assign", nas_factory=no_nas, engine=eng, log=lambda *_: None)


def test_synology_check(env, sp):
    from fc import compare, edges
    cfg, db = env
    fids, who = seed_library(db, labelled=10)
    db.execute("UPDATE faces SET y2 = y1 + 0.1")
    ada = [f for f, w in zip(fids, who) if w == "ada"]
    for f in ada[-4:]:                                       # Synology calls 4 Ada faces "Bea"
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    nas, fake = _fake_nas(cfg, sp, db)
    R.execute(cfg, "2026-11", until="assign", nas_factory=FakeNasCtx(nas),
              engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
    out = edges.build_synology_check(db, cfg, ["bea"], n=6, log=lambda *_: None)
    assert 0 < out["cards"] <= 6
    rows = db.execute("SELECT * FROM cards WHERE kind='edge'").fetchall()
    assert all(r["alt"] == "bea" for r in rows)
    assert {json.loads(r["payload"])["stratum"] for r in rows} >= {"syno:bea:accepted-other"}
    truth = dict(zip(fids, who))
    for r in rows:
        compare.answer(db, cfg, r["id"], truth[r["face"]] if truth[r["face"]] in ("ada", "bea") else "other")
    rep = edges.report(db, cfg)
    s = rep["strata"]["syno:bea:accepted-other"]
    assert s["synology right"].startswith("0/")


def test_low_score_cards_lead_to_unlinks_first(env, sp):
    from fc import compare, edges
    cfg, db = env
    fids, who = seed_library(db, labelled=10)
    db.execute("UPDATE faces SET y2 = y1 + 0.1")
    db.commit()
    nas, fake = _fake_nas(cfg, sp, db)
    R.execute(cfg, "2026-11", until="assign", nas_factory=FakeNasCtx(nas),
              engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
    named = [r[0] for r in db.execute("SELECT face FROM decisions d WHERE run='2026-11' AND status='abstained'")]
    assert named, "the seed library has abstained faces"
    db.execute("UPDATE decisions SET score=0.2 WHERE run='2026-11' AND face=?", (named[0],))
    db.commit()
    out = edges.build_low_score(db, cfg, log=lambda *_: None)
    rows = db.execute("SELECT * FROM cards WHERE kind='edge'").fetchall()
    assert out["cards"] == len(rows) and all(json.loads(r["payload"])["score"] < 0.35 for r in rows)
    assert changes.KIND_ORDER["unlink"] == changes.KIND_ORDER["label-fix"] < changes.KIND_ORDER["reassign"]


# --- generic configuration: two circles, outer from Synology, look-alikes ---------------------------------
def test_close_and_outer_circle_sections(tmp_path):
    cfg = Config({"data_dir": str(tmp_path / "d"),
                  "close_circle": {"kid": {"name": "Kid One", "born": "2020-03-01"}, "mum": {"name": "Mum"}},
                  "outer_circle": {"people": {"gran": {"name": "Gran", "died": "2019"}, "pal": "Pal"}},
                  "siblings": [["kid", "mum"]], "lookalikes": [["kid", "gran"]]})
    assert [p.key for p in cfg.close_people] == ["kid", "mum"] and cfg.people["kid"].child
    assert {p.key for p in cfg.outer_people} == {"gran", "pal"} and cfg.people["pal"].name == "Pal"
    assert not cfg.people["gran"].allowed_on("2020-01-01")
    assert cfg.lookalikes["gran"] == {"kid"} and cfg.siblings["mum"] == {"kid"}
    with pytest.raises(SystemExit):
        Config({"close_circle": {"a": {"name": "A"}}, "outer_circle": {"people": {"a": {"name": "A2"}}}})
    with pytest.raises(SystemExit):
        Config({"close_circle": {"a": {"name": "A"}}, "lookalikes": [["a", "nobody"]]})


def test_outer_circle_from_synology(tmp_path):
    from fc.config import load_config, synology_counts
    cfg = Config({"close_circle": {"kid": {"name": "Kid One"}},
                  "outer_circle": {"from_synology": True, "min_faces": 5, "exclude": ["test person"],
                                   "people": {"gran": {"name": "Gran", "died": "2019"}}}})
    added = cfg.add_synology_outer({"Kid One": 900, "Gran": 50, "Zoe Neri": 40, "Zoe-Neri": 6, "Rare": 4,
                                    "Test Person": 99, "": 300})
    assert added == ["zoe_neri", "zoe_neri_2"]
    assert cfg.people["gran"].died == "2019" and cfg.people["zoe_neri"].tier == "outer"
    assert cfg.add_synology_outer({"Zoe Neri": 40}) == []                     # idempotent
    assert Config({"outer_circle": {"from_synology": False}}).add_synology_outer({"X": 99}) == []
    # load_config reads the counts from the ledger (read-only); none before the first sync
    data = tmp_path / "data"
    db = D.connect(D.ledger_path(data))
    db.execute("INSERT INTO syno_people(id, name) VALUES(1, 'Zoe Neri')")
    for i in range(6):
        db.execute("INSERT INTO syno_faces(face_id, item, person_id) VALUES(?, 1, 1)", (100 + i,))
    db.commit()
    assert synology_counts(D.ledger_path(data)) == {"Zoe Neri": 6}
    assert synology_counts(tmp_path / "none.sqlite") == {}
    p = tmp_path / "config.yaml"
    p.write_text(f"data_dir: {data}\nclose_circle: {{kid: {{name: Kid One}}}}\n"
                 "outer_circle: {from_synology: true, min_faces: 5}\n")
    assert [x.key for x in load_config(p).outer_people] == ["zoe_neri"]


def test_pair_kinds_come_from_config(tmp_path):
    from fc import edges, study
    cfg = make_cfg(tmp_path, lookalikes=[["ada", "dad"]])
    assert edges.pair_kind(cfg, "ada", "bea") == "sibling:ada-bea"
    assert edges.pair_kind(cfg, "dad", "ada") == "lookalike"
    assert edges.pair_kind(cfg, "bea", "dad") == "other" and edges.pair_kind(cfg, "ada", None) is None
    sv = np.array(["ada", "ada", "bea", "dad", None], dtype=object)
    top = np.array(["bea", "dad", "dad", "bea", "ada"], dtype=object)
    conf = study.confusions(cfg, sv, top)
    assert conf[("ada", "bea")] == study.CONF_PAIR and conf[("dad", "ada")] == study.CONF_PAIR
    assert conf[("bea", "dad")] == study.CONF_OTHER
    assert study.study_pair(cfg, sv, top, None) == ("ada", "bea")
    with pytest.raises(SystemExit):
        study.study_pair(cfg, sv, top, ("ada", "grandpa"))                        # not in the close circle


# --- NAS write safety --------------------------------------------------------------------------------------
def _merge_plan(db, cfg):
    """Three Ada faces filed under Bea, and an unnamed group (900) of 20 Ada faces plus one stranger."""
    fids, who = seed_library(db, labelled=30)
    wrong = [f for f, w in zip(fids, who) if w == "ada"][30:33]
    for f in wrong:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    grp = build(db, [("ada", d, 900) for d in days("2018-12-18", 20, step=2)] + [("other", "2019-01-29", 900)])
    db.commit()
    _decide_all(db, cfg)
    # the decisions this plan is about are fixed here, so the test does not depend on the random embeddings
    db.execute("UPDATE decisions SET status='accepted', person='ada', score=0.8 WHERE run='2026-11' AND face IN (%s)"
               % ",".join("?" * 23), wrong + grp[:-1])
    db.execute("UPDATE decisions SET status='abstained' WHERE run='2026-11' AND face=?", (grp[-1],))
    db.commit()
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    return wrong


def _merge_status(db):
    r = db.execute("SELECT status, detail FROM changes WHERE run='2026-11' AND kind='merge'").fetchone()
    return r["status"], r["detail"] or ""


def test_merge_is_rechecked_against_live_names(env, sp):
    cfg, db = env
    _merge_plan(db, cfg)
    assert json.loads(_merge_status(db)[1]) == {"faces": 21, "target_name": "Ada"}
    edits = set(changes.FACE_KINDS)

    def attempt(mutate):
        db.execute("UPDATE changes SET status='planned', detail=? WHERE run='2026-11' AND kind='merge'",
                   (json.dumps({"faces": 21, "target_name": "Ada"}),))
        db.commit()
        fake = FakeClient(sp, db)
        mutate(fake)
        nas = library.Nas(cfg, sp, fake)
        changes.apply(db, cfg, nas, "2026-11", kinds=edits, log=lambda *_: None)        # cleaning first
        changes.apply(db, cfg, nas, "2026-11", kinds={"merge"}, log=lambda *_: None)
        return fake, _merge_status(db)

    # the owner named the group in Photos after the plan: it is somebody now, not a group to merge away
    fake, (st, why) = attempt(lambda f: f.people.__setitem__(900, "Somebody New"))
    assert st == "stale" and "was named 'Somebody New'" in why and "merge" not in [m for m, _ in fake.calls]
    # the target was renamed after the plan: the name the owner approved is not the name it would merge into
    fake, (st, why) = attempt(lambda f: f.people.__setitem__(101, "Ada Renamed"))
    assert st == "stale" and "now named 'Ada Renamed'" in why and "merge" not in [m for m, _ in fake.calls]
    fake, (st, why) = attempt(lambda f: f.people.pop(101))
    assert st == "stale" and "target person no longer exists" in why
    fake, (st, why) = attempt(lambda f: f.people.pop(900))
    assert st == "stale" and "group no longer exists" in why
    # the group gained a face after the plan
    def extra(f):
        f.faces.setdefault(999, []).append({"face_id": 777, "person_id": 900, "name": "",
                                            "face_bounding_box": library.box_dict(0.1, 0.1, 0.2, 0.2)})
    fake, (st, why) = attempt(extra)
    assert st == "stale" and "group changed" in why and "merge" not in [m for m, _ in fake.calls]
    # unchanged: it merges, and the journal and the write log carry the names read from the NAS
    fake, (st, why) = attempt(lambda f: None)
    assert st == "done" and [m for m, _ in fake.calls].count("merge") == 1
    j = [json.loads(x) for x in Path(sp.JOURNAL_PATH).read_text().splitlines() if json.loads(x)["action"] == "merge"]
    assert j[-1]["merged_name"] == "" and j[-1]["target_name"] == "Ada" and j[-1]["merged_id"] == 900
    w = db.execute("SELECT prior, params FROM writes WHERE method='merge' AND ok=1").fetchone()
    assert json.loads(w["prior"])["target_name"] == "Ada" and json.loads(w["params"])["merged_id"] == [900]


def test_duplicate_synology_name_is_not_planned(env):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    for f in [f for f, w in zip(fids, who) if w == "ada"][30:33]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.execute("INSERT INTO syno_people VALUES(555, 'Ada', 0, 'now')")          # a second Synology person called Ada
    db.commit()
    assert changes.duplicate_names(db, cfg) == {"Ada": [101, 555]} and "ada" not in changes.syno_ids(db, cfg)
    _decide_all(db, cfg)
    lines = []
    res = changes.plan(db, cfg, "2026-11", log=lines.append)
    assert res["duplicate_names"] == {"Ada": [101, 555]} and "more than one person" in lines[0]
    assert not db.execute("SELECT COUNT(*) FROM changes WHERE run='2026-11' AND (person='ada' OR to_person IN (101, 555))").fetchone()[0]
    db.execute("DELETE FROM syno_people WHERE id=555")
    db.commit()
    assert changes.plan(db, cfg, "2026-11", log=lambda *_: None)["planned"].get("reassign", 0) >= 3


def test_apply_stops_at_the_fifth_failure(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    for f in [f for f, w in zip(fids, who) if w == "ada"][30:40]:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    assert db.execute("SELECT COUNT(*) FROM changes WHERE run='2026-11' AND status='planned'").fetchone()[0] >= 8
    fake = FakeClient(sp, db)
    fake.fail = lambda method, p: method == "delete_face"
    res = changes.apply(db, cfg, library.Nas(cfg, sp, fake), "2026-11", log=lambda *_: None)
    assert res == {"failed": 5}
    assert db.execute("SELECT COUNT(*) FROM changes WHERE run='2026-11' AND status='failed'").fetchone()[0] == 5


def _one_reassign(db, cfg):
    fids, who = seed_library(db, labelled=30)
    f = [f for f, w in zip(fids, who) if w == "ada"][31]
    db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    return db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0]


def test_undo_covers_a_reassign_that_failed_between_unlink_and_link(env, sp):
    cfg, db = env
    item = _one_reassign(db, cfg)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    fake.fail = lambda method, p: method == "add_face"
    assert changes.apply(db, cfg, nas, "2026-11", log=lambda *_: None) == {"failed": 1}
    assert [x["person_id"] for x in fake.faces[item]] == [0]                 # unlinked from Bea, never linked to Ada
    assert [tuple(r) for r in db.execute("SELECT method, ok FROM writes ORDER BY id")] == [("delete_face", 1), ("add_face", 0)]
    fake.fail = None
    res = changes.plan_undo(db, "2026-11", "undo-2026-11")
    assert res == {"changes": 1, "merges_manual": [], "uncertain": 1}
    assert changes.plan_undo(db, "2026-11", "undo-2026-11")["changes"] == 1   # planning twice does not duplicate
    assert db.execute("SELECT COUNT(*) FROM changes WHERE run='undo-2026-11'").fetchone()[0] == 1
    assert changes.apply(db, cfg, nas, "undo-2026-11", log=lambda *_: None) == {"done": 1}
    assert [x["person_id"] for x in fake.faces[item]] == [102]               # back where it was
    assert changes.plan_undo(db, "2026-11", "undo-2026-11")["changes"] == 0   # nothing left to put back


def test_undo_checks_writes_with_unknown_outcome_against_live_state(env, sp):
    cfg, db = env
    item = _one_reassign(db, cfg)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    changes.apply(db, cfg, nas, "2026-11", log=lambda *_: None)
    # as if the process was stopped after the calls and before the ledger was updated
    db.execute("UPDATE writes SET ok=0, result=NULL")
    db.execute("UPDATE changes SET status='planned', detail=NULL WHERE run='2026-11'")
    db.commit()
    assert changes.plan_undo(db, "2026-11", "undo-2026-11")["uncertain"] == 1
    assert changes.apply(db, cfg, nas, "undo-2026-11", log=lambda *_: None) == {"done": 1}
    assert [x["person_id"] for x in fake.faces[item]] == [102]
    # the same log when the calls never reached the NAS: undo finds the prior state and writes nothing
    n = len(fake.calls)
    db.execute("DELETE FROM changes WHERE run='undo-2026-11'")
    db.commit()
    changes.plan_undo(db, "2026-11", "undo-2026-11")
    assert changes.apply(db, cfg, nas, "undo-2026-11", log=lambda *_: None) == {"done": 1}
    assert not [m for m, _ in fake.calls[n:] if m != "list_face"]
    # someone else changed the face since: undo leaves it alone
    fake.faces[item][0]["person_id"] = 103
    db.execute("DELETE FROM changes WHERE run='undo-2026-11'")
    db.commit()
    changes.plan_undo(db, "2026-11", "undo-2026-11")
    assert changes.apply(db, cfg, nas, "undo-2026-11", log=lambda *_: None) == {"stale": 1}
    assert fake.faces[item][0]["person_id"] == 103


def test_undo_lists_merges_for_a_manual_split(env, sp):
    cfg, db = env
    wrong = _merge_plan(db, cfg)
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    changes.apply(db, cfg, nas, "2026-11", kinds=set(changes.FACE_KINDS), log=lambda *_: None)
    changes.apply(db, cfg, nas, "2026-11", kinds={"merge"}, log=lambda *_: None)
    res = changes.plan_undo(db, "2026-11", "undo-2026-11")
    edits = db.execute("SELECT COUNT(DISTINCT change) FROM writes WHERE run='2026-11' AND method != 'merge'").fetchone()[0]
    assert edits >= len(wrong) + 1                                             # the three fixes and the group's cleaning
    assert res["merges_manual"] == [900] and res["changes"] == edits and res["uncertain"] == 0
    assert set(r[0] for r in db.execute("SELECT kind FROM changes WHERE run='undo-2026-11'")) == {"restore"}


def test_repair_reconciles_changes_that_stopped_mid_write(env, sp):
    cfg, db = env
    fids, who = seed_library(db, labelled=30)
    wrong = [f for f, w in zip(fids, who) if w == "ada"][30:33]
    for f in wrong:
        db.execute("UPDATE syno_faces SET person_id=102 WHERE face_id=(SELECT syno_face_id FROM faces WHERE id=?)", (f,))
    db.commit()
    _decide_all(db, cfg)
    changes.plan(db, cfg, "2026-11", log=lambda *_: None)
    items = [db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0] for f in wrong]
    cid = {r["item"]: r["id"] for r in db.execute("SELECT id, item FROM changes WHERE run='2026-11'")}
    fake = FakeClient(sp, db)
    nas = library.Nas(cfg, sp, fake)
    # change A: unlinked, the link failed. B: both calls went through, the ledger never heard. C: nothing was sent.
    fake.fail = lambda method, p: method == "add_face" and p["id_item"] == items[0]
    changes.apply(db, cfg, nas, "2026-11", log=lambda *_: None)
    fake.fail = None
    db.execute("UPDATE writes SET ok=0, result=NULL WHERE change=?", (cid[items[1]],))
    db.execute("UPDATE changes SET status='planned', detail=NULL WHERE id=?", (cid[items[1]],))
    fake.faces[items[2]][0]["person_id"] = 102
    db.execute("UPDATE changes SET status='failed' WHERE id=?", (cid[items[2]],))
    db.commit()
    n = len(fake.calls)
    lines = []
    dry = changes.repair(db, cfg, nas, "2026-11", write=False, log=lines.append)
    assert dry == {"would: finish the link": 1, "would: done": 1, "would: planned": 1}
    assert not [m for m, _ in fake.calls[n:] if m != "list_face"]                       # the dry run only reads
    assert [r[0] for r in db.execute("SELECT status FROM changes WHERE run='2026-11' ORDER BY id")] != ["done"] * 3
    assert changes.repair(db, cfg, nas, "2026-11", write=True, log=lambda *_: None) == {"done": 2, "planned": 1}
    st = {r["item"]: r["status"] for r in db.execute("SELECT item, status FROM changes WHERE run='2026-11'")}
    assert st == {items[0]: "done", items[1]: "done", items[2]: "planned"}
    assert [x["person_id"] for x in fake.faces[items[0]]] == [101]                      # the half-written link is finished
    assert [m for m, _ in fake.calls[n:] if m != "list_face"] == ["add_face"]
    assert changes.repair(db, cfg, nas, "2026-11", write=True, log=lambda *_: None) == {"planned": 1}   # C can be applied again


# --- the command line: previews, locks, confirmations -------------------------------------------------------
@pytest.fixture
def cli(env, sp, monkeypatch):
    """facecurator.main() on the test config, with the fake NAS instead of a login."""
    import facecurator
    cfg, db = env
    state = SimpleNamespace(cfg=cfg, db=db, fake=None, main=facecurator.main, mod=facecurator)
    real = library.Nas

    class NasCtx:
        def __init__(self, c):
            state.fake = state.fake or FakeClient(sp, db)
            self.nas = real(c, sp, state.fake)

        def __enter__(self):
            return self.nas

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(facecurator, "load_config", lambda: state.cfg)
    monkeypatch.setattr(library, "Nas", NasCtx)
    monkeypatch.setattr(R, "notify", lambda text: None)
    return state


def test_cli_apply_shows_every_merge_and_applies_merges_only_on_their_own(cli, capsys):
    cfg, db = cli.cfg, cli.db
    _merge_plan(db, cfg)
    D.set_run(db, "2026-11", stage="plan")
    cli.main(["apply", "--run", "2026-11"])
    out = capsys.readouterr().out
    assert "group 900: 21 faces -> \"Ada\" (person 101)" in out and "1 cleaning edit(s) to apply first" in out
    assert "dry run: nothing was written" in out and "--kinds merge --yes" in out
    assert cli.fake is None                                                    # a preview does not log in
    with pytest.raises(SystemExit, match="on their own"):
        cli.main(["apply", "--run", "2026-11", "--kinds", "reassign", "merge", "--yes"])
    cli.main(["apply", "--run", "2026-11", "--yes"])                           # face edits only
    out = capsys.readouterr().out
    methods = [m for m, _ in cli.fake.calls if m != "list_face"]
    assert "merge" not in methods and "add_face" in methods and "1 merge(s) wait for their own yes" in out
    assert _merge_status(db)[0] == "planned"
    cli.main(["apply", "--run", "2026-11", "--kinds", "merge"])                # its own preview
    assert "group 900: 21 faces" in capsys.readouterr().out and _merge_status(db)[0] == "planned"
    cli.main(["apply", "--run", "2026-11", "--kinds", "merge", "--yes"])
    assert _merge_status(db)[0] == "done" and [m for m, _ in cli.fake.calls].count("merge") == 1
    assert changes.KIND_ORDER["group-unlink"] < changes.KIND_ORDER["merge"] and "merge" not in changes.FACE_KINDS


def test_cli_apply_undo_and_repair_take_the_run_lock(cli, capsys):
    cfg, db = cli.cfg, cli.db
    _one_reassign(db, cfg)
    with R.lock(cfg.data_dir):
        for argv in (["apply", "--run", "2026-11", "--yes"], ["undo", "--run", "2026-11", "--yes"],
                     ["repair", "--run", "2026-11", "--yes"]):
            with pytest.raises(SystemExit, match="holds the lock"):
                cli.main(argv)
        cli.main(["apply", "--run", "2026-11"])                                # previews need no lock
    assert cli.fake is None or not [m for m, _ in cli.fake.calls if m != "list_face"]
    cli.main(["apply", "--run", "2026-11", "--yes"])
    cli.main(["undo", "--run", "2026-11"])
    assert "dry run" in capsys.readouterr().out
    cli.main(["undo", "--run", "2026-11", "--yes"])
    assert [m for m, _ in cli.fake.calls if m != "list_face"] == ["delete_face", "add_face"] * 2


def test_cli_run_ids_are_validated(cli, tmp_path):
    for bad in ("../../x", "2026-13", "2026-1", "2026-11; rm -rf ~", '2026-11" & do shell script "id', "undo-", ""):
        with pytest.raises(SystemExit):
            R.check_run_id(bad, undo=True)
        for argv in (["apply", "--run", bad], ["undo", "--run", bad], ["report", "--run", bad], ["audit", "--run", bad],
                     ["requeue", "--run", bad], ["repair", "--run", bad], ["run", "--id", bad or "x"],
                     ["progress", "--run", bad or "x"]):
            with pytest.raises(SystemExit):
                cli.main(argv)
    assert R.check_run_id("2026-11") == "2026-11" and R.check_run_id("undo-2026-11", undo=True)
    assert R.check_run_id("2026-11b") == "2026-11b" and R.check_run_id("undo-2026-11c", undo=True)
    for bad in ("2026-11bb", "2026-11B", "2026-11-", "2026-11 b"):
        with pytest.raises(SystemExit):
            R.check_run_id(bad)
    with pytest.raises(SystemExit):
        R.check_run_id("undo-2026-11")                                         # only where an undo run makes sense
    with pytest.raises(SystemExit):
        R.execute(cli.cfg, "../x", nas_factory=lambda: None, log=lambda *_: None)
    assert not list(tmp_path.rglob("x.md"))


def test_notify_passes_text_as_an_argument(monkeypatch):
    seen = []
    monkeypatch.setattr(R.subprocess, "run", lambda argv, **kw: seen.append(argv))
    evil = 'x" & (do shell script "touch /tmp/pwned") & "'
    R.notify(evil)
    argv = seen[0]
    assert argv[0] == "osascript" and argv[-2:] == ["--", evil]
    assert all(evil not in a for a in argv[:-1]) and "item 1 of argv" in " ".join(argv)


def test_debug_traceback_is_redacted(env, monkeypatch):
    cfg, db = env
    monkeypatch.setenv("FACECURATOR_DEBUG", "1")
    monkeypatch.setattr(R, "notify", lambda text: None)

    def boom(*a, **k):
        raise RuntimeError("login failed for https://nas.example/?_sid=TOPSECRET")

    monkeypatch.setattr(library, "sync_people", boom)
    lines = []
    fake_sp = SimpleNamespace(redact=lambda t: t.replace("TOPSECRET", "***"))
    R.execute(cfg, "2026-11", nas_factory=FakeNasCtx(SimpleNamespace()), sp=fake_sp, log=lines.append)
    text = "\n".join(str(x) for x in lines)
    assert "Traceback" in text and "TOPSECRET" not in text and "***" in text
    assert "TOPSECRET" not in json.dumps(D.run_stats(db, "2026-11"))


def test_cli_bumps_clear_needs_yes_and_backs_up_first(cli, capsys):
    cfg, db = cli.cfg, cli.db
    db.execute("INSERT INTO bumps VALUES('ada', '2-4y', 0.03, 1, 'now', 'card 7 (audit) answered b')")
    db.commit()
    cli.main(["bumps", "--clear", "ada", "2-4y"])
    assert "dry run" in capsys.readouterr().out and db.execute("SELECT COUNT(*) FROM bumps").fetchone()[0] == 1
    assert not list(cfg.backup_dir.glob("ledger-*.sqlite"))
    cli.main(["bumps", "--clear", "ada", "2-4y", "--yes"])
    assert db.execute("SELECT COUNT(*) FROM bumps").fetchone()[0] == 0
    (backup,) = list(cfg.backup_dir.glob("ledger-*.sqlite"))
    assert sqlite3.connect(backup).execute("SELECT COUNT(*) FROM bumps").fetchone()[0] == 1   # the hold is in the backup
    with pytest.raises(SystemExit):
        cli.main(["bumps", "--clear", "ada", "2-4y", "--yes"])


def test_cli_people_yaml_quotes_names_from_the_nas(cli, capsys):
    import yaml
    db = cli.db
    seed_library(db, labelled=0)
    evil = "Eve: {born: 1900-01-01}, x: &a [*a] #"
    db.execute("UPDATE syno_people SET name=? WHERE id=104", (evil,))
    db.execute("UPDATE syno_people SET name='Line\nbreak: yes' WHERE id=103")
    db.commit()
    cli.main(["people", "--yaml"])
    doc = yaml.safe_load(capsys.readouterr().out)["close_circle"]
    names = sorted(v["name"] for v in doc.values())
    assert evil in names and "Line\nbreak: yes" in names and all(set(v) == {"name"} for v in doc.values())


def test_cli_install_schedule_escapes_paths(cli, capsys, tmp_path, monkeypatch):
    import plistlib
    import shlex
    odd = tmp_path / "skill & <co> 'x'"
    (odd / "launchd").mkdir(parents=True)
    (odd / "launchd" / "face-curator.plist").write_text((HERE.parent / "launchd" / "face-curator.plist").read_text())
    home = tmp_path / "home & <b>"
    home.mkdir()
    monkeypatch.setattr(cli.mod, "REPO", odd)
    monkeypatch.setattr(cli.mod, "HERE", odd / "scripts")
    monkeypatch.setenv("HOME", str(home))
    cli.main(["install-schedule"])
    out = capsys.readouterr().out
    (gen,) = list((odd / "launchd" / "generated").glob("*.plist"))
    pl = plistlib.loads(gen.read_bytes())                                       # still well-formed XML
    assert pl["ProgramArguments"] == ["/bin/zsh", str(odd / "scripts" / "runner.sh")]
    assert pl["StandardOutPath"].startswith(str(home))
    assert gen.stat().st_mode & 0o777 == 0o600
    cp = next(ln for ln in out.splitlines() if ln.strip().startswith("cp "))
    assert shlex.split(cp)[1] == str(gen) and shlex.split(cp)[2].startswith(str(home))
    assert "writes nothing" not in out or cli.cfg.mode["policy"] != "auto"
    assert "WRITES to the NAS unattended" in out                                 # the test config asks for auto


def test_cli_version_and_progress(cli, capsys):
    from fc import __version__
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"facecurator {__version__}" == "facecurator 0.1.0"
    D.set_run(cli.db, "2026-11", stage="sync", status="running", stats={"current": "embed"})
    cli.main(["progress", "--run", "2026-11"])
    assert "run 2026-11: status running" in capsys.readouterr().out


# --- data at rest ------------------------------------------------------------------------------------------
def _mode(p) -> int:
    return Path(p).stat().st_mode & 0o777


def test_files_are_private(tmp_path, sp):
    old = os.umask(0o022)                       # the tests must hold even under a permissive umask
    try:
        mine = tmp_path / "shared"              # a folder the user made and pointed the backups at
        mine.mkdir(mode=0o755)
        cfg = make_cfg(tmp_path, backup_dir=str(mine))
        db = D.connect(D.ledger_path(cfg.data_dir))
        assert _mode(cfg.data_dir) == 0o700 and _mode(D.ledger_path(cfg.data_dir)) == 0o600
        for extra in ("-wal", "-shm"):
            f = Path(str(D.ledger_path(cfg.data_dir)) + extra)
            assert not f.exists() or _mode(f) == 0o600
        seed_library(db, labelled=3)
        nas, _ = _fake_nas(cfg, sp, db)
        R.notify = lambda text: None
        R.execute(cfg, "2026-11", nas_factory=FakeNasCtx(nas), engine=SimpleNamespace(models=["buffalo_l"]), log=lambda *_: None)
        row = D.run_row(db, "2026-11")
        assert _mode(row["report"]) == 0o600 and _mode(Path(row["report"]).parent) == 0o700
        exports = cfg.data_dir / "exports" / "2026-11"
        assert _mode(exports) == 0o700 and {_mode(f) for f in exports.iterdir()} == {0o600}
        backups = list(mine.glob("ledger-*.sqlite"))
        assert backups and {_mode(f) for f in backups} == {0o600}
        assert _mode(mine) == 0o755             # the user's own folder keeps the permissions the user gave it
        assert _mode(cfg.data_dir / "run.lock") == 0o600 and _mode(cfg.models_dir) == 0o700
        from fc import review
        from PIL import Image
        img = tmp_path / "p.jpg"
        Image.new("RGB", (400, 300), (120, 80, 60)).save(img)
        db.execute("UPDATE items SET local_path=?", (str(img),))
        db.commit()
        app = review.App(cfg, D.ledger_path(cfg.data_dir))
        face = db.execute("SELECT MIN(id) FROM faces").fetchone()[0]
        assert app.crop(face, 96)[:2] == b"\xff\xd8"
        crops = [f for f in (cfg.data_dir / "crops").rglob("*.jpg")]
        assert crops and {_mode(f) for f in crops} == {0o600} and _mode(cfg.data_dir / "crops") == 0o700
        assert {_mode(d) for d in cfg.data_dir.rglob("*") if d.is_dir()} == {0o700}       # every folder we made
        assert {_mode(f) for f in cfg.data_dir.rglob("*") if f.is_file()} == {0o600}      # and every file in them
        assert not list((cfg.data_dir / "crops").rglob("*.tmp"))
    finally:
        os.umask(old)


def test_cli_sets_umask_and_init_writes_a_private_config(tmp_path, monkeypatch):
    import facecurator
    from fc import config as C
    path = tmp_path / "cfgdir" / "config.yaml"
    monkeypatch.setattr(facecurator, "CONFIG_PATH", path)
    example = (HERE.parent / "references" / "config.example.yaml").read_text()
    monkeypatch.setattr(facecurator, "load_config",
                        lambda: C.Config({**__import__("yaml").safe_load(example), "data_dir": str(tmp_path / "data")}, path))
    old = os.umask(0o022)
    try:
        facecurator.main(["init"])
        assert os.umask(0o077) == 0o077                                         # main() made the process private
        assert _mode(path) == 0o600 and _mode(path.parent) == 0o700 and _mode(tmp_path / "data") == 0o700
        assert path.read_text() == example
    finally:
        os.umask(old)
    assert "umask 077" in (HERE / "runner.sh").read_text()


def test_example_config_ships_the_safe_defaults():
    import yaml
    from fc import config as C
    raw = yaml.safe_load((HERE.parent / "references" / "config.example.yaml").read_text())
    assert raw["mode"]["policy"] == "staged" and raw["outer_circle"]["from_synology"] is False
    cfg = C.Config(raw)
    assert cfg.mode["policy"] == "staged" and cfg.caps["face_edits"] <= C.HARD_CAPS["face_edits"]
    text = (HERE.parent / "references" / "config.example.yaml").read_text()
    for key in ("require_ac", "require_lan", "index_wait_minutes", "port", "rolling_fraction", "min_det",
                "default_sibling_margin", "outer_floor_bump", "group_merge_min_accepted", "audit_min_judged",
                "min_judged_per_run", "min_lower95", "auto_max_face_edits", "auto_max_merges", "merge_auto_max_faces",
                "face_cache_dir"):
        assert key in text, key
    # every default the example states in a comment is the default in code
    for section in ("detect", "assign", "changes", "sync", "runner", "review"):
        for k, v in C.DEFAULTS[section].items():
            assert f"{k}: {json.dumps(v)}" in text or f"{k}: {v}" in text, (section, k)


# --- models, previews, the synology-photos bridge ----------------------------------------------------------
def test_model_registry_pins_and_verifies(tmp_path):
    from fc import embed
    reg = embed.Registry(tmp_path)
    fake = tmp_path / "w600k_r50.onnx"
    fake.write_bytes(b"not the real weights")
    with pytest.raises(SystemExit, match="not the buffalo_l/w600k_r50.onnx this skill expects"):
        reg.register("buffalo_l/w600k_r50.onnx", fake)                           # a pinned name, another file
    assert not reg.lock and set(embed.PINNED) == {embed.DETECTOR, embed.MODELS["buffalo_l"].file}
    assert all(len(h) == 64 for h in embed.PINNED.values())
    other = tmp_path / "glintr100.onnx"
    other.write_bytes(b"a challenger the owner chose")
    e = reg.register(embed.MODELS["antelopev2"].file, other)                     # another model, by name: recorded
    assert reg.path(embed.MODELS["antelopev2"].file) == other.resolve() and e["sha256"] == embed.sha256(other)
    assert _mode(reg.lock_path) == 0o600
    other.write_bytes(b"a challenger the owner CHOSE")                            # same size, other content
    assert embed.Registry(tmp_path).verify(embed.MODELS["antelopev2"].file) is False
    with pytest.raises(SystemExit, match="does not match its registered sha256"):
        embed.Registry(tmp_path).path(embed.MODELS["antelopev2"].file)
    # a lock file edited by hand cannot make a pinned name point at another file
    reg.lock[embed.DETECTOR] = {"path": str(fake), "sha256": embed.sha256(fake), "size": fake.stat().st_size}
    with pytest.raises(SystemExit, match="does not match"):
        reg.path(embed.DETECTOR)
    with pytest.raises(SystemExit, match="not registered"):
        embed.Registry(tmp_path / "empty").path(embed.DETECTOR)


def test_preview_cache_is_atomic_and_refetches_broken_files(env):
    import io

    from PIL import Image
    cfg, db = env
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 120, 200)).save(buf, "JPEG")
    jpeg = buf.getvalue()
    db.execute("INSERT INTO items(id, type, last_seen, cache_key) VALUES(1, 'photo', 'x', 'k')")
    db.commit()
    calls = []

    class Nas:
        def __init__(self, data):
            self.data = data

        def xl(self, item, ck):
            calls.append(item)
            return self.data

    assert library.is_image(jpeg) and not library.is_image(jpeg[:-20]) and not library.is_image(b"<html>404</html>")
    with pytest.raises(RuntimeError, match="not a JPEG or PNG"):
        library.image_path(db, cfg, 1, Nas(b'{"error": {"code": 801}}'))           # an error page is never stored
    xl = cfg.data_dir / "xl" / "000" / "1.jpg"
    assert not xl.exists()
    assert library.image_path(db, cfg, 1, Nas(jpeg)) == xl and xl.read_bytes() == jpeg and _mode(xl) == 0o600
    assert library.image_path(db, cfg, 1, Nas(b"never asked")) == xl and calls == [1, 1]   # a good file is reused
    xl.write_bytes(jpeg[: len(jpeg) // 2])                                        # truncated by a crash mid-write
    assert library.image_path(db, cfg, 1, None) is None                           # not shown as if it were fine
    assert library.image_path(db, cfg, 1, Nas(jpeg)) == xl and xl.read_bytes() == jpeg and calls == [1, 1, 1]
    xl.write_bytes(b"<html>not an image</html>")
    assert library.image_path(db, cfg, 1, Nas(jpeg)) == xl and xl.read_bytes() == jpeg
    assert not list(xl.parent.glob("*.tmp"))
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, "PNG")
    assert library.is_image(png.getvalue()) and not library.is_image(png.getvalue()[:-12])


def test_synology_photos_bridge_checks_version_and_names(tmp_path):
    ok = ("__version__ = '0.1.0'\nCONFIRMED_WRITES = {'person-add', 'person-remove', 'group-clean', 'person-name'}\n"
          "class SynoClient:\n    login = logout = ok = pages = fetch = index_status = check_allowed = None\n"
          "apis = journal = group_faces = redact = tcp_ok = load_config = js = None\n")
    p = tmp_path / "synophotos.py"
    p.write_text(ok)
    assert library.load_synophotos(p).__version__ == "0.1.0"
    p.write_text(ok.replace("0.1.0", "0.0.9"))
    with pytest.raises(SystemExit, match="needs 0.1.0 or later"):
        library.load_synophotos(p)
    p.write_text(ok.replace("__version__ = '0.1.0'\n", ""))
    with pytest.raises(SystemExit, match="version unknown"):
        library.load_synophotos(p)
    p.write_text(ok.replace("apis = journal = group_faces", "apis = journal"))
    with pytest.raises(SystemExit, match="lacks group_faces"):
        library.load_synophotos(p)
    p.write_text(ok.replace("fetch = index_status", "index_status"))
    with pytest.raises(SystemExit, match="lacks SynoClient.fetch"):
        library.load_synophotos(p)
    p.write_text(ok.replace("'person-name'", "'x'"))
    with pytest.raises(SystemExit, match="does not confirm"):
        library.load_synophotos(p)
    with pytest.raises(SystemExit, match="Install the synology-photos skill"):
        library.load_synophotos(tmp_path / "missing.py")
    assert library._version("1.2.3") == (1, 2, 3) and library._version("0.2") == (0, 2, 0) and library._version("0.10.0b1") > (0, 9, 9)
