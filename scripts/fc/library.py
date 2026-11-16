"""Library state: optional face-cache import, Synology sync (one login per run), box matching, detection/embedding.

All NAS traffic goes through the synology-photos skill's own client (`SynoClient`), loaded in-process, so its
transport guard (no delete/move/copy, forbidden API prefixes) applies to every call made here. That skill's
command-line gates (dry run, `--yes`) are NOT in this path: the gates in `changes.py` and `run.py` are the only
ones between a plan and the NAS. The script path comes from the config (`synology_photos`) and is executed as
code: it is trusted configuration.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import numpy as np

from fc import db as D
from fc.config import Config

ITEM_V, FOLDER_V, PERSON_V, THUMB_V = 7, 2, 1, 2
WRITES_USED = {"person-add", "person-remove", "group-clean", "person-name"}  # must be in CONFIRMED_WRITES
SYNOPHOTOS_MIN = (0, 1, 0)
# Python internals of synology-photos this skill calls: the two skills are updated together.
SYNOPHOTOS_NAMES = ("SynoClient", "apis", "journal", "group_faces", "redact", "tcp_ok", "load_config", "js",
                    "CONFIRMED_WRITES")
SYNOPHOTOS_CLIENT_NAMES = ("login", "logout", "ok", "pages", "fetch", "index_status", "check_allowed")


def _version(text: str) -> tuple[int, ...]:
    out = []
    for part in str(text).split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))


# --- synology-photos bridge ------------------------------------------------------------------------------
def load_synophotos(path: Path) -> ModuleType:
    """Load the synology-photos CLI as a module (it is code: the path is trusted config) and check that it is
    the version and shape this skill was written against."""
    path = path.expanduser()
    need = ".".join(str(x) for x in SYNOPHOTOS_MIN)
    if not path.is_file():
        raise SystemExit(f"synology-photos CLI not found at {path} (config: synology_photos). Install the "
                         f"synology-photos skill, version {need} or later.")
    spec = importlib.util.spec_from_file_location("synophotos", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["synophotos"] = mod
    spec.loader.exec_module(mod)
    have = getattr(mod, "__version__", None)
    if have is None or _version(have) < SYNOPHOTOS_MIN:
        raise SystemExit(f"synology-photos at {path} is version {have or 'unknown'}; face-curator needs {need} "
                         "or later. Update the synology-photos skill.")
    gone = [n for n in SYNOPHOTOS_NAMES if not hasattr(mod, n)]
    gone += [f"SynoClient.{n}" for n in SYNOPHOTOS_CLIENT_NAMES if not hasattr(getattr(mod, "SynoClient", None), n)]
    if gone:
        raise SystemExit(f"synology-photos {have} at {path} lacks {', '.join(gone)}, which face-curator calls. "
                         "The two skills must be updated together: update face-curator too.")
    missing = WRITES_USED - set(getattr(mod, "CONFIRMED_WRITES", set()))
    if missing:
        raise SystemExit(f"synology-photos does not confirm {sorted(missing)}; refusing to run")
    return mod


class Nas:
    """One logged-in session for a whole run. `sp` is the synophotos module, `c` its SynoClient."""

    def __init__(self, cfg: Config, sp: ModuleType | None = None, client: Any = None):
        self.cfg = cfg
        self.sp = sp or load_synophotos(cfg.path_of("synology_photos"))
        self.space = cfg["space"]
        self.a = self.sp.apis(self.space)
        self.c = client
        self._own = client is None

    def __enter__(self) -> "Nas":
        if self.c is None:
            self.c = self.sp.SynoClient(self.sp.load_config())
            self.c.login()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._own and self.c is not None:
            self.c.logout()

    # reads. A dropped connection (the NAS closing a keep-alive socket, Wi-Fi blips) is retried for reads only,
    # with growing waits; the session stays logged in. Writes are never retried.
    READ_WAITS = (5, 15, 45, 120)

    def _read(self, fn: Callable[[], Any], what: str) -> Any:
        import requests

        for wait in (*self.READ_WAITS, None):
            try:
                return fn()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if wait is None:
                    raise
                print(f"  {what}: connection dropped ({type(e).__name__}), retrying in {wait}s", flush=True)
                self.sleep(wait)

    sleep = staticmethod(time.sleep)

    def people(self) -> list[dict]:
        return self._read(lambda: list(self.c.pages(self.a["person"], "list", PERSON_V)), "people")

    def folder_children(self, fid: int) -> list[int]:
        return self._read(lambda: [f["id"] for f in self.c.pages(self.a["folder"], "list", FOLDER_V, id=fid,
                                                                 sort_by="filename", sort_direction="asc")],
                          f"folder {fid}")

    def folder_items(self, fid: int) -> list[dict]:
        return self._read(lambda: list(self.c.pages(self.a["item"], "list", ITEM_V, folder_id=fid,
                                                    additional=["thumbnail", "person"], sort_by="takentime",
                                                    sort_direction="asc")), f"items of folder {fid}")

    def faces(self, item: int) -> list[dict]:
        return self._read(lambda: self.c.ok(self.a["item"], "list_face", ITEM_V, id_item=item).get("list", []),
                          f"faces of {item}")

    def person_items(self, pid: int) -> list[int]:
        return self._read(lambda: [it["id"] for it in self.c.pages(self.a["item"], "list", ITEM_V, person_id=pid)],
                          f"items of person {pid}")

    def xl(self, item: int, cache_key: str) -> bytes:
        return self._read(lambda: self.c.fetch(self.a["thumb"], "get", THUMB_V, id=str(item),
                                               cache_key=str(cache_key or ""), type="unit", size="xl"), f"preview {item}")

    def index_idle(self) -> bool:
        return self.c.index_status(self.a["item"]) == "idle"

    def wait_idle(self, minutes: float, poll: float = 30.0, sleep: Callable[[float], None] = time.sleep) -> bool:
        t_end = time.monotonic() + minutes * 60
        while True:
            if self.index_idle():
                return True
            if time.monotonic() > t_end:
                return False
            sleep(poll)


# --- helpers ---------------------------------------------------------------------------------------------
def syno_box(b: dict) -> tuple[float, float, float, float]:
    return (float(b["top_left"]["x"]), float(b["top_left"]["y"]), float(b["bottom_right"]["x"]),
            float(b["bottom_right"]["y"]))


def box_dict(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"top_left": {"x": round(x1, 4), "y": round(y1, 4)}, "bottom_right": {"x": round(x2, 4), "y": round(y2, 4)}}


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(local: list[tuple[int, tuple]], remote: list[tuple[int, tuple]], min_iou: float = 0.3
                ) -> dict[int, tuple[int, float]]:
    """Greedy one-to-one by IoU, highest first. Returns local id -> (remote id, iou)."""
    pairs = sorted(((iou(lb, rb), lid, rid) for lid, lb in local for rid, rb in remote), reverse=True)
    used_l, used_r, out = set(), set(), {}
    for v, lid, rid in pairs:
        if v < min_iou:
            break
        if lid in used_l or rid in used_r:
            continue
        used_l.add(lid)
        used_r.add(rid)
        out[lid] = (rid, v)
    return out


def epoch_to_iso(t: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat(timespec="seconds")[:19]
    except (TypeError, ValueError, OSError):
        return None


def rbox(b: Iterable[float]) -> tuple[float, ...]:
    return tuple(round(float(v), 4) for v in b)


# --- optional: import an existing buffalo_l face cache ---------------------------------------------------
FACE_CACHE_FILES = ("cache/ingest_index.jsonl", "cache/faces/faces.jsonl")


def import_face_cache(db: sqlite3.Connection, root: Path, model: str = "buffalo_l", log=print) -> dict:
    """Import faces and buffalo_l embeddings somebody already computed, so the first run does not embed them
    again. `root` has the layout of the photo-picker skill's library folder:

    - `cache/ingest_index.jsonl`: one JSON object per line with `photo_id` (`t<Synology item id>` for the Shared
      Space), `syno_id`, `file` (path of the local image, relative to `root`) and `taken_at`;
    - `cache/faces/faces.jsonl`: one object per photo with `photo_id` and `faces`: a list of `{bbox: [x1, y1,
      x2, y2] as fractions of the upright image, det: score, emb: base64 of 512 float16 values}`.

    Only `t<id>` photos are imported. Idempotent."""
    import base64

    root = Path(root).expanduser()
    idx_path, faces_path = (root / f for f in FACE_CACHE_FILES)
    if not idx_path.is_file() or not faces_path.is_file():
        raise SystemExit(f"no face cache under {root}: expected {' and '.join(FACE_CACHE_FILES)}")
    n_items = n_faces = n_skip = 0
    ts = D.now()
    with open(idx_path) as f:
        for line in f:
            r = json.loads(line)
            if not str(r.get("photo_id", "")).startswith("t"):
                n_skip += 1
                continue
            sid = int(r["syno_id"])
            local = str(root / r["file"]) if r.get("file") else None
            db.execute("INSERT INTO items(id, taken, local_path, first_seen, last_seen) VALUES(?,?,?,?,?) "
                       "ON CONFLICT(id) DO UPDATE SET local_path=excluded.local_path, "
                       "taken=COALESCE(items.taken, excluded.taken)", (sid, r.get("taken_at"), local, ts, ts))
            n_items += 1
    with open(faces_path) as f:
        for line in f:
            r = json.loads(line)
            pid = str(r["photo_id"])
            if not pid.startswith("t"):
                continue
            sid = int(pid[1:])
            db.execute("INSERT OR IGNORE INTO items(id, first_seen, last_seen) VALUES(?,?,?)", (sid, ts, ts))
            db.execute("UPDATE items SET detected=COALESCE(detected, 'face-cache') WHERE id=?", (sid,))
            for fc in r["faces"]:
                b = rbox(fc["bbox"])
                cur = db.execute("INSERT OR IGNORE INTO faces(item, x1, y1, x2, y2, det, source) VALUES(?,?,?,?,?,?,?)",
                                 (sid, *b, fc.get("det"), "face-cache"))
                fid = cur.lastrowid if cur.rowcount else db.execute(
                    "SELECT id FROM faces WHERE item=? AND x1=? AND y1=? AND x2=? AND y2=?", (sid, *b)).fetchone()[0]
                vec = np.frombuffer(base64.b64decode(fc["emb"]), dtype=np.float16).astype(np.float32)
                db.execute("INSERT OR IGNORE INTO embeddings(face, model, vec) VALUES(?,?,?)", (fid, model, D.pack(vec)))
                n_faces += 1
    db.commit()
    out = {"items": n_items, "faces": n_faces, "skipped_other_ids": n_skip}
    log(f"import: {out}")
    return out


# --- Synology sync ---------------------------------------------------------------------------------------
def sync_people(db: sqlite3.Connection, nas: Nas) -> int:
    ts = D.now()
    rows = nas.people()
    for p in rows:
        db.execute("INSERT INTO syno_people(id, name, item_count, seen) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                   "name=excluded.name, item_count=excluded.item_count, seen=excluded.seen",
                   (p["id"], p.get("name") or "", p.get("item_count"), ts))
    db.commit()
    return len(rows)


def sync_items(db: sqlite3.Connection, nas: Nas, root_folder: int = 1, log=print) -> dict:
    """Walk the folder tree (the confirmed way to list a whole space), record every item with its person-id
    signature. Returns {seen, new, changed}; changed items get faces_synced=NULL so their faces are re-read."""
    ts = D.now()
    queue, seen, new, changed = [root_folder], 0, 0, 0
    known = {r[0]: r[1] for r in db.execute("SELECT id, persons_sig FROM items")}
    while queue:
        fid = queue.pop()
        queue += nas.folder_children(fid)
        for it in nas.folder_items(fid):
            seen += 1
            ad = it.get("additional") or {}
            sig = ",".join(str(x) for x in sorted({p["id"] for p in ad.get("person") or []}))
            ck = (ad.get("thumbnail") or {}).get("cache_key")
            prev = known.get(it["id"], "missing")
            if prev == "missing":
                new += 1
            elif prev is not None and prev != sig:
                changed += 1
            db.execute(
                "INSERT INTO items(id, filename, taken, filesize, type, cache_key, persons_sig, first_seen, last_seen) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET filename=excluded.filename, "
                "taken=excluded.taken, filesize=excluded.filesize, type=excluded.type, cache_key=excluded.cache_key, "
                "persons_sig=excluded.persons_sig, last_seen=excluded.last_seen, "
                "faces_synced=CASE WHEN items.persons_sig IS NOT excluded.persons_sig THEN NULL ELSE items.faces_synced END",
                (it["id"], it.get("filename"), epoch_to_iso(it.get("time")), it.get("filesize"), it.get("type"), ck, sig,
                 ts, ts))
        db.commit()
    out = {"seen": seen, "new": new, "changed": changed}
    log(f"sync items: {out}")
    return out


def items_needing_faces(db: sqlite3.Connection, run_no: int, fraction: int) -> list[int]:
    """Never synced, membership changed (faces_synced NULL), or in this run's rolling slice."""
    rows = db.execute("SELECT id, faces_synced, persons_sig, type FROM items WHERE last_seen IS NOT NULL").fetchall()
    out = []
    for r in rows:
        if r["type"] == "video":
            continue
        has_people = bool(r["persons_sig"])
        has_local = db.execute("SELECT 1 FROM faces WHERE item=? LIMIT 1", (r["id"],)).fetchone() is not None
        if not (has_people or has_local):
            continue
        if r["faces_synced"] is None or (fraction and r["id"] % fraction == run_no % fraction):
            out.append(r["id"])
    return out


def store_syno_faces(db: sqlite3.Connection, item: int, faces: list[dict]) -> None:
    ts = D.now()
    db.execute("DELETE FROM syno_faces WHERE item=?", (item,))  # ledger rows only: Synology's current view
    for f in faces:
        b = syno_box(f["face_bounding_box"])
        db.execute("INSERT OR REPLACE INTO syno_faces VALUES(?,?,?,?,?,?,?,?)",
                   (f["face_id"], item, f.get("person_id") or None, *b, ts))
    match_item(db, item)


def match_item(db: sqlite3.Connection, item: int) -> None:
    local = [(r[0], (r[1], r[2], r[3], r[4])) for r in db.execute("SELECT id,x1,y1,x2,y2 FROM faces WHERE item=?", (item,))]
    remote = [(r[0], (r[1], r[2], r[3], r[4])) for r in
              db.execute("SELECT face_id,x1,y1,x2,y2 FROM syno_faces WHERE item=?", (item,))]
    m = match_boxes(local, remote)
    for lid, _ in local:
        rid, v = m.get(lid, (None, None))
        db.execute("UPDATE faces SET syno_face_id=?, iou=? WHERE id=?", (rid, v, lid))


def sync_faces(db: sqlite3.Connection, nas: Nas, items: list[int], log=print, every: int = 200) -> int:
    t0 = time.monotonic()
    for n, item in enumerate(items, 1):
        store_syno_faces(db, item, nas.faces(item))
        db.execute("UPDATE items SET faces_synced=? WHERE id=?", (D.now(), item))
        if n % every == 0:
            db.commit()
            log(f"  faces {n}/{len(items)} ({n / (time.monotonic() - t0):.1f}/s)")
    db.commit()
    return len(items)


# --- detection + embedding -------------------------------------------------------------------------------
def image_path(db: sqlite3.Connection, cfg: Config, item: int, nas: Nas | None) -> Path | None:
    r = db.execute("SELECT local_path, cache_key FROM items WHERE id=?", (item,)).fetchone()
    if r and r["local_path"] and Path(r["local_path"]).is_file() and \
            Path(r["local_path"]).suffix.lower() in (".jpg", ".jpeg", ".png"):
        return Path(r["local_path"])
    xl = cfg.data_dir / "xl" / f"{item // 1000:03d}" / f"{item}.jpg"
    if xl.is_file() and complete_image(xl):
        return xl                                   # a truncated or non-image file is fetched again, never reused
    if nas is None or r is None:
        return None
    data = nas.xl(item, r["cache_key"])
    if not is_image(data):
        raise RuntimeError(f"preview of item {item} is not a JPEG or PNG image ({len(data)} bytes)")
    D.atomic_write(xl, data)
    return xl


JPEG_MAGIC, PNG_MAGIC = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"


def is_image(data: bytes) -> bool:
    """JPEG or PNG by magic bytes, and complete: a JPEG ends with its end-of-image marker, a PNG with IEND."""
    if data[:3] == JPEG_MAGIC:
        return b"\xff\xd9" in data[-4096:]
    if data[:8] == PNG_MAGIC:
        return b"IEND" in data[-16:]
    return False


def complete_image(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(8)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read()
    except OSError:
        return False
    return size > 8 and is_image(head + tail if size > 4104 else _whole(p))


def _whole(p: Path) -> bytes:
    return p.read_bytes()


def items_needing_embedding(db: sqlite3.Connection, models: list[str]) -> list[int]:
    q = ("SELECT id FROM items WHERE last_seen IS NOT NULL AND (type IS NULL OR type != 'video') AND detected IS NULL "
         "UNION SELECT DISTINCT f.item FROM faces f WHERE " +
         " OR ".join("NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.face=f.id AND e.model=?)" for _ in models))
    return sorted(r[0] for r in db.execute(q, models))


def embed_items(db: sqlite3.Connection, cfg: Config, engine: Any, items: list[int], run: str,
                nas: Nas | None = None, log=print, every: int = 200) -> dict:
    """Detect on the upright image, match detections to known faces (IoU >= 0.5), add new faces, and compute
    every missing embedding for the active models."""
    import requests

    t0, n_new, n_emb, n_noimg, n_nopreview, streak = time.monotonic(), 0, 0, 0, 0, 0
    for n, item in enumerate(items, 1):
        try:
            p = image_path(db, cfg, item, nas)
            streak = 0
        except (requests.exceptions.HTTPError, RuntimeError) as e:
            # One photo without a preview (DSM answers 404 or an API error) is skipped and marked, so it is not
            # retried every month; a run of failures means the NAS is the problem, so the stage stops.
            streak += 1
            if streak >= 20:
                raise
            n_nopreview += 1
            db.execute("UPDATE items SET detected=? WHERE id=?", (f"no-preview:{run}", item))
            log(f"  item {item}: no preview ({type(e).__name__}), skipped")
            continue
        from fc.embed import load_image
        img = load_image(p, cfg["detect"]["max_side"]) if p else None
        if img is None:
            n_noimg += 1
            continue
        h, w = img.shape[:2]
        dets = engine.faces(img)
        known = [(r[0], (r[1], r[2], r[3], r[4])) for r in db.execute("SELECT id,x1,y1,x2,y2 FROM faces WHERE item=?", (item,))]
        m = match_boxes([(i, d["box"]) for i, d in enumerate(dets)], known, min_iou=0.5)
        face_kps: dict[int, np.ndarray] = {}
        for i, d in enumerate(dets):
            if i in m:
                fid = m[i][0]
            else:
                if d["det"] < cfg["detect"]["min_det"]:
                    continue
                cur = db.execute("INSERT OR IGNORE INTO faces(item, x1, y1, x2, y2, det, source) VALUES(?,?,?,?,?,?,?)",
                                 (item, *rbox(d["box"]), d["det"], "curator"))
                if not cur.rowcount:
                    continue
                fid = cur.lastrowid
                n_new += 1
            kf = d["kps"] / np.array([w, h], dtype=np.float32)
            db.execute("UPDATE faces SET kps=? WHERE id=?", (json.dumps(np.round(kf, 5).tolist()), fid))
            face_kps[fid] = d["kps"]
        for model in engine.models:
            have = {r[0] for r in db.execute("SELECT face FROM embeddings WHERE model=? AND face IN (%s)"
                                             % ",".join("?" * len(face_kps)), (model, *face_kps))} if face_kps else set()
            todo = [f for f in face_kps if f not in have]
            if todo:
                vecs = engine.rec(model).embed(img, [face_kps[f] for f in todo])
                for f, v in zip(todo, vecs):
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES(?,?,?)", (f, model, D.pack(v)))
                    n_emb += 1
        db.execute("UPDATE items SET detected=COALESCE(detected, ?) WHERE id=?", (run, item))
        match_item(db, item)
        if n % every == 0:
            db.commit()
            log(f"  embed {n}/{len(items)} ({n / (time.monotonic() - t0):.1f} img/s)")
    db.commit()
    out = {"items": len(items), "new_faces": n_new, "embeddings": n_emb, "no_image": n_noimg,
           "no_preview": n_nopreview}
    log(f"embed: {out}")
    return out
