"""The ledger: one SQLite file (WAL). Synology's person database is a projection of it.

Append-only where it matters: `labels` (owner truth), `writes` (every NAS write with its prior state) and
`cards`. `decisions`, `changes` and `clusters` are per run. Embeddings are float16 blobs.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY, filename TEXT, taken TEXT, filesize INTEGER, type TEXT, cache_key TEXT,
  persons_sig TEXT, first_seen TEXT, last_seen TEXT, faces_synced TEXT, local_path TEXT, detected TEXT);
CREATE TABLE IF NOT EXISTS syno_people(id INTEGER PRIMARY KEY, name TEXT, item_count INTEGER, seen TEXT);
CREATE TABLE IF NOT EXISTS syno_faces(
  face_id INTEGER PRIMARY KEY, item INTEGER, person_id INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL, seen TEXT);
CREATE INDEX IF NOT EXISTS syno_faces_item ON syno_faces(item);
CREATE INDEX IF NOT EXISTS syno_faces_person ON syno_faces(person_id);
CREATE TABLE IF NOT EXISTS faces(
  id INTEGER PRIMARY KEY, item INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL, det REAL, kps TEXT, source TEXT,
  syno_face_id INTEGER, iou REAL, UNIQUE(item, x1, y1, x2, y2));
CREATE INDEX IF NOT EXISTS faces_item ON faces(item);
CREATE INDEX IF NOT EXISTS faces_syno ON faces(syno_face_id);
CREATE TABLE IF NOT EXISTS embeddings(face INTEGER, model TEXT, vec BLOB, PRIMARY KEY(face, model));
CREATE TABLE IF NOT EXISTS labels(
  id INTEGER PRIMARY KEY, face INTEGER, value TEXT, source TEXT, via TEXT, at TEXT, run TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS labels_face ON labels(face);
CREATE TABLE IF NOT EXISTS runs(
  id TEXT PRIMARY KEY, started TEXT, finished TEXT, mode TEXT, stage TEXT, status TEXT, stats TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS calibration(
  run TEXT, model TEXT, status TEXT, floor REAL, margin REAL, sibling_margin REAL, n_pos INTEGER, n_neg INTEGER,
  accepted INTEGER, errors INTEGER, precision REAL, recall REAL, lower95 REAL, at TEXT, PRIMARY KEY(run, model));
CREATE TABLE IF NOT EXISTS decisions(
  run TEXT, face INTEGER, person TEXT, score REAL, runner TEXT, runner_score REAL, bin TEXT, status TEXT,
  reason TEXT, challenger TEXT, PRIMARY KEY(run, face));
CREATE INDEX IF NOT EXISTS decisions_status ON decisions(run, status);
CREATE TABLE IF NOT EXISTS clusters(run TEXT, face INTEGER, cluster INTEGER, pass INTEGER, PRIMARY KEY(run, face));
CREATE TABLE IF NOT EXISTS changes(
  id INTEGER PRIMARY KEY, run TEXT, seq INTEGER, kind TEXT, item INTEGER, face INTEGER, syno_face_id INTEGER,
  from_person INTEGER, to_person INTEGER, box TEXT, group_id INTEGER, person TEXT, score REAL, status TEXT,
  detail TEXT);
CREATE INDEX IF NOT EXISTS changes_run ON changes(run, status);
CREATE TABLE IF NOT EXISTS writes(
  id INTEGER PRIMARY KEY, change INTEGER, run TEXT, at TEXT, api TEXT, method TEXT, params TEXT, prior TEXT,
  result TEXT, ok INTEGER);
CREATE TABLE IF NOT EXISTS cards(
  id INTEGER PRIMARY KEY, run TEXT, kind TEXT, face INTEGER, person TEXT, alt TEXT, bin TEXT, payload TEXT,
  created TEXT, answer TEXT, answered TEXT);
CREATE INDEX IF NOT EXISTS cards_open ON cards(answer);
CREATE TABLE IF NOT EXISTS bumps(person TEXT, bin TEXT, amount REAL, staged INTEGER, at TEXT, reason TEXT,
  PRIMARY KEY(person, bin));
"""

# Exported every run as JSONL (embeddings are recomputable and large, so they are not exported).
EXPORT_TABLES = ["items", "syno_people", "syno_faces", "faces", "labels", "runs", "calibration", "decisions",
                 "changes", "writes", "cards", "bumps"]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- private files -----------------------------------------------------------------------------------------
# Everything the curator keeps (ledger, face crops, previews, exports, reports) is personal: files are created
# for this user only. Folders are made private only when the curator creates them; a folder the user pointed
# at keeps the permissions the user gave it.
def private_dir(p: Path) -> Path:
    """Create `p` (and any missing parent) for this user only. Folders that exist are left as they are."""
    missing = []
    q = p
    while not q.is_dir() and q != q.parent:
        missing.append(q)
        q = q.parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=0o700)
            os.chmod(d, 0o700)                    # mkdir's mode is cut by the umask; chmod is not
        except FileExistsError:
            pass                                  # another thread made it meanwhile
    return p


def private_file(p: Path) -> Path:
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return p


def atomic_write(p: Path, data: bytes | str) -> Path:
    """Write to a temporary name next to `p`, then rename: a reader never sees a half-written file."""
    private_dir(p.parent)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


_READY: set[str] = set()          # ledgers whose schema was checked by this process
_READY_LOCK = threading.Lock()


def connect(path: Path, timeout: float = 60, readonly: bool = False) -> sqlite3.Connection:
    """A connection to the ledger. The schema and `meta` are set up once per process, so later connections
    (one per request in the review page) never need the write lock just to open. `readonly`: the connection
    refuses writes (`PRAGMA query_only`), so it reads while a run is writing."""
    key = str(path)
    with _READY_LOCK:
        if key not in _READY or not path.is_file():
            private_dir(path.parent)
            first = sqlite3.connect(key, timeout=timeout)
            try:
                first.execute("PRAGMA journal_mode=WAL")
                first.executescript(SCHEMA)
                first.execute("INSERT OR IGNORE INTO meta VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
                first.commit()
            finally:
                first.close()
            for f in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
                if f.exists():
                    private_file(f)
            _READY.add(key)
    db = sqlite3.connect(key, timeout=timeout)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    if readonly:
        db.execute("PRAGMA query_only=ON")
    return db


def get_meta(db: sqlite3.Connection, k: str) -> str | None:
    r = db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else None


def set_meta(db: sqlite3.Connection, k: str, v: str | None) -> None:
    if v is None:
        db.execute("DELETE FROM meta WHERE k=?", (k,))
    else:
        db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))


def ledger_path(data_dir: Path) -> Path:
    return data_dir / "ledger.sqlite"


# --- vectors ---------------------------------------------------------------------------------------------
def pack(v: np.ndarray) -> bytes:
    v = np.asarray(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    return v.astype(np.float16).tobytes()


def unpack(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float16).astype(np.float32)


def vectors(db: sqlite3.Connection, model: str, faces: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(face ids, L2-normalised float32 matrix) for one model."""
    if faces is None:
        rows = db.execute("SELECT face, vec FROM embeddings WHERE model=? ORDER BY face", (model,)).fetchall()
    else:
        want = sorted(set(int(f) for f in faces))
        rows = []
        for i in range(0, len(want), 900):
            chunk = want[i:i + 900]
            rows += db.execute(f"SELECT face, vec FROM embeddings WHERE model=? AND face IN ({','.join('?' * len(chunk))}) "
                               "ORDER BY face", (model, *chunk)).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 512), dtype=np.float32)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([unpack(r[1]) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(1e-6)
    return ids, mat


# --- labels ----------------------------------------------------------------------------------------------
LABEL_SOURCES_TRUSTED = ("owner",)


def add_label(db: sqlite3.Connection, face: int, value: str, via: str, run: str | None = None,
              source: str = "owner", note: str | None = None) -> None:
    db.execute("INSERT INTO labels(face, value, source, via, at, run, note) VALUES(?,?,?,?,?,?,?)",
               (face, value, source, via, now(), run, note))


def latest_labels(db: sqlite3.Connection) -> dict[int, str]:
    """face -> latest owner label. Values: person key | 'not:<k>[,<k>]' | 'unknown' | 'not_face' | 'syno:<pid>'."""
    out: dict[int, str] = {}
    for r in db.execute("SELECT face, value FROM labels WHERE source IN (%s) ORDER BY id"
                        % ",".join("?" * len(LABEL_SOURCES_TRUSTED)), LABEL_SOURCES_TRUSTED):
        out[r[0]] = r[1]
    return out


# --- runs ------------------------------------------------------------------------------------------------
def run_row(db: sqlite3.Connection, run: str) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()


def set_run(db: sqlite3.Connection, run: str, **fields: Any) -> None:
    if not run_row(db, run):
        db.execute("INSERT INTO runs(id, started, status) VALUES(?,?,?)", (run, now(), "running"))
    for k, v in fields.items():
        if k == "stats" and not isinstance(v, str):
            old = json.loads((run_row(db, run)["stats"] or "{}"))
            old.update(v)
            v = json.dumps(old)
        db.execute(f"UPDATE runs SET {k}=? WHERE id=?", (v, run))
    db.commit()


def run_stats(db: sqlite3.Connection, run: str) -> dict:
    r = run_row(db, run)
    return json.loads(r["stats"] or "{}") if r else {}


# --- export / backup -------------------------------------------------------------------------------------
def export_jsonl(db: sqlite3.Connection, out_dir: Path) -> list[Path]:
    private_dir(out_dir)
    paths = []
    for t in EXPORT_TABLES:
        p = out_dir / f"{t}.jsonl.gz"
        tmp = p.with_name(p.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as raw, gzip.open(raw, "wt") as f:
            for r in db.execute(f"SELECT * FROM {t}"):
                f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
        os.replace(tmp, p)
        paths.append(private_file(p))
    return paths


def backup(db: sqlite3.Connection, dest: Path, keep: int = 3) -> Path:
    """A consistent copy of the ledger (SQLite backup API), written under a temporary name and renamed."""
    private_dir(dest)
    p = dest / f"ledger-{datetime.now():%Y-%m-%d}.sqlite"
    tmp = p.with_name(p.name + ".tmp")
    if tmp.exists():
        tmp.unlink()              # a leftover of an interrupted backup of ours
    os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT, 0o600))
    target = sqlite3.connect(str(tmp))
    db.backup(target)
    target.close()
    os.replace(tmp, p)
    private_file(p)
    for old in sorted(dest.glob("ledger-*.sqlite"))[:-keep]:
        old.unlink()  # our own backup copies, never photos
    return p


def integrity_ok(db: sqlite3.Connection) -> bool:
    return db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
