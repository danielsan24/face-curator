"""Configuration (no secrets): ~/.config/face-curator/config.yaml merged over DEFAULTS.

People come in two circles (see docs/design.md):
- `close_circle`: the people you label yourself (review cards). Children carry `born`, which drives age galleries.
- `outer_circle`: everyone else the curator should look after. Either listed under `outer_circle.people`, or taken
  from Synology with `outer_circle.from_synology: true` (every Synology-named person with at least `min_faces`
  tagged faces in the ledger, minus `exclude` and the close circle).
The older `people:` map with a `tier:` per person is still read.
"""

from __future__ import annotations

import math
import copy
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from fc.db import private_dir

CONFIG_PATH = Path(os.environ.get("FACECURATOR_CONFIG", "~/.config/face-curator/config.yaml")).expanduser()

DEFAULTS: dict[str, Any] = {
    "space": "team",
    "synology_photos": "~/.claude/skills/synology-photos/scripts/synophotos.py",
    "data_dir": "~/Library/Application Support/face-curator",
    "backup_dir": None,                      # default: <data_dir>/backups
    "face_cache_dir": None,                  # optional: a photo-picker library folder with a buffalo_l face cache
    "close_circle": {},
    "outer_circle": {"from_synology": False, "min_faces": 20, "exclude": [], "people": {}},
    "people": {},                            # legacy: {key: {name, born, died, tier: close|outer}}
    "siblings": [],                          # groups of look-alike siblings: a larger margin between them
    "lookalikes": [],                        # other look-alike groups (e.g. cousins): extra blind checks only
    # Age bins in months: [0,6) [6,12) [12,24) [24,48) [48,72) [72,96) [96,inf). Neighbouring bins share
    # `bin_overlap_months` of exemplars so a birthday does not split a gallery.
    "age_bins_months": [0, 6, 12, 24, 48, 72, 96],
    "bin_overlap_months": 2,
    "gallery": {"target": 30, "max": 50, "min_active": 8, "batch": 24},
    "models": {"reference": "buffalo_l", "challenger": None},
    "detect": {"min_det": 0.6, "min_face_height": 0.04, "max_side": 1600},
    "assign": {
        "precision_target": 0.995,
        # Reviewed defaults; see docs/lessons.md. Each run's calibration can only raise the floor.
        "default_floor": 0.50,
        "min_floor": 0.50,         # calibration may raise the floor, never lower it below this
        "baby_floor": 0.65,        # faces of children under baby_months: newborn relatives look alike
        "baby_months": 12,
        "default_margin": 0.08,           # calibration never goes below these two margins either
        "default_sibling_margin": 0.12,
        "no_date_floor_bump": 0.05,
        # Outer circle: gallery = the owner's labels plus Synology's own tags, filtered for self-consistency and
        # distance from the close-circle galleries. Reviewed defaults; see docs/lessons.md.
        "outer_consistency": 0.55,  # a tagged face is used if its top-5 mean to the person's other tags >= this
        "outer_close_gap": 0.05,    # ... and it is closer to them than to any close-circle gallery by this much
        "outer_max": 200,           # Synology-tagged faces per outer gallery, spread across capture years
        "outer_min": 8,             # fewer usable faces: no gallery (the person stays unassigned)
        # Reviewed default (blind check `edges build --outer`); see docs/lessons.md.
        "outer_floor_bump": 0.05,   # outer circle accepts from 0.55 (close circle from 0.50)
        "safety_bump": 0.02,
        "error_bump": 0.03,
        "min_calibration_accepts": 30,
        "reject_floor": 0.30,      # Synology says P but score for P is below this: a review card, never a write
        "impostor_sample": 5000,
    },
    "cluster": {"distance": 0.34, "schedule": [25, 10, 5, 3]},  # reviewed default; see docs/lessons.md
    "changes": {"group_merge_min_accepted": 0.95, "group_min_faces": 3, "add_missed_faces": False,
                "min_det_add": 0.75, "min_height_add": 0.06},
    # Per run. The code enforces maxima the config cannot exceed (HARD_CAPS). A merge of more than
    # merge_auto_max_faces faces is never written by an automatic run: it waits for the owner.
    "caps": {"face_edits": 3000, "merges": 20, "merge_auto_max_faces": 200},
    # Auto mode (the scheduled run writes to the NAS unattended) is off unless policy is "auto" AND every
    # condition of run.auto_gate holds. The code enforces floors (MODE_FLOORS) and ceilings (AUTO_CEILING).
    "mode": {"policy": "staged", "staged_runs_before_auto": 3, "clean_audits_for_auto": 2,
             "audit_min_judged": 200, "min_judged_per_run": 20, "min_lower95": 0.99,
             "auto_max_face_edits": 500, "auto_max_merges": 5},
    "cards": {"random": 8, "sibling_margin": 6, "disagreement": 3, "bridge": 3, "max_waiting": 40},
    "sync": {"rolling_fraction": 12},        # re-read 1/12 of the library's faces every run
    "runner": {"require_ac": True, "require_lan": True, "index_wait_minutes": 30},
    "review": {"port": 8765},
}


# Limits the config cannot loosen (docs/design.md, "Staged until earned").
HARD_CAPS = {"face_edits": 10000, "merges": 50, "merge_auto_max_faces": 200}             # config values above: clamped
MODE_FLOORS = {"staged_runs_before_auto": 3, "clean_audits_for_auto": 2, "audit_min_judged": 200,
               "min_judged_per_run": 20, "min_lower95": 0.98}                                # config values below: raised
AUTO_CEILING = {"auto_max_face_edits": 500, "auto_max_merges": 5}                           # per automatic run


def _number(kind: type, v: Any, key: str) -> Any:
    """A finite number from the config, or a clear stop: .nan, .inf or text never weaken a floor or a cap."""
    try:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError
        return kind(v)
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(f"config {key}: {v!r} is not a number") from None


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass(frozen=True)
class Person:
    key: str
    name: str                 # Synology person name (exact)
    born: date | None = None  # children: drives age bins and the birth exclusion
    died: str | None = None   # "YYYY" or "YYYY-MM": no face after this
    tier: str | None = None   # "close": owner-labelled gallery, owner cards; "outer": gallery seeded from Synology
                              # tags (filtered) plus owner labels; None: not assigned by the curator

    @property
    def gallery(self) -> bool:
        return self.tier is not None

    @property
    def outer(self) -> bool:
        return self.tier == "outer"

    @property
    def child(self) -> bool:
        return self.born is not None

    def allowed_on(self, day: str | None) -> bool:
        if not day:
            return True
        if self.born and day < self.born.isoformat():
            return False
        if self.died and day[: len(self.died)] > self.died:
            return False
        return True


TIERS = ("close", "outer")


def _tier(key: str, v: dict) -> str | None:
    t = v.get("tier")
    if t is None and "gallery" in v:          # older configs: gallery: true -> close, gallery: seed -> outer
        t = {True: "close", "seed": "outer"}.get(v["gallery"])
    if t is not None and t not in TIERS:
        raise SystemExit(f"config: people.{key}.tier must be one of {TIERS}, not {t!r}")
    return t


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "person"


class Config:
    def __init__(self, raw: dict | None = None, path: Path | None = None):
        self.path = path
        self.raw = _merge(DEFAULTS, raw or {})
        self.people: dict[str, Person] = {}
        for k, v in (self.raw.get("people") or {}).items():
            self._add(k, v, _tier(k, v))
        for k, v in (self.raw.get("close_circle") or {}).items():
            self._add(k, v, "close")
        for k, v in (self.raw["outer_circle"].get("people") or {}).items():
            self._add(k, v, "outer")
        self.siblings = self._groups("siblings")
        self.lookalikes = self._groups("lookalikes")

    def _add(self, key: str, v: dict | str, tier: str | None) -> None:
        v = {"name": v} if isinstance(v, str) else v
        if key in self.people:
            raise SystemExit(f"config: person {key!r} is defined twice")
        born = v.get("born")
        self.people[key] = Person(key=key, name=v["name"], born=date.fromisoformat(str(born)) if born else None,
                                  died=str(v["died"]) if v.get("died") else None, tier=tier)

    def _groups(self, field: str) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        groups = self.raw.get(field) or []
        unknown = [k for g in groups for k in g if k not in self.people]
        if unknown:
            raise SystemExit(f"config: {field} refer to unknown people {unknown}")
        for group in groups:
            for k in group:
                out.setdefault(k, set()).update(x for x in group if x != k)
        return out

    def add_synology_outer(self, counts: dict[str, int]) -> list[str]:
        """outer_circle.from_synology: every Synology-named person with >= min_faces faces joins the outer circle,
        unless excluded or already configured. `counts` is {Synology name: tagged faces}. Returns the added keys."""
        oc = self.raw["outer_circle"]
        if not oc.get("from_synology"):
            return []
        known = {p.name for p in self.people.values()}
        skip = {str(x).casefold() for x in oc.get("exclude") or []}
        added = []
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            if n < oc.get("min_faces", 20) or not name.strip() or name in known or name.casefold() in skip:
                continue
            key, i = slug(name), 2
            while key in self.people:
                key, i = f"{slug(name)}_{i}", i + 1
            self.people[key] = Person(key=key, name=name, tier="outer")
            added.append(key)
        return added

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    @property
    def gallery_people(self) -> list[Person]:
        """Both circles: everyone the curator assigns faces to."""
        return [p for p in self.people.values() if p.gallery]

    @property
    def close_people(self) -> list[Person]:
        """The close circle: owner-labelled galleries; the only people on owner cards, studies and edge rounds."""
        return [p for p in self.people.values() if p.tier == "close"]

    @property
    def outer_people(self) -> list[Person]:
        return [p for p in self.people.values() if p.tier == "outer"]

    def by_name(self, name: str) -> Person | None:
        return next((p for p in self.people.values() if p.name == name), None)

    def path_of(self, key: str) -> Path:
        return Path(self.raw[key]).expanduser()

    # Folders are created for this user only (0700); an existing folder keeps the permissions it has.
    @property
    def data_dir(self) -> Path:
        return private_dir(self.path_of("data_dir"))

    @property
    def backup_dir(self) -> Path:
        return private_dir(Path(self.raw["backup_dir"]).expanduser() if self.raw.get("backup_dir")
                           else self.data_dir / "backups")

    @property
    def models_dir(self) -> Path:
        return private_dir(self.data_dir / "models")

    @property
    def caps(self) -> dict[str, int]:
        """The per-run caps, never above the maxima fixed in code."""
        c = self.raw["caps"]
        return {k: max(0, min(_number(int, DEFAULTS["caps"][k] if c.get(k) is None else c[k], f"caps.{k}"),
                              HARD_CAPS[k])) for k in HARD_CAPS}

    @property
    def mode(self) -> dict[str, Any]:
        """The auto-mode settings with the floors and ceilings fixed in code applied."""
        m = self.raw["mode"]
        out: dict[str, Any] = {"policy": m.get("policy", "staged")}
        for k, floor in MODE_FLOORS.items():
            out[k] = max(_number(type(floor), DEFAULTS["mode"][k] if m.get(k) is None else m[k], f"mode.{k}"), floor)
        for k, top in AUTO_CEILING.items():
            out[k] = max(0, min(_number(int, top if m.get(k) is None else m[k], f"mode.{k}"), top))
        return out

    @property
    def active_models(self) -> list[str]:
        m = self.raw["models"]
        return [x for x in (m["reference"], m.get("challenger")) if x]


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if not path.is_file():
        raise SystemExit(f"config missing: {path} (run `facecurator init`)")
    cfg = Config(yaml.safe_load(path.read_text()) or {}, path)
    if cfg.raw["outer_circle"].get("from_synology"):
        cfg.add_synology_outer(synology_counts(cfg.path_of("data_dir") / "ledger.sqlite"))
    return cfg


def synology_counts(ledger: Path) -> dict[str, int]:
    """Tagged faces per Synology person name, from the last sync (read-only; empty before the first sync)."""
    if not ledger.is_file():
        return {}
    try:
        con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=60)
        rows = con.execute("SELECT p.name, COUNT(*) FROM syno_faces s JOIN syno_people p ON p.id=s.person_id "
                           "WHERE COALESCE(p.name, '') != '' GROUP BY p.name").fetchall()
        con.close()
    except sqlite3.Error:
        return {}
    return {name: n for name, n in rows}
