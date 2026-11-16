"""Markdown report for one run."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fc import cards
from fc import db as D
from fc.config import Config


def write(db: sqlite3.Connection, cfg: Config, run: str) -> Path:
    r = D.run_row(db, run)
    st = D.run_stats(db, run)
    L = [f"# Face Curator run {run}", "",
         f"Mode: **{r['mode'] or '?'}** · status: {r['status']} · started {r['started']} · finished {r['finished'] or '-'}", ""]
    if st.get("conditions"):
        L += [f"Conditions: {st['conditions']}", ""]
    dem = cards.demotion(db)
    if dem:
        L += [f"**Auto mode is demoted to staged** since {dem.get('at')}: {dem.get('reason')}. Only a new one-off "
              "audit (`audit --run ID`), completed without an error, re-earns it.", ""]
    if cfg.mode["policy"] == "auto" and st.get("mode_why"):
        L += ["Auto mode was not used for this run: " + "; ".join(st["mode_why"]) + ".", ""]
    for key, title in (("sync", "Sync"), ("embed", "Detection and embeddings")):
        if key in st:
            L += [f"## {title}", "", "```", json.dumps(st[key], indent=1), "```", ""]
    a = st.get("assign")
    if a:
        th = a["thresholds"]
        L += ["## Assignment", "",
              f"Thresholds ({th['status']}): floor {th['floor']}, margin {th['margin']}, sibling margin {th['sibling_margin']}.",
              ""]
        if th["status"] == "calibrated":
            L += [f"Calibration on {th.get('n_pos')} labelled faces and {th.get('n_neg')} negatives (event-separated): "
                  f"{th.get('accepted')} accepted, {th.get('errors')} errors, precision {th.get('precision')}, "
                  f"recall {th.get('recall')}, 95% lower bound {th.get('lower95')}.", ""]
        else:
            L += ["Not calibrated yet (too few owner labels): this run stays staged.", ""]
        L += ["| Status | Faces |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(a["decisions"].items())] + [""]
        L += ["### Gallery coverage (owner-confirmed faces)", ""]
        for p, row in a["coverage"].items():
            L.append(f"- **{cfg.people[p].name}**: " + ", ".join(f"{b} {n}" for b, n in row.items()))
        L.append("")
    if "cluster" in st:
        L += ["## Residual clusters", "", "```", json.dumps(st["cluster"], indent=1), "```", ""]
    if "plan" in st:
        L += ["## Change set", "", "```", json.dumps(st["plan"], indent=1), "```", ""]
        rows = db.execute("SELECT kind, status, COUNT(*) FROM changes WHERE run=? GROUP BY kind, status", (run,)).fetchall()
        L += ["| Kind | Status | Count |", "|---|---|---|"] + [f"| {x[0]} | {x[1]} | {x[2]} |" for x in rows] + [""]
        if st["plan"].get("duplicate_names"):
            L += ["Nothing is planned for " + ", ".join(f"{n} (Synology person ids {v})" for n, v in
                                                         st["plan"]["duplicate_names"].items())
                  + ": the name is used by more than one person in Synology Photos. Merge or rename them there.", ""]
        if st["plan"].get("large_merges"):
            L += ["Large merges (never written by an automatic run; the owner applies them after looking at each):", ""]
            L += [f"- group {m['group']}: {m['faces']} faces into {m['target']}" for m in st["plan"]["large_merges"]] + [""]
        if r["mode"] == "staged":
            L += [f"Staged: nothing was written. Review, then `facecurator apply --run {run}` (preview) and add `--yes` "
                  f"for the face edits; merges are applied on their own with `apply --run {run} --kinds merge --yes`.", ""]
    if "apply" in st:
        L += ["## Applied", "", "```", json.dumps(st["apply"], indent=1), "```", ""]
        if st["apply"].get("left_staged"):
            L += [f"{st['apply']['left_staged']} planned change(s) were left staged for the owner "
                  f"({st['apply'].get('left_staged_why')}): `facecurator apply --run {run}`.", ""]
    if "verify" in st:
        L += [f"Verification after writes: {st['verify']}", ""]
    waiting = db.execute("SELECT kind, COUNT(*) FROM cards WHERE answer IS NULL GROUP BY kind").fetchall()
    L += ["## Review", "", "Cards waiting: " + (", ".join(f"{k} {n}" for k, n in waiting) or "none") +
          ". Open them with `facecurator review`.", ""]
    prev = db.execute("SELECT id FROM runs WHERE id < ? AND status='done' ORDER BY id DESC LIMIT 1", (run,)).fetchone()
    if prev:
        L += [f"Audit of {prev[0]}: {cards.audit_result(db, prev[0])}", ""]
    bumps = db.execute("SELECT * FROM bumps").fetchall()
    if bumps:
        L += ["Raised thresholds (audit errors): " + ", ".join(f"{b['person']} {b['bin']} +{b['amount']}" for b in bumps), ""]
    if st.get("errors"):
        L += ["## Errors", ""] + [f"- {e}" for e in st["errors"]] + [""]
    out = D.atomic_write(cfg.data_dir / "reports" / f"{run}.md", "\n".join(L))
    D.set_run(db, run, report=str(out))
    return out
