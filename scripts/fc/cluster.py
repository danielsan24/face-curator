"""Residual clustering (faces not accepted by the supervised path), Immich-style density passes.

The distance stays fixed (cosine distance, default 0.34; on the reference library siblings merge from 0.37); only the density requirement relaxes across passes:
min neighbours 25 -> 10 -> 5 -> 3. In each pass, a face with at least k neighbours (within the distance) among
the pool is a core face; core faces within the distance of each other form one cluster; a non-core face joins
the cluster of its nearest core neighbour if that is within the distance. Clusters from stricter passes are
kept; later passes only add to them or form new clusters from what is left. Clusters only ever propose.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from fc import db as D
from fc.config import Config


def neighbours(vecs: np.ndarray, sim_min: float, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """CSR adjacency (indptr, indices) of pairs with cosine similarity >= sim_min, self excluded."""
    n = len(vecs)
    parts, deg = [], np.zeros(n, dtype=np.int64)
    for s in range(0, n, chunk):
        sims = vecs[s:s + chunk] @ vecs.T
        sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -1.0
        r, c = np.nonzero(sims >= sim_min)
        deg[s:s + sims.shape[0]] = np.bincount(r, minlength=sims.shape[0])
        parts.append(c.astype(np.int32))
    indptr = np.concatenate([[0], np.cumsum(deg)])
    return indptr, (np.concatenate(parts) if parts else np.zeros(0, np.int32))


def cluster(vecs: np.ndarray, distance: float, schedule: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cluster label per row, -1 = none; pass index that clustered it, -1 = none)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vecs)
    labels = np.full(n, -1, dtype=np.int64)
    passes = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels, passes
    indptr, indices = neighbours(vecs, 1.0 - distance)
    deg = np.diff(indptr)
    rows = np.repeat(np.arange(n, dtype=np.int32), deg)
    next_label = 0
    for pi, k in enumerate(schedule):
        core = deg >= k
        m = core[rows] & core[indices]
        g = csr_matrix((np.ones(int(m.sum()), dtype=np.int8), (rows[m], indices[m])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        new = np.nonzero(core & (labels < 0))[0]
        for cid in np.unique(comp[new]):
            members = np.nonzero((comp == cid) & core)[0]
            old = labels[members][labels[members] >= 0]
            if len(old):  # joins the earlier (stricter) cluster it touches most; earlier clusters never merge
                lab = int(np.bincount(old).argmax())
            else:
                lab, next_label = next_label, next_label + 1
            fresh = members[labels[members] < 0]
            labels[fresh], passes[fresh] = lab, pi
        # Border faces: the cluster of the nearest clustered core neighbour within the distance.
        for i in np.nonzero(~core & (labels < 0))[0]:
            nb = indices[indptr[i]:indptr[i + 1]]
            cand = nb[core[nb] & (labels[nb] >= 0)]
            if len(cand):
                j = cand[int(np.argmax(vecs[cand] @ vecs[i]))]
                labels[i], passes[i] = labels[j], pi
    return labels, passes


def run_clustering(db: sqlite3.Connection, cfg: Config, run: str, log=print) -> dict:
    model = cfg["models"]["reference"]
    pool = [r[0] for r in db.execute("SELECT face FROM decisions WHERE run=? AND status IN ('abstained','vetoed')", (run,))]
    ids, vecs = D.vectors(db, model, pool)
    C = cfg["cluster"]
    labels, passes = cluster(vecs, C["distance"], C["schedule"])
    db.execute("DELETE FROM clusters WHERE run=?", (run,))
    db.executemany("INSERT INTO clusters VALUES(?,?,?,?)",
                   [(run, int(f), int(l), int(p)) for f, l, p in zip(ids, labels, passes) if l >= 0])
    db.commit()
    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(0, int)
    out = {"pool": int(len(ids)), "clustered": int((labels >= 0).sum()), "clusters": int(len(sizes)),
           "by_pass": {str(k): int((passes == i).sum()) for i, k in enumerate(C["schedule"])},
           "largest": sorted((int(x) for x in sizes), reverse=True)[:10]}
    log(f"cluster: {out}")
    return out
