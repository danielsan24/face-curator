"""Age bins from capture date minus birth date (no age model). Months use 30.4375-day months."""

from __future__ import annotations

import math
from datetime import date

MONTH_DAYS = 30.4375


def months(born: date, day: str | None) -> float | None:
    if not day:
        return None
    try:
        d = date.fromisoformat(day[:10])
    except ValueError:
        return None
    return (d - born).days / MONTH_DAYS


def bin_index(m: float | None, edges: list[int]) -> int | None:
    if m is None or m < 0:
        return None
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else math.inf
        if lo <= m < hi:
            return i
    return None


def bin_label(i: int | None, edges: list[int]) -> str:
    if i is None:
        return "-"
    lo = edges[i]
    hi = edges[i + 1] if i + 1 < len(edges) else None

    def f(m: int) -> str:
        return f"{m}m" if m < 24 else f"{m // 12}y"

    if hi is None:
        return f"{f(lo)}+"
    if lo < 24 and hi <= 24:
        return f"{lo}-{hi}m"
    return f"{f(lo).rstrip('y')}-{f(hi)}" if lo >= 24 else f"{f(lo)}-{f(hi)}"


def window(i: int, edges: list[int], overlap: float) -> tuple[float, float]:
    """Exemplar ages (months) that may score a face in bin i: the bin plus `overlap` months on each side."""
    lo = edges[i]
    hi = edges[i + 1] if i + 1 < len(edges) else math.inf
    return lo - overlap, hi + overlap
