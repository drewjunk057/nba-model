"""Season-to-date personal fouls per player, from the cached ESPN box scores.

Counts only games played for that team before the given date, so a past
game shows what was known going into it.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
_cache: dict = {}


def _season_file(season: str) -> Path:
    # "2026-27" -> espn_2027.json.gz (ESPN labels a season by its end year)
    start = int(str(season)[:4])
    return RAW_DIR / f"espn_{start + 1}.json.gz"


def _load(path: Path):
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime)
    if _cache.get(path.name, {}).get("key") != key:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        dates = {g["game_id"]: day for day, gs in (d.get("scoreboards") or {}).items() for g in gs}
        rows = []                         # (date, team, player_id, player, fouls)
        for gid, sm in (d.get("summaries") or {}).items():
            day = dates.get(gid)
            for p in sm.get("players") or []:
                if day and "fouls" in p and not p.get("dnp") and (p.get("minutes") or 0) > 0:
                    rows.append((day, p["team"], p["player_id"], p["player"], float(p["fouls"] or 0)))
        _cache[path.name] = dict(key=key, rows=rows)
    return _cache[path.name]["rows"]


def leaders(team: str, season: str, before: str, n: int = 5):
    """Top foulers for a team this season before `before` (YYYY-MM-DD).
    None when foul data isn't downloaded yet."""
    rows = _load(_season_file(season))
    if not rows:
        return None
    agg = {}
    for day, t, pid, name, pf in rows:
        if t != team or day >= before:
            continue
        a = agg.setdefault(pid, dict(player=name, fouls=0, gp=0, fouled_out=0))
        a["fouls"] += pf
        a["gp"] += 1
        a["fouled_out"] += pf >= 6
    out = []
    for a in sorted(agg.values(), key=lambda a: (-a["fouls"], -a["fouls"] / a["gp"])):
        out.append(dict(player=a["player"], fouls=int(a["fouls"]), gp=a["gp"],
                        per_game=round(a["fouls"] / a["gp"], 1), fouled_out=a["fouled_out"]))
    return out[:n]
