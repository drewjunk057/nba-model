"""Player impact, in points of team strength per 48 minutes.

A minutes-weighted ridge regression on game margins (an adjusted plus-minus
using only who played and for how long), refit at the start of every month
on earlier games only, so a game's values never use that game or anything
after it. Tuned on the 2025-26 season: best player about +5 per 48 min.

Used for one thing: turning a missing player into "points of team strength
missing" for the injury-alert model.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from .config import DATA_DIR

log = logging.getLogger("player_value")
RAW_DIR = DATA_DIR / "raw"
ALPHA = 10.0             # shrinkage toward an average player
HALF_LIFE_DAYS = 365     # older games count less
IMPACT_COLS = ["home_missing_impact", "away_missing_impact", "impact_diff"]
_cache: dict = {}


def norm_name(n: str) -> str:
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z ]", "", n.replace("-", " "))
    return " ".join(w for w in n.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def load_minutes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(raw_dir.glob("espn_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        for gid, s in (d.get("summaries") or {}).items():
            for p in s.get("players") or []:
                if not p.get("dnp") and (p.get("minutes") or 0) > 0 and p.get("player_id"):
                    rows.append((str(gid), p["team"], str(p["player_id"]), p["player"], float(p["minutes"])))
    return pd.DataFrame(rows, columns=["game_id", "team", "player_id", "player", "minutes"])


def _fit(pm, games, before) -> pd.Series:
    g = games[(games.date < before) & games.margin.notna()]
    p = pm[pm.game_id.isin(g.game_id)]
    if len(g) < 300 or p.empty:
        return pd.Series(dtype=float)
    pids, gidx = pd.Index(p.player_id.unique()), pd.Index(g.game_id)
    home = g.set_index("game_id").home_team
    side = np.where(p.team.values == home.loc[p.game_id].values, 1.0, -1.0)
    X = sparse.csr_matrix((side * p.minutes.values / 48.0,
                           (gidx.get_indexer(p.game_id), pids.get_indexer(p.player_id))),
                          shape=(len(gidx), len(pids)))
    w = 0.5 ** ((before - g.date).dt.days.values / HALF_LIFE_DAYS)
    return pd.Series(Ridge(alpha=ALPHA).fit(X, g.margin.values, sample_weight=w).coef_, index=pids)


def _state(games: pd.DataFrame):
    files = sorted(RAW_DIR.glob("espn_*.json.gz"))
    key = tuple((f.name, f.stat().st_mtime) for f in files)
    if _cache.get("key") != key:
        pm = load_minutes()
        g = games[["game_id", "date", "home_team", "home_score", "away_score"]].copy()
        g["date"] = pd.to_datetime(g.date)
        g["margin"] = g.home_score - g.away_score
        months = pd.date_range(g.date.min().to_period("M").to_timestamp(),
                               g.date.max().to_period("M").to_timestamp(), freq="MS")
        vals = {m: _fit(pm, g, m) for m in months}
        ids = (pm.assign(k=pm.team + "|" + pm.player.map(norm_name))
                 .drop_duplicates("k", keep="last").set_index("k").player_id)
        by_name = (pm.assign(n=pm.player.map(norm_name))
                     .drop_duplicates("n", keep="last").set_index("n").player_id)
        _cache.update(key=key, vals=vals, ids=ids, by_name=by_name)
    return _cache


def current_values() -> pd.Series:
    """Latest fitted value per player_id (for display)."""
    v = _cache.get("vals") or {}
    return v[max(v)] if v else pd.Series(dtype=float)


def player_rows(inj: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Injury rows with each player's value and impact (points of team strength)."""
    st = _state(games)
    d = inj.merge(games[["game_id", "date"]], on="game_id", how="left")
    d["month"] = pd.to_datetime(d.date).dt.to_period("M").dt.to_timestamp()
    nn = d.player.map(norm_name)
    d["player_id"] = (d.team + "|" + nn).map(st["ids"]).fillna(nn.map(st["by_name"]))
    vals = st["vals"]
    last = max(vals) if vals else None
    d["value"] = [vals.get(m, vals.get(last, pd.Series(dtype=float))).get(pid, 0.0) if isinstance(pid, str) else 0.0
                  for m, pid in zip(d.month, d.player_id)]
    weight = np.where(d.status.astype(str).str.upper() == "OUT", 1.0, 0.5)
    d["impact"] = d.value * d.minutes_share * 240 / 48 * weight
    return d


def impact_frame(inj: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """(game_id, home_missing_impact, away_missing_impact, impact_diff); zeros when no data."""
    out = games[["game_id", "home_team", "away_team"]].copy()
    try:
        if inj is None or inj.empty or not list(RAW_DIR.glob("espn_*.json.gz")):
            raise ValueError("no player data")
        imp = player_rows(inj, games).groupby(["game_id", "team"], as_index=False).impact.sum()
        h = imp.rename(columns={"team": "home_team", "impact": "home_missing_impact"})
        a = imp.rename(columns={"team": "away_team", "impact": "away_missing_impact"})
        out = out.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    except Exception as e:                                   # synthetic data, no cache, etc.
        log.debug("player impact unavailable: %s", e)
        out["home_missing_impact"] = out["away_missing_impact"] = 0.0
    out[["home_missing_impact", "away_missing_impact"]] = out[["home_missing_impact", "away_missing_impact"]].fillna(0.0)
    out["impact_diff"] = out.home_missing_impact - out.away_missing_impact
    return out[["game_id"] + IMPACT_COLS]
