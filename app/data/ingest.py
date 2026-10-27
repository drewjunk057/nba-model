"""Load a data feed (CSV folder) into the database as of a given date.

This is the seam for real data. Any source (nba_api, a stats vendor, an
odds API, your own scraper) just needs to write these four CSVs:

games.csv      game_id, date, season, tip_time, prediction_timestamp,
               home_team, away_team, home_score, away_score,
               closing_home_spread, closing_total, home_moneyline, away_moneyline
team_box.csv   game_id, team, opp, is_home, pts, poss, fga, fgm, fg3m,
               fta, ftm, tov, orb, drb, opp_drb       (poss may be blank)
injuries.csv   game_id, team, player, is_star, minutes_share, status, reported_at
teams.csv      team, name, lat, lon

Ingesting "as of" a date reveals results only for games before that date,
the schedule + lines for the next `lookahead_days`, and injury reports up
to and including that date — mirroring what you would know that morning.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..db import init_db, session

GAME_COLS = ["game_id", "date", "season", "tip_time", "prediction_timestamp",
             "home_team", "away_team", "home_score", "away_score",
             "closing_home_spread", "closing_total", "home_moneyline", "away_moneyline"]
LINE_COLS = ["closing_home_spread", "closing_total", "home_moneyline", "away_moneyline"]
BOX_COLS = ["game_id", "team", "opp", "is_home", "pts", "poss", "fga", "fgm",
            "fg3m", "fta", "ftm", "tov", "orb", "drb", "opp_drb"]
INJ_COLS = ["game_id", "team", "player", "is_star", "minutes_share", "status", "reported_at"]


def _require(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def estimate_possessions(box: pd.DataFrame) -> pd.Series:
    """Standard estimate: FGA - ORB + TOV + 0.44*FTA."""
    return box["fga"] - box["orb"] + box["tov"] + 0.44 * box["fta"]


def _upsert(con, table: str, df: pd.DataFrame):
    if df.empty:
        return 0
    cols = list(df.columns)
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({','.join('?' * len(cols))})")
    rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
            for r in df.itertuples(index=False, name=None)]
    con.executemany(sql, rows)
    return len(rows)


def ingest_feed(feed_dir: Path, as_of: date, lookahead_days: int = 7) -> dict:
    init_db()
    games = pd.read_csv(feed_dir / "games.csv", dtype={"game_id": str})
    box = pd.read_csv(feed_dir / "team_box.csv", dtype={"game_id": str})
    inj = pd.read_csv(feed_dir / "injuries.csv", dtype={"game_id": str})
    teams = pd.read_csv(feed_dir / "teams.csv")
    _require(games, GAME_COLS, "games.csv")
    _require(box, [c for c in BOX_COLS if c != "poss"], "team_box.csv")
    _require(inj, INJ_COLS, "injuries.csv")

    if "poss" not in box.columns:
        box["poss"] = None
    box["poss"] = box["poss"].fillna(estimate_possessions(box))

    as_of_s = as_of.isoformat()
    horizon = (as_of + timedelta(days=lookahead_days)).isoformat()
    played = games[games["date"] < as_of_s].copy()
    upcoming = games[(games["date"] >= as_of_s) & (games["date"] <= horizon)].copy()
    upcoming[["home_score", "away_score"]] = None

    now = pd.Timestamp.now(tz="UTC")
    if as_of == date.today():
        # tonight's finished games count as played, so picks are graded the same night
        tips = pd.to_datetime(games["tip_time"], utc=True, errors="coerce")
        final_today = (games["date"] == as_of_s) & games["home_score"].notna() & (tips < now)
        if final_today.any():
            played = pd.concat([played, games[final_today]])
            upcoming = upcoming[~upcoming["game_id"].isin(games.loc[final_today, "game_id"])]
    up_tips = pd.to_datetime(upcoming["tip_time"], utc=True, errors="coerce")
    started_ids = list(upcoming.loc[up_tips <= now, "game_id"])

    played_ids = set(played["game_id"])
    visible_ids = played_ids | set(upcoming["game_id"])
    box = box[box["game_id"].isin(played_ids)]
    game_dates = games.set_index("game_id")["date"]
    inj = inj[inj["game_id"].isin(visible_ids)]
    # a report is usable once it exists: past games' reports, plus today's
    # live report attached to upcoming games
    cutoff = (as_of + timedelta(days=1)).isoformat()
    inj = inj[(inj["game_id"].map(game_dates) <= as_of_s) | (inj["reported_at"].astype(str) < cutoff)]

    with session() as con:
        before = con.execute("SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL").fetchone()[0]
        _upsert(con, "teams", teams[["team", "name", "lat", "lon"]])
        if started_ids:
            # a game in progress: keep the pregame line we already have, so live
            # in-game odds never replace it
            qs = ",".join("?" * len(started_ids))
            prior = pd.read_sql_query(
                f"SELECT game_id, {', '.join(LINE_COLS)} FROM games WHERE game_id IN ({qs})",
                con, params=started_ids).set_index("game_id")
            upcoming = upcoming.set_index("game_id")
            for c in LINE_COLS:
                keep = prior[c].dropna()
                upcoming.loc[upcoming.index.intersection(keep.index), c] = keep
            upcoming = upcoming.reset_index()
        _upsert(con, "games", pd.concat([played, upcoming])[GAME_COLS])
        _upsert(con, "team_box", box[BOX_COLS])
        _upsert(con, "injuries", inj[INJ_COLS])
        after = con.execute("SELECT COUNT(*) FROM games WHERE home_score IS NOT NULL").fetchone()[0]
    return dict(new_results=after - before, upcoming=len(upcoming), as_of=as_of_s)
