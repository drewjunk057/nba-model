"""SQLite storage. Raw tables hold what a data feed provides; feature tables
are rebuilt from them so they can never contain post-game information."""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from .config import DB_PATH

SCHEMA = """
-- One row per game. Scores are NULL until the game is final.
CREATE TABLE IF NOT EXISTS games (
    game_id              TEXT PRIMARY KEY,
    date                 TEXT NOT NULL,          -- YYYY-MM-DD
    season               TEXT NOT NULL,          -- e.g. 2025-26
    tip_time             TEXT NOT NULL,          -- ISO datetime (local)
    prediction_timestamp TEXT NOT NULL,          -- tip_time - lead hours
    home_team            TEXT NOT NULL,
    away_team            TEXT NOT NULL,
    home_score           INTEGER,
    away_score           INTEGER,
    closing_home_spread  REAL,                   -- negative = home favoured
    closing_total        REAL,
    home_moneyline       INTEGER,
    away_moneyline       INTEGER
);
CREATE INDEX IF NOT EXISTS ix_games_date ON games(date);

-- Team box score per game (source for ratings, pace, four factors).
CREATE TABLE IF NOT EXISTS team_box (
    game_id  TEXT NOT NULL,
    team     TEXT NOT NULL,
    opp      TEXT NOT NULL,
    is_home  INTEGER NOT NULL,
    pts      INTEGER, poss REAL,
    fga INTEGER, fgm INTEGER, fg3m INTEGER, fta INTEGER, ftm INTEGER,
    tov INTEGER, orb INTEGER, drb INTEGER, opp_drb INTEGER,
    PRIMARY KEY (game_id, team)
);

-- Availability as reported before prediction_timestamp.
CREATE TABLE IF NOT EXISTS injuries (
    game_id       TEXT NOT NULL,
    team          TEXT NOT NULL,
    player        TEXT NOT NULL,
    is_star       INTEGER NOT NULL,
    minutes_share REAL NOT NULL,   -- share of team minutes player normally plays
    status        TEXT NOT NULL,   -- OUT / QUESTIONABLE
    reported_at   TEXT NOT NULL,
    PRIMARY KEY (game_id, team, player)
);

CREATE TABLE IF NOT EXISTS teams (
    team TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL
);

-- Pre-game feature snapshot per team per game (home_* / away_* source).
-- Rebuilt by app.features; never edited by hand.
CREATE TABLE IF NOT EXISTS team_game_features (
    game_id TEXT NOT NULL, team TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY (game_id, team)
);

-- Every prediction the system issued, for live evaluation and drift.
CREATE TABLE IF NOT EXISTS prediction_log (
    game_id                TEXT NOT NULL,
    model_version          TEXT NOT NULL,
    predicted_at           TEXT NOT NULL,
    home_win_prob          REAL,
    home_covers_prob       REAL,
    predicted_total_points REAL,
    over_prob              REAL,
    closing_home_spread    REAL,
    closing_total          REAL,
    home_win               INTEGER,   -- filled when settled
    home_covered           INTEGER,
    total_points           INTEGER,
    settled_at             TEXT,
    PRIMARY KEY (game_id, model_version)
);

-- Injury-alert bookkeeping: when each absence was first seen, and the line then.
CREATE TABLE IF NOT EXISTS game_scans (
    game_id TEXT PRIMARY KEY, first_scan_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS injury_seen (
    game_id TEXT NOT NULL, team TEXT NOT NULL, player TEXT NOT NULL, status TEXT NOT NULL,
    impact REAL, first_seen_at TEXT NOT NULL, home_spread_then REAL,
    PRIMARY KEY (game_id, team, player, status)
);
CREATE TABLE IF NOT EXISTS alerts (
    game_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT,
    side_team TEXT NOT NULL, side_is_home INTEGER NOT NULL, home_spread REAL NOT NULL,
    edge_points REAL, unpriced_points REAL, players TEXT, reason TEXT,
    result TEXT, closing_home_spread REAL, settled_at TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_at TEXT PRIMARY KEY, as_of TEXT, ingested INTEGER, settled INTEGER,
    predicted INTEGER, retrained INTEGER, model_version TEXT, notes TEXT
);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@contextmanager
def session():
    con = connect()
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with session() as con:
        con.executescript(SCHEMA)


def read_df(sql, params=()):
    with session() as con:
        return pd.read_sql_query(sql, con, params=params)


def reset_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()
