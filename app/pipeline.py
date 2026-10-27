"""Nightly pipeline: ingest -> features -> settle -> (retrain) -> predict -> drift."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .config import DATA_DIR, MODEL_DIR, RETRAIN_EVERY_DAYS
from .data.ingest import ingest_feed
from .db import init_db, read_df, session
from .features import build_all
from .models import DRIFT_FEATURES, ModelBundle, evaluate, psi, train_production

SYNTHETIC_FEED = DATA_DIR / "feed"
REAL_FEED = DATA_DIR / "real"
RAW_DIR = DATA_DIR / "raw"
FIRST_SEASON = 2021          # 2021-22 onward; earlier seasons had COVID scheduling


def default_feed() -> Path:
    return REAL_FEED if (REAL_FEED / "games.csv").exists() else SYNTHETIC_FEED


def data_source() -> str:
    return "real" if default_feed() == REAL_FEED else "synthetic"


FEED_DIR = default_feed()


def _needs_retrain(as_of: date) -> bool:
    cur = MODEL_DIR / "current.json"
    if not cur.exists():
        return True
    meta = ModelBundle().meta
    trained = date.fromisoformat(meta.get("trained_as_of", meta["trained_through"]))
    return (as_of - trained).days >= RETRAIN_EVERY_DAYS


def settle() -> int:
    with session() as con:
        n = con.execute("""
            UPDATE prediction_log SET
              home_win = (SELECT CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END FROM games g WHERE g.game_id = prediction_log.game_id),
              home_covered = (SELECT CASE WHEN g.home_score - g.away_score + g.closing_home_spread > 0 THEN 1
                                          WHEN g.home_score - g.away_score + g.closing_home_spread < 0 THEN 0 END
                              FROM games g WHERE g.game_id = prediction_log.game_id),
              total_points = (SELECT g.home_score + g.away_score FROM games g WHERE g.game_id = prediction_log.game_id),
              settled_at = ?
            WHERE settled_at IS NULL AND game_id IN (SELECT game_id FROM games WHERE home_score IS NOT NULL)
        """, (datetime.now().isoformat(timespec="seconds"),)).rowcount
    return n


def log_predictions(gf: pd.DataFrame, bundle: ModelBundle, day: date) -> int:
    rows = gf[(gf.date == day.isoformat()) & gf.home_score.isna()]
    # never (re)log a prediction once the game has tipped
    tips = pd.to_datetime(rows.tip_time, utc=True, errors="coerce")
    live_rows = rows.tip_time.astype(str).str.endswith("Z")
    rows = rows[~live_rows | (tips > pd.Timestamp.now(tz="UTC"))]
    if rows.empty:
        return 0
    preds = bundle.predict(rows)
    now = datetime.now().isoformat(timespec="seconds")
    with session() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prediction_log (game_id, model_version, predicted_at, home_win_prob,"
            " home_covers_prob, predicted_total_points, over_prob, closing_home_spread, closing_total)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [(r.game_id, p["model_version"], now, p["home_win_prob"], p["home_covers_prob"],
              p["predicted_total_points"], p.get("over_prob"), r.closing_home_spread, r.closing_total)
             for r, p in zip(rows.itertuples(), preds)])
    return len(rows)


def drift_report(gf: pd.DataFrame, bundle: ModelBundle, window: int = 300) -> dict:
    played = gf[gf.home_score.notna()].tail(window)
    ref = bundle.meta["drift_reference"]
    feat_psi = {f: round(psi(ref[f], played[f].dropna().values), 4) for f in DRIFT_FEATURES if f in ref}
    out = dict(feature_psi=feat_psi, psi_alert=[f for f, v in feat_psi.items() if v > 0.2])

    log = read_df("SELECT * FROM prediction_log WHERE settled_at IS NOT NULL")
    if len(log) >= 30:
        out["live"] = dict(
            n=len(log),
            win_brier=float(brier_score_loss(log.home_win, log.home_win_prob)),
            win_accuracy=float(((log.home_win_prob > 0.5) == (log.home_win == 1)).mean()),
            total_mae=float((log.total_points - log.predicted_total_points).abs().mean()))
        ev = bundle.meta["evaluation"]["models"]["win"]["brier"]
        out["live"]["win_brier_vs_eval"] = round(out["live"]["win_brier"] - ev, 4)
    return out


def update_real_data(today: date | None = None, backfill: bool = False, log=print) -> dict:
    """Download new games, lines, injuries and rosters from ESPN and rebuild
    the real feed CSVs. Only new games are downloaded; the first run backfills
    every season from FIRST_SEASON (about 6,000 box scores, 10-20 minutes)."""
    from .data import espn
    today = today or date.today()
    seasons = espn.season_end_years(FIRST_SEASON, today)
    if backfill:
        for p in RAW_DIR.glob("espn_*.json.gz"):
            p.unlink()
    summary = espn.fetch(REAL_FEED, RAW_DIR, seasons, today=today, progress=log)
    return dict(feed=summary)


def refresh_live(log=print) -> dict:
    """Quick daytime refresh: latest lines, injuries and results; re-log today's predictions."""
    upd = update_real_data(log=log)
    s = run_nightly(date.today(), feed_dir=REAL_FEED, allow_retrain=False)
    return dict(update=upd["feed"], pipeline=s)


def run_nightly(as_of: date | None = None, force_retrain: bool = False,
                feed_dir: Path | None = None, allow_retrain: bool = True) -> dict:
    as_of = as_of or date.today()
    feed_dir = feed_dir or default_feed()
    init_db()
    ing = ingest_feed(feed_dir, as_of)
    gf = build_all(persist=True)
    settled = settle()

    retrained = force_retrain or (allow_retrain and _needs_retrain(as_of))
    if retrained:
        report = evaluate(gf)
        train_production(gf, report, as_of.isoformat())
    bundle = ModelBundle()
    predicted = log_predictions(gf, bundle, as_of)
    from . import alerts
    settled += alerts.settle()
    try:
        new_alerts = alerts.scan(gf, bundle)
    except Exception as e:                      # alerts must never break the update
        import logging
        logging.getLogger("alerts").warning("alert scan failed: %r", e)
        new_alerts = []
    drift = drift_report(gf, bundle)

    summary = dict(as_of=as_of.isoformat(), ingested=ing["new_results"], upcoming=ing["upcoming"],
                   settled=settled, predicted=predicted, retrained=bool(retrained), alerts=len(new_alerts),
                   model_version=bundle.version, drift=drift)
    with session() as con:
        con.execute("INSERT OR REPLACE INTO pipeline_runs VALUES (?,?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), summary["as_of"], ing["new_results"],
                     settled, predicted, int(retrained), bundle.version, json.dumps(drift)))
    return summary
