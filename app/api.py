"""FastAPI backend.  Run:  uvicorn app.api:app --reload"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DB_PATH, MODEL_DIR, PREDICTION_LEAD_HOURS
from .db import init_db, read_df
from .features import game_features, load_raw, team_features
from .models import ModelBundle
from .pipeline import REAL_FEED, data_source, refresh_live, run_nightly

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="NBA Predictor", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ------------------------------------------------------------ cached state
_lock = threading.Lock()
_cache: dict = {}


def state():
    """Feature frame + model, reloaded when the DB or model pointer changes."""
    key = (DB_PATH.stat().st_mtime if DB_PATH.exists() else 0,
           (MODEL_DIR / "current.json").read_text() if (MODEL_DIR / "current.json").exists() else "")
    with _lock:
        if _cache.get("key") != key:
            if not key[1]:
                raise HTTPException(503, "No trained model yet. Run go_live.bat (real data) or python -m scripts.setup (demo)")
            games, box, inj, teams = load_raw()
            tf = team_features(games, box, inj, teams)
            out = {}
            for r in inj.sort_values("minutes_share", ascending=False).itertuples():
                if r.minutes_share >= 0.04 or r.is_star:
                    out.setdefault((r.game_id, r.team), []).append(
                        dict(player=r.player, status=r.status, star=int(r.is_star)))
            _cache.update(key=key, raw=(games, box, inj, teams), tf=tf, gf=game_features(games, tf),
                          bundle=ModelBundle(), out=out)
        return _cache


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


def _fair_ml(p):
    p = min(max(p, 0.01), 0.99)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _game_payload(row, pred, out_map=None) -> dict:
    out_map = out_map or {}
    played = pd.notna(row.home_score)
    mkt_p = _f(row.mkt_home_prob, 4)
    out = dict(
        game_id=row.game_id, date=row.date, season=row.season, tip_time=row.tip_time,
        prediction_timestamp=row.prediction_timestamp,
        home_team=row.home_team, away_team=row.away_team,
        market=dict(home_spread=_f(row.closing_home_spread, 1), total=_f(row.closing_total, 1),
                    home_moneyline=None if pd.isna(row.home_moneyline) else int(row.home_moneyline),
                    away_moneyline=None if pd.isna(row.away_moneyline) else int(row.away_moneyline),
                    home_win_prob=mkt_p),
        prediction={k: v for k, v in pred.items() if k != "drivers"},
        drivers=pred["drivers"],
        context=dict(
            home_net_rtg=_f(row.home_net_rtg, 1), away_net_rtg=_f(row.away_net_rtg, 1),
            home_rest=int(row.home_days_rest), away_rest=int(row.away_days_rest),
            home_b2b=int(row.home_b2b), away_b2b=int(row.away_b2b),
            home_star_out=int(row.home_star_out), away_star_out=int(row.away_star_out),
            home_streak=int(row.home_win_streak), away_streak=int(row.away_win_streak),
            internal_spread=_f(row.internal_spread, 1), internal_total=_f(row.internal_total, 1),
            home_out=out_map.get((row.game_id, row.home_team), []),
            away_out=out_map.get((row.game_id, row.away_team), [])),
    )
    p = pred
    edges = dict(model_fair_home_ml=_fair_ml(p["home_win_prob"]))
    if mkt_p is not None:
        edges["win_prob_edge"] = round(p["home_win_prob"] - mkt_p, 4)
    edges["cover_edge"] = round(p["home_covers_prob"] - 0.5, 4)
    if pd.notna(row.closing_total):
        edges["total_edge_points"] = round(p["predicted_total_points"] - row.closing_total, 1)
    out["edges"] = edges
    if played:
        out["result"] = dict(home_score=int(row.home_score), away_score=int(row.away_score),
                             home_win=int(row.home_win), total_points=int(row.total_points),
                             home_covered=None if pd.isna(row.home_covers) else int(row.home_covers),
                             went_over=None if pd.isna(row.over) else int(row.over))
    return out


# ------------------------------------------------------------------ routes
@app.on_event("startup")
def _startup():
    init_db()


@app.get("/")
def index():
    # always serve the newest dashboard after an update
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    s = state()
    return dict(ok=True, model_version=s["bundle"].version, games=int(len(s["gf"])))


@app.get("/api/meta")
def meta():
    s = state()
    gf = s["gf"]
    played = gf[gf.home_score.notna()]
    upcoming = gf[gf.home_score.isna()]
    runs = read_df("SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT 1")
    return dict(
        model_version=s["bundle"].version,
        trained_through=s["bundle"].meta["trained_through"],
        last_result_date=played.date.max() if len(played) else None,
        next_game_date=upcoming.date.min() if len(upcoming) else None,
        first_date=gf.date.min(),
        last_date=gf.date.max(),
        seasons=sorted(gf.season.unique().tolist()),
        teams=sorted(s["raw"][3].team.tolist()),
        last_run=runs.to_dict("records")[0] if len(runs) else None,
        data_source=data_source(),
        feed=_feed_summary(),
    )


def _feed_summary():
    p = REAL_FEED / "fetch_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/api/teams")
def teams():
    """Current team ratings (as of each team's next game) plus rosters."""
    s = state()
    gf, tf = s["gf"], s["tf"]
    games_df = s["raw"][0]
    names = s["raw"][3].set_index("team")["name"].to_dict()
    cur_season = gf.season.max()
    up = gf[gf.home_score.isna()]
    snap = {}
    for side in ("home", "away"):
        for r in gf.sort_values("date").itertuples():
            snap[getattr(r, f"{side}_team")] = (r.date, r.game_id)
    # prefer each team's next game, else its latest
    for r in up.sort_values("date", ascending=False).itertuples():
        snap[r.home_team] = (r.date, r.game_id)
        snap[r.away_team] = (r.date, r.game_id)
    tfi = tf.set_index(["game_id", "team"])

    def record(season):
        g = games_df[(games_df.season == season) & games_df.home_score.notna()]
        rec = {}
        for r in g.itertuples():
            hw = r.home_score > r.away_score
            for t, w in ((r.home_team, hw), (r.away_team, not hw)):
                rec.setdefault(t, [0, 0])[0 if w else 1] += 1
        return rec
    seasons = sorted(games_df.season.unique())
    prev_season = seasons[-2] if len(seasons) > 1 else None
    rec_cur, rec_prev = record(cur_season), record(prev_season) if prev_season else {}

    rosters = {}
    rp = REAL_FEED / "rosters.csv"
    try:
        rdf = pd.read_csv(rp, dtype={"player_id": str}) if rp.exists() and data_source() == "real" else None
    except pd.errors.EmptyDataError:      # roster download failed on the last update
        rdf = None
    if rdf is not None and {"team", "mpg", "prev_mpg"} <= set(rdf.columns):
        rdf = rdf.sort_values(["team", "mpg", "prev_mpg"], ascending=[True, False, False], na_position="last")
        for t, g in rdf.groupby("team"):
            rosters[t] = json.loads(g.drop(columns=["team"]).to_json(orient="records"))

    out = []
    for t in sorted(names):
        gid = snap.get(t, (None, None))[1]
        f = tfi.loc[(gid, t)] if gid is not None and (gid, t) in tfi.index else None
        row = dict(team=t, name=names[t], as_of=snap.get(t, (None,))[0],
                   record=rec_cur.get(t, [0, 0]), prev_record=rec_prev.get(t, [0, 0]),
                   roster=rosters.get(t, []))
        if f is not None:
            row.update({k: _f(f[k], 2) for k in ("gp", "off_rtg", "def_rtg", "net_rtg", "pace", "efg", "tov_pct",
                                                 "orb_pct", "ft_rate", "sos", "roll_net_10", "win_pct_10",
                                                 "win_streak", "days_rest")})
        out.append(row)
    return dict(season=cur_season, prev_season=prev_season, data_source=data_source(), teams=out)


@app.get("/api/games")
def games(date: str):
    s = state()
    gf = s["gf"]
    rows = gf[gf.date == date]
    if rows.empty:
        return dict(date=date, games=[])
    preds = s["bundle"].predict(rows)
    return dict(date=date, games=[_game_payload(r, p, s["out"]) for r, p in zip(rows.itertuples(), preds)])


class InjuryIn(BaseModel):
    team: str
    player: str
    is_star: bool = False
    minutes_share: float = Field(0.1, ge=0, le=0.5)
    status: str = "OUT"


class PredictIn(BaseModel):
    game_id: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    date: str | None = None
    closing_home_spread: float | None = None
    closing_total: float | None = None
    home_moneyline: int | None = None
    away_moneyline: int | None = None
    injuries: list[InjuryIn] = []


@app.post("/api/predict")
def predict(body: PredictIn):
    """Predict a scheduled game by id, or any hypothetical matchup.

    For a hypothetical matchup, features are rebuilt with that game inserted
    into the schedule, so rest, travel, form and ratings are all as of its date.
    """
    s = state()
    if body.game_id:
        rows = s["gf"][s["gf"].game_id == body.game_id]
        if rows.empty:
            raise HTTPException(404, "Unknown game_id")
        return _game_payload(next(rows.itertuples()), s["bundle"].predict(rows)[0], s["out"])

    if not (body.home_team and body.away_team):
        raise HTTPException(422, "Provide game_id, or home_team and away_team")
    games_df, box, inj, teams = s["raw"]
    valid = set(teams.team)
    if body.home_team not in valid or body.away_team not in valid or body.home_team == body.away_team:
        raise HTTPException(422, "Unknown or identical teams")
    d = body.date or (s["gf"][s["gf"].home_score.isna()].date.min() or date.today().isoformat())
    season = games_df[games_df.date <= d].season.max() or games_df.season.max()

    # default market lines from our own rating view if not supplied
    tip = datetime.fromisoformat(d + "T19:00")
    gid = "ADHOC"
    new = dict(game_id=gid, date=d, season=season, tip_time=tip.isoformat(),
               prediction_timestamp=(tip - timedelta(hours=PREDICTION_LEAD_HOURS)).isoformat(),
               home_team=body.home_team, away_team=body.away_team, home_score=None, away_score=None,
               closing_home_spread=body.closing_home_spread, closing_total=body.closing_total,
               home_moneyline=body.home_moneyline, away_moneyline=body.away_moneyline)
    g2 = pd.concat([games_df, pd.DataFrame([new])], ignore_index=True).sort_values(["date", "tip_time"])
    inj2 = pd.concat([inj, pd.DataFrame([dict(game_id=gid, team=i.team, player=i.player,
                                              is_star=int(i.is_star), minutes_share=i.minutes_share,
                                              status=i.status.upper(), reported_at=tip.isoformat())
                                         for i in body.injuries])], ignore_index=True) if body.injuries else inj
    gf2 = game_features(g2, team_features(g2, box, inj2, teams))
    row = gf2[gf2.game_id == gid].copy()
    # fill missing market inputs from internal estimates
    if pd.isna(row.closing_home_spread.iat[0]):
        row["closing_home_spread"] = round(row.internal_spread.iat[0] * 2) / 2
        row["spread_gap"] = row.internal_spread - row.closing_home_spread
    if pd.isna(row.closing_total.iat[0]):
        row["closing_total"] = round(row.internal_total.iat[0] * 2) / 2
        row["total_gap"] = row.internal_total - row.closing_total
    if pd.isna(row.mkt_home_prob.iat[0]):
        from scipy.stats import norm
        row["mkt_home_prob"] = norm.cdf(-row.closing_home_spread.iat[0] / 12.5)
    if pd.isna(row.fav_rating_gap.iat[0]) or body.closing_home_spread is None:
        fav_home = row.closing_home_spread.iat[0] <= 0
        row["fav_rating_gap"] = row.net_rtg_diff if fav_home else -row.net_rtg_diff
    pred = s["bundle"].predict(row)[0]
    payload = _game_payload(next(row.itertuples()), pred)
    payload["note"] = "Hypothetical matchup; market lines estimated from ratings where not provided."
    return payload


@app.get("/api/model")
def model_info():
    s = state()
    m = dict(s["bundle"].meta)
    m.pop("drift_reference", None)
    m["versions"] = sorted((p.name for p in MODEL_DIR.glob("v*") if p.is_dir()), reverse=True)
    return m


@app.get("/api/backtest")
def backtest():
    s = state()
    bt = pd.DataFrame(s["bundle"].backtest())
    if bt.empty:
        return dict(games=[], monthly=[])
    bt["home_win"] = (bt.home_score > bt.away_score).astype(int)
    bt["month"] = bt.date.str[:7]
    bt["err_total"] = (bt.home_score + bt.away_score) - bt.predicted_total_points
    # plain column math + agg: works on every pandas 2.x
    bt["win_se"] = (bt.home_win_prob - bt.home_win) ** 2
    bt["mkt_se"] = (bt.mkt_home_prob - bt.home_win) ** 2
    bt["win_hit"] = ((bt.home_win_prob > 0.5) == (bt.home_win == 1)).astype(float)
    bt["abs_err"] = bt.err_total.abs()
    monthly = bt.groupby("month", as_index=False).agg(
        n=("home_win", "size"), win_brier=("win_se", "mean"), market_brier=("mkt_se", "mean"),
        win_acc=("win_hit", "mean"), total_mae=("abs_err", "mean"))
    return dict(season=s["bundle"].meta["evaluation"]["test_season"],
                monthly=monthly.round(4).to_dict("records"))


@app.get("/api/log")
def prediction_log(limit: int = 200):
    log = read_df("SELECT l.*, g.date, g.home_team, g.away_team FROM prediction_log l "
                  "JOIN games g USING(game_id) ORDER BY g.date DESC LIMIT ?", (limit,))
    runs = read_df("SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT 20")
    runs["notes"] = runs.notes.apply(lambda x: json.loads(x) if x else None)
    return dict(predictions=json.loads(log.to_json(orient="records")),
                runs=json.loads(runs.to_json(orient="records")))


@app.get("/api/record")
def record():
    """Our live record: the latest pick logged for each game before tip-off,
    graded against the line that was posted when the pick was made."""
    log = read_df(
        "SELECT l.*, g.date, g.home_team, g.away_team, g.home_score, g.away_score "
        "FROM prediction_log l JOIN games g USING(game_id) "
        "WHERE l.predicted_at = (SELECT MAX(predicted_at) FROM prediction_log x WHERE x.game_id = l.game_id)")
    win_pay = 100 / 110
    tallies = {k: dict(w=0, l=0, p=0, units=0.0) for k in ("winner", "spread", "total")}
    picks, pending = [], 0

    def grade(kind, outcome):
        t = tallies[kind]
        if outcome is None:
            return None
        t[outcome] += 1
        if kind != "winner":
            t["units"] += win_pay if outcome == "w" else (-1.0 if outcome == "l" else 0.0)
        return outcome

    for r in log.sort_values(["date", "game_id"], ascending=False).itertuples():
        home_side = r.home_win_prob >= 0.5
        pick = dict(game_id=r.game_id, date=r.date, home=r.home_team, away=r.away_team,
                    winner_pick=r.home_team if home_side else r.away_team,
                    win_prob=round(max(r.home_win_prob, 1 - r.home_win_prob), 3))
        if pd.notna(r.closing_home_spread):
            cover_home = r.home_covers_prob >= 0.5
            line = r.closing_home_spread if cover_home else -r.closing_home_spread
            pick["spread_pick"] = f"{r.home_team if cover_home else r.away_team} {line:+g}"
        if pd.notna(r.closing_total) and pd.notna(r.over_prob):
            pick["total_pick"] = f"{'Over' if r.over_prob >= 0.5 else 'Under'} {r.closing_total:g}"
        if pd.isna(r.home_score):
            pending += 1
            pick["status"] = "pending"
            picks.append(pick)
            continue
        margin = r.home_score - r.away_score
        pick["final"] = f"{r.away_team} {int(r.away_score)}, {r.home_team} {int(r.home_score)}"
        pick["winner"] = grade("winner", "w" if (margin > 0) == home_side else "l")
        if "spread_pick" in pick:
            m = margin + r.closing_home_spread
            pick["spread"] = grade("spread", "p" if m == 0 else ("w" if (m > 0) == cover_home else "l"))
        if "total_pick" in pick:
            tot = r.home_score + r.away_score
            pick["total"] = grade("total", "p" if tot == r.closing_total else
                                  ("w" if (tot > r.closing_total) == (r.over_prob >= 0.5) else "l"))
        pick["status"] = "graded"
        picks.append(pick)

    for t in tallies.values():
        n = t["w"] + t["l"]
        t["pct"] = round(t["w"] / n, 3) if n else None
        t["units"] = round(t["units"], 2)
    return dict(record=tallies, pending=pending, graded=len(picks) - pending, picks=picks[:200],
                break_even=0.524)


_refresh_lock = threading.Lock()


@app.post("/api/refresh")
def refresh():
    """Pull the latest results, lines, injuries and rosters, then re-predict today."""
    if data_source() != "real":
        raise HTTPException(400, "The app is on demo data. Run go_live.bat to switch to real NBA data.")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(409, "An update is already running.")
    try:
        msgs = []
        res = refresh_live(log=msgs.append)
    finally:
        _refresh_lock.release()
    _cache.clear()
    return dict(feed=res["update"], pipeline={k: v for k, v in res["pipeline"].items() if k != "drift"}, log=msgs[-6:])


@app.post("/api/pipeline/run")
def pipeline_run(as_of: str | None = None, retrain: bool = False):
    summary = run_nightly(date.fromisoformat(as_of) if as_of else None, force_retrain=retrain)
    _cache.clear()
    return summary
