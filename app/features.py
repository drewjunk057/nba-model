"""Feature engineering.

Every team statistic is computed from games strictly before the game being
predicted (shift(1) on time-ordered team logs), so features for a game
are identical whether or not that game has been played yet.

Outputs
  team_features(): one row per (game_id, team) — the pre-game snapshot
  game_features(): one row per game with home_* / away_* columns, diffs,
                   market info, model-specific extras and targets
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import (LEAGUE_AVG_PACE, LEAGUE_AVG_RTG, PRIOR_GAMES,
                     PRIOR_REGRESSION, ROLLING_WINDOWS)
from .db import read_df, session

HCA_POINTS = 2.8        # home court in points, used only for the internal spread
LEAGUE_FF = dict(efg=0.535, tov_pct=0.13, orb_pct=0.27, ft_rate=0.195)

TEAM_FEATURES = (
    ["gp", "off_rtg", "def_rtg", "net_rtg", "pace", "efg", "tov_pct", "orb_pct",
     "ft_rate", "opp_efg", "sos"]
    + [f"roll_{s}_{w}" for w in ROLLING_WINDOWS for s in ("off", "def", "net")]
    + ["roll_pace_10", "win_pct_10", "win_streak",
       "days_rest", "b2b", "games_last_7", "road_trip_len", "travel_km",
       "star_out", "star_questionable", "missing_minutes", "rotation_continuity"]
)


# ---------------------------------------------------------------- helpers
def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(h))


_erf = np.vectorize(math.erf, otypes=[float])


def american_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def load_raw():
    games = read_df("SELECT * FROM games ORDER BY date, tip_time, game_id")
    box = read_df("SELECT * FROM team_box")
    inj = read_df("SELECT * FROM injuries")
    teams = read_df("SELECT * FROM teams")
    return games, box, inj, teams


# ---------------------------------------------------------- team features
def team_features(games, box, inj, teams) -> pd.DataFrame:
    g = games[["game_id", "date", "season", "tip_time", "home_team", "away_team"]]
    long = pd.concat([
        g.assign(team=g.home_team, opp=g.away_team, is_home=1),
        g.assign(team=g.away_team, opp=g.home_team, is_home=0),
    ])[["game_id", "date", "season", "tip_time", "team", "opp", "is_home", "home_team"]]

    stats = box[["game_id", "team", "pts", "poss", "fga", "fgm", "fg3m", "fta",
                 "ftm", "tov", "orb", "opp_drb"]]
    opp_stats = box[["game_id", "team", "pts", "fga", "fgm", "fg3m"]].rename(
        columns={"team": "opp", "pts": "pts_against", "fga": "o_fga",
                 "fgm": "o_fgm", "fg3m": "o_fg3m"})
    long = long.merge(stats, on=["game_id", "team"], how="left")
    long = long.merge(opp_stats, on=["game_id", "opp"], how="left")
    long["played"] = long["pts"].notna().astype(int)
    long["won"] = np.where(long.played == 1, (long.pts > long.pts_against).astype(float), np.nan)
    long = long.sort_values(["team", "tip_time", "game_id"]).reset_index(drop=True)

    sum_cols = ["pts", "pts_against", "poss", "fga", "fgm", "fg3m", "fta", "ftm",
                "tov", "orb", "opp_drb", "o_fga", "o_fgm", "o_fg3m", "played"]
    filled = long[sum_cols].fillna(0)

    # ---- season-to-date (pre-game) -------------------------------------
    cum = filled.groupby([long.team, long.season]).cumsum() - filled
    gp = cum["played"]
    raw_off = 100 * cum.pts / cum.poss.replace(0, np.nan)
    raw_def = 100 * cum.pts_against / cum.poss.replace(0, np.nan)
    raw_pace = cum.poss / gp.replace(0, np.nan)

    # prior = last season's final rating, regressed to the mean
    finals = (long[long.played == 1].groupby(["team", "season"])
              .agg(pts=("pts", "sum"), pa=("pts_against", "sum"), poss=("poss", "sum"),
                   n=("played", "sum"), fga=("fga", "sum"), fgm=("fgm", "sum"),
                   fg3m=("fg3m", "sum"), fta=("fta", "sum"), ftm=("ftm", "sum"),
                   tov=("tov", "sum"), orb=("orb", "sum"), opp_drb=("opp_drb", "sum"),
                   o_fga=("o_fga", "sum"), o_fgm=("o_fgm", "sum"), o_fg3m=("o_fg3m", "sum"))
              .reset_index())
    finals["f_off"] = 100 * finals.pts / finals.poss
    finals["f_def"] = 100 * finals.pa / finals.poss
    finals["f_pace"] = finals.poss / finals.n
    # last season's four factors, so early-season values are team-specific
    finals["f_efg"] = (finals.fgm + 0.5 * finals.fg3m) / finals.fga
    finals["f_tov_pct"] = finals.tov / (finals.fga + 0.44 * finals.fta + finals.tov)
    finals["f_orb_pct"] = finals.orb / (finals.orb + finals.opp_drb)
    finals["f_ft_rate"] = finals.ftm / finals.fga
    finals["f_opp_efg"] = (finals.o_fgm + 0.5 * finals.o_fg3m) / finals.o_fga
    seasons = sorted(long.season.unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    finals["season"] = finals.season.map(nxt)
    prior_cols = ["f_off", "f_def", "f_pace", "f_efg", "f_tov_pct", "f_orb_pct", "f_ft_rate", "f_opp_efg"]
    long = long.merge(finals[["team", "season"] + prior_cols], on=["team", "season"], how="left")
    pr_off = LEAGUE_AVG_RTG + PRIOR_REGRESSION * (long.f_off.fillna(LEAGUE_AVG_RTG) - LEAGUE_AVG_RTG)
    pr_def = LEAGUE_AVG_RTG + PRIOR_REGRESSION * (long.f_def.fillna(LEAGUE_AVG_RTG) - LEAGUE_AVG_RTG)
    pr_pace = LEAGUE_AVG_PACE + 0.7 * (long.f_pace.fillna(LEAGUE_AVG_PACE) - LEAGUE_AVG_PACE)

    w = gp / (gp + PRIOR_GAMES)
    out = long[["game_id", "team", "opp", "is_home", "date", "season", "home_team", "won"]].copy()
    out["gp"] = gp
    out["off_rtg"] = w * raw_off.fillna(0) + (1 - w) * pr_off
    out["def_rtg"] = w * raw_def.fillna(0) + (1 - w) * pr_def
    out["net_rtg"] = out.off_rtg - out.def_rtg
    out["pace"] = w * raw_pace.fillna(0) + (1 - w) * pr_pace

    def ff(num, den, key, prior_col):
        v = num / den.replace(0, np.nan)
        lg = LEAGUE_FF[key]
        prior = lg + PRIOR_REGRESSION * (long[prior_col].fillna(lg) - lg)
        return w * v.fillna(0) + (1 - w) * prior
    out["efg"] = ff(cum.fgm + 0.5 * cum.fg3m, cum.fga, "efg", "f_efg")
    out["tov_pct"] = ff(cum.tov, cum.fga + 0.44 * cum.fta + cum.tov, "tov_pct", "f_tov_pct")
    out["orb_pct"] = ff(cum.orb, cum.orb + cum.opp_drb, "orb_pct", "f_orb_pct")
    out["ft_rate"] = ff(cum.ftm, cum.fga, "ft_rate", "f_ft_rate")
    out["opp_efg"] = ff(cum.o_fgm + 0.5 * cum.o_fg3m, cum.o_fga, "efg", "f_opp_efg")

    # ---- rolling over last N played games (crosses seasons) -------------
    played_only = long[long.played == 1][["team", "pts", "pts_against", "poss", "won", "game_id"]]
    pg = played_only.groupby("team")
    rolled = played_only[["game_id", "team"]].copy()
    for wdw in ROLLING_WINDOWS:
        sums = pg[["pts", "pts_against", "poss"]].transform(
            lambda s: s.rolling(wdw, min_periods=1).sum())
        rolled[f"roll_off_{wdw}"] = 100 * sums.pts / sums.poss
        rolled[f"roll_def_{wdw}"] = 100 * sums.pts_against / sums.poss
        rolled[f"roll_net_{wdw}"] = rolled[f"roll_off_{wdw}"] - rolled[f"roll_def_{wdw}"]
    rolled["roll_pace_10"] = pg["poss"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    rolled["win_pct_10"] = pg["won"].transform(lambda s: s.rolling(10, min_periods=1).mean())

    def streak(s):
        res, cur = [], 0
        for v in s:
            cur = (cur + 1 if cur >= 0 else 1) if v == 1 else (cur - 1 if cur <= 0 else -1)
            res.append(cur)
        return pd.Series(res, index=s.index)
    rolled["win_streak"] = pg["won"].transform(streak)
    # these are POST-game values for played games; carry the latest one
    # forward to each team's NEXT game (pre-game view)
    roll_cols = [c for c in rolled.columns if c not in ("game_id", "team")]
    out = out.merge(rolled, on=["game_id", "team"], how="left")
    out = out.sort_values(["team", "date", "game_id"])
    out[roll_cols] = out.groupby("team")[roll_cols].transform(lambda s: s.shift(1).ffill())
    defaults = {c: LEAGUE_AVG_RTG for c in roll_cols if c.startswith(("roll_off", "roll_def"))}
    defaults.update({c: 0.0 for c in roll_cols if c.startswith("roll_net")})
    defaults.update(roll_pace_10=LEAGUE_AVG_PACE, win_pct_10=0.5, win_streak=0)
    out = out.fillna(defaults)

    # ---- strength of schedule: mean pre-game net rating of opponents faced
    opp_net = out[["game_id", "team", "net_rtg"]].rename(columns={"team": "opp", "net_rtg": "opp_net"})
    out = out.merge(opp_net, on=["game_id", "opp"], how="left")
    out = out.sort_values(["team", "date", "game_id"])
    played_mask = out.won.notna()
    opp_played = out.opp_net.where(played_mask, 0)
    c_sum = opp_played.groupby([out.team, out.season]).cumsum() - opp_played
    c_n = played_mask.astype(int).groupby([out.team, out.season]).cumsum() - played_mask.astype(int)
    out["sos"] = (c_sum / (c_n + 5)).fillna(0)   # shrunk toward 0

    # ---- fatigue & travel ------------------------------------------------
    venues = teams.set_index("team")[["lat", "lon"]]
    out["v_lat"] = out.home_team.map(venues.lat)
    out["v_lon"] = out.home_team.map(venues.lon)
    out["d"] = pd.to_datetime(out.date)
    grp = out.groupby("team")
    prev_d = grp["d"].shift(1)
    same_season = grp["season"].shift(1) == out.season
    out["days_rest"] = np.where(same_season, (out.d - prev_d).dt.days, 7).clip(0, 7)
    out["b2b"] = (out.days_rest == 1).astype(int)
    out["travel_km"] = np.where(
        same_season,
        _haversine(grp["v_lat"].shift(1), grp["v_lon"].shift(1), out.v_lat, out.v_lon), 0)
    out["travel_km"] = out.travel_km.fillna(0)
    def _last7(s):
        v = s.values
        idx = np.searchsorted(v, v - np.timedelta64(7, "D"), side="left")
        return pd.Series(np.arange(len(v)) - idx, index=s.index)
    out["games_last_7"] = grp["d"].transform(_last7)
    def _trip(s):
        res, cur = [], 0
        for v in s:
            cur = 0 if v == 1 else cur + 1      # consecutive away games incl. this one
            res.append(cur)
        return pd.Series(res, index=s.index)
    out["road_trip_len"] = grp["is_home"].transform(_trip)

    # ---- availability ----------------------------------------------------
    inj = inj.copy()
    is_out = inj.status == "OUT"
    inj["star_out"] = ((inj.is_star == 1) & is_out).astype(int)
    inj["star_questionable"] = ((inj.is_star == 1) & ~is_out).astype(int)
    inj["missing_minutes"] = inj.minutes_share * np.where(is_out, 1.0, 0.5)
    agg = inj.groupby(["game_id", "team"], as_index=False).agg(
        star_out=("star_out", "max"), star_questionable=("star_questionable", "max"),
        missing_minutes=("missing_minutes", "sum"))
    out = out.merge(agg, on=["game_id", "team"], how="left")
    out["star_out"] = out.star_out.fillna(0).astype(int)
    out["star_questionable"] = out.star_questionable.fillna(0).astype(int)
    out["missing_minutes"] = out.missing_minutes.fillna(0.0)
    out = out.sort_values(["team", "date", "game_id"])

    # rotation continuity = 1 - minutes share of players whose availability
    # changed since the team's previous game (newly out OR just returned)
    outs: dict = {}
    for gid, team, player, share in inj.loc[is_out, ["game_id", "team", "player", "minutes_share"]].itertuples(index=False):
        outs.setdefault((gid, team), {})[player] = share
    cont, prev_key, prev = [], None, {}
    for gid, team, season in zip(out.game_id, out.team, out.season):
        cur = outs.get((gid, team), {})
        if (team, season) != prev_key:
            # first game of a team's season: no previous game to compare with
            prev_key, prev = (team, season), cur
        changed = sum(cur[p] for p in cur.keys() - prev.keys()) + sum(prev[p] for p in prev.keys() - cur.keys())
        cont.append(max(0.0, 1.0 - changed))
        prev = cur
    out["rotation_continuity"] = cont

    return out[["game_id", "team", "is_home"] + TEAM_FEATURES].reset_index(drop=True)


# ---------------------------------------------------------- game features
SHARED = (
    [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES]
    + ["net_rtg_diff", "roll_net_10_diff", "rest_diff", "travel_diff", "form_diff",
       "streak_diff", "continuity_diff", "missing_minutes_diff", "pace_avg",
       "closing_home_spread", "closing_total", "mkt_home_prob"]
)
SPREAD_EXTRA = ["internal_spread", "spread_gap", "fav_rating_gap"]
TOTAL_EXTRA = ["home_exp_ortg", "away_exp_ortg", "off_def_interaction",
               "pace_interaction", "expected_possessions", "internal_total",
               "total_gap", "fatigue_scoring"]

FEATURE_SETS = {
    "win": SHARED,
    "spread": SHARED + SPREAD_EXTRA,
    "total": SHARED + TOTAL_EXTRA,
}


def game_features(games: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    tfi = tf.drop(columns="is_home")
    home = tfi.add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_team": "home_team"})
    away = tfi.add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_team": "away_team"})
    df = games.merge(home, on=["game_id", "home_team"]).merge(away, on=["game_id", "away_team"])

    df["net_rtg_diff"] = df.home_net_rtg - df.away_net_rtg
    df["roll_net_10_diff"] = df.home_roll_net_10 - df.away_roll_net_10
    df["rest_diff"] = df.home_days_rest - df.away_days_rest
    df["travel_diff"] = df.home_travel_km - df.away_travel_km
    df["form_diff"] = df.home_win_pct_10 - df.away_win_pct_10
    df["streak_diff"] = df.home_win_streak - df.away_win_streak
    df["continuity_diff"] = df.home_rotation_continuity - df.away_rotation_continuity
    df["missing_minutes_diff"] = df.home_missing_minutes - df.away_missing_minutes
    df["pace_avg"] = (df.home_pace + df.away_pace) / 2

    ph, pa = american_to_prob(df.home_moneyline), american_to_prob(df.away_moneyline)
    df["mkt_home_prob"] = ph / (ph + pa)            # vig removed

    # spread extras: our own rating-based line vs the market's
    exp_poss = df.home_pace * df.away_pace / LEAGUE_AVG_PACE
    df["internal_spread"] = -(df.net_rtg_diff * exp_poss / 100 + HCA_POINTS)
    df["home_exp_ortg"] = df.home_off_rtg + df.away_def_rtg - LEAGUE_AVG_RTG
    df["away_exp_ortg"] = df.away_off_rtg + df.home_def_rtg - LEAGUE_AVG_RTG
    df["internal_total"] = exp_poss * (df.home_exp_ortg + df.away_exp_ortg) / 100

    # upcoming games without a posted line yet: use our own rating-based line
    # (flagged so the UI can show it). Played games keep missing lines as NaN.
    unplayed = df.home_score.isna()
    df["lines_estimated"] = 0
    est_s = unplayed & df.closing_home_spread.isna()
    df.loc[est_s, "closing_home_spread"] = (df.loc[est_s, "internal_spread"] * 2).round() / 2
    est_t = unplayed & df.closing_total.isna()
    df.loc[est_t, "closing_total"] = (df.loc[est_t, "internal_total"] * 2).round() / 2
    est_p = unplayed & df.mkt_home_prob.isna()
    df.loc[est_p, "mkt_home_prob"] = 0.5 * (1 + _erf(-df.loc[est_p, "closing_home_spread"] / (12.5 * math.sqrt(2))))
    df.loc[est_s | est_t | est_p, "lines_estimated"] = 1

    df["spread_gap"] = df.internal_spread - df.closing_home_spread
    fav_home = df.closing_home_spread <= 0
    df["fav_rating_gap"] = np.where(fav_home, df.net_rtg_diff, -df.net_rtg_diff)

    # total extras
    df["off_def_interaction"] = (df.home_off_rtg * df.away_def_rtg
                                 + df.away_off_rtg * df.home_def_rtg) / 1000
    df["pace_interaction"] = df.home_pace * df.away_pace / 100
    df["expected_possessions"] = exp_poss
    df["total_gap"] = df.internal_total - df.closing_total
    df["fatigue_scoring"] = df.home_b2b + df.away_b2b + (df.home_travel_km + df.away_travel_km) / 2000

    # targets (NaN for unplayed games)
    played = df.home_score.notna()
    margin = df.home_score - df.away_score
    df["home_win"] = np.where(played, (margin > 0).astype(float), np.nan)
    df["home_margin_vs_spread"] = margin + df.closing_home_spread
    has_spread, has_total = df.closing_home_spread.notna(), df.closing_total.notna()
    df["home_covers"] = np.where(played & has_spread & (df.home_margin_vs_spread != 0),
                                 (df.home_margin_vs_spread > 0).astype(float), np.nan)  # pushes excluded
    df["total_points"] = df.home_score + df.away_score
    df["over"] = np.where(played & has_total & (df.total_points != df.closing_total),
                          (df.total_points > df.closing_total).astype(float), np.nan)
    return df.sort_values(["date", "tip_time", "game_id"]).reset_index(drop=True)


def build_all(persist: bool = True):
    games, box, inj, teams = load_raw()
    tf = team_features(games, box, inj, teams)
    gf = game_features(games, tf)
    if persist:
        with session() as con:
            con.execute("DELETE FROM team_game_features")
            con.executemany(
                "INSERT INTO team_game_features VALUES (?,?,?)",
                [(r.game_id, r.team, json.dumps({k: (None if pd.isna(v) else float(v))
                                                 for k, v in zip(TEAM_FEATURES, r[3:])}))
                 for r in tf.itertuples(index=False)])
    return gf
