"""Real NBA data from ESPN's public JSON endpoints.

Writes the same four CSVs as the synthetic feed (see app/data/ingest.py)
into data/feed_real/, so the rest of the app is unchanged.

Sources (no API key needed)
  scoreboard  /scoreboard?dates=YYYYMMDD   schedule, scores, current DraftKings lines
  summary     /summary?event=ID            team box score, every player's minutes,
                                           who did not play and why, open/close lines
  injuries    /injuries                    today's league-wide injury report

Player availability for past games comes from the box score: a player
listed as "did not play" for any reason except a coach's decision is
recorded as OUT. His minutes share and star status are computed only from
games before that one, so there is no look-ahead.

Parsed games are cached in data/espn_cache/, so later runs only download
new games.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .synthetic import TEAMS

log = logging.getLogger("espn")

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
# books in order of preference for the core odds fallback; live feeds and
# model sites (accuscore, betegy) are never used
BOOK_PREFERENCE = ["draftkings", "draft kings", "espn bet", "consensus", "mgm", "caesars", "fanduel",
                   "betmgm", "pointsbet", "sugarhouse", "unibet", "betfair"]
NOT_BOOKS = ("live", "accuscore", "betegy", "numberfire", "teamrankings")
HEADERS = {"User-Agent": "Mozilla/5.0 (NBA Predictor personal project)",
           "Accept": "application/json", "Accept-Encoding": "gzip"}

# ESPN abbreviations that differ from the standard ones used in the app
ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
        "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
VALID = set(TEAMS)
ESPN_TEAM_IDS = {
    "ATL": 1, "BOS": 2, "NOP": 3, "CHI": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10,
    "IND": 11, "LAC": 12, "LAL": 13, "MIA": 14, "MIL": 15, "MIN": 16, "BKN": 17, "NYK": 18, "ORL": 19,
    "PHI": 20, "PHX": 21, "POR": 22, "SAC": 23, "SAS": 24, "OKC": 25, "UTA": 26, "WAS": 27, "TOR": 28,
    "MEM": 29, "CHA": 30,
}

STAR_MIN_GAMES = 5      # games needed before a player can count as a star
STAR_MIN_MPG = 24.0
ROT_WINDOW = 10         # a rotation player appeared in 3+ of the team's last 10 games
ROT_MIN_APPS = 3
ROT_MIN_MPG = 12.0      # ...averaging 12+ minutes in those appearances
TEAM_MINUTES = 240.0


# ----------------------------------------------------------------- HTTP
def _get(url: str, retries: int = 4) -> dict:
    delay = 1.5
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            code = getattr(e, "code", None)
            if code == 404 or attempt == retries:
                raise
            time.sleep(delay * (attempt + 1))
    raise RuntimeError("unreachable")


# --------------------------------------------------------------- parsing
def team_abbr(a: str | None) -> str | None:
    if not a:
        return None
    a = ABBR.get(a.upper(), a.upper())
    return a if a in VALID else None


def _num(v):
    """'-11.5' / '+410' / 'o227.5' / 'EVEN' / 'PK' -> float or None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().lstrip("ou").replace("+", "")
    if s in ("even", "ev"):
        return 100.0
    if s in ("pk", "pick"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _iso_utc(s: str | None) -> str | None:
    """'2026-10-21T23:00Z' -> '2026-10-21T23:00:00Z' (sortable, JS-parsable)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _side(obj: dict, *path):
    for k in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def parse_odds(o: dict | None) -> dict:
    """Works for scoreboard competitions[].odds[0] and summary pickcenter[0]."""
    out = dict(closing_home_spread=None, closing_total=None, home_moneyline=None,
               away_moneyline=None, open_home_spread=None, open_total=None, line_provider=None)
    if not o:
        return out
    out["line_provider"] = _side(o, "provider", "name")

    def close_open(*path):
        node = _side(o, *path) or {}
        c = node.get("close") or node.get("current") or {}
        op = node.get("open") or {}
        return c, op

    c, op = close_open("pointSpread", "home")
    out["closing_home_spread"] = _num(c.get("line"))
    out["open_home_spread"] = _num(op.get("line"))
    if out["closing_home_spread"] is None and o.get("spread") is not None:
        out["closing_home_spread"] = _num(o.get("spread"))     # home perspective on ESPN

    c, op = close_open("total", "over")
    out["closing_total"] = _num(c.get("line"))
    out["open_total"] = _num(op.get("line"))
    if out["closing_total"] is None:
        out["closing_total"] = _num(o.get("overUnder"))

    for side in ("home", "away"):
        c, _ = close_open("moneyline", side)
        ml = _num(c.get("odds"))
        if ml is None:
            ml = _num(_side(o, f"{side}TeamOdds", "moneyLine"))
        out[f"{side}_moneyline"] = None if ml is None else int(ml)
    return out


def _spread_from_details(details, home: str, away: str):
    """'DEN -4.5' -> home spread, using which team the line names."""
    if not details:
        return None
    d = str(details).strip().upper()
    if d in ("EVEN", "PK", "PICK", "PICK'EM"):
        return 0.0
    parts = d.split()
    if len(parts) != 2:
        return None
    team, val = team_abbr(parts[0]), _num(parts[1])
    if val is None or team is None:
        return None
    if team == home:
        return val
    if team == away:
        return -val
    return None


def parse_core_odds(js: dict, home: str, away: str) -> dict:
    """ESPN core odds endpoint (many books per game) -> one closing line."""
    out = dict(closing_home_spread=None, closing_total=None, home_moneyline=None,
               away_moneyline=None, open_home_spread=None, open_total=None, line_provider=None)
    items = []
    for it in js.get("items", []):
        name = (_side(it, "provider", "name") or "").lower()
        if any(x in name for x in NOT_BOOKS):
            continue
        if it.get("spread") is None and it.get("overUnder") is None and not it.get("details"):
            continue
        rank = next((i for i, b in enumerate(BOOK_PREFERENCE) if b in name), len(BOOK_PREFERENCE))
        items.append((rank, it))
    if not items:
        return out
    items.sort(key=lambda x: x[0])
    it = items[0][1]
    spread = _spread_from_details(it.get("details"), home, away)
    if spread is None and it.get("spread") is not None:
        spread = _num(it.get("spread"))
    out.update(closing_home_spread=spread, closing_total=_num(it.get("overUnder")),
               line_provider=_side(it, "provider", "name"))
    op = _side(it, "homeTeamOdds", "open", "pointSpread", "american")
    out["open_home_spread"] = _num(op)
    out["open_total"] = _num(_side(it, "open", "total", "american"))
    for _, cand in items:                      # first book with both moneylines
        hm, am = _num(_side(cand, "homeTeamOdds", "moneyLine")), _num(_side(cand, "awayTeamOdds", "moneyLine"))
        if hm is not None and am is not None:
            out["home_moneyline"], out["away_moneyline"] = int(hm), int(am)
            break
    return out


def parse_scoreboard(js: dict, day: date) -> list[dict]:
    games = []
    for ev in js.get("events", []):
        season = ev.get("season", {})
        if season.get("type") != 2:                 # regular season only
            continue
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        h, a = team_abbr(_side(teams["home"], "team", "abbreviation")), team_abbr(_side(teams["away"], "team", "abbreviation"))
        if not h or not a:
            continue                                # All-Star, exhibitions
        status = _side(comp, "status", "type") or {}
        yr = int(season.get("year"))
        g = dict(game_id=str(ev["id"]), date=day.isoformat(),
                 season=f"{yr - 1}-{str(yr)[2:]}", tip_time=_iso_utc(ev.get("date")),
                 home_team=h, away_team=a, neutral=int(bool(comp.get("neutralSite"))),
                 completed=bool(status.get("completed")), state=status.get("state"),
                 status_name=status.get("name"),
                 home_score=_num(teams["home"].get("score")), away_score=_num(teams["away"].get("score")))
        g.update(parse_odds((comp.get("odds") or [None])[0]))
        games.append(g)
    return games


def _stat_lookup(stats: list[dict]) -> dict:
    out = {}
    for s in stats or []:
        for k in (s.get("name"), s.get("label"), s.get("abbreviation")):
            if k:
                out[k] = s.get("displayValue")
    return out


def _made_att(v):
    try:
        m, a = str(v).split("-")
        return int(m), int(a)
    except (ValueError, AttributeError):
        return None, None


def parse_summary(js: dict) -> dict:
    """Team box, player lines and closing odds for one completed game."""
    box = {}
    for t in _side(js, "boxscore", "teams") or []:
        ab = team_abbr(_side(t, "team", "abbreviation"))
        s = _stat_lookup(t.get("statistics"))
        fgm, fga = _made_att(s.get("fieldGoalsMade-fieldGoalsAttempted") or s.get("FG"))
        fg3m, _ = _made_att(s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted") or s.get("3PT"))
        ftm, fta = _made_att(s.get("freeThrowsMade-freeThrowsAttempted") or s.get("FT"))
        tov = _num(s.get("totalTurnovers") or s.get("Total Turnovers") or s.get("turnovers") or s.get("Turnovers"))
        box[ab] = dict(fgm=fgm, fga=fga, fg3m=fg3m, ftm=ftm, fta=fta, tov=tov,
                       orb=_num(s.get("offensiveRebounds") or s.get("Offensive Rebounds")),
                       drb=_num(s.get("defensiveRebounds") or s.get("Defensive Rebounds")))

    players = []
    for t in _side(js, "boxscore", "players") or []:
        ab = team_abbr(_side(t, "team", "abbreviation"))
        for grp in t.get("statistics") or []:
            keys = grp.get("keys") or grp.get("names") or []
            idx = {k: i for i, k in enumerate(keys)}
            i_min = idx.get("minutes", idx.get("MIN"))
            i_pts = idx.get("points", idx.get("PTS"))
            for ath in grp.get("athletes") or []:
                st = ath.get("stats") or []
                mins = _num(st[i_min]) if i_min is not None and i_min < len(st) else None
                pts = _num(st[i_pts]) if i_pts is not None and i_pts < len(st) else None
                players.append(dict(
                    team=ab, player_id=str(_side(ath, "athlete", "id") or ""),
                    player=_side(ath, "athlete", "displayName") or "",
                    dnp=bool(ath.get("didNotPlay")) or not st,
                    reason=(ath.get("reason") or "").upper(),
                    minutes=mins or 0.0, points=pts or 0.0))

    odds = parse_odds((js.get("pickcenter") or [None])[0])
    return dict(box=box, players=players, odds=odds)


# ------------------------------------------------------------- the feed
class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"scoreboards": {}, "summaries": {}}
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self.data = json.load(f)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(self.data, f)
        tmp.replace(self.path)


def season_dates(end_year: int, today: date, lookahead: int):
    start, end = date(end_year - 1, 10, 1), date(end_year, 4, 20)
    end = min(end, today + timedelta(days=lookahead))
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def season_end_years(first_start_year: int, today: date) -> list[int]:
    """ESPN labels a season by its end year: 2025-26 -> 2026."""
    return list(range(first_start_year + 1, current_season_end_year(today) + 1))


def current_season_end_year(today: date) -> int:
    return today.year + 1 if today.month >= 8 else today.year


def fetch(out_dir: Path, cache_dir: Path, seasons: list[int], today: date | None = None,
          lookahead: int = 7, workers: int = 8, progress=print) -> dict:
    """Download/refresh the given seasons (ESPN end years, e.g. 2026 = 2025-26)."""
    today = today or date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_games, summaries = [], {}

    for yr in seasons:
        cache = Cache(cache_dir / f"espn_{yr}.json.gz")
        sb, sm = cache.data["scoreboards"], cache.data["summaries"]
        days = list(season_dates(yr, today, lookahead))
        # past days with all games final never change: fetch once
        need = [d for d in days if d.isoformat() not in sb or d >= today - timedelta(days=2)]
        progress(f"{yr - 1}-{str(yr)[2:]}: {len(days)} days, fetching {len(need)} scoreboards")
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_get, f"{BASE}/scoreboard?dates={d:%Y%m%d}&limit=50"): d for d in need}
            for f in as_completed(futs):
                d = futs[f]
                try:
                    games = parse_scoreboard(f.result(), d)
                except Exception as e:                       # keep going, log it
                    log.warning("scoreboard %s failed: %s", d, e)
                    continue
                if d < today - timedelta(days=2) and any(not g["completed"] for g in games
                                                          if g["status_name"] not in ("STATUS_POSTPONED", "STATUS_CANCELED")):
                    continue                                  # retry later
                sb[d.isoformat()] = games
        games = [g for d in days for g in sb.get(d.isoformat(), [])]

        todo = [g["game_id"] for g in games if g["completed"] and g["game_id"] not in sm]
        progress(f"{yr - 1}-{str(yr)[2:]}: {len(games)} games, fetching {len(todo)} box scores")
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_get, f"{BASE}/summary?event={gid}"): gid for gid in todo}
            for f in as_completed(futs):
                gid = futs[f]
                try:
                    sm[gid] = parse_summary(f.result())
                except Exception as e:
                    log.warning("summary %s failed: %s", gid, e)
                done += 1
                if done % 200 == 0:
                    cache.save()
                    progress(f"  {done}/{len(todo)} box scores")
        cache.save()

        # closing lines missing from the game summary (ESPN dropped them from
        # summaries for much of 2023-25): use the multi-book odds endpoint
        co = cache.data.setdefault("core_odds", {})
        by_id = {g["game_id"]: g for g in games}
        todo = [gid for gid, v in sm.items() if gid in by_id and v["odds"].get("closing_home_spread") is None
                and v["odds"].get("closing_total") is None and gid not in co]
        if todo:
            progress(f"{yr - 1}-{str(yr)[2:]}: fetching closing lines for {len(todo)} games from the odds feed")
        done = 0
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(_get, f"{CORE}/events/{gid}/competitions/{gid}/odds?limit=50"): gid for gid in todo}
            for f in as_completed(futs):
                gid = futs[f]
                try:
                    g = by_id[gid]
                    co[gid] = parse_core_odds(f.result(), g["home_team"], g["away_team"])
                except Exception as e:
                    log.warning("core odds %s failed: %s", gid, e)
                done += 1
                if done % 300 == 0:
                    cache.save()
                    progress(f"  {done}/{len(todo)} lines")
        cache.save()
        for gid, o in co.items():
            if gid in sm and sm[gid]["odds"].get("closing_home_spread") is None:
                sm[gid]["odds"] = o
        all_games += games
        summaries.update(sm)

    inj_report = []
    try:
        inj_report = parse_injury_report(_get(f"{BASE}/injuries"))
        progress(f"Injury report: {len(inj_report)} players listed")
    except Exception as e:
        log.warning("injury report failed: %s", e)
        progress(f"WARNING: injury report unavailable ({e})")

    rosters = []
    def _roster(team):
        js = _get(f"{BASE}/teams/{ESPN_TEAM_IDS[team]}/roster")
        return [parse_roster_player(a, team) for a in js.get("athletes", [])
                if isinstance(a, dict) and a.get("id")]
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_roster, t): t for t in ESPN_TEAM_IDS}
        for f in as_completed(futs):
            try:
                rosters += f.result()
            except Exception as e:
                log.warning("roster %s failed: %s", futs[f], e)
    progress(f"Rosters: {len(rosters)} players on {len({r['team'] for r in rosters})} teams")

    return write_feed(all_games, summaries, inj_report, out_dir, today, rosters)


def parse_roster_player(a: dict, team: str) -> dict:
    inj = (a.get("injuries") or [{}])[0] or {}
    return dict(team=team, player_id=str(a.get("id")), player=a.get("displayName", ""),
                position=_side(a, "position", "abbreviation"), jersey=a.get("jersey"),
                age=a.get("age"), height=a.get("displayHeight"), weight=a.get("displayWeight"),
                experience=_side(a, "experience", "years"), injury_status=inj.get("status"))


def parse_injury_report(js: dict) -> list[dict]:
    rows = []
    for team in js.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            ab = team_abbr(_side(ath, "team", "abbreviation"))
            if not ab:
                continue
            status = (inj.get("status") or "").lower()
            fantasy = (_side(inj, "details", "fantasyStatus", "abbreviation") or "").upper()
            st = "OUT" if status == "out" or fantasy in ("OUT", "OFS") else "QUESTIONABLE"
            rows.append(dict(team=ab, player=ath.get("displayName", ""), status=st,
                             reported_at=inj.get("date") or datetime.utcnow().isoformat(),
                             comment=inj.get("shortComment") or ""))
    return rows


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return " ".join(n.replace(".", "").replace("'", "").replace(" jr", "").replace(" iii", "")
                    .replace(" ii", "").split())


class PlayerState:
    """Running per-player averages, updated only after each game (no look-ahead)."""

    def __init__(self):
        self.p = {}          # player_id -> dict

    def roll_season(self, pid, season):
        s = self.p[pid]
        if s["season"] != season:
            if s["gp"] >= 5:
                s["prev_mpg"], s["prev_ppg"] = s["min"] / s["gp"], s["pts"] / s["gp"]
            s.update(season=season, gp=0, min=0.0, pts=0.0)

    def get(self, pid, season):
        if pid not in self.p:
            return None
        self.roll_season(pid, season)
        return self.p[pid]

    def mpg(self, s):
        if s["gp"] >= 3:
            return s["min"] / s["gp"]
        return s.get("prev_mpg") or (s["min"] / s["gp"] if s["gp"] else 0.0)

    def ppg(self, s):
        if s["gp"] >= STAR_MIN_GAMES:
            return s["pts"] / s["gp"]
        return s.get("prev_ppg") or 0.0

    def update(self, pid, name, team, season, minutes, points):
        s = self.p.setdefault(pid, dict(season=season, gp=0, min=0.0, pts=0.0, name=name))
        self.roll_season(pid, season)
        s.update(team=team, name=name)
        s["gp"] += 1
        s["min"] += minutes
        s["pts"] += points

    def stars(self, team, season):
        cands = []
        for pid, s in self.p.items():
            if s.get("team") != team:
                continue
            self.roll_season(pid, season)
            if self.mpg(s) >= STAR_MIN_MPG and (s["gp"] >= STAR_MIN_GAMES or s.get("prev_ppg")):
                cands.append((self.ppg(s), pid))
        return {pid for _, pid in sorted(cands, reverse=True)[:2]}


def write_feed(games, summaries, inj_report, out_dir: Path, today: date, rosters=None) -> dict:
    games = sorted({g["game_id"]: g for g in games}.values(), key=lambda g: (g["date"], g["tip_time"] or "", g["game_id"]))
    game_rows, box_rows, inj_rows = [], [], []
    state = PlayerState()
    recent = {}          # team -> deque of {player_id: (name, minutes)} for its last games
    missing_box = missing_lines = 0

    for g in games:
        sm = summaries.get(g["game_id"])
        played = g["completed"] and sm is not None
        if g["completed"] and sm is None:
            missing_box += 1
            continue
        if not g["completed"] and g["date"] < today.isoformat():
            continue                                    # postponed / cancelled
        odds = dict(g)
        if sm and sm["odds"].get("closing_home_spread") is not None:
            odds.update({k: v for k, v in sm["odds"].items() if v is not None})
        if odds.get("closing_home_spread") is None:
            missing_lines += 1
        tip = g["tip_time"] or f"{g['date']}T23:00:00Z"
        tip_dt = datetime.fromisoformat(tip.replace("Z", "+00:00"))
        game_rows.append(dict(
            game_id=g["game_id"], date=g["date"], season=g["season"], tip_time=tip,
            prediction_timestamp=(tip_dt - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            home_team=g["home_team"], away_team=g["away_team"],
            home_score=int(g["home_score"]) if played else None,
            away_score=int(g["away_score"]) if played else None,
            closing_home_spread=odds.get("closing_home_spread"), closing_total=odds.get("closing_total"),
            home_moneyline=odds.get("home_moneyline"), away_moneyline=odds.get("away_moneyline"),
            open_home_spread=odds.get("open_home_spread"), open_total=odds.get("open_total"),
            line_provider=odds.get("line_provider"), neutral=g.get("neutral", 0)))

        if not played:
            continue
        b = sm["box"]
        h, a = g["home_team"], g["away_team"]
        if h in b and a in b and all(b[t].get(k) is not None for t in (h, a) for k in ("fga", "fgm", "fta", "orb", "drb", "tov")):
            for t, o, is_home, pts in ((h, a, 1, g["home_score"]), (a, h, 0, g["away_score"])):
                box_rows.append(dict(game_id=g["game_id"], team=t, opp=o, is_home=is_home, pts=int(pts), poss=None,
                                     fga=b[t]["fga"], fgm=b[t]["fgm"], fg3m=b[t]["fg3m"] or 0,
                                     fta=b[t]["fta"], ftm=b[t]["ftm"], tov=int(b[t]["tov"]),
                                     orb=int(b[t]["orb"]), drb=int(b[t]["drb"]), opp_drb=int(b[o]["drb"])))

        # availability BEFORE this game. Two signals, because older ESPN box
        # scores don't list injured inactives:
        #  1. listed as did-not-play for any reason except a coach's decision
        #  2. a rotation player over the team's last 10 games who didn't appear
        for t in (h, a):
            stars = state.stars(t, g["season"])
            team_players = [p for p in sm["players"] if p["team"] == t]
            appeared = {p["player_id"] for p in team_players if not p["dnp"] and p["minutes"] > 0}
            scratched = {p["player_id"] for p in team_players if p["dnp"] and "COACH" in p["reason"]}
            absent = {p["player_id"]: (p["player"], p["reason"].title()) for p in team_players
                      if p["dnp"] and "COACH" not in p["reason"]}
            apps, mins, names = {}, {}, {}
            for gm in recent.get(t, ()):
                for pid, (nm, mn) in gm.items():
                    apps[pid] = apps.get(pid, 0) + 1
                    mins[pid] = mins.get(pid, 0.0) + mn
                    names[pid] = nm
            for pid, n in apps.items():
                if (n >= ROT_MIN_APPS and mins[pid] / n >= ROT_MIN_MPG and pid not in appeared
                        and pid not in scratched and pid not in absent):
                    absent[pid] = (names[pid], "Did not play")
            for pid, (name, reason) in absent.items():
                s = state.get(pid, g["season"])
                if s is None:
                    continue                            # never played: no minutes to lose
                share = round(min(state.mpg(s) / TEAM_MINUTES, 0.25), 4)
                if share < 0.02:
                    continue
                inj_rows.append(dict(game_id=g["game_id"], team=t, player=name,
                                     is_star=int(pid in stars), minutes_share=share,
                                     status="OUT", reported_at=tip, reason=reason))
            dq = recent.setdefault(t, deque(maxlen=ROT_WINDOW))
            dq.append({p["player_id"]: (p["player"], p["minutes"]) for p in team_players
                       if not p["dnp"] and p["minutes"] > 0})
        for p in sm["players"]:
            if not p["dnp"] and p["minutes"] > 0 and p["team"]:
                state.update(p["player_id"], p["player"], p["team"], g["season"], p["minutes"], p["points"])

    # live injury report -> every upcoming game in the window for that team
    by_name = {}
    for pid, s in state.p.items():
        by_name[(s.get("team"), _norm(s["name"]))] = pid
        by_name.setdefault((None, _norm(s["name"])), pid)
    upcoming = [r for r in game_rows if r["home_score"] is None]
    cur_season = upcoming[0]["season"] if upcoming else None
    matched = 0
    for r in inj_report:
        pid = by_name.get((r["team"], _norm(r["player"]))) or by_name.get((None, _norm(r["player"])))
        s = state.get(pid, cur_season) if pid else None
        share = round(min(state.mpg(s) / TEAM_MINUTES, 0.25), 4) if s else 0.0
        is_star = int(bool(pid) and pid in state.stars(r["team"], cur_season)) if cur_season else 0
        matched += bool(pid)
        for gm in upcoming:
            if r["team"] in (gm["home_team"], gm["away_team"]):
                inj_rows.append(dict(game_id=gm["game_id"], team=r["team"], player=r["player"],
                                     is_star=is_star, minutes_share=share, status=r["status"],
                                     reported_at=r["reported_at"], reason=r["comment"][:120]))

    # current rosters joined to each player's real averages
    roster = []
    for r in rosters or []:
        s = state.get(r["player_id"], cur_season) if cur_season else state.p.get(r["player_id"])
        stars = state.stars(r["team"], cur_season) if cur_season else set()
        row = dict(r, gp=0, mpg=None, ppg=None, prev_mpg=None, prev_ppg=None, is_star=0)
        if s:
            row.update(gp=s["gp"], mpg=round(s["min"] / s["gp"], 1) if s["gp"] else None,
                       ppg=round(s["pts"] / s["gp"], 1) if s["gp"] else None,
                       prev_mpg=round(s["prev_mpg"], 1) if s.get("prev_mpg") else None,
                       prev_ppg=round(s["prev_ppg"], 1) if s.get("prev_ppg") else None,
                       is_star=int(r["player_id"] in stars))
        roster.append(row)

    gdf = pd.DataFrame(game_rows)
    pd.DataFrame(game_rows).to_csv(out_dir / "games.csv", index=False)
    pd.DataFrame(box_rows).to_csv(out_dir / "team_box.csv", index=False)
    idf = pd.DataFrame(inj_rows, columns=["game_id", "team", "player", "is_star", "minutes_share",
                                          "status", "reported_at", "reason"])
    idf = idf.drop_duplicates(["game_id", "team", "player"])
    idf.to_csv(out_dir / "injuries.csv", index=False)
    pd.DataFrame([dict(team=t, name=v[0], lat=v[1], lon=v[2]) for t, v in TEAMS.items()]
                 ).to_csv(out_dir / "teams.csv", index=False)
    pd.DataFrame(roster).to_csv(out_dir / "rosters.csv", index=False)

    played = gdf[gdf.home_score.notna()] if len(gdf) else gdf
    summary = dict(
        games=len(gdf), completed=len(played), upcoming=int(gdf.home_score.isna().sum()) if len(gdf) else 0,
        box_scores=len(box_rows) // 2, missing_box_scores=missing_box, games_without_lines=missing_lines,
        availability_rows=len(idf), injury_report_players=len(inj_report),
        injury_report_matched=matched, roster_players=len(roster),
        line_coverage=round(1 - missing_lines / max(len(gdf), 1), 3),
        seasons=played.groupby("season").size().to_dict() if len(played) else {},
        last_result=played.date.max() if len(played) else None,
        next_games=gdf[gdf.home_score.isna()].date.min() if len(gdf) else None)
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
