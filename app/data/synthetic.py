"""Synthetic NBA data feed.

Produces CSVs in exactly the format the real-data importer expects
(see app/data/ingest.py), so the whole system can be built, trained and
tested before a real data source is wired in.

The simulation has hidden team strengths that drift through a season,
star players who miss games (more often on back-to-backs), role-player
absences, rest and travel effects, home court, and a sharp betting
market that sees the same information. Because the market is sharp, a
model should NOT beat the spread or total by much on this data — which is
also what you should expect against real closing lines.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = {
    "ATL": ("Atlanta", 33.75, -84.39), "BOS": ("Boston", 42.36, -71.06),
    "BKN": ("Brooklyn", 40.68, -73.97), "CHA": ("Charlotte", 35.23, -80.84),
    "CHI": ("Chicago", 41.88, -87.67), "CLE": ("Cleveland", 41.50, -81.69),
    "DAL": ("Dallas", 32.79, -96.81), "DEN": ("Denver", 39.75, -105.01),
    "DET": ("Detroit", 42.34, -83.06), "GSW": ("Golden State", 37.77, -122.39),
    "HOU": ("Houston", 29.75, -95.36), "IND": ("Indiana", 39.76, -86.16),
    "LAC": ("LA Clippers", 33.94, -118.34), "LAL": ("LA Lakers", 34.04, -118.27),
    "MEM": ("Memphis", 35.14, -90.05), "MIA": ("Miami", 25.78, -80.19),
    "MIL": ("Milwaukee", 43.05, -87.92), "MIN": ("Minnesota", 44.98, -93.28),
    "NOP": ("New Orleans", 29.95, -90.08), "NYK": ("New York", 40.75, -73.99),
    "OKC": ("Oklahoma City", 35.46, -97.52), "ORL": ("Orlando", 28.54, -81.38),
    "PHI": ("Philadelphia", 39.90, -75.17), "PHX": ("Phoenix", 33.45, -112.07),
    "POR": ("Portland", 45.53, -122.67), "SAC": ("Sacramento", 38.58, -121.50),
    "SAS": ("San Antonio", 29.43, -98.44), "TOR": ("Toronto", 43.64, -79.38),
    "UTA": ("Utah", 40.77, -111.90), "WAS": ("Washington", 38.90, -77.02),
}

LEAGUE_RTG = 114.0
LEAGUE_PACE = 99.0
HCA = 1.4            # ortg points per 100 added to home, removed from away
SEASON_DAYS = 174
GAMES_PER_TEAM = 82


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(h))


def _american(p: float, vig: float = 0.0225) -> int:
    p = min(max(p + vig, 0.02), 0.98)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _schedule(rng, start: date):
    """Greedy day-by-day schedule: ~82 games per team, roughly even home/away."""
    teams = list(TEAMS)
    remaining = {t: GAMES_PER_TEAM for t in teams}
    homes = {t: 0 for t in teams}
    played_yday: set[str] = set()
    games = []
    for d in range(SEASON_DAYS):
        days_left = SEASON_DAYS - d
        avail = []
        for t in teams:
            if remaining[t] <= 0:
                continue
            p = remaining[t] / days_left * 1.9
            if t in played_yday:
                p *= 0.45          # back-to-backs happen, but less often
            if rng.random() < min(p, 0.95):
                avail.append(t)
        rng.shuffle(avail)
        today = set()
        for i in range(0, len(avail) - 1, 2):
            a, b = avail[i], avail[i + 1]
            # whoever has fewer home games hosts (with some randomness)
            if homes[a] + rng.normal(0, 1.5) < homes[b]:
                h, aw = a, b
            else:
                h, aw = b, a
            homes[h] += 1
            remaining[a] -= 1
            remaining[b] -= 1
            today.update((a, b))
            games.append((start + timedelta(days=d), h, aw))
        played_yday = today
    return games


def generate(feed_dir: Path, first_season: int = 2021, last_season: int = 2026,
             seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    feed_dir.mkdir(parents=True, exist_ok=True)

    off = {t: rng.normal(0, 3) for t in TEAMS}
    dfn = {t: rng.normal(0, 3) for t in TEAMS}
    pace = {t: rng.normal(LEAGUE_PACE, 2.5) for t in TEAMS}

    game_rows, box_rows, inj_rows = [], [], []
    for yr in range(first_season, last_season + 1):
        season = season_label(yr)
        # off-season: partial carry-over, new stars, pace drift
        for t in TEAMS:
            off[t] = 0.55 * off[t] + rng.normal(0, 2.6)
            dfn[t] = 0.55 * dfn[t] + rng.normal(0, 2.6)
            pace[t] = 0.7 * pace[t] + 0.3 * LEAGUE_PACE + rng.normal(0, 1.2)
        star = {t: (f"{t} Star", rng.uniform(3, 7)) for t in TEAMS}
        star_out_left = {t: 0 for t in TEAMS}
        last_game = {t: None for t in TEAMS}     # (date, city)

        start = date(yr, 10, 19 if yr != 2026 else 20)
        for n, (gd, h, a) in enumerate(_schedule(rng, start)):
            gid = f"{yr}{n:05d}"
            tip = datetime(gd.year, gd.month, gd.day, 19, 0) + timedelta(
                minutes=int(rng.choice([0, 30, 60, 90])))
            ctx = {}
            for t, is_home in ((h, True), (a, False)):
                # latent drift during the season
                off[t] += rng.normal(0, 0.12)
                dfn[t] += rng.normal(0, 0.12)
                prev = last_game[t]
                rest = (gd - prev[0]).days if prev else 5
                city = TEAMS[h][1:]
                travel = haversine_km(prev[1], city) if prev else 0.0
                b2b = rest == 1

                # availability
                star_out = False
                if star_out_left[t] > 0:
                    star_out = True
                    star_out_left[t] -= 1
                elif rng.random() < 0.035:           # new injury
                    star_out = True
                    star_out_left[t] = int(rng.geometric(0.2))
                elif b2b and rng.random() < 0.22:    # load management
                    star_out = True
                injured = []
                if star_out:
                    injured.append((star[t][0], 1, 0.30, "OUT"))
                for k in range(rng.poisson(0.7)):
                    injured.append((f"{t} Role {k + 1}", 0,
                                    round(float(rng.uniform(0.04, 0.16)), 3),
                                    "OUT" if rng.random() < 0.75 else "QUESTIONABLE"))
                # questionable players play half the time
                impact = 0.0
                for name, is_star, share, status in injured:
                    plays = status == "QUESTIONABLE" and rng.random() < 0.5
                    if not plays:
                        impact += star[t][1] if is_star else share * 6
                    inj_rows.append(dict(
                        game_id=gid, team=t, player=name, is_star=is_star,
                        minutes_share=share, status=status,
                        reported_at=(tip - timedelta(hours=3)).isoformat()))
                fatigue = (2.2 if b2b else 0) + 0.45 * travel / 1000 - (0.4 if rest >= 3 else 0)
                ctx[t] = dict(off=off[t] - 0.7 * impact - 0.5 * fatigue,
                              dfn=dfn[t] - 0.3 * impact - 0.5 * fatigue,
                              pace=pace[t] - (0.6 if b2b else 0))
                last_game[t] = (gd, city)

            exp_poss = (ctx[h]["pace"] + ctx[a]["pace"]) / 2
            exp_h = LEAGUE_RTG + ctx[h]["off"] - ctx[a]["dfn"] + HCA
            exp_a = LEAGUE_RTG + ctx[a]["off"] - ctx[h]["dfn"] - HCA
            exp_margin = (exp_h - exp_a) * exp_poss / 100
            exp_total = (exp_h + exp_a) * exp_poss / 100

            # sharp market with a little noise
            mkt_margin = exp_margin + rng.normal(0, 1.3)
            spread = -round(mkt_margin * 2) / 2
            total = round((exp_total + rng.normal(0, 2.4)) * 2) / 2
            p_home = 0.5 * (1 + math.erf(mkt_margin / (12.5 * math.sqrt(2))))

            poss = exp_poss + rng.normal(0, 3.2)
            shared = rng.normal(0, 3.0)            # game-level shooting env.
            rtg = {h: exp_h + shared + rng.normal(0, 9.5),
                   a: exp_a + shared + rng.normal(0, 9.5)}
            pts = {t: int(round(poss * rtg[t] / 100)) for t in (h, a)}
            if pts[h] == pts[a]:                    # overtime
                ot_poss = poss * 5 / 48
                for t in (h, a):
                    pts[t] += int(round(ot_poss * (rtg[t] + rng.normal(0, 25)) / 100))
                if pts[h] == pts[a]:
                    pts[h if rng.random() < 0.5 else a] += 2

            boxes = {t: _box(rng, pts[t], poss) for t in (h, a)}
            # defensive rebounds = opponent misses not offensively rebounded
            drb = {t: boxes[o]["miss"] - boxes[o]["orb"] for t, o in ((h, a), (a, h))}
            for t, o, is_home in ((h, a, 1), (a, h, 0)):
                b = boxes[t]
                box_rows.append(dict(game_id=gid, team=t, opp=o, is_home=is_home,
                                     pts=pts[t], poss=round(poss, 1),
                                     **{k: b[k] for k in ("fga", "fgm", "fg3m", "fta", "ftm", "tov", "orb")},
                                     drb=drb[t], opp_drb=drb[o]))

            game_rows.append(dict(
                game_id=gid, date=gd.isoformat(), season=season,
                tip_time=tip.isoformat(),
                prediction_timestamp=(tip - timedelta(hours=2)).isoformat(),
                home_team=h, away_team=a, home_score=pts[h], away_score=pts[a],
                closing_home_spread=spread, closing_total=total,
                home_moneyline=_american(p_home), away_moneyline=_american(1 - p_home)))

    pd.DataFrame(game_rows).to_csv(feed_dir / "games.csv", index=False)
    pd.DataFrame(box_rows).to_csv(feed_dir / "team_box.csv", index=False)
    pd.DataFrame(inj_rows).to_csv(feed_dir / "injuries.csv", index=False)
    pd.DataFrame([dict(team=t, name=v[0], lat=v[1], lon=v[2]) for t, v in TEAMS.items()]
                 ).to_csv(feed_dir / "teams.csv", index=False)
    return dict(games=len(game_rows), seasons=last_season - first_season + 1)


def _box(rng, pts: int, poss: float) -> dict:
    """Box score consistent with points and possessions."""
    tov = int(round(poss * min(max(rng.normal(0.135, 0.022), 0.07), 0.22)))
    ftr = min(max(rng.normal(0.25, 0.05), 0.1), 0.45)
    orbp = min(max(rng.normal(0.27, 0.05), 0.12), 0.45)
    fga = int(round((poss - tov) / (1 + 0.44 * ftr - orbp * 0.53)))
    fta = int(round(fga * ftr))
    ftm = int(round(fta * rng.normal(0.78, 0.05)))
    fg3a = fga * rng.normal(0.40, 0.04)
    fg3m = int(round(fg3a * min(max(rng.normal(0.36, 0.05), 0.15), 0.6)))
    fgm = int(round((pts - ftm - fg3m) / 2))
    fgm = max(fgm, fg3m)
    ftm = pts - 2 * fgm - fg3m
    if ftm < 0:                       # rebalance via threes if needed
        fg3m = max(0, fg3m + ftm)
        ftm = pts - 2 * fgm - fg3m
        if ftm < 0:
            fgm += ftm // 2
            ftm = pts - 2 * fgm - fg3m
    fta = max(fta, ftm)
    fga = max(fga, fgm + 1)
    miss = fga - fgm
    orb = int(round(miss * orbp))
    return dict(fga=fga, fgm=fgm, fg3m=fg3m, fta=fta, ftm=ftm, tov=tov, orb=orb, miss=miss)
