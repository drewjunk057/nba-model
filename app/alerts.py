"""Injury alerts: the one edge our testing supports.

Backtests (2024-25, 2025-26) showed a stats model that knows who's missing
beats the OPENING line 54-56% of the time with 2+ points of disagreement,
and the line then moves our way 61-69% of the time; without the injury
information that edge disappears. So the edge is news the line hasn't
absorbed yet. An alert fires when, for a game that hasn't tipped:

  1. a player is newly ruled OUT since we started watching the game,
     worth >= MIN_IMPACT points of team strength (player-impact model);
  2. the line has moved less than that impact since we first saw the news
     (UNPRICED_MIN points or more still unpriced); and
  3. the margin model (stats + who's missing, no market inputs) disagrees
     with the current line by >= MIN_EDGE points toward the other team.

Each alert is recorded once, at the line when it fired, and graded and
compared with the closing line separately from the regular picks.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .db import read_df, session

log = logging.getLogger("alerts")
MIN_IMPACT = 1.0      # points of team strength lost
UNPRICED_MIN = 1.0    # points of that impact the line hasn't absorbed
MIN_EDGE = 2.0        # model vs current line, points


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan(gf: pd.DataFrame, bundle, now: str | None = None) -> list[dict]:
    now = now or _now()
    if "margin" not in bundle.boosters:
        return []
    tips = pd.to_datetime(gf.tip_time, utc=True, errors="coerce")
    nowts = pd.Timestamp(now)
    up = gf[gf.home_score.isna() & (tips > nowts) & (tips < nowts + pd.Timedelta(hours=30))]
    if up.empty:
        return []
    ids = list(up.game_id)
    qs = ",".join("?" * len(ids))
    with session() as con:
        con.executemany("INSERT OR IGNORE INTO game_scans VALUES (?, ?)", [(g, now) for g in ids])
    first = read_df(f"SELECT * FROM game_scans WHERE game_id IN ({qs})", ids).set_index("game_id").first_scan_at

    inj = read_df(f"SELECT * FROM injuries WHERE game_id IN ({qs})", ids)
    if not inj.empty:
        from .player_value import player_rows
        pr = player_rows(inj, gf)
        spread = up.set_index("game_id").closing_home_spread
        with session() as con:
            con.executemany(
                "INSERT OR IGNORE INTO injury_seen VALUES (?,?,?,?,?,?,?)",
                [(r.game_id, r.team, r.player, str(r.status).upper(), float(r.impact) if pd.notna(r.impact) else 0.0,
                  now, float(spread.get(r.game_id)) if pd.notna(spread.get(r.game_id)) else None)
                 for r in pr.itertuples()])
    seen = read_df(f"SELECT * FROM injury_seen WHERE game_id IN ({qs})", ids)
    existing = set(read_df(f"SELECT game_id FROM alerts WHERE game_id IN ({qs})", ids).game_id)
    preds = {r.game_id: p for r, p in zip(up.itertuples(), bundle.predict(up))}

    new_alerts = []
    for g in up.itertuples():
        if g.game_id in existing or pd.isna(g.closing_home_spread) or getattr(g, "lines_estimated", 0):
            continue
        news = seen[(seen.game_id == g.game_id) & (seen.status == "OUT") & (seen.first_seen_at > first[g.game_id])
                    & (seen.impact >= MIN_IMPACT)]
        if news.empty:
            continue
        edge = preds[g.game_id].get("line_edge_points")
        if edge is None:
            continue
        for team, n in news.groupby("team"):
            hurt_home = team == g.home_team
            impact = float(n.impact.sum())
            then = n.home_spread_then.dropna()
            then = float(then.iloc[0]) if len(then) else float(g.closing_home_spread)
            moved = (g.closing_home_spread - then) if hurt_home else (then - g.closing_home_spread)
            unpriced = impact - max(moved, 0.0)
            toward_other = -edge if hurt_home else edge
            if unpriced >= UNPRICED_MIN and toward_other >= MIN_EDGE:
                side_home = not hurt_home
                side_team = g.home_team if side_home else g.away_team
                line = g.closing_home_spread if side_home else -g.closing_home_spread
                players = ", ".join(n.player)
                reason = (f"{players} ruled out ({team}), worth about {impact:.1f} pts. Line has moved "
                          f"{max(moved, 0):.1f} since; model sees {toward_other:.1f} pts of value on {side_team} {line:+g}.")
                row = dict(game_id=g.game_id, created_at=now, model_version=bundle.version, side_team=side_team,
                           side_is_home=int(side_home), home_spread=float(g.closing_home_spread),
                           edge_points=round(toward_other, 1), unpriced_points=round(unpriced, 1),
                           players=players, reason=reason)
                with session() as con:
                    con.execute("INSERT OR IGNORE INTO alerts (game_id, created_at, model_version, side_team, side_is_home,"
                                " home_spread, edge_points, unpriced_points, players, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                tuple(row.values()))
                new_alerts.append(dict(row, away=g.away_team, home=g.home_team, tip_time=g.tip_time))
                break
    for a in new_alerts:
        notify(f"Injury edge: {a['side_team']} {a['home_spread'] if a['side_is_home'] else -a['home_spread']:+g}",
               a["reason"])
    return new_alerts


def settle() -> int:
    with session() as con:
        rows = con.execute("""SELECT a.game_id, a.side_is_home, a.home_spread, g.home_score, g.away_score,
                                     g.closing_home_spread FROM alerts a JOIN games g USING(game_id)
                              WHERE a.settled_at IS NULL AND g.home_score IS NOT NULL""").fetchall()
        for gid, side_home, line, hs, as_, close in rows:
            m = hs - as_ + line
            res = "p" if m == 0 else ("w" if (m > 0) == bool(side_home) else "l")
            con.execute("UPDATE alerts SET result=?, closing_home_spread=?, settled_at=? WHERE game_id=?",
                        (res, close, _now(), gid))
    return len(rows)


def notify(title: str, body: str):
    """Windows toast notification; silently skipped elsewhere."""
    if os.name != "nt":
        return
    esc = lambda s: s.replace("'", "''")
    ps = ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; "
          "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
          "$x = $t.GetElementsByTagName('text'); "
          f"$x.Item(0).AppendChild($t.CreateTextNode('{esc(title)}')) > $null; "
          f"$x.Item(1).AppendChild($t.CreateTextNode('{esc(body)}')) > $null; "
          "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
          "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe')"
          ".Show([Windows.UI.Notifications.ToastNotification]::new($t))")
    try:
        subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                       capture_output=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as e:
        log.warning("notification failed: %s", e)


def summary() -> dict:
    a = read_df("SELECT a.*, g.date, g.home_team, g.away_team, g.home_score, g.away_score "
                "FROM alerts a JOIN games g USING(game_id) ORDER BY a.created_at DESC")
    rec = dict(w=0, l=0, p=0, units=0.0, clv_n=0, moved_our_way=0, clv_points=0.0)
    out = []
    for r in a.itertuples():
        line = r.home_spread if r.side_is_home else -r.home_spread
        item = dict(game_id=r.game_id, date=r.date, home=r.home_team, away=r.away_team, created_at=r.created_at,
                    pick=f"{r.side_team} {line:+g}", edge_points=r.edge_points, reason=r.reason, result=r.result)
        if r.result in ("w", "l", "p"):
            rec[r.result] += 1
            rec["units"] += (100 / 110) if r.result == "w" else (-1.0 if r.result == "l" else 0.0)
            item["final"] = f"{r.away_team} {int(r.away_score)}, {r.home_team} {int(r.home_score)}"
        if pd.notna(r.closing_home_spread):
            clv = (r.home_spread - r.closing_home_spread) if r.side_is_home else (r.closing_home_spread - r.home_spread)
            item["clv_points"] = round(float(clv), 1)
            rec["clv_n"] += 1
            rec["clv_points"] += float(clv)
            rec["moved_our_way"] += int(clv > 0)
        out.append(item)
    n = rec["w"] + rec["l"]
    rec["pct"] = round(rec["w"] / n, 3) if n else None
    rec["units"] = round(rec["units"], 2)
    rec["avg_clv_points"] = round(rec["clv_points"] / rec["clv_n"], 2) if rec["clv_n"] else None
    rec["moved_our_way_pct"] = round(rec["moved_our_way"] / rec["clv_n"], 3) if rec["clv_n"] else None
    return dict(record=rec, alerts=out[:100])


def active(game_ids: list[str]) -> dict:
    if not game_ids:
        return {}
    qs = ",".join("?" * len(game_ids))
    a = read_df(f"SELECT * FROM alerts WHERE game_id IN ({qs})", game_ids)
    return {r.game_id: dict(side_team=r.side_team, side_is_home=int(r.side_is_home), home_spread=r.home_spread,
                            edge_points=r.edge_points, reason=r.reason, created_at=r.created_at, result=r.result)
            for r in a.itertuples()}
