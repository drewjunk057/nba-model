"""Move a pick record from one copy of the app to another.

export_record() writes seed/record.json (the prediction log plus any pick
corrections) from this copy's database. Published with the app, it's
imported automatically by every other copy on its next update:
import_record() adds any picks that copy doesn't already have (its own picks
always win) and merges pick corrections. Safe to run any number of times.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from .config import DATA_DIR, ROOT
from .db import read_df, session

SEED_RECORD = ROOT / "seed" / "record.json"
MARKER = DATA_DIR / "record_imported.txt"
OVERRIDES = DATA_DIR / "pick_overrides.json"
COLS = ["game_id", "model_version", "predicted_at", "home_win_prob", "home_covers_prob",
        "predicted_total_points", "over_prob", "closing_home_spread", "closing_total",
        "home_win", "home_covered", "total_points", "settled_at"]


def export_record() -> int:
    log = read_df(f"SELECT {', '.join(COLS)} FROM prediction_log ORDER BY predicted_at")
    try:
        ov = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    except Exception:
        ov = {}
    SEED_RECORD.parent.mkdir(parents=True, exist_ok=True)
    SEED_RECORD.write_text(json.dumps(dict(
        exported_at=datetime.now().isoformat(timespec="seconds"),
        picks=json.loads(log.to_json(orient="records")), overrides=ov), indent=1))
    return len(log)


def import_record() -> int:
    if not SEED_RECORD.exists():
        return 0
    raw = SEED_RECORD.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if MARKER.exists() and MARKER.read_text().strip() == digest:
        return 0
    data = json.loads(raw)
    rows = [tuple(p.get(c) for c in COLS) for p in data.get("picks", [])]
    with session() as con:
        before = con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
        con.executemany(f"INSERT OR IGNORE INTO prediction_log ({', '.join(COLS)}) "
                        f"VALUES ({', '.join('?' * len(COLS))})", rows)
        added = con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0] - before
    ov_in = data.get("overrides") or {}
    if ov_in:
        try:
            ov = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
        except Exception:
            ov = {}
        for k, v in ov_in.items():
            ov.setdefault(k, v)                 # this copy's own corrections win
        OVERRIDES.write_text(json.dumps(ov, indent=2))
    MARKER.write_text(digest)
    return added
