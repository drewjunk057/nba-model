"""Features for a game must not change once its result is known.

Builds the DB as of day D (games on D unplayed) and as of D+1 (games on D
final) and checks every model feature for D's games is identical.

    python -m pytest tests -q
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "data" / "feed"

SNAPSHOT = """
import sys
from datetime import date
from pathlib import Path
from app.data.ingest import ingest_feed
from app.db import reset_db
from app.features import build_all, FEATURE_SETS
reset_db()
ingest_feed(Path(sys.argv[1]), date.fromisoformat(sys.argv[2]))
gf = build_all(persist=False)
cols = sorted(set(sum(FEATURE_SETS.values(), [])))
gf[gf.date == sys.argv[3]][['game_id'] + cols].to_pickle(sys.argv[4])
"""


def _snapshot(as_of, day, out, tmp):
    env = dict(os.environ, NBA_DB_PATH=str(Path(tmp) / f"{as_of}.db"))
    subprocess.run([sys.executable, "-c", SNAPSHOT, str(FEED), as_of, day, out],
                   cwd=ROOT, env=env, check=True)
    return pd.read_pickle(out).set_index("game_id").sort_index()


def test_no_leakage_mid_season():
    if not FEED.exists():
        from app.data.synthetic import generate
        generate(FEED)
    day = "2026-01-15"
    with tempfile.TemporaryDirectory() as tmp:
        before = _snapshot(day, day, str(Path(tmp) / "a.pkl"), tmp)
        after = _snapshot("2026-01-16", day, str(Path(tmp) / "b.pkl"), tmp)
    assert len(before) > 0
    pd.testing.assert_frame_equal(before, after, check_exact=False, rtol=1e-9)
