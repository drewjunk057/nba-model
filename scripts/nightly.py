"""Nightly job: ingest new results, rebuild features, settle and log
predictions, retrain when due, and record drift.

    python -m scripts.nightly                       # as of today
    python -m scripts.nightly --as-of 2026-10-25 --retrain
    python -m scripts.nightly --feed /path/to/real/csvs

Schedule it with cron, e.g.  15 4 * * *  cd /path/nba_predictor && python -m scripts.nightly
"""
import argparse
import json
from datetime import date
from pathlib import Path

from app.pipeline import FEED_DIR, run_nightly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--feed", type=Path, default=FEED_DIR)
    a = ap.parse_args()
    print(json.dumps(run_nightly(a.as_of, force_retrain=a.retrain, feed_dir=a.feed), indent=2))


if __name__ == "__main__":
    main()
