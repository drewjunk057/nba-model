"""One-time setup: generate the synthetic feed, build the DB, train v1.

    python -m scripts.setup                 # as of today
    python -m scripts.setup --as-of 2026-11-15
"""
import argparse
from datetime import date

from app.data.synthetic import generate
from app.db import reset_db
from app.pipeline import FEED_DIR, run_nightly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print("Generating synthetic feed ...", generate(FEED_DIR, seed=a.seed))
    reset_db()
    s = run_nightly(a.as_of, force_retrain=True)
    print({k: v for k, v in s.items() if k != "drift"})


if __name__ == "__main__":
    main()
