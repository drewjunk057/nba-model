"""Pull real NBA data from ESPN and run the pipeline on it.

    python -m scripts.update_data --go-live     first time: backfill 2021-22 onward,
                                               wipe the demo database, retrain on real games
    python -m scripts.update_data               nightly: new results, lines, injuries, rosters
    python -m scripts.update_data --no-pipeline just refresh the CSVs in data/real/

Everything is logged to data/update.log as well as the screen.
"""
import argparse
import json
import logging
import shutil
import sys
import time
from datetime import date, datetime

from app.config import DATA_DIR, MODEL_DIR

LOG = DATA_DIR / "update.log"


def say(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def archive_demo_models():
    """Move models trained on synthetic data out of the way."""
    cur = MODEL_DIR / "current.json"
    versions = [p for p in MODEL_DIR.glob("v*") if p.is_dir()]
    if not versions:
        return
    dest = MODEL_DIR / "demo_synthetic"
    dest.mkdir(exist_ok=True)
    for p in versions:
        meta = p / "meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("data_source") == "real":
            continue
        shutil.move(str(p), str(dest / p.name))
    if cur.exists():
        cur.unlink()
    say(f"Archived {len(versions)} demo model version(s) to models/demo_synthetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-live", action="store_true", help="first-time backfill, reset DB, retrain")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(LOG, encoding="utf-8")])
    from app.pipeline import REAL_FEED, run_nightly, update_real_data

    t0 = time.time()
    say("=== update started" + (" (go-live)" if a.go_live else ""))
    try:
        res = update_real_data(today=a.as_of, log=say)
    except Exception as e:
        say(f"FAILED to download data: {e!r}")
        raise
    fs = res["feed"]
    say("Feed: " + json.dumps(fs, default=str))
    if fs.get("completed", 0) < 2000:
        say("Too few completed games for a reliable model; stopping before the pipeline.")
        sys.exit(2)
    if a.no_pipeline:
        return

    if a.go_live:
        from app.db import reset_db
        try:
            reset_db()
        except PermissionError:
            say("The database is in use. Close the NBA Predictor window, then run this again.")
            sys.exit(3)
        archive_demo_models()
        say("Database reset; training on real games (1-3 minutes)...")
    try:
        s = run_nightly(a.as_of, force_retrain=a.go_live or a.retrain, feed_dir=REAL_FEED)
    except Exception:
        import traceback
        say("FAILED in pipeline:\n" + traceback.format_exc())
        raise
    say("Pipeline: " + json.dumps({k: v for k, v in s.items() if k != "drift"}, default=str))
    if s["retrained"]:
        meta = json.loads((MODEL_DIR / s["model_version"] / "meta.json").read_text())
        ev = meta["evaluation"]["models"]
        w, sp, t = ev["win"], ev["spread"], ev["total"]
        say(f"Holdout {meta['evaluation']['test_season']}: win Brier {w['brier']:.4f} "
            f"(market {w['market_baseline']['brier']:.4f} on {w['market_baseline']['n']} games), "
            f"ATS acc {sp['accuracy']:.3f} Brier {sp['brier']:.4f}, "
            f"total MAE {t['mae']:.2f} (market {t['market_mae']:.2f})")
    say(f"=== done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
