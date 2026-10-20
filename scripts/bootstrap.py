"""Run by start.bat before the server starts.

1. First run on a new PC: copy the starter models and downloaded game history
   from seed/ into models/ and data/raw/, so setup takes about a minute
   instead of a 20-minute download and retrain.
2. Refresh games, lines, injuries and rosters if the last update is more than
   3 hours old (or there is no database yet).

Exit code 0 means the dashboard can start.
"""
import shutil
import subprocess
import sys
import time

from app.config import DATA_DIR, DB_PATH, MODEL_DIR, ROOT

SEED = ROOT / "seed"
RAW = DATA_DIR / "raw"
SUMMARY = DATA_DIR / "real" / "fetch_summary.json"
FRESH_HOURS = 3


def copy_seed():
    if not (MODEL_DIR / "current.json").exists() and (SEED / "models" / "current.json").exists():
        for p in (SEED / "models").iterdir():
            dest = MODEL_DIR / p.name
            if p.is_dir():
                shutil.copytree(p, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dest)
        print("Installed the starter models.")
    if (SEED / "raw").exists():
        RAW.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in (SEED / "raw").glob("*.json.gz"):
            if not (RAW / p.name).exists():
                shutil.copy2(p, RAW / p.name)
                n += 1
        if n:
            print(f"Installed {n} seasons of saved game data.")


def run_update(args):
    return subprocess.call([sys.executable, "-m", "scripts.update_data", *args], cwd=ROOT)


def main():
    copy_seed()
    have_models = (MODEL_DIR / "current.json").exists()
    have_db = DB_PATH.exists() and SUMMARY.exists()

    if not have_models:
        print("No trained model found: downloading five seasons and training (10-25 minutes)...")
        return 0 if run_update(["--go-live"]) == 0 else 1

    if not have_db:
        print("First run: downloading the latest games, lines and injuries (about a minute)...")
        if run_update([]) != 0:
            print("Couldn't download the NBA data. Check your internet connection and run start.bat again.")
            return 1
        return 0

    age_h = (time.time() - SUMMARY.stat().st_mtime) / 3600
    if age_h < FRESH_HOURS:
        print(f"Data was updated {age_h * 60:.0f} minutes ago; skipping the refresh.")
        return 0
    print("Updating games, lines and injuries...")
    if run_update([]) != 0:
        print("The update failed, so the app will open with the data from the last update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
