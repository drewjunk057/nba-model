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
from pathlib import Path

from app.config import DATA_DIR, DB_PATH, MODEL_DIR, ROOT

SEED = ROOT / "seed"
RAW = DATA_DIR / "raw"
SUMMARY = DATA_DIR / "real" / "fetch_summary.json"
FRESH_HOURS = 3


def _trained(version_dir: Path) -> str:
    import json
    try:
        m = json.loads((version_dir / "meta.json").read_text())
        return m.get("trained_as_of") or m.get("trained_through") or ""
    except Exception:
        return ""


def _seed_is_newer() -> bool:
    """True when the published model was trained after the one in use here."""
    import json
    try:
        seed_v = json.loads((SEED / "models" / "current.json").read_text())["model_version"]
        cur_v = json.loads((MODEL_DIR / "current.json").read_text())["model_version"]
    except Exception:
        return False
    return seed_v != cur_v and _trained(SEED / "models" / seed_v) > _trained(MODEL_DIR / cur_v)


def copy_seed():
    newer = (MODEL_DIR / "current.json").exists() and _seed_is_newer()
    if newer:
        print("Installing the newer published model.")
    if (newer or not (MODEL_DIR / "current.json").exists()) and (SEED / "models" / "current.json").exists():
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


TASK_NAME = "NBA Predictor update"
SCHEDULE_VERSION = "schedule-v2"      # bump when the triggers change


def ensure_schedule():
    """Register the task if it's missing or older than SCHEDULE_VERSION."""
    status = DATA_DIR / "schedule_status.txt"
    try:
        current = status.read_text(encoding="utf-8")
    except OSError:
        current = ""
    if SCHEDULE_VERSION not in current or "registered" not in current:
        schedule_updates()


def schedule_updates():
    """Register (or refresh) a Windows scheduled task that updates the data at
    4:00 AM and every 30 minutes from 8 AM to 8 PM (injury alerts), hidden,
    catching up after sleep."""
    import os
    import subprocess as sp
    from datetime import datetime
    if os.name != "nt":
        return
    status = DATA_DIR / "schedule_status.txt"
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    exe = pyw if pyw.exists() else exe
    q = lambda v: "'" + str(v).replace("'", "''") + "'"
    ps = "; ".join([
        f"$a = New-ScheduledTaskAction -Execute {q(exe)} -Argument '-m scripts.scheduled_update' -WorkingDirectory {q(ROOT)}",
        "$t1 = New-ScheduledTaskTrigger -Daily -At 4:00AM",
        "$t2 = New-ScheduledTaskTrigger -Daily -At 8:00AM",
        # every 30 minutes from 8 AM to 8 PM: catches injury news before tip-off
        "$t2.Repetition = (New-ScheduledTaskTrigger -Once -At 8:00AM -RepetitionInterval "
        "(New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Hours 12)).Repetition",
        "$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries "
        "-DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 50) -MultipleInstances IgnoreNew",
        f"Register-ScheduledTask -TaskName {q(TASK_NAME)} -Action $a -Trigger $t1,$t2 -Settings $s "
        "-Description 'Updates NBA Predictor games, lines and injuries, grades picks and logs new ones.' "
        "-Force | Out-Null",
        f"(Get-ScheduledTask -TaskName {q(TASK_NAME)}).State",
    ])
    try:
        r = sp.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                   capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0
        status.write_text(f"{datetime.now():%Y-%m-%d %H:%M}  {'registered' if ok else 'FAILED'}  {SCHEDULE_VERSION}  "
                          f"exe={exe}\n{r.stdout}{r.stderr}", encoding="utf-8")
        if ok:
            print("Automatic updates are on: 4:00 AM, then every 30 minutes from 8 AM to 8 PM.")
        else:
            print("Couldn't set up automatic updates (details in data\\schedule_status.txt).")
    except Exception as e:  # never block the app over this
        status.write_text(f"{datetime.now():%Y-%m-%d %H:%M}  FAILED  {e!r}", encoding="utf-8")


def run_update(args):
    return subprocess.call([sys.executable, "-m", "scripts.update_data", *args], cwd=ROOT)


def main():
    copy_seed()
    schedule_updates()                 # always refresh on app start (folder may have moved)
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
