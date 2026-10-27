"""Entry point for the Windows scheduled task (runs hidden under pythonw).

Downloads new results, lines, injuries and rosters, grades finished picks,
logs picks for today's games and retrains weekly. Output goes to
data/update.log (and data/scheduled.log for anything unexpected).
"""
import runpy
import sys

from app.config import DATA_DIR

out = open(DATA_DIR / "scheduled.log", "a", encoding="utf-8")
sys.stdout = sys.stderr = out          # pythonw has no console
import os
if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".dev")):
    print("development copy: the scheduled task does nothing here")
    sys.exit(0)
try:                                    # production: install a newer published version first
    from scripts.self_update import run as self_update
    self_update()
except Exception as e:
    print("self-update failed:", repr(e))
try:                                    # pick up schedule changes without a restart
    from scripts.bootstrap import ensure_schedule
    ensure_schedule()
except Exception as e:
    print("schedule check failed:", repr(e))
sys.argv = ["scripts.update_data"]
try:
    runpy.run_module("scripts.update_data", run_name="__main__")
except SystemExit:
    pass
