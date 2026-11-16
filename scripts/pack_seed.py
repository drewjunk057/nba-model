"""Run by "Publish to GitHub.bat": copy the current model and saved game data
into seed/ so a new or updating copy starts from them.

Copies over what's there instead of deleting folders first, so a file that
OneDrive or another program has open can't make publishing fail.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED, MODELS, RAW = ROOT / "seed", ROOT / "models", ROOT / "data" / "raw"


def main() -> int:
    cur = MODELS / "current.json"
    if not cur.exists():
        print("No trained model yet. Run start.bat once first.")
        return 1
    version = json.loads(cur.read_text())["model_version"]

    (SEED / "raw").mkdir(parents=True, exist_ok=True)
    for f in RAW.glob("*.json.gz"):
        shutil.copy2(f, SEED / "raw" / f.name)

    (SEED / "models").mkdir(parents=True, exist_ok=True)
    shutil.copytree(MODELS / version, SEED / "models" / version, dirs_exist_ok=True)
    shutil.copy2(cur, SEED / "models" / "current.json")

    # remove older model versions from seed/ (skip any file that's locked)
    for old in (SEED / "models").iterdir():
        if old.is_dir() and old.name != version:
            shutil.rmtree(old, ignore_errors=True)
            if old.exists():
                print(f"  note: couldn't remove old {old.name} from seed (in use); it's harmless")
    print("  model", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
