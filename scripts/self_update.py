"""Keep a production copy up to date with the published app on GitHub.

Runs from the scheduled task (and start.bat). Each time it:
  1. asks GitHub for the latest commit of drewjunk057/nba-model (one small request);
  2. if it's new, downloads that version and test-imports it in a temporary
     folder -- a broken version is never installed;
  3. backs up the current code to data/code_backup/ and copies the new code
     over it. data/ and models/ (the database, pick history, record, trained
     models) are never touched; a newer published model is installed by the
     usual seed check in bootstrap;
  4. reinstalls packages only if requirements.txt changed.

The running dashboard reloads itself when app/ changes.

A folder containing a file named `.dev` is a development copy and is never
updated (so it can't overwrite work in progress).

    python -m scripts.self_update            # check and update
    python -m scripts.self_update --force    # reinstall the latest even if current
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "drewjunk057/nba-model"
VERSION_FILE = ROOT / "data" / "app_version.json"
CODE_ITEMS = ["app", "scripts", "tests", "seed", "requirements.txt", "README.md",
              "start.bat", "nightly.bat", "go_live.bat", ".gitignore"]
UA = {"User-Agent": "nba-predictor-self-update", "Accept": "application/vnd.github+json"}


def say(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  [self-update] {msg}"
    print(line, flush=True)
    try:
        with open(ROOT / "data" / "update.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def is_dev_copy() -> bool:
    return (ROOT / ".dev").exists()


def installed() -> dict:
    try:
        return json.loads(VERSION_FILE.read_text())
    except Exception:
        return {}


def latest_sha() -> str:
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/commits/main", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["sha"]


def download(sha: str) -> bytes:
    req = urllib.request.Request(f"https://github.com/{REPO}/archive/{sha}.zip", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def extract(blob: bytes, dest: Path) -> Path:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    tops = [p for p in dest.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise RuntimeError("unexpected archive layout")
    return tops[0]


def smoke_test(src: Path) -> None:
    """Import the new code in its own folder; raise if it's broken."""
    code = "import app.api, app.pipeline, app.models, app.features, scripts.bootstrap, scripts.update_data"
    exe = Path(sys.executable)
    py = exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists() else exe
    r = subprocess.run([str(py), "-c", code], cwd=src, capture_output=True, text=True, timeout=300,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError("new version failed its import test:\n" + (r.stderr or r.stdout)[-1500:])


def install(src: Path, sha: str) -> None:
    backup = ROOT / "data" / "code_backup"
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True)
    old_req = (ROOT / "requirements.txt").read_text() if (ROOT / "requirements.txt").exists() else ""
    for item in CODE_ITEMS:
        cur, new = ROOT / item, src / item
        if cur.exists():
            (shutil.copytree if cur.is_dir() else shutil.copy2)(cur, backup / item)
        if new.exists():
            if item == "start.bat" and os.environ.get("NBA_IN_START_BAT"):
                shutil.copy2(new, ROOT / "start.bat.new")    # swapped in by the next background run
                continue
            if new.is_dir():
                shutil.copytree(new, cur, dirs_exist_ok=True)
            else:
                shutil.copy2(new, cur)
    if (ROOT / "requirements.txt").read_text() != old_req:
        say("requirements changed; installing packages")
        exe = Path(sys.executable)
        py = exe.with_name("python.exe") if exe.with_name("python.exe").exists() else exe
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=ROOT,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps({"sha": sha, "installed_at": datetime.now().isoformat(timespec="seconds")}))


def swap_pending_launcher() -> None:
    new = ROOT / "start.bat.new"
    if new.exists() and not os.environ.get("NBA_IN_START_BAT"):
        try:
            os.replace(new, ROOT / "start.bat")
            say("installed the updated start.bat")
        except OSError:
            pass                              # start.bat in use; try next run


def run(force: bool = False) -> str:
    if is_dev_copy():
        return "dev copy: skipped"
    swap_pending_launcher()
    try:
        sha = latest_sha()
    except Exception as e:
        say(f"couldn't reach GitHub ({e!r}); keeping the current version")
        return "offline"
    if not force and installed().get("sha") == sha:
        return "up to date"
    say(f"new version {sha[:7]} found; downloading")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = extract(download(sha), Path(tmp))
            smoke_test(src)
            install(src, sha)
    except Exception as e:
        say(f"update skipped, current version kept: {e}")
        return "failed"
    try:                                     # pick up a newer published model right away
        from scripts.bootstrap import copy_seed
        copy_seed()
    except Exception as e:
        say(f"model check after update failed: {e!r}")
    say(f"updated to {sha[:7]}")
    return "updated"


if __name__ == "__main__":
    print(run(force="--force" in sys.argv))
