#!/usr/bin/env python3
"""Fetch only the pinned engine source; no imports/builds/GPU work."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / "runtime.lock.json").read_text())["exllamav3"]
destination = ROOT / "work/exllamav3"
if not destination.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", lock["repo"], str(destination)], check=True)
elif subprocess.check_output(["git", "-C", str(destination), "status", "--porcelain"], text=True).strip():
    raise SystemExit("Engine checkout contains changes; refusing to replace them")
subprocess.run(["git", "-C", str(destination), "fetch", "origin", lock["revision"]], check=True)
subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock["revision"]], check=True)
assert subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip() == lock["revision"]
print("Pinned source ready; GPU untouched")
