#!/usr/bin/env python3
"""Persistent download followed only by CPU checks."""
from download import main as download
from finalize import main as finalize
from download import ROOT, atomic_json
import json

try:
    download()
except Exception as exc:
    path = ROOT / "work/download-state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(status="failed", error=str(exc))
    atomic_json(path, state)
    raise
finalize()
