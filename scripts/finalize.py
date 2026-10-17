#!/usr/bin/env python3
"""Finish staging with CPU validation only; never start the model."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.preflight import check
    from runtime.indexed_store import inventory, MARKER, read_header
    state = check()
    # A concurrent compiler may still be running; weights can be checked independently.
    pending = [p for p in state["pending"] if p not in ("Runtime image build is pending", "CPU staging validation is pending")]
    if pending:
        raise RuntimeError("Pinned downloads are not yet verified: " + "; ".join(pending[:3]))
    records, report = inventory(ROOT)
    markers = 0
    fds = {}
    try:
        for record in records.values():
            for name, spec in record["tensors"].items():
                if not name.endswith(".mul1"):
                    continue
                fn = spec["source_file"]
                if fn not in fds:
                    fds[fn] = os.open(ROOT / "work/models/exl3" / fn, os.O_RDONLY)
                value = os.pread(fds[fn], 4, spec["source_offset"])
                if len(value) != 4 or struct.unpack("<i", value)[0] != MARKER:
                    raise ValueError(f"Invalid codebook marker in {fn}")
                markers += 1
    finally:
        for fd in fds.values():
            os.close(fd)
    engram = {}
    for layer, shard in ((1, 47), (14, 48)):
        path = ROOT / "work/models/engram" / f"model-{shard:05d}-of-00048.safetensors"
        _, header = read_header(path)
        for suffix in ("weight", "scale"):
            name = f"layers.{layer}.engram.embed.{suffix}"
            engram[name] = header[name]
    report.update(payload_verified=True, verified_mul1_markers=markers, engram_tables=engram,
        completed_at=datetime.now(timezone.utc).isoformat(), gpu_used=False,
        gpu_validation="pending; nothing starts automatically")
    path = ROOT / "results/staging-validation.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)
    print(f"CPU staging validation complete: {len(records)} experts, {markers} MUL1 markers. GPU remains idle from this recipe.", flush=True)


if __name__ == "__main__":
    if "--wait-for-download" in sys.argv:
        while True:
            receipt = ROOT / "work/download-state.json"
            if receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete":
                break
            status = subprocess.check_output(["systemctl", "--user", "show", "deepseek41-exl3-download.service",
                "-p", "ActiveState", "-p", "Result"], text=True)
            fields = dict(line.split("=", 1) for line in status.splitlines() if "=" in line)
            if fields.get("ActiveState") in ("active", "activating", "deactivating"):
                time.sleep(10)
                continue
            if fields.get("Result") != "success":
                raise SystemExit("Download unit failed; leaving GPU untouched")
            break
    main()
