#!/usr/bin/env python3
"""Read-only CPU preflight. No torch import and no GPU probes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    lock = json.loads((root / "models.lock.json").read_text())
    state_path = root / "work/download-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    errors = []
    verified_bytes = 0
    for model in lock["models"]:
        for spec in model["files"]:
            key = f"{model['id']}/{spec['name']}"
            path = root / "work/models" / key
            receipt = state.get("verified", {}).get(key)
            if not path.exists() or not receipt:
                errors.append(f"Pending: {key}")
                continue
            st = path.stat()
            if receipt["identity"] != [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] or receipt["digest"] != (spec["sha256"] or spec["git_blob"]):
                errors.append(f"Changed since verification: {key}")
                continue
            verified_bytes += st.st_size
    if not (root / "results/runtime-image.json").exists():
        errors.append("Runtime image build is pending")
    if not (root / "results/staging-validation.json").exists():
        errors.append("CPU staging validation is pending")
    return {"ready_for_gpu_validation": not errors, "gpu_used": False,
        "download_status": state.get("status", "not_started"), "verified_bytes": verified_bytes,
        "total_bytes": sum(f["bytes"] for m in lock["models"] for f in m["files"]), "pending": errors}


if __name__ == "__main__":
    report = check()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ready_for_gpu_validation"] else 1)
