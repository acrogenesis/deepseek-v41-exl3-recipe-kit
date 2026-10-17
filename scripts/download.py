#!/usr/bin/env python3
"""Stage pinned checkpoint bytes on disk. Never imports a GPU runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "2")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def identity(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def verify(path, spec):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Size mismatch: {path}")
    h = hashlib.sha256() if spec["sha256"] else hashlib.sha1()
    if not spec["sha256"]:
        h.update(f"blob {spec['bytes']}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    expected = spec["sha256"] or spec["git_blob"]
    if h.hexdigest() != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return {"identity": identity(path), "digest": expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    lock = json.loads((ROOT / "models.lock.json").read_text())
    state_path = ROOT / "work/download-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"verified": {}}
    tasks = [(m, f) for m in lock["models"] for f in m["files"]
             if not args.metadata_only or not f["name"].endswith(".safetensors")]
    tasks.sort(key=lambda task: (task[1]["name"].endswith(".safetensors"), task[0]["id"] == "engram", task[1]["name"]))
    remaining = sum(f["bytes"] for m, f in tasks
                    if not (ROOT / "work/models" / m["id"] / f["name"]).exists())
    if shutil.disk_usage(ROOT).free < remaining + lock["reserve_disk_bytes"]:
        raise RuntimeError("Not enough free disk for remaining files plus 50 GiB reserve")
    state.update(status="downloading", gpu_used=False, started_at=time.time())
    atomic_json(state_path, state)
    for model, spec in tasks:
        key = f"{model['id']}/{spec['name']}"
        destination = ROOT / "work/models" / model["id"]
        path = destination / spec["name"]
        old = state["verified"].get(key)
        if old and path.exists() and old["identity"] == identity(path) and old["digest"] == (spec["sha256"] or spec["git_blob"]):
            continue
        if shutil.disk_usage(ROOT).free < lock["reserve_disk_bytes"]:
            raise RuntimeError("Disk reserve reached; stopped before downloading another file")
        state["current_file"] = key
        atomic_json(state_path, state)
        print(f"Downloading {key} ({spec['bytes'] / 1e9:.3f} GB)", flush=True)
        for attempt in range(6):
            try:
                hf_hub_download(model["repo"], spec["name"], revision=model["revision"], local_dir=destination)
                state["verified"][key] = verify(path, spec)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(10 * (attempt + 1), 50))
        atomic_json(state_path, state)
        print(f"Verified {key}", flush=True)
    state.update(status="metadata_ready" if args.metadata_only else "complete", completed_at=time.time(), current_file=None)
    atomic_json(state_path, state)
    print(state["status"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        path = ROOT / "work/download-state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status="failed", error=str(exc))
        atomic_json(path, state)
        raise
