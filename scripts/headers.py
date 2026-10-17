#!/usr/bin/env python3
"""Read small safetensors headers, never tensor payloads or CUDA."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_range(url, start, length, total):
    # Distinct URLs also avoid proxy caches that do not vary on the Range header.
    url += f"?download=true&header_range={start}-{start + length - 1}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=90) as r:
        expected = f"bytes {start}-{start + length - 1}/{total}"
        if r.status != 206 or r.headers.get("Content-Range") != expected:
            raise ValueError(f"Server did not return the exact requested range: {r.status}, {r.headers.get('Content-Range')}")
        data = r.read(length + 1)
    if len(data) != length:
        raise ValueError("Incorrect range length")
    return data


def fetch(task):
    model, spec = task
    out = ROOT / "work/headers" / model["id"] / (spec["name"] + ".header")
    if out.exists():
        return str(out.relative_to(ROOT))
    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{spec['name']}"
    first = read_range(url, 0, 8, spec["bytes"])
    length = struct.unpack("<Q", first)[0]
    if not 2 <= length <= 64 * 2**20 or length + 8 > spec["bytes"]:
        raise ValueError("Invalid safetensors header size")
    data = first + read_range(url, 8, length, spec["bytes"])
    json.loads(data[8:])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return str(out.relative_to(ROOT))


def main():
    lock = json.loads((ROOT / "models.lock.json").read_text())
    tasks = [(m, f) for m in lock["models"] for f in m["files"] if f["name"].endswith(".safetensors")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for name in pool.map(fetch, tasks):
            print(name, flush=True)
    receipt = {"gpu_used": False, "headers": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "work/headers").rglob("*.header")}}
    (ROOT / "results/header-download.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
