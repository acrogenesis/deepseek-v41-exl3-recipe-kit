"""Compare direct-I/O cache bytes with independent reads of verified real shards."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.indexed_store import inventory, IndexedExpertStore

records, _ = inventory(ROOT, cached_headers=True)
keys = sys.argv[1:] or ["0:0", "0:1"]
selected = {key: records[key] for key in keys}
capacity = max(r["bytes"] for r in selected.values())
store = IndexedExpertStore(ROOT / "work/models/exl3", selected, capacity, reserve_bytes=0)
state = json.loads((ROOT / "work/download-state.json").read_text())
digests = {}
try:
    for key in keys + keys:
        with store.read(key) as data:
            for name, spec in selected[key]["tensors"].items():
                path = ROOT / "work/models/exl3" / spec["source_file"]
                st = path.stat()
                receipt = state["verified"]["exl3/" + spec["source_file"]]
                assert receipt["identity"] == [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
                fd = os.open(path, os.O_RDONLY)
                try:
                    expected = os.pread(fd, spec["bytes"], spec["source_offset"])
                finally:
                    os.close(fd)
                actual = data[spec["offset"]:spec["offset"] + spec["bytes"]]
                assert actual == expected, (key, name)
                digests[f"{key}/{name}"] = hashlib.sha256(actual).hexdigest()
                actual.release()
    print(json.dumps({"gpu_used": False, "experts": keys, "all_tensor_bytes_equal": True,
        "cache_capacity_bytes": capacity, "stats": store.stats, "tensor_sha256": digests}, indent=2))
finally:
    store.close()
