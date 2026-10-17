import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import unittest

from runtime.indexed_store import IndexedExpertStore, make_records, read_header, MARKER


def fixture(path):
    header, weight_map, values = {}, {}, {}
    payload = bytearray()
    for e in range(3):
        for p in ("w1", "w2", "w3"):
            for suffix, dtype, shape, width in (("trellis", "I16", [8, 8, 32 + 16 * (e % 2)], 2),
                    ("suh", "F16", [128], 2), ("svh", "F16", [128], 2), ("mul1", "I32", [1], 4)):
                name = f"layers.0.ffn.experts.{e}.{p}.{suffix}"
                size = math.prod(shape) * width
                value = struct.pack("<i", MARKER) if suffix == "mul1" else bytes([e + 1]) * size
                begin = len(payload)
                payload.extend(value)
                header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
                weight_map[name] = path.name
                values[name] = value
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    base, parsed = read_header(path)
    headers = {path.name: (base, parsed, path.stat().st_size)}
    records = make_records(weight_map, headers, layers=1, experts=3, hidden=128, intermediate=128)
    return weight_map, headers, records, values


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "model.safetensors"
        self.weights, self.headers, self.records, self.values = fixture(self.path)
        self.capacity = max(r["bytes"] for r in self.records.values())

    def tearDown(self):
        self.temp.cleanup()

    def exercise(self, direct):
        store = IndexedExpertStore(self.root, self.records, self.capacity, direct=direct, reserve_bytes=0)
        try:
            for key in ("0:0", "0:0", "0:1", "0:2", "0:0"):
                with store.read(key) as data:
                    for name, spec in self.records[key]["tensors"].items():
                        actual = bytes(data[spec["offset"]:spec["offset"] + spec["bytes"]])
                        self.assertEqual(actual, self.values[f"layers.0.ffn.experts.{key.split(':')[1]}.{name}"])
                    self.assertLessEqual(store.resident, self.capacity)
            self.assertEqual(store.stats["hits"], 1)
            self.assertEqual(store.stats["misses"], 4)
            self.assertEqual(store.stats["evictions"], 3)
        finally:
            store.close()

    def test_buffered_roundtrip_eviction(self):
        self.exercise(False)

    def test_direct_roundtrip_eviction(self):
        self.exercise(True)

    def test_missing_expert(self):
        weights = {k: v for k, v in self.weights.items() if ".experts.1." not in k}
        with self.assertRaisesRegex(ValueError, "ownership"):
            make_records(weights, self.headers, layers=1, experts=3, hidden=128, intermediate=128)

    def test_incomplete_projection(self):
        weights = dict(self.weights)
        weights.pop(next(iter(weights)))
        with self.assertRaisesRegex(ValueError, "twelve"):
            make_records(weights, self.headers, layers=1, experts=3, hidden=128, intermediate=128)

    def test_out_of_bounds(self):
        headers = copy.deepcopy(self.headers)
        next(iter(headers[self.path.name][1].values()))["data_offsets"][1] = 10**9
        with self.assertRaisesRegex(ValueError, "extent"):
            make_records(self.weights, headers, layers=1, experts=3, hidden=128, intermediate=128)

    def test_wrong_codebook(self):
        spec = self.records["0:0"]["tensors"]["w1.mul1"]
        with self.path.open("r+b") as f:
            f.seek(spec["source_offset"])
            f.write(b"\0" * 4)
        store = IndexedExpertStore(self.root, self.records, self.capacity, direct=True, reserve_bytes=0)
        try:
            with self.assertRaisesRegex(ValueError, "MUL1"):
                with store.read("0:0"):
                    pass
            self.assertEqual(store.resident, 0)
            self.assertFalse(store.cache)
        finally:
            store.close()

    def test_cancel(self):
        store = IndexedExpertStore(self.root, self.records, self.capacity, reserve_bytes=0)
        event = threading.Event()
        event.set()
        try:
            with self.assertRaises(InterruptedError):
                with store.read("0:0", cancel=event):
                    pass
            self.assertEqual(store.resident, 0)
        finally:
            store.close()

    def test_changed_file(self):
        store = IndexedExpertStore(self.root, self.records, self.capacity, direct=False, reserve_bytes=0)
        try:
            with store.read("0:0"):
                pass
            with self.path.open("ab") as f:
                f.write(b"x")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                with store.read("0:1"):
                    pass
        finally:
            store.close()

    def test_truncated_header(self):
        self.path.write_bytes(struct.pack("<Q", 100) + b"{}")
        with self.assertRaisesRegex(ValueError, "Truncated"):
            read_header(self.path)


if __name__ == "__main__":
    unittest.main()
