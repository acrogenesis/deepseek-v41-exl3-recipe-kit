import unittest
import random
from runtime.engram_cache import DecodeRowCache, RowCache


class RowCacheTests(unittest.TestCase):
    def test_order_duplicates_hits_and_handle_identity(self):
        cache = RowCache(8)
        a, b = object(), object()
        calls = []
        def read(handle, ids, width):
            calls.append((handle, ids, width))
            return b''.join(bytes([i+(0 if handle is a else 100)])*width for i in ids)
        self.assertEqual(cache.gather(read,a,[2,1,2],2), b'\2\2\1\1\2\2')
        self.assertEqual(cache.gather(read,a,[1,2],2), b'\1\1\2\2')
        self.assertEqual(cache.gather(read,b,[1],2), b'ee')
        self.assertEqual([c[1] for c in calls], [[2,1],[1]])
        self.assertEqual(cache.gather(read,a,[1],1), b'\1')
        self.assertEqual(cache.payload_bytes,7)

    def test_batch_larger_than_capacity_and_lru(self):
        cache=RowCache(2)
        handle=object()
        read=lambda h,ids,n: bytes(ids)
        self.assertEqual(cache.gather(read,handle,[1,2,3,1],1),bytes([1,2,3,1]))
        self.assertEqual(len(cache.rows),2)
        cache.gather(read,handle,[2],1)
        cache.gather(read,handle,[4],1)
        self.assertEqual([k[1] for k in cache.rows], [2,4])
        self.assertEqual(cache.payload_bytes,2)
        cache.clear()
        self.assertEqual(cache.payload_bytes,0)
        self.assertEqual(len(cache.rows),0)

    def test_failed_read_does_not_cache_partial_bytes(self):
        cache=RowCache(8)
        with self.assertRaises(IOError):
            cache.gather(lambda *a:b'x',object(),[0,1],2)
        self.assertEqual(len(cache.rows),0)
        self.assertEqual(cache.payload_bytes,0)


class DecodeRowCacheTests(unittest.TestCase):
    def test_generation_rows_survive_large_prefill_scan(self):
        cache = DecodeRowCache(6)
        calls = []
        def read(handle, ids, width):
            calls.append(ids)
            return bytes(ids)
        cache.set_phase('decode')
        self.assertEqual(cache.gather(read, 1, [10, 11, 12], 1), bytes([10, 11, 12]))
        cache.set_phase('prefill')
        scan = list(range(20, 120))
        self.assertEqual(cache.gather(read, 1, scan, 1), bytes(scan))
        cache.set_phase('decode')
        self.assertEqual(cache.gather(read, 1, [12, 10, 11, 12], 1), bytes([12, 10, 11, 12]))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(cache.rows), 6)

    def test_prefill_does_not_refresh_generation_recency(self):
        cache = DecodeRowCache(4)
        read = lambda h, ids, n: bytes(ids)
        cache.set_phase('decode')
        cache.gather(read, 1, [1, 2], 1)
        cache.set_phase('prefill')
        cache.gather(read, 1, [1], 1)
        cache.set_phase('decode')
        cache.gather(read, 1, [3], 1)
        cache.set_phase('prefill')
        cache.gather(read, 1, list(range(10, 20)), 1)
        self.assertNotIn((1, 1, 1), cache.rows)
        self.assertIn((1, 2, 1), cache.rows)
        self.assertIn((1, 3, 1), cache.rows)

    def test_mixed_phases_preserve_exact_bytes_and_hard_capacity(self):
        rng = random.Random(7219)
        def read(handle, ids, width):
            return b''.join(bytes([(row + handle) % 256]) * width for row in ids)
        for capacity in (1, 2, 7, 64):
            cache = DecodeRowCache(capacity)
            for step in range(500):
                cache.set_phase('prefill' if rng.randrange(3) == 0 else 'decode')
                handle, width = rng.randrange(2), rng.choice((1, 2, 256))
                ids = [rng.randrange(100) for _ in range(rng.randrange(60))]
                self.assertEqual(cache.gather(read, handle, ids, width), read(handle, ids, width))
                self.assertLessEqual(len(cache.rows), capacity)
                self.assertLessEqual(len(cache.protected), capacity // 2)
                self.assertEqual(set(cache.rows), set(cache.protected) | set(cache.probation))
                self.assertFalse(set(cache.protected) & set(cache.probation))
                self.assertEqual(cache.payload_bytes, sum(map(len, cache.rows.values())))
                if step % 117 == 0:
                    cache.clear()
                    self.assertFalse(cache.rows or cache.protected or cache.probation)
                    self.assertEqual(cache.payload_bytes, 0)

    def test_failed_read_keeps_existing_bytes(self):
        cache = DecodeRowCache(2)
        cache.set_phase('decode')
        cache.gather(lambda *a: b'AB', 1, [0], 2)
        before = dict(cache.rows)
        with self.assertRaises(IOError):
            cache.gather(lambda *a: b'x', 1, [0, 1, 2], 2)
        self.assertEqual(dict(cache.rows), before)
        self.assertEqual(cache.payload_bytes, 2)
        with self.assertRaises(ValueError):
            cache.set_phase('unknown')


if __name__ == '__main__':
    unittest.main()
