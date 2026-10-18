import unittest
from runtime.engram_cache import RowCache


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


if __name__ == '__main__':
    unittest.main()
