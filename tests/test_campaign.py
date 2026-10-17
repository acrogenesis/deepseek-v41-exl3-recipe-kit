import copy
import unittest
from scripts.campaign import compare


class ComparisonTests(unittest.TestCase):
    def data(self, gib, speed):
        return {"runs": [{"id": "test", "phase": "fresh", "token_ids": [1, 2], "passed": True}],
            "profile": {"gpu_expert_cache_bytes": gib * 2**30}, "fresh_median_tok_s": speed, "repeat_median_tok_s": speed}

    def test_larger_cache_requires_unchanged_output(self):
        a, b = self.data(16, 10), self.data(22, 12)
        self.assertEqual(compare(a, b)["recommended_cache_gib"], 22)
        b["runs"][0]["token_ids"] = [1, 3]
        self.assertEqual(compare(a, b)["recommended_cache_gib"], 16)

    def test_small_or_negative_change_does_not_change_recommendation(self):
        a = self.data(16, 10)
        for rate in (9, 10, 10.4):
            self.assertEqual(compare(a, self.data(22, rate))["recommended_cache_gib"], 16)
