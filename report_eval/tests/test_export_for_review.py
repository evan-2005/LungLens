"""Tests for the parser-review export."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from report_eval.export_for_review import rows_for_review  # noqa: E402


def _rec(gen, key, error=None, claim=None):
    return {"generator": gen, "condition": "none", "key": key, "error": error,
            "text": "Patchy  opacity\nin the right lower zone.", "claim": claim}


class ExportTest(unittest.TestCase):
    def test_only_free_text_outputs_without_errors_are_sampled(self):
        recs = [_rec("s1", "a"), _rec("s3", "b"), _rec("s4", "c", error="x"),
                _rec("s4", "d", claim={"side": "right lung", "zone": "lower", "extent": None})]
        rows = rows_for_review(recs, n=10)
        self.assertEqual([r["key"] for r in rows], ["b", "d"])
        self.assertEqual(rows[1]["parsed_side"], "right lung")
        self.assertEqual(rows[1]["parsed_extent"], "")
        self.assertEqual(rows[0]["text"], "Patchy opacity in the right lower zone.")

    def test_sample_size_and_reproducibility(self):
        recs = [_rec("s3", f"k{i}") for i in range(100)]
        a, b = rows_for_review(recs, n=40, seed=1), rows_for_review(recs, n=40, seed=1)
        self.assertEqual(len(a), 40)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
