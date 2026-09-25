"""Unit tests for the E1 evaluation helpers (no dataset or checkpoint needed)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

from lung_attention.eval_lungattn import gt_lung_mask, region_stats, summarise_heat  # noqa: E402

SIZE = 224


def _left_lung():
    lung = np.zeros((SIZE, SIZE), bool)
    lung[:, : SIZE // 2] = True
    return lung


class RegionStatsTest(unittest.TestCase):
    def test_hot_blob_inside_lung_scores_fully_inside(self):
        heat = np.zeros((SIZE, SIZE), np.float32)
        heat[80:140, 20:80] = 1.0
        s = region_stats(heat, _left_lung())
        self.assertTrue(s["has_region"])
        self.assertAlmostEqual(s["region_in_lung"], 1.0, places=3)
        self.assertAlmostEqual(s["mass_in_lung"], 1.0, places=3)

    def test_hot_blob_outside_lung_scores_zero(self):
        heat = np.zeros((SIZE, SIZE), np.float32)
        heat[80:140, 150:210] = 1.0
        s = region_stats(heat, _left_lung())
        self.assertAlmostEqual(s["region_in_lung"], 0.0, places=3)

    def test_cold_map_has_no_region(self):
        heat = np.full((SIZE, SIZE), 0.1, np.float32)  # below the display floor
        s = region_stats(heat, _left_lung())
        self.assertFalse(s["has_region"])
        self.assertEqual(s["region_area"], 0.0)


class SummariseHeatTest(unittest.TestCase):
    def test_counts_majority_outside_and_empty_regions(self):
        rows = [
            {"gt": {"has_region": True, "region_in_lung": 0.9, "mass_in_lung": 0.9, "region_area": 0.1}},
            {"gt": {"has_region": True, "region_in_lung": 0.2, "mass_in_lung": 0.3, "region_area": 0.2}},
            {"gt": {"has_region": False, "region_in_lung": float("nan"),
                    "mass_in_lung": float("nan"), "region_area": 0.0}},
            {"pred_mask": {}},  # no gt key: excluded from the gt summary
        ]
        s = summarise_heat(rows, "gt")
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["with_region"], 2)
        self.assertAlmostEqual(s["no_region_frac"], 1 / 3)
        self.assertAlmostEqual(s["region_majority_outside_frac"], 0.5)

    def test_empty_input(self):
        self.assertEqual(summarise_heat([], "gt"), {"n": 0})


class GtLungMaskTest(unittest.TestCase):
    def test_non_radiography_path_has_no_ground_truth(self):
        self.assertIsNone(gt_lung_mask(os.path.join("somewhere", "Shenzhen", "CHNCXR_0001_0.png")))

    def test_missing_mask_file_returns_none(self):
        path = os.path.join("x", "COVID-19_Radiography_Dataset", "COVID", "images", "nope.png")
        self.assertIsNone(gt_lung_mask(path))


if __name__ == "__main__":
    unittest.main()
