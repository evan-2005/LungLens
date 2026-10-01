"""Tests for computing descriptors live from a region and lung masks."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from report_eval.descriptors import NO_REGION, from_region  # noqa: E402

H = W = 100


def _lungs():
    """Anatomical right lung on the IMAGE left, left lung on the image right."""
    right = np.zeros((H, W), bool)
    left = np.zeros((H, W), bool)
    right[20:80, 10:45] = True
    left[20:80, 55:90] = True
    return right, left


def _box(y0, y1, x0, x1):
    r = np.zeros((H, W), np.uint8)
    r[y0:y1, x0:x1] = 1
    return r


class FromRegionTest(unittest.TestCase):
    def test_region_in_image_left_is_patient_right_lung(self):
        d = from_region(_box(25, 35, 15, 30), *_lungs())
        self.assertEqual(d.side, "right lung")
        self.assertEqual(d.zone, "upper")
        self.assertEqual(d.extent, "focal")
        self.assertEqual(d.focality, "single")
        self.assertAlmostEqual(d.in_lung, 1.0)
        self.assertTrue(d.gate_pass)

    def test_region_between_lungs_is_outside_and_fails_gate(self):
        d = from_region(_box(40, 60, 46, 54), *_lungs())
        self.assertEqual(d.side, "outside both")
        self.assertFalse(d.gate_pass)

    def test_two_blobs_spanning_both_lungs(self):
        region = _box(60, 75, 15, 40) | _box(60, 75, 60, 85)
        d = from_region(region, *_lungs())
        self.assertEqual(d.focality, "multifocal")
        self.assertTrue(d.spans_both)
        self.assertEqual(d.zone, "lower")

    def test_empty_region(self):
        self.assertEqual(from_region(np.zeros((H, W), np.uint8), *_lungs()), NO_REGION)
        self.assertEqual(from_region(None, *_lungs()), NO_REGION)

    def test_without_lung_masks_no_location_is_possible(self):
        d = from_region(_box(25, 35, 15, 30), None, None)
        self.assertTrue(d.has_region)
        self.assertIsNone(d.side)
        self.assertIsNone(d.in_lung)
        self.assertFalse(d.gate_pass)


if __name__ == "__main__":
    unittest.main()
