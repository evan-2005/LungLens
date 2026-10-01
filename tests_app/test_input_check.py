"""Tests for rejecting non-chest-X-ray uploads (improvement #5)."""
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from input_check import check_upload, colourfulness  # noqa: E402


def _grey(value=128, size=64):
    return Image.fromarray(np.full((size, size), value, np.uint8)).convert("RGB")


def _colour(size=64):
    a = np.zeros((size, size, 3), np.uint8)
    a[: size // 2, :, 0] = 255
    a[size // 2:, :, 2] = 255
    return Image.fromarray(a)


def _lung(fraction, size=100):
    m = np.zeros((size, size), bool)
    m.flat[: int(fraction * m.size)] = True
    return m


class ColourfulnessTest(unittest.TestCase):
    def test_grey_is_zero_and_colour_is_high(self):
        self.assertAlmostEqual(colourfulness(_grey()), 0.0, places=3)
        self.assertGreater(colourfulness(_colour()), 45)


class CheckUploadTest(unittest.TestCase):
    def test_plausible_xray_passes_without_warning(self):
        r = check_upload(_grey(), lung_mask=_lung(0.33))
        self.assertTrue(r.ok)
        self.assertEqual(r.warning, "")

    def test_no_lungs_at_all_is_rejected(self):
        r = check_upload(_colour(), lung_mask=_lung(0.01))
        self.assertFalse(r.ok)
        self.assertIn("lung", r.reason)

    def test_small_lung_area_is_analysed_with_warning(self):
        r = check_upload(_grey(), lung_mask=_lung(0.078))  # measured: an infant film
        self.assertTrue(r.ok)
        self.assertIn("unusual", r.warning)

    def test_implausibly_large_lung_area_warns(self):
        r = check_upload(_grey(), lung_mask=_lung(0.9))
        self.assertTrue(r.ok)
        self.assertTrue(r.warning)

    def test_tinted_xray_is_analysed_with_warning(self):
        r = check_upload(_colour(), lung_mask=_lung(0.30))  # measured: a cyan-tinted scan
        self.assertTrue(r.ok)
        self.assertIn("tinted", r.warning)

    def test_without_lung_segmenter_nothing_is_rejected(self):
        r = check_upload(_grey(), lung_mask=None)
        self.assertTrue(r.ok)
        self.assertFalse(r.lung_checked)


if __name__ == "__main__":
    unittest.main()
