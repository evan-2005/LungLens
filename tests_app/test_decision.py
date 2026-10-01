"""Tests for calibrated, pneumonia-sensitive decisions (improvement #10)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from decision import DecisionConfig, decide, fit_temperature  # noqa: E402

NORMAL, PNEU, TB, COVID = range(4)


class DecideTest(unittest.TestCase):
    def test_identity_config_is_plain_softmax_argmax(self):
        logits = np.array([2.0, 1.0, 0.0, -1.0])
        probs, idx = decide(logits, DecisionConfig())
        self.assertEqual(idx, NORMAL)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_temperature_softens_but_keeps_argmax(self):
        logits = np.array([4.0, 1.0, 0.0, 0.0])
        sharp, _ = decide(logits, DecisionConfig(temperature=1.0))
        soft, idx = decide(logits, DecisionConfig(temperature=2.0))
        self.assertLess(soft[NORMAL], sharp[NORMAL])
        self.assertEqual(idx, NORMAL)

    def test_pneumonia_override_only_replaces_normal(self):
        cfg = DecisionConfig(pneumonia_threshold=0.30)
        probs, idx = decide(np.log(np.array([0.60, 0.35, 0.03, 0.02])), cfg)
        self.assertEqual(idx, PNEU)
        _, idx = decide(np.log(np.array([0.10, 0.35, 0.53, 0.02])), cfg)
        self.assertEqual(idx, TB)  # never overrides another disease
        _, idx = decide(np.log(np.array([0.75, 0.20, 0.03, 0.02])), cfg)
        self.assertEqual(idx, NORMAL)  # below the threshold


class FitTemperatureTest(unittest.TestCase):
    def test_recovers_temperature_of_overconfident_logits(self):
        rng = np.random.default_rng(0)
        true_logits = rng.normal(size=(4000, 4)) * 1.5
        labels = np.array([rng.choice(4, p=np.exp(l) / np.exp(l).sum()) for l in true_logits])
        t = fit_temperature(true_logits * 3.0, labels)  # 3x overconfident
        self.assertAlmostEqual(t, 3.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
