"""Unit tests for the lung-constrained attention loss.

Run from the repository root:
    python -m unittest discover -s lung_attention/tests -t .
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

from lung_attention.attention_loss import (  # noqa: E402
    class_activation_map,
    forward_with_features,
    lung_masks_from_batch,
    outside_lung_fraction,
)

SIZE = 32


def _half_mask(batch=2):
    """Lung mask covering the left half of the frame."""
    m = torch.zeros(batch, SIZE, SIZE)
    m[:, :, : SIZE // 2] = 1.0
    return m


class OutsideLungFractionTest(unittest.TestCase):
    def test_zero_when_all_activation_inside_lung(self):
        cam = torch.zeros(2, SIZE, SIZE)
        cam[:, :, : SIZE // 4] = 1.0
        loss = outside_lung_fraction(cam, _half_mask())
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_one_when_all_activation_outside_lung(self):
        cam = torch.zeros(2, SIZE, SIZE)
        cam[:, :, -SIZE // 4:] = 1.0
        loss = outside_lung_fraction(cam, _half_mask())
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_half_when_activation_uniform(self):
        cam = torch.ones(2, SIZE, SIZE)
        loss = outside_lung_fraction(cam, _half_mask())
        self.assertAlmostEqual(loss.item(), 0.5, places=5)

    def test_upsamples_low_resolution_cam(self):
        cam = torch.zeros(1, 4, 4)
        cam[:, :, :2] = 1.0
        loss = outside_lung_fraction(cam, _half_mask(batch=1))
        self.assertLess(loss.item(), 0.1)

    def test_invalid_samples_are_ignored(self):
        cam = torch.zeros(2, SIZE, SIZE)
        cam[0, :, : SIZE // 4] = 1.0   # inside
        cam[1, :, -SIZE // 4:] = 1.0   # outside, but marked invalid
        valid = torch.tensor([True, False])
        loss = outside_lung_fraction(cam, _half_mask(), valid=valid)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_all_invalid_returns_zero_that_keeps_the_graph(self):
        cam = torch.ones(2, SIZE, SIZE, requires_grad=True)
        loss = outside_lung_fraction(cam, _half_mask(), valid=torch.tensor([False, False]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # must not raise
        self.assertIsNotNone(cam.grad)

    def test_zero_activation_sample_does_not_produce_nan(self):
        cam = torch.zeros(2, SIZE, SIZE)
        loss = outside_lung_fraction(cam, _half_mask())
        self.assertFalse(torch.isnan(loss))

    def test_gradient_pushes_activation_into_lung(self):
        cam = torch.ones(1, SIZE, SIZE, requires_grad=True)
        outside_lung_fraction(cam, _half_mask(batch=1)).backward()
        grad = cam.grad[0]
        # Increasing outside activation raises the loss; inside lowers it.
        self.assertGreater(grad[:, -1].mean().item(), 0.0)
        self.assertLess(grad[:, 0].mean().item(), 0.0)


class ClassActivationMapTest(unittest.TestCase):
    def test_selects_weights_of_requested_class_and_rectifies(self):
        feats = torch.ones(2, 3, 2, 2)
        weight = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        cam = class_activation_map(feats, weight, torch.tensor([0, 1]))
        self.assertEqual(cam.shape, (2, 2, 2))
        self.assertTrue(torch.allclose(cam[0], torch.full((2, 2), 3.0)))
        self.assertTrue(torch.allclose(cam[1], torch.zeros(2, 2)))


class ForwardWithFeaturesTest(unittest.TestCase):
    def test_logits_match_the_served_model_forward(self):
        import app
        torch.manual_seed(0)
        model = app.CNNModel(classCount=4, isTrained=False).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected = model(x)
            logits, feats = forward_with_features(model, x)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))
        self.assertEqual(feats.shape[1], 1024)
        self.assertGreaterEqual(feats.min().item(), 0.0)


class _ConstantLungNet(nn.Module):
    """Stand-in lung segmenter: positive logit on the left `frac` of the frame."""
    def __init__(self, frac):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        out = torch.full_like(x, -10.0)
        out[..., : int(x.shape[-1] * self.frac)] = 10.0
        return out


class LungMasksFromBatchTest(unittest.TestCase):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def test_plausible_area_is_valid_and_dilation_grows_the_mask(self):
        images = torch.zeros(2, 3, SIZE, SIZE)
        mask, valid = lung_masks_from_batch(
            _ConstantLungNet(0.4), images, self.MEAN, self.STD, dilate_px=2)
        self.assertEqual(mask.shape, (2, SIZE, SIZE))
        self.assertTrue(valid.all())
        self.assertGreater(mask.mean().item(), 0.4)

    def test_implausible_areas_are_invalid(self):
        images = torch.zeros(1, 3, SIZE, SIZE)
        _, tiny = lung_masks_from_batch(
            _ConstantLungNet(0.03), images, self.MEAN, self.STD, dilate_px=0)
        _, huge = lung_masks_from_batch(
            _ConstantLungNet(0.95), images, self.MEAN, self.STD, dilate_px=0)
        self.assertFalse(tiny.item())
        self.assertFalse(huge.item())


if __name__ == "__main__":
    unittest.main()
