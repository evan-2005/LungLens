"""Tests for the app's grounded summary (improvements #1 to #4)."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from grounded_report import (  # noqa: E402
    BADGE_GROUNDED,
    BADGE_LLM_REJECTED,
    BADGE_LLM_UNAVAILABLE,
    BADGE_NO_LOCATION,
    DISCLAIMER,
    build_report,
)
from report_eval.descriptors import NO_REGION, Descriptors  # noqa: E402
from report_eval.llm_backends import BackendError, LLMResult  # noqa: E402
from report_eval.templates import Prediction  # noqa: E402

COVID = Prediction("Covid-19", 0.97, {"Normal": 0.01, "Pneumonia": 0.01, "Tuberculosis": 0.01,
                                      "Covid-19": 0.97}, "segmentation")
NORMAL = Prediction("Normal", 0.95, {"Normal": 0.95, "Pneumonia": 0.03, "Tuberculosis": 0.01,
                                     "Covid-19": 0.01}, "segmentation")
UNCERTAIN = Prediction("Pneumonia", 0.5, {"Normal": 0.4, "Pneumonia": 0.5, "Tuberculosis": 0.05,
                                          "Covid-19": 0.05}, "gradcam")
IN_LUNG = Descriptors(True, "left lung", "mid", "patchy", "multifocal", 0.8, False)
OUT_LUNG = Descriptors(True, "outside both", "mid", "extensive", "single", 0.2, False)


class Backend:
    name, model = "fake", "fake-1"

    def __init__(self, reply=None, error=False):
        self.reply, self.error = reply, error

    def generate(self, system, user, images=None, schema=None, max_tokens=1024):
        if self.error:
            raise BackendError("offline")
        return LLMResult(text=json.dumps(self.reply), model=self.model, latency_s=0.0)


class GroundedModeTest(unittest.TestCase):
    def test_confident_finding_in_lung_states_location_with_badge(self):
        r = build_report(COVID, IN_LUNG, region_drawn=True)
        self.assertIn("left lung, mid zone", r.text)
        self.assertEqual(r.badge, BADGE_GROUNDED)
        self.assertIn(DISCLAIMER, r.text)

    def test_no_hardcoded_bilateral(self):
        for d in (IN_LUNG, OUT_LUNG, None):
            self.assertNotIn("bilateral", build_report(COVID, d, region_drawn=d is not None).text)

    def test_no_phantom_outline_claim_when_nothing_was_outlined(self):
        r = build_report(COVID, NO_REGION, region_drawn=False)
        self.assertNotIn("outline marks", r.text)
        self.assertIn("No heatmap region was strong enough to outline", r.text)
        self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_region_outside_lungs_is_reported_but_not_located(self):
        r = build_report(COVID, OUT_LUNG, region_drawn=True)
        self.assertIn("outside the lung fields", r.text)
        self.assertNotIn("left lung", r.text)
        self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_normal_and_uncertain_never_state_a_location(self):
        for pred in (NORMAL, UNCERTAIN):
            r = build_report(pred, IN_LUNG, region_drawn=True)
            self.assertIsNone(r.claim)
            self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_region_centred_between_lungs_names_no_side(self):
        between = Descriptors(True, "outside both", "mid", "extensive", "single", 0.6, False)
        r = build_report(COVID, between, region_drawn=True)
        self.assertNotIn("outside both", r.text)
        self.assertIn("centred between the lungs", r.text)
        self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_region_spanning_both_lungs_says_both_lungs(self):
        both = Descriptors(True, "outside both", "mid", "extensive", "multifocal", 0.7, True)
        r = build_report(COVID, both, region_drawn=True)
        self.assertIn("in both lungs, mid zone", r.text)
        self.assertEqual(r.claim.side, "bilateral")
        self.assertEqual(r.badge, BADGE_GROUNDED)

    def test_manually_selected_other_class_states_no_location(self):
        r = build_report(COVID, None, region_drawn=False, visualised_cls="Tuberculosis")
        self.assertIn("Tuberculosis (selected manually)", r.text)
        self.assertNotIn("strong enough to outline", r.text)
        self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_original_mode_is_the_deployed_template(self):
        r = build_report(COVID, IN_LUNG, region_drawn=True, mode="original")
        self.assertIn("bilateral", r.text)  # kept verbatim for comparison


class LLMModeTest(unittest.TestCase):
    GOOD = {"state_location": True, "side": "left lung", "zone": "mid", "extent": "patchy",
            "sentence": "The heatmap highlights a patchy region in the left lung mid zone."}

    def test_accepted_llm_sentence_is_used(self):
        r = build_report(COVID, IN_LUNG, region_drawn=True, mode="llm", backend=Backend(self.GOOD))
        self.assertIn("The heatmap highlights a patchy region", r.text)
        self.assertEqual(r.badge, BADGE_GROUNDED)

    def test_rejected_llm_sentence_falls_back_with_badge(self):
        bad = dict(self.GOOD, sentence="Patchy opacity in the left lung mid zone.")
        r = build_report(COVID, IN_LUNG, region_drawn=True, mode="llm", backend=Backend(bad))
        self.assertNotIn("opacity", r.text)
        self.assertEqual(r.badge, BADGE_LLM_REJECTED)

    def test_llm_outage_falls_back_with_badge(self):
        r = build_report(COVID, IN_LUNG, region_drawn=True, mode="llm",
                         backend=Backend(error=True))
        self.assertIn("left lung, mid zone", r.text)
        self.assertEqual(r.badge, BADGE_LLM_UNAVAILABLE)

    def test_llm_is_not_called_when_there_is_nothing_to_locate(self):
        r = build_report(COVID, OUT_LUNG, region_drawn=True, mode="llm",
                         backend=Backend(error=True))
        self.assertIn("outside the lung fields", r.text)
        self.assertEqual(r.badge, BADGE_NO_LOCATION)

    def test_llm_mode_without_backend_falls_back(self):
        r = build_report(COVID, IN_LUNG, region_drawn=True, mode="llm", backend=None)
        self.assertEqual(r.badge, BADGE_LLM_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
