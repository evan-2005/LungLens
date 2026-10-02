"""Unit tests for descriptors, templates, claim parsing and scoring."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from report_eval.claims import Claim, claim_from_descriptors, parse_claim  # noqa: E402
from report_eval.descriptors import NO_REGION, Descriptors, from_table_row  # noqa: E402
from report_eval.films import derangement  # noqa: E402
from report_eval.score import score_output, summarise  # noqa: E402
from report_eval.templates import Prediction, s0_template, s1_descriptor  # noqa: E402

LEFT_MID = Descriptors(has_region=True, side="left lung", zone="mid", extent="patchy",
                       focality="multifocal", in_lung=0.8)
OUTSIDE = Descriptors(has_region=True, side="outside both", zone="mid", extent="extensive",
                      focality="single", in_lung=0.3)


def _row(**kw):
    base = {"side": "left lung", "cy": "0.5", "mask_area_frac": "0.1",
            "compactness": "0.7", "in_lung_frac": "0.8"}
    base.update(kw)
    return base


class DescriptorsTest(unittest.TestCase):
    def test_table_row_maps_to_zone_extent_focality(self):
        d = from_table_row(_row(cy="0.2", mask_area_frac="0.03", compactness="1.0"))
        self.assertEqual((d.zone, d.extent, d.focality), ("upper", "focal", "single"))
        d = from_table_row(_row(cy="0.8", mask_area_frac="0.4", compactness="0.5"))
        self.assertEqual((d.zone, d.extent, d.focality), ("lower", "extensive", "multifocal"))

    def test_no_region(self):
        d = from_table_row(_row(side="none", cy="nan", mask_area_frac="0", compactness="nan",
                                in_lung_frac="nan"))
        self.assertEqual(d, NO_REGION)
        self.assertFalse(d.gate_pass)

    def test_gate(self):
        self.assertTrue(LEFT_MID.gate_pass)
        self.assertFalse(OUTSIDE.gate_pass)


class TemplatesTest(unittest.TestCase):
    PNEU = Prediction(top_cls="Pneumonia", top_prob=0.9,
                      probs={"Normal": 0.05, "Pneumonia": 0.9, "Tuberculosis": 0.03, "Covid-19": 0.02},
                      mask_source="segmentation")

    def test_s0_ignores_descriptors(self):
        self.assertEqual(s0_template(self.PNEU), s0_template(self.PNEU))
        self.assertIn("The outline marks the most influential area", s0_template(self.PNEU))

    def test_s0_normal_visualises_next_abnormal_class(self):
        p = Prediction("Normal", 0.8, {"Normal": 0.8, "Pneumonia": 0.05, "Tuberculosis": 0.1,
                                       "Covid-19": 0.05}, "gradcam")
        self.assertIn("assessed for Tuberculosis", s0_template(p))

    def test_s0_uncertain(self):
        p = Prediction("Covid-19", 0.5, {"Normal": 0.3, "Pneumonia": 0.1, "Tuberculosis": 0.1,
                                         "Covid-19": 0.5}, "segmentation")
        self.assertIn("**Prediction:** Uncertain", s0_template(p))

    def test_s1_states_location_only_when_gate_passes(self):
        self.assertIn("left lung, mid zone, patchy, multifocal", s1_descriptor(self.PNEU, LEFT_MID))
        self.assertIn("no location is reported", s1_descriptor(self.PNEU, OUTSIDE))
        self.assertIn("no location is reported", s1_descriptor(self.PNEU, NO_REGION))
        self.assertIn("no location is reported", s1_descriptor(self.PNEU, None))


class ClaimParserTest(unittest.TestCase):
    def test_parses_side_zone_extent(self):
        c = parse_claim("Patchy opacity in the right lower zone.")
        self.assertEqual(c, Claim(side="right lung", zone="lower", extent="patchy"))

    def test_synonyms(self):
        self.assertEqual(parse_claim("Focal apical lesion, left.").zone, "upper")
        self.assertEqual(parse_claim("Diffuse bilateral perihilar change").side, "bilateral")
        self.assertEqual(parse_claim("Diffuse bilateral perihilar change").zone, "mid")

    def test_no_location(self):
        self.assertIsNone(parse_claim("Findings consistent with pneumonia."))
        self.assertIsNone(parse_claim("No location is reported."))

    def test_zone_words_outside_anatomy_are_ignored(self):
        self.assertIsNone(parse_claim("The confidence is lower than usual."))
        self.assertEqual(parse_claim("Opacity in the right mid-zone.").zone, "mid")
        self.assertEqual(parse_claim("Left upper lobe consolidation.").zone, "upper")

    def test_locations_in_negated_sentences_are_not_claims(self):
        # Real Qwen2.5-VL outputs from the pilot.
        self.assertIsNone(parse_claim("The findings supporting the prediction of Normal are "
                                      "throughout both lungs, with no visible abnormalities."))
        self.assertIsNone(parse_claim("The findings are located throughout the lungs, with no "
                                      "specific area showing a higher likelihood of abnormality."))
        c = parse_claim("No abnormality in the right lung. Patchy opacity in the left lower zone.")
        self.assertEqual((c.side, c.zone), ("left lung", "lower"))

    def test_zone_with_several_words_before_the_place(self):
        c = parse_claim("located in the lower left and right lung fields.")
        self.assertEqual((c.side, c.zone), ("bilateral", "lower"))
        self.assertEqual(parse_claim("located in the upper regions of both lungs.").zone, "upper")

    def test_ambiguous_sides_are_not_guessed(self):
        self.assertEqual(parse_claim("left upper and right lower opacities").side, "bilateral")

    def test_claim_from_descriptors(self):
        self.assertEqual(claim_from_descriptors(LEFT_MID), Claim("left lung", "mid", "patchy"))
        self.assertIsNone(claim_from_descriptors(OUTSIDE))


class ResponseKindTest(unittest.TestCase):
    def test_categories_from_pilot_outputs(self):
        from report_eval.claims import response_kind
        self.assertEqual(response_kind("Located in the lower left lung."), "specific")
        self.assertEqual(response_kind("The findings are located throughout the lungs."),
                         "non_specific")
        self.assertEqual(response_kind("The findings that support this prediction are located "
                                       "in the entire lungs."), "non_specific")
        self.assertEqual(response_kind("The image does not show any visible abnormalities that "
                                       "would support a diagnosis of Tuberculosis."), "no_evidence")
        self.assertEqual(response_kind("Findings consistent with pneumonia."), "other")


class ScoringTest(unittest.TestCase):
    def test_correct_partial_and_wrong_claims(self):
        right = score_output(Claim("left lung", "mid", "patchy"), LEFT_MID)
        partial = score_output(Claim("left lung", None, None), LEFT_MID)
        wrong = score_output(Claim("right lung", "mid", "patchy"), LEFT_MID)
        self.assertTrue(right["claim_correct"])
        self.assertTrue(partial["claim_correct"])
        self.assertFalse(wrong["claim_correct"])
        self.assertTrue(wrong["zone_correct"])

    def test_phantom_claim_when_no_region(self):
        s = score_output(Claim("left lung", "mid", None), NO_REGION)
        self.assertTrue(s["phantom"])
        self.assertFalse(s["claim_correct"])

    def test_summary_rates(self):
        rows = [score_output(Claim("left lung", "mid", "patchy"), LEFT_MID),
                score_output(None, LEFT_MID),
                score_output(Claim("right lung", "mid", None), LEFT_MID),
                score_output(None, NO_REGION)]
        s = summarise(rows)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["claims"], 2)
        self.assertAlmostEqual(s["claim_rate"], 0.5)
        self.assertAlmostEqual(s["grounding_precision"], 0.5)


class DerangementTest(unittest.TestCase):
    def test_no_fixed_points_and_reproducible(self):
        keys = [f"k{i}" for i in range(50)]
        a = derangement(keys, seed=42)
        self.assertEqual(a, derangement(keys, seed=42))
        self.assertEqual(sorted(a.values()), sorted(keys))
        self.assertTrue(all(k != v for k, v in a.items()))


if __name__ == "__main__":
    unittest.main()
