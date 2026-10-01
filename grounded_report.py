"""
The app's summary text, grounded in the heatmap it shows (improvements #1 to #4).

Replaces the class-keyed template that never read the heatmap. Rules:
  - A location is stated only for a confident, abnormal prediction whose region
    was actually outlined AND lies mostly inside the lung fields (the gate).
  - Otherwise the text says plainly why no location is reported.
  - No class sentence asserts a location (the old Covid-19 line always said
    "bilateral", whatever the heatmap showed).
  - The heatmap is described as where the model looked, never as a finding.
  - Every summary carries a badge and a fixed research-use disclaimer.

Modes: "grounded" (default, template), "llm" (constrained local LLM via
report_eval.s2_llm, falling back to the template), "original" (the old deployed
text, kept only for side-by-side comparison).
"""
from dataclasses import dataclass, replace
from typing import Optional

from report_eval.claims import Claim, claim_from_descriptors
from report_eval.llm_backends import BackendError
from report_eval.templates import s0_template

MODES = ("grounded", "llm", "original")
MODE_LABELS = {"Grounded (recommended)": "grounded", "Local LLM wording": "llm",
               "Original template (for comparison)": "original"}

CLASS_LINES = {
    "Normal": "The classifier did not find a pattern it associates with pneumonia, "
              "tuberculosis or Covid-19.",
    "Pneumonia": "The classifier's most likely class is pneumonia.",
    "Tuberculosis": "The classifier's most likely class is tuberculosis.",
    "Covid-19": "The classifier's most likely class is Covid-19.",
}
DISCLAIMER = ("_Research tool, not a diagnosis. The heatmap shows where the model looked, "
              "not a confirmed finding; confirm with a clinician and the appropriate test._")
BADGE_GROUNDED = "Grounded: the stated location matches the outlined heatmap region"
BADGE_NO_LOCATION = "No location claimed"
BADGE_LLM_REJECTED = "Unsupported LLM wording removed; template sentence used"
BADGE_LLM_UNAVAILABLE = "Local LLM unavailable; template sentence used"


@dataclass(frozen=True)
class Report:
    text: str
    badge: str
    claim: Optional[Claim]


def _header(p):
    if p.is_uncertain:
        return (f"**Prediction:** Uncertain\n\nThe top class is {p.top_cls} at "
                f"{p.top_prob * 100:.1f}%, below the reporting threshold. Review the class "
                f"confidence values; a repeat or higher quality image may help.")
    return f"**Prediction:** {p.top_cls} ({p.top_prob * 100:.1f}% confidence)\n\n{CLASS_LINES[p.top_cls]}"


def _location_line(p, d, region_drawn, visualised_cls=None):
    """(sentence, descriptors the sentence may describe or None)."""
    if visualised_cls and visualised_cls != p.top_cls:
        return (f"Heatmap: areas the model weighed for {visualised_cls} (selected manually). "
                f"No location is reported."), None
    if p.top_cls == "Normal":
        # The old template added "none reached an abnormal level", a claim about
        # the heatmap that was never measured; it is dropped.
        return (f"Heatmap: areas the model assessed for {p.viz_cls} (the next most likely "
                f"class). No location is reported."), None
    if p.is_uncertain:
        return ("Heatmap: areas the model weighed while uncertain. No location is reported."), None
    if not region_drawn or d is None or not d.has_region:
        return "No heatmap region was strong enough to outline; no location is reported.", None
    if d.in_lung is None:
        return ("A region is outlined, but the lung fields could not be found, so no location "
                "is reported."), None
    if not d.gate_pass:
        return ("The outlined region lies mostly outside the lung fields, so no location is "
                "reported. This can mean the model relied on cues outside the lungs."), None
    # Stricter than the paper's S1 gate: the region must also be centred on a
    # lung (or clearly span both), so the text never names "outside both".
    if d.spans_both:
        side_text, claim_side = "both lungs", "bilateral"
    elif d.side in ("right lung", "left lung"):
        side_text, claim_side = d.side, d.side
    else:
        return ("The outlined region is centred between the lungs, so no side is reported."), None
    return (f"The outlined heatmap region is in {'the ' if claim_side != 'bilateral' else ''}"
            f"{side_text}, {d.zone} zone ({d.extent}, {d.focality})."), replace(d, side=claim_side)


def _provenance(p):
    if p.mask_source == "segmentation":
        return "_Overlay source: disease-segmentation U-Net._"
    return "_Overlay source: Grad-CAM._" if p.mask_source else ""


def _assemble(p, sentence, badge, claim):
    parts = [_header(p), sentence, f"**Summary check:** {badge}", _provenance(p), DISCLAIMER]
    return Report(text="\n\n".join(x for x in parts if x), badge=badge, claim=claim)


def build_report(prediction, descriptors, region_drawn, mode="grounded", backend=None,
                 visualised_cls=None):
    """`visualised_cls` is set when the user picked the overlay class by hand."""
    if mode not in MODES:
        raise ValueError(f"unknown summary mode {mode!r}")
    if mode == "original":
        return Report(text=s0_template(prediction), badge="Original template", claim=None)

    sentence, claimable = _location_line(prediction, descriptors, region_drawn, visualised_cls)
    claim = claim_from_descriptors(claimable)
    badge = BADGE_GROUNDED if claim else BADGE_NO_LOCATION
    # With nothing to locate, the template already explains why; an LLM call
    # would only add latency and a chance of inventing a location.
    if mode == "grounded" or claimable is None:
        return _assemble(prediction, sentence, badge, claim)

    if backend is None:
        return _assemble(prediction, sentence, BADGE_LLM_UNAVAILABLE, claim)
    from report_eval.llm_generators import s2_llm
    try:
        out = s2_llm(prediction, claimable, backend)
    except BackendError:
        return _assemble(prediction, sentence, BADGE_LLM_UNAVAILABLE, claim)
    if out.rejected:
        return _assemble(prediction, sentence, BADGE_LLM_REJECTED, claim)
    return _assemble(prediction, out.sentence, badge, claim)
