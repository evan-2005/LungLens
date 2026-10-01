"""
Template generators.

S0 reproduces the deployed summary (app.py, predict_image) verbatim, including
its branch logic, so it can be scored without running the app. It never reads
the heatmap. S1 is the descriptor-gated template from the paper.
"""
from dataclasses import dataclass
from typing import Dict

CLASSES = ("Normal", "Pneumonia", "Tuberculosis", "Covid-19")
CONFIDENCE_THRESHOLD = 0.60     # app.py
OVERLAY_MIN_PROB = 0.15         # app.py

DESCRIPTIONS = {
    "Normal":       "No abnormal opacities detected in the lung fields.",
    "Pneumonia":    "Findings consistent with pneumonia. Warmer areas indicate possible consolidation.",
    "Tuberculosis": "Findings consistent with tuberculosis. Warmer areas indicate possible focal lesions or cavitation.",
    "Covid-19":     "Findings consistent with Covid-19. Warmer areas indicate possible bilateral ground-glass opacities.",
}
NO_REGION_TEXT = "No heatmap region met the reporting threshold; no location is reported."
OUTSIDE_LUNG_TEXT = ("The heatmap region lies mostly outside the lung fields; "
                     "no location is reported.")


@dataclass(frozen=True)
class Prediction:
    top_cls: str
    top_prob: float
    probs: Dict[str, float]
    mask_source: str                # "segmentation" | "gradcam" | "" when no mask

    @property
    def is_uncertain(self):
        return self.top_prob < CONFIDENCE_THRESHOLD

    @property
    def viz_cls(self):
        """Auto mode: the prediction, or the most likely abnormal class for Normal."""
        if self.top_cls != "Normal":
            return self.top_cls
        return max((c for c in CLASSES if c != "Normal"), key=lambda c: self.probs[c])

    @property
    def is_confident_finding(self):
        return (not self.is_uncertain and self.viz_cls == self.top_cls
                and self.top_cls != "Normal" and self.probs[self.viz_cls] >= OVERLAY_MIN_PROB)


def prediction_line(p):
    if p.is_uncertain:
        return (f"**Prediction:** Uncertain\n\n"
                f"The top class is {p.top_cls} at {p.top_prob * 100:.1f}%, below the "
                f"{CONFIDENCE_THRESHOLD * 100:.0f}% reporting threshold. Review the class "
                f"confidence values; a repeat or higher quality image may help.")
    return (f"**Prediction:** {p.top_cls} ({p.top_prob * 100:.1f}% confidence)\n\n"
            f"{DESCRIPTIONS[p.top_cls]}")


def provenance(p):
    if not p.mask_source:
        return ""
    return ("\n\n_Overlay source: disease-segmentation U-Net._" if p.mask_source == "segmentation"
            else "\n\n_Overlay source: Grad-CAM._")


def s0_template(p):
    """The deployed summary. Depends only on the prediction, never on the heatmap."""
    txt = prediction_line(p)
    viz_prob = p.probs[p.viz_cls]
    if not p.mask_source:
        txt += "\n\nNo attention map is available for this model."
    elif p.is_confident_finding:
        txt += (f"\n\nHeatmap: region driving the {p.viz_cls} prediction. "
                f"The outline marks the most influential area.")
    elif p.top_cls == "Normal":
        txt += (f"\n\nHeatmap: areas the model assessed for {p.viz_cls} "
                f"(the next most likely class); none reached an abnormal level.")
    elif p.is_uncertain:
        txt += (f"\n\nHeatmap: areas the model weighed for {p.viz_cls}. "
                f"Interpret with caution while the prediction is uncertain.")
    else:
        txt += f"\n\nHeatmap: model attention for {p.viz_cls} ({viz_prob * 100:.1f}%)."
    return txt + provenance(p)


def location_sentence(d):
    """The S1 location line for one set of descriptors (None = no heatmap given)."""
    if d is None or not d.has_region:
        return NO_REGION_TEXT
    if not d.gate_pass:
        return OUTSIDE_LUNG_TEXT
    return f"Heatmap evidence: {d.side}, {d.zone} zone, {d.extent}, {d.focality}."


def s1_descriptor(p, d):
    """Descriptor-gated template: states a location only when the gate passes."""
    return prediction_line(p) + "\n\n" + location_sentence(d) + provenance(p)
