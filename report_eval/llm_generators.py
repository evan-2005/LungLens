"""
LLM and VLM report generators.

S2  text LLM, CONSTRAINED: receives the prediction and the heatmap descriptors
    as JSON and must return JSON matching a schema. A deterministic checker then
    verifies every field and every location mentioned in its sentence against
    the input. Any mismatch rejects the reply, and the film falls back to the S1
    template. Rejections are counted and reported, never hidden.
S3  VLM, FREE, radiograph only: says where it sees the evidence.
S4  VLM, FREE, radiograph plus the heatmap overlay.

S2 tests whether a fluent writer stays faithful when it may only rephrase
evidence; S3 vs S4 is "with vs. without the heatmap" for a VLM.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from report_eval.claims import Claim, parse_claim
from report_eval.llm_backends import BackendError
from report_eval.templates import prediction_line, provenance, location_sentence

S2_SYSTEM = (
    "You write the location sentence of a chest X-ray decision-support summary. "
    "You receive the classifier's prediction and measured properties of its heatmap region. "
    "Use ONLY those fields. If gate_pass is false, set state_location to false and write one "
    "sentence saying no location is reported; do not name any side, zone or extent. "
    "If gate_pass is true, set state_location to true, copy side, zone and extent exactly "
    "from the input, and write one plain sentence stating them. The heatmap shows where the "
    "model looked, not a finding: describe it as the heatmap region, and never call it an "
    "abnormality, lesion, opacity or any other finding. Never add sides, zones, findings or "
    "diagnoses that are not in the input. This is research software, not a clinical report."
)
# A heatmap region is where the model looked, not evidence of disease. A sentence
# that turns it into a finding overstates the evidence, even with the right location.
FINDING_WORDS = re.compile(
    r"\b(abnormalit\w*|lesions?|opacit\w*|consolidat\w*|nodules?|mass(es)?|infiltrat\w*|"
    r"effusions?|cavit\w*|ground[- ]glass|pneumonia|tuberculosis|covid\w*|infection)\b", re.I)
S2_SCHEMA = {
    "type": "object",
    "properties": {
        "state_location": {"type": "boolean"},
        "side": {"type": "string", "enum": ["right lung", "left lung", "bilateral", "outside both", "none"]},
        "zone": {"type": "string", "enum": ["upper", "mid", "lower", "none"]},
        "extent": {"type": "string", "enum": ["focal", "patchy", "extensive", "none"]},
        "sentence": {"type": "string"},
    },
    "required": ["state_location", "side", "zone", "extent", "sentence"],
    "additionalProperties": False,
}

VLM_SYSTEM = (
    "You assist with a research study on chest X-ray decision-support software. "
    "Answer in one or two short sentences. Refer to the patient's left and right: on this "
    "frontal film the patient's right lung appears on the image's left. If you cannot see an "
    "abnormality supporting the prediction, say so instead of guessing a location. "
    "This is not a clinical report."
)
S3_PROMPT = ("The classifier predicts {cls} ({prob:.1f}%). Where in the lungs are the findings "
             "that support this prediction located?")
S4_PROMPT = ("The classifier predicts {cls} ({prob:.1f}%). The first image is the radiograph. The "
             "second image overlays the classifier's heatmap: warmer colours mark the region that "
             "drove the prediction. Where in the lungs are the findings that support this "
             "prediction located?")
VLM_MAX_TOKENS = 200


@dataclass(frozen=True)
class GeneratedReport:
    text: str
    claim: Optional[Claim]
    model: str = ""
    latency_s: float = 0.0
    raw: str = ""
    rejected: bool = False
    reject_reason: str = ""
    sentence: str = ""          # S2 only: the location sentence on its own


def _s2_problems(reply, d):
    """Why an S2 reply is not a faithful restatement of the descriptors ([] if it is)."""
    problems = []
    gate = d is not None and d.gate_pass
    if reply["state_location"] != gate:
        problems.append(f"state_location={reply['state_location']} but gate_pass={gate}")
    if gate:
        for f in ("side", "zone", "extent"):
            if reply[f] != getattr(d, f):
                problems.append(f"{f}={reply[f]!r} but input {getattr(d, f)!r}")
    finding = FINDING_WORDS.search(reply["sentence"])
    if finding:
        problems.append(f"sentence asserts a finding ({finding.group(0)!r})")
    stated = parse_claim(reply["sentence"])
    expected = Claim(d.side, d.zone, d.extent) if gate else None
    if stated is not None and expected is None:
        problems.append("sentence states a location while the gate blocks it")
    elif stated is not None and expected is not None:
        for f in ("side", "zone", "extent"):
            said = getattr(stated, f)
            if said is not None and said != getattr(expected, f):
                problems.append(f"sentence says {f}={said!r}, input {getattr(expected, f)!r}")
    return problems


def s2_llm(prediction, descriptors, backend):
    """Constrained text-LLM summary; `descriptors` is None when no heatmap is given."""
    fields = {"predicted_class": prediction.top_cls,
              "confidence": round(prediction.top_prob, 3),
              **(descriptors.as_prompt_fields() if descriptors is not None
                 else {"region_present": False, "gate_pass": False, "side": None,
                       "zone": None, "extent": None, "focality": None})}
    result = backend.generate(S2_SYSTEM, json.dumps(fields), schema=S2_SCHEMA, max_tokens=400)
    fallback = location_sentence(descriptors)
    try:
        reply = json.loads(result.text)
        if not isinstance(reply, dict) or set(S2_SCHEMA["required"]) - set(reply):
            raise ValueError("missing fields")
        problems = _s2_problems(reply, descriptors)
    except (ValueError, TypeError) as e:
        problems = [f"invalid json: {e}"]
    rejected = bool(problems)
    sentence = fallback if rejected else reply["sentence"].strip()
    text = prediction_line(prediction) + "\n\n" + sentence + provenance(prediction)
    return GeneratedReport(text=text, claim=parse_claim(sentence), model=result.model,
                           latency_s=result.latency_s, raw=result.text, rejected=rejected,
                           reject_reason="; ".join(problems), sentence=sentence)


def _vlm(prompt, images, backend):
    result = backend.generate(VLM_SYSTEM, prompt, images=images, max_tokens=VLM_MAX_TOKENS)
    text = result.text.strip()
    return GeneratedReport(text=text, claim=parse_claim(text), model=result.model,
                           latency_s=result.latency_s, raw=result.text)


def s3_vlm(prediction, radiograph_png, backend):
    prompt = S3_PROMPT.format(cls=prediction.top_cls, prob=prediction.top_prob * 100)
    return _vlm(prompt, [radiograph_png], backend)


def s4_vlm(prediction, radiograph_png, overlay_png, backend):
    prompt = S4_PROMPT.format(cls=prediction.top_cls, prob=prediction.top_prob * 100)
    return _vlm(prompt, [radiograph_png, overlay_png], backend)


__all__ = ["BackendError", "GeneratedReport", "s2_llm", "s3_vlm", "s4_vlm"]
