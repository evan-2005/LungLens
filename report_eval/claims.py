"""
Location claims: extract them from generated text and compare them with the heatmap.

A claim exists when the text names a side or a zone. Extent alone ("focal
lesions") is class boilerplate, not a location, and does not count.

The parser is deliberately simple and conservative: when a text names two
different zones it reports no zone rather than guessing. Its accuracy on real
VLM output must be checked by hand (PAPER_PLAN.md, Section 3.5) before the S3/S4
numbers are reported.
"""
import re
from dataclasses import dataclass
from typing import Optional

_SIDE_BOTH = re.compile(r"\b(bilateral|bilaterally|both lungs|both sides|both lung fields)\b", re.I)
_SIDE_RIGHT = re.compile(r"\bright\b", re.I)
_SIDE_LEFT = re.compile(r"\bleft\b", re.I)
_OUTSIDE = re.compile(r"\boutside both\b", re.I)
# Zone words count only when they describe an anatomical place ("lower zone",
# "left upper lobe", "mid-zone", "lower left and right lung fields"), so
# "confidence is lower" is not a claim.
_PLACE = (r"[\s-]+(?:\w+[\s-]+){0,3}?"
          r"(?:zones?|lobes?|lungs?|fields?|thirds?|regions?|areas?|portions?|"
          r"parts?|halves|half|quadrants?|sections?|image)\b")
# A sentence that says nothing was found does not locate anything, even if it
# names the lungs ("throughout both lungs, with no visible abnormalities").
_NEGATION = re.compile(
    r"\b(no|not|without|cannot|can't|unable)\b[^.]{0,40}?"
    r"\b(visible|abnormalit\w*|specific|evidence|clear|obvious|significant|show|see|seen|"
    r"identif\w*|support\w*|focal)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_NON_SPECIFIC = re.compile(r"\b(throughout|entire|whole|all (of )?the|diffusely|both lungs|"
                           r"the lungs)\b", re.I)
_ZONES = {
    "upper": re.compile(r"\b(?:upper" + _PLACE + r"|apical|apex|apices)", re.I),
    "mid": re.compile(r"\b(?:(?:mid|middle)" + _PLACE + r"|perihilar|hilar)", re.I),
    "lower": re.compile(r"\b(?:lower" + _PLACE + r"|basal|bases|lung base)", re.I),
}
_EXTENTS = {
    "focal": re.compile(r"\b(focal|small|locali[sz]ed)\b", re.I),
    "patchy": re.compile(r"\bpatchy\b", re.I),
    "extensive": re.compile(r"\b(extensive|diffuse|widespread)\b", re.I),
}


@dataclass(frozen=True)
class Claim:
    side: Optional[str] = None      # "right lung" | "left lung" | "bilateral" | "outside both"
    zone: Optional[str] = None
    extent: Optional[str] = None


def _one_of(patterns, text):
    hits = [name for name, p in patterns.items() if p.search(text)]
    return hits[0] if len(hits) == 1 else None


def _affirmative(text):
    """The text with every negated sentence removed."""
    return " ".join(s for s in _SENTENCE.split(text or "") if not _NEGATION.search(s))


def parse_claim(text):
    """Claim stated in free text, or None if it names no side and no zone.

    Sentences that say nothing was found are ignored first.
    """
    text = _affirmative(text)
    if not text:
        return None
    if _OUTSIDE.search(text):
        side = "outside both"
    elif _SIDE_BOTH.search(text) or (_SIDE_RIGHT.search(text) and _SIDE_LEFT.search(text)):
        side = "bilateral"
    elif _SIDE_RIGHT.search(text):
        side = "right lung"
    elif _SIDE_LEFT.search(text):
        side = "left lung"
    else:
        side = None
    zone = _one_of(_ZONES, text)
    if side is None and zone is None:
        return None
    return Claim(side=side, zone=zone, extent=_one_of(_EXTENTS, text))


def response_kind(text):
    """
    What kind of answer a free-text generator gave:
      specific      names a side or zone (a scoreable claim)
      non_specific  points at the lungs generally ("throughout the lungs")
      no_evidence   says it sees nothing supporting the prediction
      other         none of the above
    """
    if parse_claim(text) is not None:
        return "specific"
    if any(_NEGATION.search(s) for s in _SENTENCE.split(text or "")):
        return "no_evidence"
    if _NON_SPECIFIC.search(text or ""):
        return "non_specific"
    return "other"


def claim_from_descriptors(d):
    """The claim a descriptor-gated generator makes, or None when the gate blocks it."""
    if d is None or not d.gate_pass:
        return None
    return Claim(side=d.side, zone=d.zone, extent=d.extent)


def side_matches(claimed, truth):
    if claimed == "bilateral":
        return truth.spans_both
    return claimed == truth.side
