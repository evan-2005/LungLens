"""
Region descriptors: what a report generator is allowed to know about the heatmap.

Each served overlay is reduced to laterality, vertical zone, extent, focality,
in-lung fraction and whether the region spans both lungs. The first three and
the in-lung fraction are the paper's descriptors (Section 4); spans_both is
added so a free-text "bilateral" claim can be scored at all.

Laterality is ANATOMICAL (patient's side): on a PA film the image-left half is
the patient's right lung. fig7_work/gen_overlays.py already made that
translation when it wrote the localisation table.
"""
import math
from dataclasses import dataclass
from typing import Optional

UPPER_ZONE_MAX = 1.0 / 3.0      # centroid y as a fraction of image height
MID_ZONE_MAX = 2.0 / 3.0
FOCAL_AREA_MAX = 0.05           # region area as a fraction of the frame
PATCHY_AREA_MAX = 0.25
GATE_IN_LUNG_MIN = 0.5          # below this the region is mostly outside the lungs
SPANS_BOTH_MIN_SHARE = 0.15     # each lung must hold at least this share of the region
SINGLE_COMPONENT = 0.999999     # compactness = largest component / region area


@dataclass(frozen=True)
class Descriptors:
    has_region: bool
    side: Optional[str] = None          # "right lung" | "left lung" | "outside both"
    zone: Optional[str] = None          # "upper" | "mid" | "lower"
    extent: Optional[str] = None        # "focal" | "patchy" | "extensive"
    focality: Optional[str] = None      # "single" | "multifocal"
    in_lung: Optional[float] = None
    spans_both: bool = False

    @property
    def gate_pass(self):
        """True when a location may be stated: a region exists and sits mostly in the lungs."""
        return self.has_region and self.in_lung is not None and self.in_lung >= GATE_IN_LUNG_MIN

    def as_prompt_fields(self):
        """The fields an LLM generator receives. No raw coordinates."""
        return {"region_present": self.has_region, "gate_pass": self.gate_pass,
                "side": self.side, "zone": self.zone, "extent": self.extent,
                "focality": self.focality}


NO_REGION = Descriptors(has_region=False)


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def zone_of(cy):
    if cy < UPPER_ZONE_MAX:
        return "upper"
    return "mid" if cy < MID_ZONE_MAX else "lower"


def extent_of(area):
    if area < FOCAL_AREA_MAX:
        return "focal"
    return "patchy" if area <= PATCHY_AREA_MAX else "extensive"


def from_region(region, lung_right, lung_left):
    """
    Descriptors computed live from a binary region and the two lung masks.

    lung_right / lung_left are ANATOMICAL (patient's side), all arrays HxW of
    the same shape. Mirrors fig7_work/gen_overlays.py and rescore.py so the app
    and the paper's analysis describe a region identically. Without lung masks
    the region is still reported, but no side, no in-lung fraction and so no
    location claim (the gate cannot pass).
    """
    import cv2
    import numpy as np

    if region is None:
        return NO_REGION
    r = np.asarray(region).astype(bool)
    area = int(r.sum())
    if area == 0:
        return NO_REGION
    h, w = r.shape
    ys, xs = np.nonzero(r)
    cy, cx = float(ys.mean()), float(xs.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(r.astype(np.uint8), 8)
    largest = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n)), default=0)
    focality = "single" if largest / area >= SINGLE_COMPONENT else "multifocal"
    base = dict(has_region=True, zone=zone_of(cy / h), extent=extent_of(area / (h * w)),
                focality=focality)
    if lung_right is None or lung_left is None:
        return Descriptors(**base)
    lr, ll = np.asarray(lung_right, bool), np.asarray(lung_left, bool)
    ci, cj = min(max(int(round(cy)), 0), h - 1), min(max(int(round(cx)), 0), w - 1)
    side = "right lung" if lr[ci, cj] else "left lung" if ll[ci, cj] else "outside both"
    share_r, share_l = (r & lr).sum() / area, (r & ll).sum() / area
    return Descriptors(**base, side=side, in_lung=float(share_r + share_l),
                       spans_both=bool(share_r >= SPANS_BOTH_MIN_SHARE
                                       and share_l >= SPANS_BOTH_MIN_SHARE))


def from_table_row(row):
    """Descriptors from one row of fig7_work/localisation_table.csv."""
    if row.get("side", "none") == "none":
        return NO_REGION
    cy, area, comp = _num(row["cy"]), _num(row["mask_area_frac"]), _num(row["compactness"])
    in_lung = _num(row["in_lung_frac"])
    share_r, share_l = _num(row.get("share_right")), _num(row.get("share_left"))
    spans = (not math.isnan(share_r) and not math.isnan(share_l)
             and share_r >= SPANS_BOTH_MIN_SHARE and share_l >= SPANS_BOTH_MIN_SHARE)
    return Descriptors(
        has_region=True,
        side=row["side"],
        zone=zone_of(cy),
        extent=extent_of(area),
        focality="single" if comp >= SINGLE_COMPONENT else "multifocal",
        in_lung=None if math.isnan(in_lung) else in_lung,
        spans_both=spans,
    )
