"""
Catch uploads that are not chest X-rays (improvement #5).

The lung-field U-Net's lung area (fraction of the border-cropped image) decides,
in three bands set from measurements:

  < 0.03        rejected: no lungs at all. Images with no radiograph in them
                (charts, diagrams, UI without an X-ray, lung-MASK images, in
                colour or grey) never exceeded 0.017.
  0.03 to 0.10  analysed with a visible warning. Screenshots and figures that
  or > 0.75     CONTAIN an X-ray scored 0.057 to 0.092, and so did genuine but
                unusual films: an infant film filling half the frame (0.078).
  0.10 to 0.75  normal. 565 genuine X-rays (320 validation films from every
                source plus the 245 scored test films) ranged from 0.078 to
                0.629, all but one above 0.13.

Colour never rejects: a cyan-tinted scan of a real film measured 46.6
(Hasler-Suesstrunk), above most colour charts. It only adds a warning.

The negative set is small and built from this project's own images: this is a
safety net, not a validated out-of-distribution detector.
"""
from dataclasses import dataclass

import numpy as np

REJECT_LUNG_AREA = 0.03
MIN_TYPICAL_LUNG_AREA = 0.10
MAX_TYPICAL_LUNG_AREA = 0.75
TINT_COLOURFULNESS = 45.0
_SAMPLE = 128


@dataclass(frozen=True)
class UploadCheck:
    ok: bool
    reason: str = ""        # why it was rejected
    warning: str = ""       # shown above the result when analysed anyway
    lung_checked: bool = True


def colourfulness(img):
    """Hasler & Suesstrunk (2003) colourfulness on a 128x128 thumbnail."""
    a = np.asarray(img.convert("RGB").resize((_SAMPLE, _SAMPLE)), np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def check_upload(img, lung_mask):
    """`lung_mask` is the segmenter's HxW boolean output, or None if unavailable."""
    warnings = []
    if colourfulness(img) > TINT_COLOURFULNESS:
        warnings.append("The image is strongly tinted; X-rays are normally greyscale, so "
                        "results may be less reliable.")
    if lung_mask is None:
        return UploadCheck(True, warning=" ".join(warnings), lung_checked=False)
    area = float(np.asarray(lung_mask, bool).mean())
    if area < REJECT_LUNG_AREA:
        return UploadCheck(False, reason="No lung fields were found, so this does not look like "
                                         "a frontal chest X-ray. Please upload a chest radiograph.")
    if not MIN_TYPICAL_LUNG_AREA <= area <= MAX_TYPICAL_LUNG_AREA:
        warnings.append("The lung fields look unusual for a frontal chest X-ray (very small, "
                        "cropped, or part of a screenshot). Check the upload is a single "
                        "frontal radiograph.")
    return UploadCheck(True, warning=" ".join(warnings))
