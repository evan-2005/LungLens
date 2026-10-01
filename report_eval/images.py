"""
Image inputs for the VLM generators.

The overlay is drawn with the app's own build_heatmap, so a VLM sees the same
colouring a user sees. For the permuted control, film A's radiograph is drawn
with film B's heatmap. Images are downscaled so the long side is at most
MAX_SIDE pixels, which keeps VLM cost and latency predictable.
"""
import os

import cv2
import numpy as np

MAX_SIDE = 768


def _encode_png(rgb, max_side=MAX_SIDE):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def _read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def radiograph_png(film, max_side=MAX_SIDE):
    return _encode_png(_read_rgb(film.orig_png), max_side)


def overlay_png(film, heatmap_film, max_side=MAX_SIDE):
    """`film`'s radiograph with `heatmap_film`'s served heatmap drawn on it."""
    os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
    import app  # heavy import (Gradio); only needed for VLM conditions
    orig = _read_rgb(film.orig_png)
    mask = np.load(heatmap_film.mask_npy)
    return _encode_png(app.build_heatmap(orig, mask), max_side)
