"""
Shortlist Fig. 1 candidates. CPU only: reads pixels, screens for burned-in
markers, and writes contact sheets to look at by eye.

Artefact score components (all computed on the BorderCrop(0.08) frame the app
and the figure both use):
  bright_frac  : fraction of near-saturated pixels (>=250) - text/stamps burn in white
  corner_bright: near-saturated pixels in the four corner boxes (L/R markers live there)
  n_blobs      : count of small saturated connected components (letters, arrows)
"""
import os, sys, csv, glob, re, json, ast, random
import numpy as np
from PIL import Image
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BORDER = 0.08

def load_cropped(p, size=512):
    im = Image.open(p).convert("L")
    w, h = im.size
    dx, dy = int(round(w*BORDER)), int(round(h*BORDER))
    if w-2*dx > 0 and h-2*dy > 0:
        im = im.crop((dx, dy, w-dx, h-dy))
    return im.resize((size, size), Image.BILINEAR)

def artefact_score(im):
    a = np.array(im)
    sat = (a >= 250).astype(np.uint8)
    bright_frac = sat.mean()
    H, W = a.shape
    k = int(H*0.18)
    corners = [sat[:k,:k], sat[:k,-k:], sat[-k:,:k], sat[-k:,-k:]]
    corner_bright = float(np.mean([c.mean() for c in corners]))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sat, 8)
    blobs = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 30 <= area <= 4000 and max(w_, h_) <= H*0.25:
            blobs += 1
    return dict(bright_frac=float(bright_frac),
                corner_bright=corner_bright, n_blobs=int(blobs))

def contact_sheet(items, out, cols=6, cell=260, label_h=34):
    import matplotlib
    matplotlib.use("Agg")
    from PIL import ImageDraw
    rows = (len(items) + cols - 1)//cols
    sheet = Image.new("RGB", (cols*cell, rows*(cell+label_h)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(items):
        r, c = divmod(i, cols)
        try:
            im = load_cropped(p, cell).convert("RGB")
        except Exception:
            im = Image.new("RGB", (cell, cell), "red")
        sheet.paste(im, (c*cell, r*(cell+label_h)))
        d.text((c*cell+4, r*(cell+label_h)+cell+4), lab, fill="black")
    sheet.save(out)
    print("wrote", out, len(items), "tiles")
