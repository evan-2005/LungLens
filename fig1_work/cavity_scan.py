"""
Rank TBX11K upper-zone TB films by an objective cavitation proxy.

Inside each annotated lesion box: find dark, roughly-round connected components
whose surrounding annulus is markedly brighter. A cavity on a plain film is
exactly that - a lucency ringed by a thick opaque wall inside consolidated lung.
Also require the box to be brighter than its mirror-image counterpart, i.e. the
lesion really is a consolidation, not just normal lung.
"""
import os, sys, csv, ast, json
import numpy as np, cv2
from PIL import Image

T = (r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified"
     r"\versions\1\tbx11k-simplified")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

boxes, meta = {}, {}
with open(os.path.join(T, "data.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if r["image_type"] != "tb":
            continue
        meta[r["fname"]] = r
        if r["bbox"] not in ("", "none"):
            boxes.setdefault(r["fname"], []).append(ast.literal_eval(r["bbox"]))

test = set()
with open(os.path.join(ROOT, "fig7_work", "test_manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test.add(os.path.normcase(os.path.abspath(r["path"])))

def cavity_score(a, box):
    x0, y0 = int(box["xmin"]), int(box["ymin"])
    x1, y1 = int(box["xmin"]+box["width"]), int(box["ymin"]+box["height"])
    H, W = a.shape
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    if x1-x0 < 24 or y1-y0 < 24:
        return 0.0, 0.0, None
    roi = a[y0:y1, x0:x1].astype(np.float32)
    # consolidation: box brighter than its mirror
    mx0, mx1 = W-x1, W-x0
    mirror = a[y0:y1, mx0:mx1].astype(np.float32)
    consol = float(roi.mean() - mirror.mean()) if mirror.size else 0.0

    blur = cv2.GaussianBlur(roi, (0, 0), 2.0)
    thr = blur.mean() - 0.55*blur.std()
    dark = (blur < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    best, bbox = 0.0, None
    area_roi = roi.size
    for i in range(1, n):
        ar = stats[i, cv2.CC_STAT_AREA]
        if not (0.008*area_roi <= ar <= 0.30*area_roi):
            continue
        w_, h_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if min(w_, h_) < 8:
            continue
        if max(w_, h_)/max(min(w_, h_), 1) > 2.2:      # cavities are roundish
            continue
        circ = ar/(w_*h_)
        if circ < 0.55:
            continue
        comp = (lab == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((9, 9), np.uint8)) - comp
        if ring.sum() < 30:
            continue
        rim = float(roi[ring > 0].mean() - roi[comp > 0].mean())   # wall contrast
        s = rim * circ
        if s > best:
            best, bbox = s, (int(x0+stats[i, 0]), int(y0+stats[i, 1]), int(w_), int(h_))
    return best, consol, bbox

rows = []
for fname, bs in boxes.items():
    fp = os.path.join(T, "images", fname)
    if not os.path.exists(fp):
        continue
    H = float(meta[fname]["image_height"]); W = float(meta[fname]["image_width"])
    cys = [(b["ymin"]+b["height"]/2)/H for b in bs]
    if max(cys) > 0.45 or len(bs) > 3:
        continue
    a = np.array(Image.open(fp).convert("L"))
    best, consol, bb = 0.0, 0.0, None
    for b in bs:
        s, c, bx = cavity_score(a, b)
        if s > best:
            best, consol, bb = s, c, bx
    if best <= 0:
        continue
    rows.append(dict(fname=fname, path=fp, cav=float(best), consol=float(consol),
                     cavbox=bb, in_test=os.path.normcase(os.path.abspath(fp)) in test,
                     nbox=len(bs), tb_type=meta[fname]["tb_type"]))

rows = [r for r in rows if r["consol"] > 4]
rows.sort(key=lambda r: -r["cav"])
print(f"{len(rows)} films with a cavity-like lucency inside an upper-zone lesion box")
for r in rows[:24]:
    print(f"  {r['fname']:12s} cav={r['cav']:6.1f} consol={r['consol']:6.1f} "
          f"test={int(r['in_test'])} nbox={r['nbox']} {r['tb_type']}")
json.dump(rows[:40], open(os.path.join(HERE, "tb_cavity.json"), "w"), indent=1)
