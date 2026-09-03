"""Add per-lung mass share to the localisation table, from the saved masks."""
import os, sys, csv
import numpy as np, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ["LUNGLENS_SKIP_STARTUP"] = "1"
import app
from lungfield import lung_mask, split_lungs

OV = os.path.join(HERE, "overlays")
rows = list(csv.DictReader(open(os.path.join(HERE, "localisation_table.csv"), encoding="utf-8")))
out = []
for i, r in enumerate(rows):
    disp = np.load(os.path.join(OV, f"{r['key']}_mask.npy"))
    orig = np.array(Image.open(os.path.join(OV, f"{r['key']}_orig.png")))
    thr = app.MASK_DISPLAY_THRESHOLD if r["mask_source"] == "segmentation" else app.OVERLAY_THRESHOLD
    region = app.clean_region(disp, thr, orig.shape)
    lf = lung_mask(Image.fromarray(orig), orig.shape)
    Limg, Rimg = split_lungs(lf)
    if region is None:
        r.update(share_right="", share_left="", share_outside="", dominant="none")
    else:
        rb = region.astype(bool); a = max(int(rb.sum()), 1)
        sr = float((rb & Limg).sum())/a      # image-left = patient's RIGHT lung
        sl = float((rb & Rimg).sum())/a
        so = 1.0 - sr - sl
        dom = "right lung" if sr > max(sl, so) else ("left lung" if sl > max(sr, so) else "outside both")
        r.update(share_right=f"{sr:.3f}", share_left=f"{sl:.3f}",
                 share_outside=f"{max(so,0):.3f}", dominant=dom)
    out.append(r)
    if (i+1) % 50 == 0: print(" ", i+1, flush=True)

cols = list(rows[0].keys())
with open(os.path.join(HERE, "localisation_table.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for r in out: w.writerow(r)
print("updated", len(out), "rows")
