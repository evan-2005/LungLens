import os, sys, csv, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fig7_work"))
from screen import contact_sheet
import importlib.util
spec = importlib.util.spec_from_file_location("cr", os.path.join(HERE, "candidates_rest.py"))
# reuse feats by copy: simpler to re-import pieces
from screen import artefact_score
from lungfield import lung_mask, split_lungs, border_crop
from PIL import Image
import cv2

rows = list(csv.DictReader(open(os.path.join(ROOT, "fig7_work", "test_manifest.csv"), encoding="utf-8")))
# Pneumonia films whose label is unambiguous: the paediatric pneumonia set and the
# radiography DB's "Viral Pneumonia" folder. Lung_Opacity is excluded here - it is
# mapped to Pneumonia by the repo's label map but is a broader radiological finding,
# so it is a poor choice for a figure captioned "Pneumonia".
pool = [r for r in rows if int(r["true_idx"]) == 1 and int(r["pred_idx"]) == 1
        and (r["source"] == "pneu_ds" or "Viral Pneumonia" in r["path"])
        and max(float(r[k]) for k in ("p_normal","p_pneumonia","p_tb","p_covid")) > 0.98]
print("pool:", len(pool))
out = []
for r in pool:
    im = border_crop(Image.open(r["path"]).convert("L"))
    a = np.array(im.resize((384, 384), Image.BILINEAR), np.float32)
    m = lung_mask(im, a.shape)
    if m.sum() < 384*384*0.05: continue
    L, R = split_lungs(m)
    asym = float(a[L].mean() - a[R].mean())
    lv = a[m]; thr = np.percentile(lv, 88)
    dense = cv2.morphologyEx(((a > thr) & m).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dense, 8)
    biggest = max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0)
    out.append(dict(path=r["path"], source=r["source"], asym=asym,
                    compact=float(biggest/max(dense.sum(),1)),
                    std=float(lv.std()), **artefact_score(im.resize((512,512)))))
out = [f for f in out if f["n_blobs"] <= 1 and f["corner_bright"] < 0.0008]
out.sort(key=lambda f: -abs(f["asym"])*(0.5+f["compact"]))
top = out[:24]
for f in top:
    print(f"  {os.path.basename(f['path'])[:32]:32s} {f['source']:9s} asym={f['asym']:+6.1f} "
          f"compact={f['compact']:.2f} std={f['std']:5.1f}")
json.dump(top, open(os.path.join(HERE, "pneumonia2_cands.json"), "w"), indent=1)
contact_sheet([(f["path"], f"{i}:{os.path.basename(f['path'])[:24]}") for i, f in enumerate(top)],
              os.path.join(HERE, "sheet_pneumonia2.png"))
