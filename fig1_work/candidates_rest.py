"""Shortlist Pneumonia / Covid-19 / Normal panels for Fig. 1 from the test split."""
import os, sys, csv, json
import numpy as np, cv2
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fig7_work"))
from screen import artefact_score, contact_sheet
from lungfield import lung_mask, split_lungs, border_crop

rows = list(csv.DictReader(open(os.path.join(ROOT, "fig7_work", "test_manifest.csv"),
                                encoding="utf-8")))
WANT = {"pneumonia": 1, "covid": 3, "normal": 0}
which = sys.argv[1]
cls = WANT[which]

# only correctly and confidently classified films, so the panel is not itself a
# mislabelled or borderline case
pool = [r for r in rows
        if int(r["true_idx"]) == cls and int(r["pred_idx"]) == cls
        and max(float(r[k]) for k in ("p_normal","p_pneumonia","p_tb","p_covid")) > 0.95]
print(f"{which}: {len(pool)} confident correct test films")

def feats(path):
    im = border_crop(Image.open(path).convert("L"))
    a = np.array(im.resize((384, 384), Image.BILINEAR), np.float32)
    m = lung_mask(im, a.shape)
    if m.sum() < 384*384*0.05:
        return None
    L, R = split_lungs(m)                       # image-left, image-right
    lm, rm = a[L].mean(), a[R].mean()
    core = cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    periph = m & ~core
    per_minus_core = a[periph].mean() - a[core].mean() if core.sum() > 100 else 0.0
    # compact dense focus inside a lung: brightest 12% of lung pixels, how clustered
    lv = a[m]
    thr = np.percentile(lv, 88)
    dense = ((a > thr) & m).astype(np.uint8)
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dense, 8)
    biggest = max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0)
    compact = biggest/max(dense.sum(), 1)
    return dict(l_mean=float(lm), r_mean=float(rm), asym=float(lm-rm),
                per_core=float(per_minus_core), compact=float(compact),
                lung_std=float(lv.std()), exposure=float(a.mean()),
                lung_frac=float(m.mean()), **artefact_score(im.resize((512, 512))))

out = []
for i, r in enumerate(pool):
    try:
        f = feats(r["path"])
    except Exception as e:
        continue
    if f is None:
        continue
    f.update(path=r["path"], source=r["source"],
             conf=max(float(r[k]) for k in ("p_normal","p_pneumonia","p_tb","p_covid")))
    out.append(f)
    if (i+1) % 200 == 0: print("  ", i+1, flush=True)

# clean films only
clean = [f for f in out if f["n_blobs"] <= 2 and f["corner_bright"] < 0.0015]
print(f"  {len(clean)}/{len(out)} pass the burned-in-marker screen")

if which == "pneumonia":
    clean.sort(key=lambda f: -abs(f["asym"]) * (0.5 + f["compact"]))
elif which == "covid":
    clean = [f for f in clean if abs(f["asym"]) < 12]
    clean.sort(key=lambda f: -(f["per_core"] + 0.15*f["lung_std"]))
else:
    clean.sort(key=lambda f: (f["lung_std"], abs(f["asym"])))

top = clean[:24]
for f in top:
    print(f"  {os.path.basename(f['path'])[:34]:34s} {f['source']:14s} "
          f"asym={f['asym']:+6.1f} per-core={f['per_core']:+5.1f} "
          f"compact={f['compact']:.2f} std={f['lung_std']:5.1f} conf={f['conf']:.3f}")
json.dump(top, open(os.path.join(HERE, f"{which}_cands.json"), "w"), indent=1)
contact_sheet([(f["path"], f"{i}:{os.path.basename(f['path'])[:22]} {f['source'][:8]}")
               for i, f in enumerate(top)],
              os.path.join(HERE, f"sheet_{which}.png"))
