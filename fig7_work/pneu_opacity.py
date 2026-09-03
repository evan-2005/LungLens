"""
Is there anything radiographically there on the 102 pneumonia films the model
calls Normal?

Objective proxy, measured inside the lung fields only (same lung-field U-Net,
val Dice 0.982), on the border-cropped frame the model sees:

  asym      left-right mean intensity difference, absolute. Unilateral
            consolidation drives this up.
  dense     area fraction of the lung field above (lung median + 1 SD of a
            reference Normal population), i.e. how much of the lung is
            abnormally opaque.
  std       intensity spread inside the lung field.

Three groups, all from the same held-out test split:
  A  the 102 pneumonia films predicted Normal
  B  pneumonia films predicted Pneumonia   (200 sampled)
  C  Normal films predicted Normal         (200 sampled)

If A sits with C rather than with B, the films look normal, and a model that
calls them Normal is agreeing with the pixels rather than failing on them.
"""
import os, sys, csv, json, random
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ["LUNGLENS_SKIP_STARTUP"] = "1"
from lungfield import lung_mask, split_lungs, border_crop

rows = list(csv.DictReader(open(os.path.join(HERE, "test_manifest.csv"),
                                encoding="utf-8")))
rng = random.Random(11)
A = [r for r in rows if r["true_idx"] == "1" and r["pred_idx"] == "0"]
B = rng.sample([r for r in rows if r["true_idx"] == "1" and r["pred_idx"] == "1"], 200)
C = rng.sample([r for r in rows if r["true_idx"] == "0" and r["pred_idx"] == "0"], 200)
print(f"A={len(A)}  B={len(B)}  C={len(C)}", flush=True)


def feats(path):
    im = border_crop(Image.open(path).convert("L"))
    a = np.array(im.resize((384, 384), Image.BILINEAR), np.float32)
    m = lung_mask(im, a.shape)
    if m.sum() < 384*384*0.05:
        return None
    L, R = split_lungs(m)
    lv = a[m]
    med, sd = float(np.median(lv)), float(lv.std())
    # "abnormally opaque" is scored against each film's own lung distribution,
    # so it survives the wide exposure differences between the source datasets.
    dense = float(((a > med + 1.0*sd) & m).sum()) / float(m.sum())
    return dict(asym=abs(float(a[L].mean() - a[R].mean())), dense=dense, std=sd)


def run(name, group):
    out = []
    for r in group:
        f = feats(r["path"])
        if f:
            f["path"] = r["path"]
            out.append(f)
    arr = {k: np.array([f[k] for f in out]) for k in ("asym", "dense", "std")}
    print(f"\n{name}  n={len(out)}")
    for k in ("asym", "dense", "std"):
        v = arr[k]
        print(f"  {k:6s} median {np.median(v):7.3f}   mean {v.mean():7.3f}   "
              f"p75 {np.percentile(v,75):7.3f}   p90 {np.percentile(v,90):7.3f}")
    return out, arr


oA, aA = run("A  pneumonia -> Normal        ", A)
oB, aB = run("B  pneumonia -> Pneumonia     ", B)
oC, aC = run("C  Normal    -> Normal        ", C)

print("\n--- how many of group A exceed group C's 90th percentile? ---")
for k in ("asym", "dense", "std"):
    thr = float(np.percentile(aC[k], 90))
    nA = int((aA[k] > thr).sum()); nB = int((aB[k] > thr).sum())
    print(f"  {k:6s} threshold {thr:7.3f}:  A {nA:3d}/{len(aA[k])} "
          f"({100*nA/len(aA[k]):4.1f}%)   B {nB:3d}/{len(aB[k])} "
          f"({100*nB/len(aB[k]):4.1f}%)   C 10.0% by construction")

json.dump({"A": oA, "B": oB, "C": oC},
          open(os.path.join(HERE, "pneu_opacity.json"), "w"))

# a seeded sample of the errors to look at by eye
sample = random.Random(5).sample(oA, 16)
CELL = 340
sh = Image.new("RGB", (CELL*4, (CELL+22)*4), "white")
from PIL import ImageDraw
d = ImageDraw.Draw(sh)
for i, f in enumerate(sample):
    im = border_crop(Image.open(f["path"]).convert("L"))
    w, h = im.size; s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    r_, c_ = divmod(i, 4)
    sh.paste(im.resize((CELL, CELL), Image.LANCZOS).convert("RGB"),
             (c_*CELL, r_*(CELL+22)))
    d.text((c_*CELL+3, r_*(CELL+22)+CELL+3),
           f"{i}: {os.path.basename(f['path'])[:26]} dense={f['dense']:.3f}", fill="black")
sh.save(os.path.join(HERE, "sheet_pneu_as_normal_sample.png"))
json.dump([os.path.basename(f["path"]) for f in sample],
          open(os.path.join(HERE, "pneu_sample16.json"), "w"), indent=1)
print("\nwrote sheet_pneu_as_normal_sample.png (seeded sample of 16)")
