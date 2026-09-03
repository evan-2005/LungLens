"""
Fig. 7 step 1+2: run the SERVED inference path over the candidate pool, save the
original / heatmap / overlay for each, and score the overlay's localisation.

The overlay is produced by calling app.predict_image() itself - the same
function the Gradio button calls - so the composite is byte-for-byte what a user
sees. The intermediate maps are captured by wrapping app.build_heatmap and
app.apply_heatmap_floor rather than by re-implementing the pipeline, so this
script cannot drift from the app.

The checkpoint loaded is the git-HEAD chest_model_4class.pth, which is the one
that reproduces chest_classifier_metrics.json exactly (96.1757% / macro-F1
95.7709%); the working-tree file is a different, weaker model (91.20%).
"""
import os, sys, csv, json, random, argparse
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.environ["LUNGLENS_SKIP_STARTUP"] = "1"
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch
import app
from lungfield import lung_mask, split_lungs

CLASSES = app.CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="verified classifier checkpoint")
ap.add_argument("--out", default=os.path.join(HERE, "overlays"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ── load the verified checkpoint into the app's own globals ──────────────────
app.model_path = args.ckpt
app.seg_model_path = os.path.join(ROOT, "chest_segmentation_model.pth")
app.seg_model, app.cls_model = app.load_models_from_disk()
_m = json.load(open(os.path.join(ROOT, "chest_classifier_metrics.json"), encoding="utf-8"))
app.seg_disease_model = bool(app.seg_model is not None
                             and _m.get("segmentation") == "gradcam_distilled")
print(f"cls={app.cls_model is not None} seg={app.seg_model is not None} "
      f"seg_disease={app.seg_disease_model} device={app.device}", flush=True)

# ── capture the app's own intermediate maps ──────────────────────────────────
CAP = {}
_orig_floor, _orig_heat = app.apply_heatmap_floor, app.build_heatmap

def floor_spy(mask):
    CAP["raw_mask"] = np.array(mask, copy=True)
    out = _orig_floor(mask)
    CAP["display_mask"] = np.array(out, copy=True)
    return out

def heat_spy(orig_np, mask, alpha=0.5):
    out = _orig_heat(orig_np, mask, alpha)
    CAP["orig_np"] = np.array(orig_np, copy=True)
    CAP["heatmap"] = np.array(out, copy=True)
    return out

app.apply_heatmap_floor = floor_spy
app.build_heatmap = heat_spy

# ── candidate pool ───────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(os.path.join(HERE, "test_manifest.csv"), encoding="utf-8")))
PK = ("p_normal", "p_pneumonia", "p_tb", "p_covid")

pool, why = [], {}
def add(r, tag):
    k = r["path"]
    if k not in why:
        pool.append(r); why[k] = tag
    else:
        why[k] += "+" + tag

for r in rows:
    if int(r["true_idx"]) == 1 and int(r["pred_idx"]) == 0:
        add(r, "pneu_as_normal")
for r in rows:
    if r["source"] in ("montgomery_tb", "shenzhen_tb"):
        add(r, r["source"])

correct = [r for r in rows if r["true_idx"] == r["pred_idx"]]
rng = random.Random(7)
for c in range(4):
    sub = [r for r in correct if int(r["true_idx"]) == c]
    for r in rng.sample(sub, min(10, len(sub))):
        add(r, "correct_sample")
print(f"pool = {len(pool)} images", flush=True)

# ── score one overlay ────────────────────────────────────────────────────────
def score(row, key):
    CAP.clear()
    pil = Image.open(row["path"])
    txt, probs, upd = app.predict_image(pil, app.AUTO_OVERLAY)
    if not txt:
        return None
    overlay = np.array(upd["value"]) if isinstance(upd, dict) else np.array(upd.value)
    orig_np = CAP["orig_np"]
    disp = CAP["display_mask"]

    mask_source = "segmentation" if "disease-segmentation U-Net" in txt else "gradcam"
    thr = app.MASK_DISPLAY_THRESHOLD if mask_source == "segmentation" else app.OVERLAY_THRESHOLD
    region = app.clean_region(disp, thr, orig_np.shape)

    lf = lung_mask(Image.fromarray(orig_np), orig_np.shape)
    Limg, Rimg = split_lungs(lf)          # image-left, image-right

    H, W = orig_np.shape[:2]
    if region is None:
        rec = dict(mask_area_frac=0.0, in_lung_frac=float("nan"), side="none",
                   compactness=float("nan"), cx=float("nan"), cy=float("nan"))
    else:
        rb = region.astype(bool)
        area = int(rb.sum())
        in_lung = float((rb & lf).sum())/max(area, 1)
        ys, xs = np.nonzero(rb)
        cx, cy = float(xs.mean()), float(ys.mean())
        ci, cj = int(round(cy)), int(round(cx))
        ci = min(max(ci, 0), H-1); cj = min(max(cj, 0), W-1)
        # image-left is the patient's RIGHT lung on a PA film
        if Limg[ci, cj]:   side = "right lung"
        elif Rimg[ci, cj]: side = "left lung"
        else:              side = "outside both"
        n, lab, st, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
        big = max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0)
        rec = dict(mask_area_frac=area/(H*W), in_lung_frac=in_lung, side=side,
                   compactness=big/max(area, 1), cx=cx/W, cy=cy/H)

    Image.fromarray(orig_np).save(os.path.join(args.out, f"{key}_orig.png"))
    Image.fromarray(CAP["heatmap"]).save(os.path.join(args.out, f"{key}_heat.png"))
    Image.fromarray(overlay).save(os.path.join(args.out, f"{key}_overlay.png"))
    np.save(os.path.join(args.out, f"{key}_mask.npy"), disp.astype(np.float32))

    top = max(probs, key=probs.get)
    rec.update(key=key, path=row["path"], source=row["source"], group=why[row["path"]],
               true=CLASSES[int(row["true_idx"])], pred=CLASSES[int(row["pred_idx"])],
               app_pred=top, conf=float(probs[top]),
               uncertain=bool(probs[top] < app.CONFIDENCE_THRESHOLD),
               mask_source=mask_source, lung_frac=float(lf.mean()),
               outlined=bool("The outline marks" in txt))
    return rec

recs = []
for i, r in enumerate(pool):
    key = f"{i:03d}_" + os.path.splitext(os.path.basename(r["path"]))[0][:40].replace(" ", "_")
    try:
        rec = score(r, key)
    except Exception as e:
        print(f"  !! {r['path']}: {type(e).__name__}: {e}", flush=True)
        continue
    if rec: recs.append(rec)
    if (i+1) % 25 == 0: print(f"  {i+1}/{len(pool)}", flush=True)

with open(os.path.join(HERE, "localisation_table.csv"), "w", newline="", encoding="utf-8") as fh:
    cols = ["key","group","source","true","pred","app_pred","conf","uncertain","mask_source",
            "in_lung_frac","mask_area_frac","side","compactness","cx","cy","outlined","path"]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in recs: w.writerow(r)
print("wrote localisation_table.csv:", len(recs), "rows")
