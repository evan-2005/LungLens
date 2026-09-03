"""
Three extra measurements the qualitative section needs, all over the same 245
scored test films.

A. Border contact. Does the region reach the edge of the analysed frame, i.e. is
   the classifier keying on something the 8% BorderCrop failed to remove?
B. Wrong lung. For the TBX11K films the dataset ships lesion bounding boxes, so
   for those we can ask whether a CORRECT tuberculosis call put its region on the
   side the lesion is actually on.
C. Distilled U-Net versus live Grad-CAM. The reported Dice of 0.48 is against the
   teacher on the segmentation val split; this measures how far apart the two
   maps are on the films that actually get served.
"""
import os, sys, csv, json
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

OV = os.path.join(HERE, "overlays")
rows = list(csv.DictReader(open(os.path.join(HERE, "localisation_table.csv"),
                                encoding="utf-8")))

app.model_path = os.path.join(HERE, "chest_model_4class_HEAD.pth")
app.seg_model_path = os.path.join(ROOT, "chest_segmentation_model.pth")
app.seg_model, app.cls_model = app.load_models_from_disk()
app.seg_disease_model = True
print(f"device={app.device}", flush=True)

tf = __import__("torchvision").transforms.Compose([
    __import__("torchvision").transforms.Resize((app.IMG_SIZE, app.IMG_SIZE)),
    __import__("torchvision").transforms.ToTensor(),
    app._normalize_transform()])


def region_of(disp, src, shape):
    thr = app.MASK_DISPLAY_THRESHOLD if src == "segmentation" else app.OVERLAY_THRESHOLD
    return app.clean_region(disp, thr, shape)


# ── A. border contact ────────────────────────────────────────────────────────
def border_stats():
    touch, outer_mass, n = 0, [], 0
    worst = []
    for r in rows:
        if r["side"] == "none":
            continue
        disp = np.load(os.path.join(OV, f"{r['key']}_mask.npy"))
        orig = np.array(Image.open(os.path.join(OV, f"{r['key']}_orig.png")))
        reg = region_of(disp, r["mask_source"], orig.shape)
        if reg is None:
            continue
        rb = reg.astype(bool)
        H, W = rb.shape
        n += 1
        edge = np.zeros_like(rb)
        k = max(2, int(round(0.02 * min(H, W))))          # outer 2% frame
        edge[:k, :] = edge[-k:, :] = edge[:, :k] = edge[:, -k:] = True
        if (rb & edge).any():
            touch += 1
        band = np.zeros_like(rb)
        kb = max(3, int(round(0.10 * min(H, W))))         # outer 10% band
        band[:kb, :] = band[-kb:, :] = band[:, :kb] = band[:, -kb:] = True
        f = float((rb & band).sum()) / max(int(rb.sum()), 1)
        outer_mass.append(f)
        worst.append((f, r["key"], r["true"], r["pred"], r["group"]))
    outer_mass = np.array(outer_mass)
    print("\n=== A. border contact (analysed frame, i.e. after BorderCrop 0.08) ===")
    print(f"  regions scored                      : {n}")
    print(f"  touching the outer 2% frame         : {touch} ({100*touch/n:.1f}%)")
    print(f"  median mass in the outer 10% band   : {np.median(outer_mass):.3f}")
    print(f"  >25% of region in the outer 10% band: {int((outer_mass>0.25).sum())} "
          f"({100*(outer_mass>0.25).mean():.1f}%)")
    worst.sort(reverse=True)
    print("  most edge-heavy regions:")
    for f, k, t, p, g in worst[:8]:
        print(f"    {k:26s} {f:.2f} outer-band  T={t:13s} P={p:13s} {g}")


# ── B. wrong lung, using the TBX11K lesion boxes ─────────────────────────────
def wrong_lung():
    import ast
    T = (r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified"
         r"\versions\1\tbx11k-simplified")
    boxes = {}
    with open(os.path.join(T, "data.csv"), newline="") as fh:
        for rr in csv.DictReader(fh):
            if rr["image_type"] == "tb" and rr["bbox"] not in ("", "none"):
                boxes.setdefault(rr["fname"], []).append(
                    (ast.literal_eval(rr["bbox"]), float(rr["image_width"])))
    print("\n=== B. wrong lung (TBX11K films, which ship lesion boxes) ===")
    agree = disagree = nocall = 0
    detail = []
    for r in rows:
        fn = os.path.basename(r["path"])
        if fn not in boxes or r["true"] != "Tuberculosis":
            continue
        if r["side"] == "none":
            nocall += 1
            continue
        # lesion side in image coordinates, then to anatomy: image-left = patient right
        sides = set()
        for b, W in boxes[fn]:
            cx = (b["xmin"] + b["width"]/2)/W
            sides.add("right lung" if cx < 0.5 else "left lung")
        sr, sl = float(r["share_right"] or 0), float(r["share_left"] or 0)
        if max(sr, sl) < 0.25:
            dom = "neither"
        else:
            dom = "right lung" if sr > sl else "left lung"
        ok = dom in sides
        if dom == "neither":
            nocall += 1
        elif ok:
            agree += 1
        else:
            disagree += 1
        detail.append((r["key"], r["pred"], sorted(sides), dom, ok))
    print(f"  TB films with lesion boxes in the pool: {len(detail)}")
    print(f"    region on a lesion-bearing side     : {agree}")
    print(f"    region on the opposite side         : {disagree}")
    print(f"    region on neither lung (>75% off)   : {nocall}")
    for k, p, s, d, ok in detail:
        print(f"    {k:26s} pred={p:13s} lesion={','.join(s):22s} region={d:11s} "
              f"{'ok' if ok else 'MISMATCH' if d != 'neither' else '-'}")


# ── C. distilled U-Net versus live Grad-CAM on the served films ──────────────
def unet_vs_gradcam():
    print("\n=== C. distilled U-Net mask vs live Grad-CAM, same film, same class ===")
    dices, cdist, sameside = [], [], []
    for i, r in enumerate(rows):
        im = app.BorderCrop()(Image.open(r["path"]).convert("RGB"))
        t = tf(im).unsqueeze(0).to(app.device)
        with torch.no_grad():
            _, pred_masks = app.seg_model(t)
            probs = torch.softmax(app.cls_model(t), 1)[0].cpu()
        top = int(torch.argmax(probs))
        ab = probs.clone(); ab[0] = -1.0
        viz = top if top != 0 else int(torch.argmax(ab))
        if viz == 0:
            continue
        u = pred_masks[0, viz].cpu().numpy()
        gt = tf(im).unsqueeze(0).to(app.device); gt.requires_grad_(True)
        g = app.GradCAM(app.cls_model, app.get_gradcam_layer(app.cls_model)
                        ).generate_heatmap(gt, viz)
        ub = app.apply_heatmap_floor(u) > app.MASK_DISPLAY_THRESHOLD
        gb = app.apply_heatmap_floor(g) > app.OVERLAY_THRESHOLD
        inter = float((ub & gb).sum())
        denom = float(ub.sum() + gb.sum())
        if denom == 0:
            continue
        dices.append(2*inter/denom)
        if ub.any() and gb.any():
            uy, ux = np.nonzero(ub); gy, gx = np.nonzero(gb)
            d = np.hypot(ux.mean()-gx.mean(), uy.mean()-gy.mean())/ub.shape[1]
            cdist.append(d)
            sameside.append((ux.mean() < ub.shape[1]/2) == (gx.mean() < gb.shape[1]/2))
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)
    d = np.array(dices); c = np.array(cdist); s = np.array(sameside)
    print(f"  films compared                 : {d.size}")
    print(f"  Dice(U-Net region, Grad-CAM region): median {np.median(d):.3f}  "
          f"mean {d.mean():.3f}")
    print(f"    Dice = 0 (no overlap at all) : {int((d==0).sum())} ({100*(d==0).mean():.1f}%)")
    print(f"    Dice > 0.5                   : {int((d>0.5).sum())} ({100*(d>0.5).mean():.1f}%)")
    print(f"  centroid distance / image width: median {np.median(c):.3f}")
    print(f"  same side of the image         : {int(s.sum())}/{s.size} "
          f"({100*s.mean():.1f}%)")
    json.dump({"dice": d.tolist(), "cdist": c.tolist(),
               "same_side": [bool(x) for x in s]},
              open(os.path.join(HERE, "unet_vs_gradcam.json"), "w"))


border_stats()
wrong_lung()
unet_vs_gradcam()
