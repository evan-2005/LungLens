"""
Evaluate a classifier checkpoint on classification AND on where its served
heatmap lands (E1 evaluation).

The heatmap scored is the one the app shows: Grad-CAM at
denseblock4.denselayer16.conv2 for the predicted class, passed through the
app's own apply_heatmap_floor and clean_region at OVERLAY_THRESHOLD. It is not
the final-layer CAM used by the training penalty.

Lung fields come from two instruments, reported separately:
  gt   - ground-truth lung masks shipped with the COVID-19 Radiography Database
         (independent of the segmenter that supervises E1 training);
  pred - the lung-field U-Net in fig7_work (the same instrument as training,
         so these numbers are optimistic for E1 and must be read as such).

Usage (repository root; --samples must match the training run):
    python -m lung_attention.eval_lungattn --ckpt runs/lungattn/lam1/classifier.pth \
        --split val --out runs/lungattn/lam1/eval_val.json
    python -m lung_attention.eval_lungattn --ckpt fig7_work/chest_model_4class_HEAD.pth \
        --split val --out runs/lungattn/baseline_eval_val.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fig7_work"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, recall_score  # noqa: E402

import app  # noqa: E402
import lungfield  # noqa: E402
from lung_attention.train_lungattn import get_split  # noqa: E402

RADIOGRAPHY_MARKER = "COVID-19_Radiography_Dataset"
MAJORITY_OUTSIDE = 0.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--split-manifest", default=None,
                   help="frozen split CSV; see split_manifest.py")
    p.add_argument("--samples", type=int, default=32000)
    p.add_argument("--max-images", type=int, default=0, help="0 = whole split")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def split_records(split, samples, manifest):
    return get_split(manifest, samples)[split]


def gt_lung_mask(path):
    """Radiography-Database lung mask in the app's frame, or None if absent."""
    if RADIOGRAPHY_MARKER not in path:
        return None
    mask_path = path.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
    if mask_path == path or not os.path.exists(mask_path):
        return None
    m = app.BorderCrop()(Image.open(mask_path).convert("L"))
    m = m.resize((app.IMG_SIZE, app.IMG_SIZE), Image.NEAREST)
    return np.array(m) > 127


def region_stats(heat, lung):
    """In-lung share of the thresholded region and of the floored heat mass."""
    floored = app.apply_heatmap_floor(heat)
    region = app.clean_region(floored, app.OVERLAY_THRESHOLD, (app.IMG_SIZE, app.IMG_SIZE))
    mass = float(floored.sum())
    mass_in = float(floored[lung].sum()) / mass if mass > 0 else float("nan")
    if region is None:
        return {"has_region": False, "region_in_lung": float("nan"), "mass_in_lung": mass_in,
                "region_area": 0.0}
    r = region.astype(bool)
    return {"has_region": True, "region_in_lung": float(r[lung].sum()) / float(r.sum()),
            "mass_in_lung": mass_in, "region_area": float(r.mean())}


def score_image(net, path, eval_tf):
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(app.device)
    with torch.no_grad():
        pred = int(net(x).argmax(1).item())
    heat = app.gradcam_soft_mask(net, x, pred)
    row = {"pred": pred}
    gt = gt_lung_mask(path)
    if gt is not None:
        row["gt"] = region_stats(heat, gt)
    pred_lung = lungfield.lung_mask(app.BorderCrop()(img), out_shape=heat.shape)
    row["pred_mask"] = region_stats(heat, pred_lung)
    return row


def summarise_heat(rows, key):
    sub = [r[key] for r in rows if key in r]
    if not sub:
        return {"n": 0}
    with_region = [s for s in sub if s["has_region"]]
    rin = np.array([s["region_in_lung"] for s in with_region])
    mass = np.array([s["mass_in_lung"] for s in sub if not np.isnan(s["mass_in_lung"])])
    out = {"n": len(sub), "with_region": len(with_region),
           "no_region_frac": 1.0 - len(with_region) / len(sub)}
    if with_region:
        out.update({"region_in_lung_median": float(np.median(rin)),
                    "region_majority_outside_frac": float(np.mean(rin < MAJORITY_OUTSIDE)),
                    "region_area_median": float(np.median([s["region_area"] for s in with_region]))})
    if mass.size:
        out["mass_in_lung_median"] = float(np.median(mass))
    return out


def summarise(rows, labels, sources):
    y_true, y_pred = labels, [r["pred"] for r in rows]
    recall = recall_score(y_true, y_pred, labels=list(range(app.NUM_CLASSES)),
                          average=None, zero_division=0)
    return {
        "n": len(rows),
        "accuracy": 100.0 * float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        # None, not 0, for a class absent from the evaluated images.
        "recall": {app.CLASSES[i]: (100.0 * float(v) if i in set(y_true) else None)
                   for i, v in enumerate(recall)},
        "per_source": app.per_source_accuracy(y_true, y_pred, sources),
        "confusion": confusion_matrix(y_true, y_pred,
                                      labels=list(range(app.NUM_CLASSES))).tolist(),
        "heatmap_gt_lung": summarise_heat(rows, "gt"),
        "heatmap_pred_lung": summarise_heat(rows, "pred_mask"),
    }


def main(argv=None):
    args = parse_args(argv)
    paths, labels, sources = split_records(args.split, args.samples, args.split_manifest)
    if args.max_images:
        paths, labels, sources = (paths[:args.max_images], labels[:args.max_images],
                                  sources[:args.max_images])
    net = app.CNNModel(classCount=app.NUM_CLASSES, isTrained=False)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    net.to(app.device).eval()
    _, eval_tf = app.build_transforms()

    rows = []
    for i, path in enumerate(paths):
        rows.append(score_image(net, path, eval_tf))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(paths)}", flush=True)

    result = {"ckpt": args.ckpt, "split": args.split, "samples": args.samples,
              "split_manifest": args.split_manifest,
              **summarise(rows, labels, sources)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    short = {k: result[k] for k in ("n", "accuracy", "macro_f1")}
    print(json.dumps({**short, "heatmap_gt_lung": result["heatmap_gt_lung"],
                      "heatmap_pred_lung": result["heatmap_pred_lung"]}, indent=2))


if __name__ == "__main__":
    main()
