"""
Fit temperature scaling and the pneumonia threshold on the validation split.

Selection rule for the threshold (stated up front, validation only): the lowest
threshold, i.e. the highest pneumonia recall, such that validation macro-F1
stays within MAX_F1_DROP points and Normal recall within MAX_NORMAL_DROP points
of plain argmax. If none qualifies, no threshold is saved.

Usage (repository root):
    python -m calibration.fit_decision                 # fit on val, write decision_config.json
    python -m calibration.fit_decision --test          # also score the test split, once
"""
import argparse
import json
import os
import sys

os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, recall_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import app  # noqa: E402
from decision import (CONFIG_PATH, DecisionConfig, decide_batch, fit_temperature,  # noqa: E402
                      save_config, sha256_of, softmax)
from lung_attention.split_manifest import load_split  # noqa: E402

DEFAULT_CKPT = os.path.join(ROOT, "fig7_work", "chest_model_4class_HEAD.pth")
DEFAULT_MANIFEST = os.path.join(ROOT, "lung_attention", "splits", "split_s32000_seed42.csv")
CACHE_DIR = os.path.join(ROOT, "runs", "calibration")
THRESHOLDS = np.round(np.arange(0.05, 0.61, 0.01), 2)
MAX_F1_DROP = 0.5        # macro-F1 points
MAX_NORMAL_DROP = 2.0    # Normal recall points
ECE_BINS = 15


@torch.no_grad()
def logits_for(ckpt, paths, labels, batch=64):
    net = app.CNNModel(classCount=app.NUM_CLASSES, isTrained=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.to(app.device).eval()
    _, eval_tf = app.build_transforms()
    loader = DataLoader(app.ChestXRayDataset(paths, labels, eval_tf), batch_size=batch)
    return np.concatenate([net(x.to(app.device)).float().cpu().numpy() for x, _ in loader])


def cached_logits(split_name, ckpt, split):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"logits_{split_name}_{sha256_of(ckpt)[:12]}.npz")
    paths, labels, _ = split[split_name]
    if os.path.exists(path):
        data = np.load(path)
        return data["logits"], data["labels"]
    logits = logits_for(ckpt, paths, labels)
    np.savez(path, logits=logits, labels=np.asarray(labels))
    return logits, np.asarray(labels)


def ece(probs, labels, bins=ECE_BINS):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(total)


def metrics(labels, preds):
    rec = recall_score(labels, preds, labels=range(4), average=None, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=range(4))
    return {"accuracy": 100 * float((labels == preds).mean()),
            "macro_f1": 100 * f1_score(labels, preds, average="macro", zero_division=0),
            "recall": {c: 100 * float(r) for c, r in zip(app.CLASSES, rec)},
            "pneumonia_as_normal": int(cm[1, 0]), "normal_as_pneumonia": int(cm[0, 1])}


def choose_threshold(logits, labels, temperature):
    base = metrics(labels, decide_batch(logits, DecisionConfig(temperature=temperature)))
    curve, chosen = [], None
    for t in THRESHOLDS:
        m = metrics(labels, decide_batch(logits, DecisionConfig(temperature, float(t))))
        ok = (m["macro_f1"] >= base["macro_f1"] - MAX_F1_DROP
              and m["recall"]["Normal"] >= base["recall"]["Normal"] - MAX_NORMAL_DROP)
        curve.append({"threshold": float(t), "ok": ok, **m})
        if ok and chosen is None:
            chosen = float(t)
    return base, chosen, curve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out", default=os.path.join(ROOT, CONFIG_PATH))
    p.add_argument("--test", action="store_true", help="also score the test split (once)")
    args = p.parse_args(argv)

    split = load_split(args.manifest)
    v_logits, v_labels = cached_logits("val", args.ckpt, split)
    temp = fit_temperature(v_logits, v_labels)
    base, threshold, curve = choose_threshold(v_logits, v_labels, temp)
    cfg = DecisionConfig(temperature=temp, pneumonia_threshold=threshold,
                         checkpoint_sha256=sha256_of(args.ckpt))
    report = {
        "fitted_on": "validation split", "n_val": int(len(v_labels)),
        "selection_rule": f"lowest threshold keeping macro-F1 within {MAX_F1_DROP} and Normal "
                          f"recall within {MAX_NORMAL_DROP} points of argmax",
        "val_ece_before": ece(softmax(v_logits), v_labels),
        "val_ece_after": ece(softmax(v_logits, temp), v_labels),
        "val_argmax": base,
        "val_with_threshold": next((c for c in curve if c["threshold"] == threshold), None),
    }
    if args.test:
        t_logits, t_labels = cached_logits("test", args.ckpt, split)
        report["test_argmax"] = metrics(t_labels, decide_batch(t_logits, DecisionConfig(temp)))
        report["test_with_threshold"] = metrics(t_labels, decide_batch(t_logits, cfg))
        report["test_ece_before"] = ece(softmax(t_logits), t_labels)
        report["test_ece_after"] = ece(softmax(t_logits, temp), t_labels)
    save_config(cfg, args.out, extra={"report": report})
    with open(os.path.join(CACHE_DIR, "threshold_curve_val.json"), "w", encoding="utf-8") as fh:
        json.dump(curve, fh, indent=2)
    print(json.dumps({"temperature": temp, "pneumonia_threshold": threshold, **report}, indent=2))


if __name__ == "__main__":
    main()
