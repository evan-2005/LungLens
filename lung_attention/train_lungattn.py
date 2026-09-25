"""
Train the DenseNet-121 classifier with a lung-constrained attention penalty (E1).

Same data, split, transforms, class weights, optimiser, scheduler and
checkpoint rule (best validation macro-F1) as app.run_training_thread, plus
lambda * outside_lung_fraction. lambda = 0 through this same script is the
control run, so any difference is due to the penalty alone.

Checkpoints go to --out, never to app.model_path, so the served model is never
touched. The test split is not used here; evaluate with eval_lungattn.py.

Usage (repository root):
    python -m lung_attention.train_lungattn --lam 1.0 --out runs/lungattn/lam1
    python -m lung_attention.train_lungattn --lam 0 --out runs/lungattn/lam0
Smoke test:
    python -m lung_attention.train_lungattn --samples 400 --epochs 1 --batch 8 \
        --workers 0 --out runs/lungattn/smoke
"""
import argparse
import datetime
import json
import os
import sys

os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fig7_work"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import app  # noqa: E402
import lungfield  # noqa: E402
from lung_attention.attention_loss import (  # noqa: E402
    DEFAULT_DILATE_PX,
    class_activation_map,
    forward_with_features,
    lung_masks_from_batch,
    outside_lung_fraction,
)
from lung_attention.split_manifest import load_split  # noqa: E402

CHECKPOINT_NAME = "classifier.pth"
METRICS_NAME = "train_metrics.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split-manifest", default=None,
                   help="frozen split CSV (required for results comparable to the paper "
                        "on any machine other than the one that exported it)")
    p.add_argument("--samples", type=int, default=32000,
                   help="subsample size when re-deriving the split without a manifest")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lam", type=float, default=1.0, help="attention-loss weight")
    p.add_argument("--warmup-epochs", type=int, default=1,
                   help="epochs trained with lambda = 0 before the penalty starts")
    p.add_argument("--dilate", type=int, default=DEFAULT_DILATE_PX,
                   help="lung-mask dilation in pixels at 224")
    p.add_argument("--init", choices=["imagenet", "checkpoint"], default="imagenet")
    p.add_argument("--init-ckpt", default=None,
                   help="weights to start from when --init checkpoint")
    p.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.lam < 0:
        p.error("--lam must be >= 0")
    if args.init == "checkpoint" and not args.init_ckpt:
        p.error("--init checkpoint needs --init-ckpt")
    return args


def get_split(manifest, samples):
    """
    {train, val, test} -> (paths, labels, sources).

    With a manifest the split is rebuilt exactly, on any OS. Without one it is
    re-derived with the app's pipeline, which only reproduces the paper's split
    on the machine that produced it (see split_manifest.py).
    """
    if manifest:
        return load_split(manifest)
    print("[lungattn] WARNING: no --split-manifest; re-deriving the split, which "
          "differs across operating systems.", flush=True)
    paths, labels, groups, sources = app.collect_dataset()
    if len(paths) < 20:
        raise RuntimeError("Not enough images found to train (need at least 20).")
    paths, labels, groups, sources = app.stratified_subsample(
        paths, labels, groups, sources, samples)
    tr, va, te = app.patient_grouped_split(paths, labels, groups, sources)
    return {"train": tr, "val": va, "test": te}


def build_loaders(args):
    """The app's own data pipeline and transforms, on the paper's split."""
    split = get_split(args.split_manifest, args.samples)
    (tr_p, tr_l, _), (va_p, va_l, _), (te_p, _, _) = split["train"], split["val"], split["test"]
    train_tf, eval_tf = app.build_transforms()
    kw = dict(num_workers=args.workers, pin_memory=(app.device.type == "cuda"))
    train = DataLoader(app.ChestXRayDataset(tr_p, tr_l, train_tf),
                       batch_size=args.batch, shuffle=True, **kw)
    val = DataLoader(app.ChestXRayDataset(va_p, va_l, eval_tf),
                     batch_size=args.batch, shuffle=False, **kw)
    sizes = {"train": len(tr_p), "val": len(va_p), "test": len(te_p)}
    return train, val, tr_l, sizes


def build_model(args):
    net = app.CNNModel(classCount=app.NUM_CLASSES, isTrained=(args.init == "imagenet"))
    if args.init == "checkpoint":
        net.load_state_dict(torch.load(args.init_ckpt, map_location="cpu", weights_only=True))
    return net.to(app.device)


def class_weights(train_labels):
    counts = np.bincount(train_labels, minlength=app.NUM_CLASSES).astype(np.float64)
    inv = 1.0 / np.clip(counts, 1.0, None)
    return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=app.device), counts


def attention_loss(net, feats, labels, images, lung_net, dilate):
    lung, valid = lung_masks_from_batch(lung_net, images, app.IMAGENET_MEAN,
                                        app.IMAGENET_STD, dilate_px=dilate)
    # Under --amp the features arrive in half precision; the head weight is fp32.
    cam = class_activation_map(feats.float(), net.cnnmodel.classifier.weight, labels)
    return outside_lung_fraction(cam, lung, valid), valid


def train_epoch(net, loader, criterion, optimizer, scaler, lung_net, lam, args):
    net.train()
    sums = {"ce": 0.0, "attn": 0.0, "valid": 0, "seen": 0, "correct": 0, "batches": 0}
    use_amp = args.amp and app.device.type == "cuda"
    for images, labels in loader:
        images, labels = images.to(app.device), labels.to(app.device)
        with torch.autocast("cuda", enabled=use_amp):
            logits, feats = forward_with_features(net, images)
            ce = criterion(logits.float(), labels)
        attn, valid = attention_loss(net, feats, labels, images, lung_net, args.dilate)
        loss = ce + lam * attn
        if not torch.isfinite(loss):
            print("  WARNING: non-finite loss, batch skipped", flush=True)
            continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        sums["ce"] += ce.item()
        sums["attn"] += attn.item()
        sums["valid"] += int(valid.sum().item())
        sums["seen"] += labels.size(0)
        sums["correct"] += int(logits.argmax(1).eq(labels).sum().item())
        sums["batches"] += 1
    b = max(1, sums["batches"])
    n = max(1, sums["seen"])
    return {"ce": sums["ce"] / b, "attn": sums["attn"] / b,
            "train_acc": 100.0 * sums["correct"] / n,
            "lung_mask_valid_frac": sums["valid"] / n}


@torch.no_grad()
def validate(net, loader, lung_net, dilate):
    """Macro-F1 for checkpoint selection, plus the CAM outside-lung fraction."""
    net.eval()
    y_true, y_pred, attn = [], [], []
    for images, labels in loader:
        images, labels = images.to(app.device), labels.to(app.device)
        logits, feats = forward_with_features(net, images)
        preds = logits.argmax(1)
        a, _ = attention_loss(net, feats, preds, images, lung_net, dilate)
        attn.append(a.item())
        y_true += labels.tolist()
        y_pred += preds.tolist()
    acc = 100.0 * float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"val_acc": acc, "val_f1": f1, "val_cam_outside_lung": float(np.mean(attn))}


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, CHECKPOINT_NAME)
    if os.path.abspath(ckpt_path) == os.path.abspath(app.model_path):
        raise RuntimeError("Refusing to overwrite the served checkpoint.")

    train_loader, val_loader, train_labels, sizes = build_loaders(args)
    print(f"[lungattn] split {sizes} | lambda={args.lam} | device={app.device}", flush=True)
    net = build_model(args)
    lung_net = lungfield.net()
    for p in lung_net.parameters():
        p.requires_grad_(False)

    weights, counts = class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and app.device.type == "cuda"))

    history, best, best_f1, stale = [], None, -1.0, 0
    for epoch in range(args.epochs):
        lam = 0.0 if epoch < args.warmup_epochs else args.lam
        tr = train_epoch(net, train_loader, criterion, optimizer, scaler, lung_net, lam, args)
        va = validate(net, val_loader, lung_net, args.dilate)
        scheduler.step(va["val_f1"])
        row = {"epoch": epoch + 1, "lam": lam, "lr": optimizer.param_groups[0]["lr"], **tr, **va}
        history.append(row)
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in row.items()), flush=True)
        if va["val_f1"] > best_f1:
            best_f1, stale, best = va["val_f1"], 0, row
            app.save_checkpoint_atomic(net.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= app.EARLY_STOP_PATIENCE:
                print("[lungattn] early stopping", flush=True)
                break

    meta = {"args": vars(args), "split_sizes": sizes,
            "class_counts": {app.CLASSES[i]: int(c) for i, c in enumerate(counts)},
            "best": best, "history": history,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(args.out, METRICS_NAME), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[lungattn] best epoch {best and best['epoch']} -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
