"""Generate the Stage-2 disease-segmentation overlay from the ALREADY-trained
classifier, without retraining the classifier.

The full training run's Stage 2 stalled by overflowing a 6 GB GPU; this reruns
only the fixed segmentation stage against the saved chest_model_4class.pth, then
writes chest_segmentation_model.pth and honest metrics (recomputed on the
deterministic held-out test split).

Usage:
    python seg_recover.py <seg_cap> <seg_epochs> <save 0|1>
"""
import os, sys, json, datetime
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, confusion_matrix

import app

NUM_SAMPLES = 32000  # matches the interrupted run's split


def main():
    seg_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seg_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    do_save = bool(int(sys.argv[3])) if len(sys.argv) > 3 else 0

    print(f"[seg_recover] device={app.device} seg_cap={seg_cap} "
          f"seg_epochs={seg_epochs} save={do_save}", flush=True)

    paths, labels, groups, sources = app.collect_dataset()
    paths, labels, groups, sources = app.stratified_subsample(
        paths, labels, groups, sources, NUM_SAMPLES)
    (train_paths, train_labels, _), (vp, vl_, vs), \
        (test_paths, test_labels, test_sources) = app.patient_grouped_split(
            paths, labels, groups, sources)
    print(f"Split -> train {len(train_paths)} | val {len(vp)} | test {len(test_paths)}",
          flush=True)

    _, eval_tf = app.build_transforms()

    cls_net = app.CNNModel(classCount=app.NUM_CLASSES, isTrained=False)
    cls_net.load_state_dict(
        torch.load(app.model_path, map_location=app.device, weights_only=True))
    cls_net.to(app.device).eval()

    # Grad-CAM distillation into the U-Net (memory-fixed Stage 2).
    seg_net, seg_dice = app.train_segmentation_head(
        cls_net, train_paths, train_labels, eval_tf,
        seg_cap, seg_epochs, batch_size=app.SEG_MAX_BATCH, num_workers=0,
        log=lambda m: print(m, flush=True))
    print(f"[seg_recover] seg done, best Val Dice = {seg_dice:.4f}", flush=True)

    if seg_net is None:
        print("[seg_recover] segmentation could not run; nothing saved.")
        return

    if not do_save:
        print("[seg_recover] smoke run only (save=0); not writing checkpoints.")
        return

    # Save the U-Net overlay.
    app.save_checkpoint_atomic(seg_net.state_dict(), app.seg_model_path)
    print(f"[seg_recover] saved {app.seg_model_path}", flush=True)

    # Recompute honest test metrics on the deterministic held-out split.
    test_ds = app.ChestXRayDataset(test_paths, test_labels, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    y_true, y_pred = app.evaluate_classifier(cls_net, test_loader)
    test_acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    test_f1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_recall = recall_score(
        y_true, y_pred, labels=list(range(app.NUM_CLASSES)),
        average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(app.NUM_CLASSES)))
    src_acc = app.per_source_accuracy(y_true, y_pred, test_sources)

    # Carry the Stage-1 fields forward from whatever the training run recorded
    # rather than inventing them. This script does not train the classifier, so it
    # cannot know its val_acc / val_f1 / train_acc / epoch / batch_size / lr; an
    # earlier version hardcoded plausible-looking literals for exactly those keys,
    # which then surfaced in the UI (and in the write-up) as if measured.
    metrics = {}
    if os.path.exists(app.metrics_path):
        try:
            with open(app.metrics_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                metrics = loaded
                print(f"[seg_recover] carrying Stage-1 fields forward from "
                      f"{app.metrics_path}", flush=True)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[seg_recover] could not read existing metrics ({e}); "
                  f"writing only what this run measured.", flush=True)

    # Only these keys are actually measured here. Per-class recall comes from the
    # TEST loader above, so it is named for the test split, not val.
    metrics.update({
        "test_per_class_recall": {app.CLASSES[i]: 100.0 * per_class_recall[i]
                                  for i in range(app.NUM_CLASSES)},
        "train_samples": len(train_paths), "val_samples": len(vp),
        "test_samples": len(test_paths),
        "test_acc": test_acc, "test_f1": test_f1,
        "test_per_source_acc": src_acc, "confusion_matrix": cm.tolist(),
        "segmentation": "gradcam_distilled", "seg_val_dice": seg_dice,
        "seg_train_images": min(seg_cap, len(train_paths)),
        "metrics_source": "seg_recover.py (classifier re-scored, not retrained)",
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    app.save_training_metrics(metrics)
    print(f"[seg_recover] wrote {app.metrics_path} "
          f"(test acc {test_acc:.2f}%, macro-F1 {test_f1:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
