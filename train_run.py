"""Headless training driver.

Runs the same classifier + Grad-CAM-distilled segmentation pipeline the Training
tab uses, but from the command line so a long run survives with no browser tab
open. Hyperparameters come from argv; everything else (data collection, splits,
early stopping, checkpointing) is app.run_training_thread's job.

Usage:
    python train_run.py <num_samples> <epochs> <lr> <batch_size> <num_workers> \
                        <train_seg 0|1> <seg_cap> <seg_epochs>

The __main__ guard is required: DataLoader workers under Windows spawn re-import
this module, and without the guard each worker would re-launch training.
"""
import os
import sys

# Must be set before importing app: skips the startup model load+warmup, both in
# this process and in every DataLoader worker that re-imports app on Windows spawn
# (workers inherit this environment).
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

import app


def main():
    a = sys.argv[1:]
    num_samples = int(a[0]) if len(a) > 0 else 32000
    epochs      = int(a[1]) if len(a) > 1 else 40
    lr          = float(a[2]) if len(a) > 2 else 1e-4
    batch_size  = int(a[3]) if len(a) > 3 else 32
    num_workers = int(a[4]) if len(a) > 4 else 4
    train_seg   = bool(int(a[5])) if len(a) > 5 else True
    seg_cap     = int(a[6]) if len(a) > 6 else 3000
    seg_epochs  = int(a[7]) if len(a) > 7 else 10

    print(f"[train_run] device={app.device} samples={num_samples} epochs={epochs} "
          f"lr={lr} batch={batch_size} workers={num_workers} "
          f"train_seg={train_seg} seg_cap={seg_cap} seg_epochs={seg_epochs}",
          flush=True)

    app.run_training_thread(num_samples, epochs, lr, batch_size, num_workers,
                            train_seg=train_seg, seg_cap=seg_cap, seg_epochs=seg_epochs)

    print(f"[train_run] final status: {app.training_status}", flush=True)


if __name__ == "__main__":
    main()
