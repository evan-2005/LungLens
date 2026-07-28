"""
eval_classifier_perclass.py
===========================
Definitively test chest_model_4class.pth on the held-out test split to verify
whether it is the good epoch-11 model or a corrupted/different checkpoint.

Uses the EXACT same transforms, BorderCrop, and patient-grouped split logic as
train_full3 / app.py, so the per-class numbers should match the log exactly if
the checkpoint is the genuine epoch-11 result.

Run with:
    python eval_classifier_perclass.py
"""

import os, sys, random, json, datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# ── Config (must match app.py exactly) ────────────────────────────────────────
CLASSES         = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
NUM_CLASSES     = 4
IMG_SIZE        = 224
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
BORDER_CROP_FRAC = 0.08
TEST_FRACTION   = 0.15
VAL_FRACTION    = 0.15
MODEL_PATH      = "chest_model_4class.pth"
METRICS_PATH    = "chest_classifier_metrics.json"
BATCH_SIZE      = 64
SEED            = 42
# MUST match the `num_samples` the checkpoint was trained with. app.py subsamples
# the pool to this size BEFORE splitting, so evaluating on a split derived from the
# full pool yields a different test set whose images were in that run's TRAINING
# set. Override from argv[1] when a run used a different size.
NUM_SAMPLES     = int(sys.argv[1]) if len(sys.argv) > 1 else 32000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── BorderCrop (identical to app.py) ─────────────────────────────────────────
class BorderCrop:
    def __init__(self, frac=BORDER_CROP_FRAC):
        self.frac = frac
    def __call__(self, img):
        w, h = img.size
        dx, dy = int(round(w * self.frac)), int(round(h * self.frac))
        if w - 2*dx < 1 or h - 2*dy < 1:
            return img
        return img.crop((dx, dy, w-dx, h-dy))

val_transform = transforms.Compose([
    BorderCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ── Model (identical to app.py CNNModel) ──────────────────────────────────────
class CNNModel(nn.Module):
    def __init__(self, classCount, isTrained=True):
        super().__init__()
        self.cnnmodel = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if isTrained else None)
        self.cnnmodel.classifier = nn.Linear(
            self.cnnmodel.classifier.in_features, classCount)

    def forward(self, x):
        return self.cnnmodel(x)

# ── Load data using the same pipeline as app.py ───────────────────────────────
print("Loading dataset paths from app.py collect_dataset()...")
sys.path.insert(0, os.path.dirname(__file__))

# Import only the data-gathering parts; avoid launching Gradio.
# We'll replicate collect_dataset() manually using the same kagglehub paths.
# Importing app otherwise loads BOTH checkpoints and runs a warm-up forward pass
# through each, which this script does not need (it builds its own model below).
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

try:
    import kagglehub
    from app import collect_dataset, stratified_subsample
    print("Importing collect_dataset from app.py...")
    paths, labels, groups, sources = collect_dataset()
    print(f"  Total samples in pool: {len(paths)}")
    # Reproduce the training run's subsample BEFORE splitting. Skipping this was a
    # real defect: the resulting "test" set overlapped the checkpoint's training
    # data, so the verdict below could pass a checkpoint that had simply memorised
    # the images it was being scored on.
    paths, labels, groups, sources = stratified_subsample(
        paths, labels, groups, sources, NUM_SAMPLES, seed=SEED)
    print(f"  After stratified subsample to {NUM_SAMPLES}: {len(paths)}")
except Exception as e:
    print(f"ERROR importing from app.py: {e}")
    sys.exit(1)

# ── Replicate the patient-grouped split (same as train_full3) ─────────────────
def _group_split(indices, labels, groups, test_size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    sub_labels = [labels[i] for i in indices]
    sub_groups = [groups[i] for i in indices]
    a_idx, b_idx = next(gss.split(indices, sub_labels, sub_groups))
    return ([indices[i] for i in a_idx], [indices[i] for i in b_idx])

all_idx = list(range(len(paths)))
trainval_idx, test_idx = _group_split(all_idx, labels, groups, TEST_FRACTION, SEED)
val_rel = VAL_FRACTION / (1.0 - TEST_FRACTION)
train_idx, val_idx = _group_split(trainval_idx, labels, groups, val_rel, SEED)

test_paths  = [paths[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]
test_sources = [sources[i] for i in test_idx]

print(f"  Test set: {len(test_paths)} samples")

# Per-class counts in test set
from collections import Counter
cls_counts = Counter(test_labels)
for ci, cn in enumerate(CLASSES):
    print(f"    {cn}: {cls_counts[ci]}")

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {MODEL_PATH}")
stat = os.stat(MODEL_PATH)
mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
print(f"  File size:    {stat.st_size:,} bytes")
print(f"  Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

model = CNNModel(NUM_CLASSES, isTrained=False)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
# The checkpoint may be the raw state_dict or wrapped in a dict
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
    saved_epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint type: wrapped dict  (epoch={saved_epoch})")
elif isinstance(ckpt, dict) and any(k.startswith("cnnmodel.") for k in ckpt.keys()):
    state_dict = ckpt
    print(f"  Checkpoint type: raw state_dict")
else:
    state_dict = ckpt
    print(f"  Checkpoint type: unknown, attempting raw load")

model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()
print("  Checkpoint loaded successfully.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
test_ds = ChestXRayDataset(test_paths, test_labels, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=(device.type == "cuda"))

all_preds, all_labels, all_probs = [], [], []
correct = 0
print(f"\nRunning inference on {len(test_ds)} test samples...")

with torch.no_grad():
    for i, (imgs, lbls) in enumerate(test_loader):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(lbls.tolist())
        all_probs.extend(probs.cpu().tolist())
        correct += (preds.cpu() == lbls).sum().item()
        if (i+1) % 10 == 0 or (i+1) == len(test_loader):
            print(f"  Batch {i+1}/{len(test_loader)} done", end="\r")

print()

acc = 100.0 * correct / len(test_ds)
macro_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n{'='*60}")
print(f"  Overall Test Accuracy : {acc:.4f}%")
print(f"  Macro F1              : {macro_f1:.4f}%")
print(f"{'='*60}")

# Per-class report
report = classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=4, zero_division=0)
print("\nPer-class classification report:")
print(report)

# Per-class recall (to match metrics JSON format)
print("Per-class recall:")
for ci, cn in enumerate(CLASSES):
    tp = cm[ci, ci]
    fn = cm[ci, :].sum() - tp
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print(f"  {cn:15s}: {recall:.4f}%")

# Confusion matrix
print("\nConfusion matrix (rows=true, cols=pred; order: Normal, Pneumonia, TB, Covid-19):")
print(cm)

# ── Per-source accuracy ────────────────────────────────────────────────────────
print("\nPer-source accuracy:")
src_groups = {}
for pred, lbl, src in zip(all_preds, all_labels, test_sources):
    if src not in src_groups:
        src_groups[src] = {"correct": 0, "total": 0}
    src_groups[src]["total"] += 1
    if pred == lbl:
        src_groups[src]["correct"] += 1

for src, info in sorted(src_groups.items()):
    src_acc = 100.0 * info["correct"] / info["total"]
    print(f"  {src:20s}: {src_acc:.2f}%  (n={info['total']})")

# ── Compare to the recorded metrics for this checkpoint ───────────────────────
# The reference is read from the metrics file rather than hardcoded. The previous
# hardcoded literals (96.1859% / 95.8291%) went stale the moment a new run
# finished, at which point the verdict compared the checkpoint against numbers
# belonging to a different model.
print("\n" + "="*60)
ref_acc = ref_f1 = None
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_acc = ref.get("test_acc")
        ref_f1 = ref.get("test_f1")
        print(f"REFERENCE ({METRICS_PATH}, recorded "
              f"{ref.get('saved_at', 'unknown')}, epoch {ref.get('epoch', '?')}):")
        if ref_acc is not None:
            f1_txt = "not recorded" if ref_f1 is None else f"{ref_f1:.4f}%"
            print(f"  Test Acc  = {ref_acc:.4f}%  |  Macro F1 = {f1_txt}")
        print(f"  Recorded split sizes: train {ref.get('train_samples', '?')} / "
              f"val {ref.get('val_samples', '?')} / test {ref.get('test_samples', '?')}")
    except (OSError, json.JSONDecodeError) as e:
        print(f"REFERENCE unavailable (could not read {METRICS_PATH}: {e})")
else:
    print(f"REFERENCE unavailable ({METRICS_PATH} not found).")
print("="*60)
print("CURRENT CHECKPOINT:")
print(f"  Test Acc  = {acc:.4f}%  |  Macro F1 = {macro_f1:.4f}%")
print(f"  This run's test split: {len(test_ds)} images "
      f"(subsampled pool = {NUM_SAMPLES})")

# ── Verdict ───────────────────────────────────────────────────────────────────
print()
if ref_acc is None:
    print("VERDICT: no recorded reference to compare against. Absolute score "
          f"is {acc:.2f}%; judge it against the per-class report above.")
elif ref.get("test_samples") not in (None, len(test_ds)):
    print(f"WARNING: this run scored {len(test_ds)} test images but the metrics "
          f"file records {ref.get('test_samples')}. The splits differ, so the "
          f"comparison below is not apples to apples. Re-run with the correct "
          f"num_samples as argv[1].")
elif abs(acc - ref_acc) < 0.5:
    print(f"PASS: within 0.5% of the recorded {ref_acc:.2f}%. This is the "
          f"checkpoint the metrics file describes.")
elif acc < 70.0:
    print("FAIL: accuracy is severely degraded (<70%). The checkpoint is likely "
          "corrupted or is a very early, biased epoch.")
else:
    print(f"MISMATCH: {acc:.2f}% does not match the recorded {ref_acc:.2f}%. "
          f"Either the checkpoint is not the one described by the metrics file, "
          f"or the split differs. Investigate before citing either number.")
