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
BATCH_SIZE      = 64
SEED            = 42

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
try:
    import kagglehub
    from app import collect_dataset, patient_grouped_split
    print("Importing collect_dataset from app.py...")
    paths, labels, groups, sources = collect_dataset()
    print(f"  Total samples: {len(paths)}")
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

# ── Compare to logged metrics (epoch-11 reference) ────────────────────────────
print("\n" + "="*60)
print("REFERENCE (from train_full3 log / metrics JSON at epoch 11):")
print("  Test Acc  = 96.1859%  |  Macro F1 = 95.8291%")
print("  Normal=98.46%  Pneumonia=93.49%  TB=94.27%  Covid=95.51%")
print("="*60)
print("CURRENT CHECKPOINT:")
print(f"  Test Acc  = {acc:.4f}%  |  Macro F1 = {macro_f1:.4f}%")

# ── Verdict ───────────────────────────────────────────────────────────────────
print()
REF_ACC = 96.19
if abs(acc - REF_ACC) < 0.5:
    print("✅ VERDICT: Checkpoint matches epoch-11 reference within 0.5% — "
          "chest_model_4class.pth is the GOOD model from train_full3.")
elif acc < 70.0:
    print("❌ VERDICT: Accuracy is severely degraded (<70%) — "
          "checkpoint is likely corrupted or is a very early biased epoch.")
elif acc < 90.0:
    print("⚠️  VERDICT: Accuracy is below expectation — "
          "may be an early epoch (e.g., epoch 1 from app.py run).")
else:
    print(f"⚠️  VERDICT: Accuracy {acc:.2f}% is plausible but doesn't closely "
          f"match the logged 96.19% — investigate further.")
