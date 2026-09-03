"""
Reproduce the exact test split the served checkpoint was scored on, then run
BOTH candidate checkpoints (working tree vs git HEAD) over it so we can tell
which one produced chest_classifier_metrics.json.

Writes fig7_work/test_manifest.csv for the winning checkpoint:
    path, source, true_idx, pred_idx, p_normal, p_pneu, p_tb, p_covid
"""
import os, sys, csv, json, datetime
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix

os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
IMG_SIZE, SEED = 224, 42
TEST_FRACTION = VAL_FRACTION = 0.15
NUM_SAMPLES = 32000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

from app import (collect_dataset, stratified_subsample, BorderCrop,
                 IMAGENET_MEAN, IMAGENET_STD)

val_tf = transforms.Compose([
    BorderCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class DS(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224), (0, 0, 0))
        return val_tf(im), self.labels[i]

class CNNModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.cnnmodel = models.densenet121(weights=None)
        self.cnnmodel.classifier = nn.Linear(self.cnnmodel.classifier.in_features, n)
    def forward(self, x): return self.cnnmodel(x)

print("collecting dataset...", flush=True)
paths, labels, groups, sources = collect_dataset()
print("  pool:", len(paths), flush=True)
paths, labels, groups, sources = stratified_subsample(
    paths, labels, groups, sources, NUM_SAMPLES, seed=SEED)
print("  after subsample:", len(paths), flush=True)

def gsplit(idx, test_size):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    a, b = next(gss.split(idx, [labels[i] for i in idx], [groups[i] for i in idx]))
    return [idx[i] for i in a], [idx[i] for i in b]

all_idx = list(range(len(paths)))
trainval_idx, test_idx = gsplit(all_idx, TEST_FRACTION)
train_idx, val_idx = gsplit(trainval_idx, VAL_FRACTION / (1 - TEST_FRACTION))
print(f"  train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)

te_paths  = [paths[i] for i in test_idx]
te_labels = [labels[i] for i in test_idx]
te_src    = [sources[i] for i in test_idx]

loader = DataLoader(DS(te_paths, te_labels), batch_size=64, shuffle=False,
                    num_workers=0, pin_memory=(device.type == "cuda"))

def score(ckpt_path):
    m = CNNModel(4)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    P, Y = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            pr = torch.softmax(m(imgs.to(device)), 1).cpu().numpy()
            P.append(pr); Y.extend(lbls.tolist())
    P = np.concatenate(P)
    pred = P.argmax(1)
    Y = np.array(Y)
    acc = 100.0 * (pred == Y).mean()
    f1 = 100.0 * f1_score(Y, pred, average="macro")
    return P, pred, Y, acc, f1

SCRATCH = sys.argv[1]
cands = {"working_tree": os.path.join(ROOT, "chest_model_4class.pth"),
         "git_HEAD":     os.path.join(SCRATCH, "ckpt_head.pth")}

results = {}
for name, p in cands.items():
    print(f"\nscoring {name} ...", flush=True)
    P, pred, Y, acc, f1 = score(p)
    cm = confusion_matrix(Y, pred, labels=[0,1,2,3])
    print(f"  acc={acc:.4f}%  macroF1={f1:.4f}%")
    print("  confusion matrix:\n", cm)
    results[name] = (P, pred, Y, acc, f1, cm)

REF = json.load(open(os.path.join(ROOT, "chest_classifier_metrics.json")))
ref_cm = np.array(REF["confusion_matrix"])
print("\nreference (chest_classifier_metrics.json):")
print(f"  acc={REF['test_acc']:.4f}%  macroF1={REF['test_f1']:.4f}%  n={ref_cm.sum()}")
print("  confusion matrix:\n", ref_cm)

winner = None
for name, (P, pred, Y, acc, f1, cm) in results.items():
    exact = np.array_equal(cm, ref_cm)
    close = abs(acc - REF["test_acc"]) < 0.01
    print(f"\n{name}: cm_exact_match={exact}  acc_match={close}")
    if exact:
        winner = name

print("\nWINNER:", winner)
if winner is None:
    print("NEITHER checkpoint reproduces the recorded confusion matrix.")
    sys.exit(2)

P, pred, Y, acc, f1, cm = results[winner]
out = os.path.join(HERE, "test_manifest.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["path","source","true_idx","pred_idx",
                "p_normal","p_pneumonia","p_tb","p_covid"])
    for i in range(len(te_paths)):
        w.writerow([te_paths[i], te_src[i], int(Y[i]), int(pred[i]),
                    f"{P[i,0]:.6f}", f"{P[i,1]:.6f}", f"{P[i,2]:.6f}", f"{P[i,3]:.6f}"])
json.dump({"winner": winner, "acc": acc, "f1": f1,
           "confusion_matrix": cm.tolist(), "n": int(len(te_paths))},
          open(os.path.join(HERE, "manifest_meta.json"), "w"), indent=2)
print("wrote", out, len(te_paths), "rows")
