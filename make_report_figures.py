"""
make_report_figures.py
======================
Regenerate every numeric figure in the Part II report directly from
chest_classifier_metrics.json (written by app.py's training run) and
train_full3.log, so no figure can drift from the checkpoint that is served.

Outputs into report_figures/:
    fig1_pipeline.png          Pipeline diagram (8-source dataset)
    fig2_confusion_matrix.png  Held-out test confusion matrix (n = 5,637)
    fig3_per_class.png         Per-class recall and precision
    fig4_per_source.png        Accuracy by source dataset (7 sources)
    fig6_training_curves.png   Train/val accuracy and val macro-F1 per epoch
    table3_per_class.csv       Per-class table (precision/recall/F1/spec/support)

Run:
    python make_report_figures.py
"""

import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_figures")
os.makedirs(OUT, exist_ok=True)

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]

with open(os.path.join(HERE, "chest_classifier_metrics.json"), encoding="utf-8") as fh:
    M = json.load(fh)

cm = np.array(M["confusion_matrix"], dtype=int)
N = int(cm.sum())

# ── Per-class metrics derived from the confusion matrix ──────────────────────
tp = np.diag(cm).astype(float)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = N - tp - fp - fn
prec = tp / np.maximum(tp + fp, 1)
rec = tp / np.maximum(tp + fn, 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
spec = tn / np.maximum(tn + fp, 1)
supp = cm.sum(axis=1)
acc = tp.sum() / N

print(f"n = {N}  accuracy = {100*acc:.2f}%  macro-F1 = {100*f1.mean():.2f}%")
for i, c in enumerate(CLASSES):
    print(f"  {c:14s} P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} "
          f"Spec={spec[i]:.3f} n={supp[i]}")
print(f"  {'Macro avg.':14s} P={prec.mean():.3f} R={rec.mean():.3f} "
      f"F1={f1.mean():.3f} Spec={spec.mean():.3f} n={N}")

with open(os.path.join(OUT, "table3_per_class.csv"), "w", encoding="utf-8") as fh:
    fh.write("Class,Prec.,Recall,F1,Spec.,Supp.\n")
    for i, c in enumerate(CLASSES):
        fh.write(f"{c},{prec[i]:.3f},{rec[i]:.3f},{f1[i]:.3f},{spec[i]:.3f},{supp[i]}\n")
    fh.write(f"Macro avg.,{prec.mean():.3f},{rec.mean():.3f},{f1.mean():.3f},"
             f"{spec.mean():.3f},{N}\n")

# ── Fig 2: confusion matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200)
im = ax.imshow(cm, cmap="Blues", vmax=cm.max())
ax.set_title(f"DenseNet-121 confusion matrix (held-out test, n={N:,})", fontsize=12)
ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
thr = cm.max() / 2.0
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > thr else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_confusion_matrix.png"), bbox_inches="tight")
plt.close(fig)

# ── Fig 3: per-class recall and precision ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w/2, rec, w, label="Recall (sensitivity)", color="#1b5e8a")
b2 = ax.bar(x + w/2, prec, w, label="Precision", color="#7fb3d5")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
            f"{b.get_height():.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.set_title("Per-class recall and precision (held-out test)")
ax.legend(loc="lower left", framealpha=0.95)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_per_class.png"), bbox_inches="tight")
plt.close(fig)

# ── Fig 4: per-source accuracy ───────────────────────────────────────────────
ps = M["test_per_source_acc"]
order = sorted(ps, key=lambda k: ps[k]["n"])          # smallest at bottom
vals = [ps[k]["acc"] / 100.0 for k in order]
ns = [ps[k]["n"] for k in order]
fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
bars = ax.barh(range(len(order)), vals, color="#c0632a", height=0.68)
for b, v, n in zip(bars, vals, ns):
    ax.text(v + 0.008, b.get_y() + b.get_height()/2,
            f"{v:.3f} (n={n:,})", va="center", fontsize=9)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=9.5)
ax.set_xlim(0, 1.22); ax.set_xlabel("Accuracy")
ax.set_title("Accuracy by source dataset\n(confounding probe)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_per_source.png"), bbox_inches="tight")
plt.close(fig)

# ── Fig 6: training curves from the real log ─────────────────────────────────
# Pick the most recent train*.log rather than a fixed name, so the curves always
# describe the same run as the metrics file instead of silently plotting an older
# one after a retrain.
import glob as _glob
_logs = sorted(_glob.glob(os.path.join(HERE, "train*.log")), key=os.path.getmtime)
log = _logs[-1] if _logs else os.path.join(HERE, "train_full3.log")
if os.path.exists(log):
    print(f"training curves from: {os.path.basename(log)}")
    ep, tr, va, vf = [], [], [], []
    pat = re.compile(r"^Epoch (\d+)/\d+ \| Train Acc: ([\d.]+)% \| Val Acc: "
                     r"([\d.]+)% \| Val macro-F1: ([\d.]+)%")
    for line in open(log, encoding="utf-8", errors="ignore"):
        m = pat.match(line)
        if m:
            ep.append(int(m.group(1))); tr.append(float(m.group(2)))
            va.append(float(m.group(3))); vf.append(float(m.group(4)))
    if ep:
        best = int(np.argmax(vf))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
        axes[0].plot(ep, tr, "o-", color="#1f77b4", label="Train acc.")
        axes[0].plot(ep, va, "o-", color="#d62728", label="Val acc.")
        axes[0].axvline(ep[best], ls="--", c="grey", lw=1)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Classification accuracy"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(ep, vf, "o-", color="#2ca02c", label="Val macro-F1")
        axes[1].axvline(ep[best], ls="--", c="grey", lw=1)
        axes[1].annotate(f"best epoch {ep[best]}\n{vf[best]:.2f}%",
                         (ep[best], vf[best]), textcoords="offset points",
                         xytext=(-96, -30), fontsize=9)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1 (%)")
        axes[1].set_title("Validation macro-F1"); axes[1].legend(); axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig6_training_curves.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"training curves: {len(ep)} epochs, best epoch {ep[best]}")

# ── Fig 1: pipeline diagram ──────────────────────────────────────────────────
BLUE_D, BLUE_L = "#1b5e8a", "#dbe9f4"
BRN_D, BRN_L, BRN_M = "#8c3a12", "#f0e0d6", "#d9b8a5"
GREY = "#eef1f4"

def box(ax, x, y, w, h, text, fc, ec, tc="black", bold=False, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.01",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x1, y1, x2, y2, c="#4a5a68", label=None, fs=8.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color=c))
    if label:
        ax.text((x1+x2)/2 + 0.012, (y1+y2)/2, label, fontsize=fs, color=c,
                ha="left", va="center", style="italic")

fig, ax = plt.subplots(figsize=(13.2, 7.4), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.02, 0.965, "Stage 1: Train the served classifier", fontsize=14,
        fontweight="bold", color=BLUE_D)
w, h, y = 0.168, 0.135, 0.775
xs = [0.02, 0.216, 0.412, 0.608, 0.804]
box(ax, xs[0], y, w, h, "8 public CXR sources\n(TB · Pneumonia ·\nCOVID · Normal)\n37,553 unique images", BLUE_L, BLUE_D)
box(ax, xs[1], y, w, h, "Content-hash dedup +\nfolder-relative labels +\npatient-grouped\nstratified split", BLUE_L, BLUE_D)
box(ax, xs[2], y, w, h, "Border crop 8% ·\nresize 224 · crop ·\nflip · rotate ·\njitter · blur", BLUE_L, BLUE_D)
box(ax, xs[3], y, w, h, "DenseNet-121\nclassifier\n(ImageNet init)", BLUE_D, BLUE_D, "white", True, 11)
box(ax, xs[4], y, w, h, "Select on val\nmacro-F1 +\nearly stopping\n(best: epoch 11)", BLUE_L, BLUE_D)
for i in range(4):
    arrow(ax, xs[i] + w + 0.004, y + h/2, xs[i+1] - 0.004, y + h/2)
arrow(ax, xs[3] + w/2, y - 0.006, xs[3] + w/2, 0.585, BRN_D, "trained\nweights")

ax.text(0.02, 0.545, "Stage 2: Distil Grad-CAM into a disease-segmentation head",
        fontsize=14, fontweight="bold", color=BRN_D)
y2, w2 = 0.335, 0.215
x2 = [0.135, 0.395, 0.655]
box(ax, x2[0], y2, w2, 0.145, "Grad-CAM of the trained\nDenseNet (3,000 images,\ndenseblock4.denselayer16)", BRN_L, BRN_D)
box(ax, x2[1], y2, w2, 0.145, "Soft target masks\n(disease-focused,\nnot anatomy)", BRN_M, BRN_D)
box(ax, x2[2], y2, w2, 0.145, "U-Net segmentation head\n(distillation, Dice-BCE)\nval Dice 0.162", BRN_D, BRN_D, "white", True, 10)
arrow(ax, x2[0] + w2 + 0.004, y2 + 0.0725, x2[1] - 0.004, y2 + 0.0725, BRN_D)
arrow(ax, x2[1] + w2 + 0.004, y2 + 0.0725, x2[2] - 0.004, y2 + 0.0725, BRN_D)

ax.text(0.02, 0.275, "Inference (Gradio web app, CPU)", fontsize=14, fontweight="bold")
y3, w3 = 0.055, 0.205
x3 = [0.02, 0.265, 0.51, 0.755]
box(ax, x3[0], y3, w3, 0.145, "Upload CXR", GREY, "#5a6b78")
box(ax, x3[1], y3, w3, 0.145, "DenseNet-121\n→ class + confidence\n(<60% → Uncertain)", BLUE_D, BLUE_D, "white", True, 10)
box(ax, x3[2], y3, w3, 0.145, "Overlay: Grad-CAM or\ndistilled U-Net\ndisease mask", BRN_D, BRN_D, "white", True, 10)
box(ax, x3[3], y3, w3, 0.145, "Predicted class +\nclinical-language\nsummary", GREY, "#5a6b78")
for i in range(3):
    arrow(ax, x3[i] + w3 + 0.004, y3 + 0.0725, x3[i+1] - 0.004, y3 + 0.0725)
ax.text(x3[1] + w3/2, y3 + 0.155, "uses trained classifier", fontsize=9,
        ha="center", style="italic", color=BLUE_D)
ax.text(x3[2] + w3/2, y3 + 0.155, "uses distilled seg head", fontsize=9,
        ha="center", style="italic", color=BRN_D)

fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nWrote figures to {OUT}")
