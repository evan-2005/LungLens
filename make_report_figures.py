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
# Drawn with standard flowchart notation (process / predefined-process /
# decision / data-store / terminator shapes) in black ink on white, the way a
# pipeline is drawn by hand for a paper, rather than as a stack of uniformly
# coloured slide boxes. Every number is read from chest_classifier_metrics.json
# so the figure cannot drift from the served run.
from matplotlib.patches import Rectangle, Polygon, Ellipse, Circle

INK      = "#1a1a1a"
GREY_TXT = "#5a5a5a"
RULE     = "#c9c9c9"

seg_dice = M.get("seg_val_dice", 0.0)
seg_n    = M.get("seg_train_images", "?")
epoch, epochs_req = M.get("epoch", "?"), M.get("epochs_requested", "?")
n_train = M.get("train_samples", 0)
n_val   = M.get("val_samples", 0)
n_test  = M.get("test_samples", 0)
n_total = n_train + n_val + n_test

def section_rule(ax, x, y, w, label_):
    ax.text(x, y, label_, fontsize=11.5, fontweight="bold", color=INK, family="sans-serif")
    ax.plot([x, x + w], [y - 0.016, y - 0.016], color=RULE, lw=1.0)

_step = [0]
def tag(ax, x, y):
    """Small numbered marker at a shape's corner, the way a paper figure
    numbers its own stages for reference in the body text."""
    _step[0] += 1
    ax.add_patch(Circle((x, y), 0.0105, fc="white", ec=INK, lw=0.8, zorder=5))
    ax.text(x, y, str(_step[0]), ha="center", va="center", fontsize=6.3, zorder=6)

def label(ax, cx, y, h, title, note, fs=8.8, bold=False):
    ty = y + h - 0.028 if note else y + h / 2
    ax.text(cx, ty, title, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.2, wrap=True)
    if note:
        ax.text(cx, y + 0.022, note, ha="center", va="center", fontsize=6.8,
                color=GREY_TXT, family="monospace", linespacing=1.2)

def rect(ax, x, y, w, h, title, note=None, numbered=True):
    """Plain flowchart process box."""
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=1.05))
    if numbered:
        tag(ax, x + 0.008, y + h - 0.008)
    label(ax, x + w / 2, y, h, title, note)

def predef(ax, x, y, w, h, title, note=None):
    """Flowchart 'predefined process': double side-bars mark a step that is
    itself a whole subsystem, used only for the two trained networks."""
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=1.3))
    inset = 0.010
    ax.plot([x + inset, x + inset], [y, y + h], color=INK, lw=0.9)
    ax.plot([x + w - inset, x + w - inset], [y, y + h], color=INK, lw=0.9)
    tag(ax, x + 0.008, y + h - 0.008)
    label(ax, x + w / 2, y, h, title, note, fs=9.0, bold=True)

def cyl(ax, x, y, w, h, title, note):
    """Flowchart data-store cylinder, used once for the raw dataset."""
    eh = h * 0.26
    ax.plot([x, x], [y + eh / 2, y + h - eh / 2], color=INK, lw=1.05)
    ax.plot([x + w, x + w], [y + eh / 2, y + h - eh / 2], color=INK, lw=1.05)
    ax.add_patch(Rectangle((x, y + eh / 2), w, h - eh, fc="white", ec="none", zorder=1))
    ax.add_patch(Ellipse((x + w / 2, y + eh / 2), w, eh, fc="white", ec=INK, lw=1.05, zorder=2))
    ax.add_patch(Ellipse((x + w / 2, y + h - eh / 2), w, eh, fc="white", ec=INK, lw=1.05, zorder=3))
    tag(ax, x + 0.01, y + h - eh / 2 - 0.006)
    # Single-line title and note, positioned clear of both ellipse curves. A
    # two-line title in this box height collided with either the note below
    # or the top ellipse above; one line each fits the available band cleanly.
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=8.6)
    ax.text(x + w / 2, y + h * 0.33, note, ha="center", va="center", fontsize=6.8,
            color=GREY_TXT, family="monospace")

def parallelogram(ax, x, y, w, h, title, note=None, slant=0.026):
    pts = [(x + slant, y), (x + w, y), (x + w - slant, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=INK, lw=1.05))
    tag(ax, x + slant + 0.012, y + h - 0.008)
    label(ax, x + w / 2, y, h, title, note)

def diamond(ax, cx, cy, w, h, title):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=INK, lw=1.05))
    # Placed just inside the top vertex rather than above it, so the tag reads
    # as unambiguously belonging to the diamond even when another shape (the
    # "yes" branch box) sits directly above with only a small gap.
    tag(ax, cx, cy + h / 2 - 0.026)
    ax.text(cx, cy, title, ha="center", va="center", fontsize=7.3, linespacing=1.1, wrap=True)

def fileicon(ax, x, y, w, h, filename):
    """A folded-corner document icon for the two on-disk checkpoints, so a
    trained artefact reads as a concrete file rather than an abstract label."""
    fold = 0.02
    pts = [(x, y), (x + w, y), (x + w, y + h - fold), (x + w - fold, y + h), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=INK, lw=1.05))
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w, y + h - fold), (x + w - fold, y + h - fold)],
                         closed=True, fc="#e6e6e6", ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, filename, ha="center", va="center",
            fontsize=6.6, family="monospace")

def go(ax, x1, y1, x2, y2, label_=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK,
                                mutation_scale=7.5, shrinkA=0, shrinkB=0))
    if label_:
        ax.text((x1 + x2) / 2 + 0.012, (y1 + y2) / 2, label_, fontsize=6.8,
                color=GREY_TXT, ha="left", va="center", style="italic")

def elbow(ax, x1, y1, x2, y2, bend=None, label_=None):
    ymid = bend if bend is not None else (y1 + y2) / 2
    ax.plot([x1, x1], [y1, ymid], color=INK, lw=1.0)
    ax.plot([x1, x2], [ymid, ymid], color=INK, lw=1.0)
    if label_:
        ax.text((x1 + x2) / 2, ymid + 0.012, label_, fontsize=6.6, color=GREY_TXT,
                ha="center", style="italic")
    go(ax, x2, ymid, x2, y2)

def row_var(widths, gap, x0=0.015, x1=0.985):
    """Space boxes proportional to the given (hand-chosen, unequal) widths,
    scaled to exactly fill x0..x1 so nothing overflows the frame."""
    scale = (x1 - x0 - (len(widths) - 1) * gap) / sum(widths)
    ws = [w_ * scale for w_ in widths]
    xs_, cur = [], x0
    for w_ in ws:
        xs_.append(cur); cur += w_ + gap
    return xs_, ws

fig, ax = plt.subplots(figsize=(13.2, 10.0), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---- Stage 1 ----
section_rule(ax, 0.015, 0.975, 0.97, "STAGE 1: TRAIN THE SERVED CLASSIFIER")
h1, y1 = 0.125, 0.815
xs1, ws1 = row_var([0.15, 0.21, 0.15, 0.19, 0.21], 0.018)
cyl(ax, xs1[0], y1, ws1[0], h1, "Public CXR sources", f"8 sources, n={n_total:,}")
rect(ax, xs1[1], y1, ws1[1], h1, "Dedup + label + split", "content-hash dedup\nsorted, patient-grouped")
rect(ax, xs1[2], y1, ws1[2], h1, "Preprocess", "224px, crop 8%\naugment")
predef(ax, xs1[3], y1, ws1[3], h1, "DenseNet-121", "class-weighted CE")
rect(ax, xs1[4], y1, ws1[4], h1, "Select checkpoint", f"best val macro-F1\nepoch {epoch}/{epochs_req}")
for i in range(4):
    go(ax, xs1[i] + ws1[i], y1 + h1 / 2, xs1[i + 1], y1 + h1 / 2)

fx, fy, fw, fh = xs1[3] + ws1[3] / 2 - 0.075, 0.665, 0.15, 0.075
elbow(ax, xs1[3] + ws1[3] / 2, y1, fx + fw / 2, fy + fh)
fileicon(ax, fx, fy, fw, fh, "chest_model_4class.pth")

# ---- Stage 2 ----
section_rule(ax, 0.015, 0.615, 0.97,
             "STAGE 2: DISTIL GRAD-CAM INTO A DISEASE-SEGMENTATION HEAD")
h2, y2 = 0.125, 0.44
xs2, ws2 = row_var([0.32, 0.32, 0.32], 0.022)
# Explicit bend at 0.585: strictly between the rule at 0.599 and the box top
# at 0.565, so the jog passes under the header instead of through its text.
elbow(ax, fx + fw / 2, fy, xs2[0] + ws2[0] / 2, y2 + h2, bend=0.585)
rect(ax, xs2[0], y2, ws2[0], h2, "Grad-CAM of classifier",
     f"n={seg_n:,} images\ndenseblock4.denselayer16.conv2")
rect(ax, xs2[1], y2, ws2[1], h2, "Soft target masks",
     "continuous 0-1 heatmap\ndisease-focused, not anatomy")
predef(ax, xs2[2], y2, ws2[2], h2, "U-Net segmentation head",
       f"Dice-BCE loss, val Dice {seg_dice:.3f}")
go(ax, xs2[0] + ws2[0], y2 + h2 / 2, xs2[1], y2 + h2 / 2)
go(ax, xs2[1] + ws2[1], y2 + h2 / 2, xs2[2], y2 + h2 / 2)

# ---- Inference ----
section_rule(ax, 0.015, 0.40, 0.97, "INFERENCE: GRADIO WEB APP, CPU ONLY")
ymain, hmain = 0.155, 0.11
cy = ymain + hmain / 2
xs3, ws3 = row_var([0.15, 0.205, 0.15, 0.205, 0.175], 0.02)
parallelogram(ax, xs3[0], ymain, ws3[0], hmain, "Upload CXR")
predef(ax, xs3[1], ymain, ws3[1], hmain, "DenseNet-121 forward pass",
       "class + confidence\n<60% -> Uncertain")
go(ax, xs3[0] + ws3[0], cy, xs3[1], cy)

# Diamond geometry computed before the arrow that feeds it, so the arrow can
# target its exact left vertex instead of a guessed point that overshot into
# the diamond's interior.
dcx, dw, dh = xs3[2] + ws3[2] / 2, ws3[2] * 0.95, 0.155
go(ax, xs3[1] + ws3[1], cy, dcx - dw / 2, cy)
diamond(ax, dcx, cy, dw, dh, "channel\nconfident?")

ub_x, ub_y, ub_w, ub_h = dcx - 0.09, 0.305, 0.18, 0.075
db_x, db_y, db_w, db_h = dcx - 0.09, 0.02, 0.18, 0.075
# Both branches run straight up/down at x=dcx, so a plain arrow is used
# instead of elbow(): elbow's forced jog overshot past the target and back,
# which drew a short zigzag straight through the box's note text.
go(ax, dcx, cy + dh / 2, dcx, ub_y)
ax.text(dcx + 0.02, (cy + dh / 2 + ub_y) / 2, "yes", fontsize=6.8, color=GREY_TXT, style="italic")
rect(ax, ub_x, ub_y, ub_w, ub_h, "U-Net disease mask", "distilled channel")
go(ax, dcx, cy - dh / 2, dcx, db_y + db_h)
ax.text(dcx + 0.02, (cy - dh / 2 + db_y + db_h) / 2, "no", fontsize=6.8, color=GREY_TXT, style="italic")
rect(ax, db_x, db_y, db_w, db_h, "Live Grad-CAM", "classifier backward pass")

rect(ax, xs3[3], ymain, ws3[3], hmain, "Overlay", "colour map + region outline")
elbow(ax, ub_x + ub_w, ub_y + ub_h / 2, xs3[3] + ws3[3] / 2, ymain + hmain, bend=ub_y + ub_h / 2)
elbow(ax, db_x + db_w, db_y + db_h / 2, xs3[3] + ws3[3] / 2, ymain, bend=db_y + db_h / 2)

parallelogram(ax, xs3[4], ymain, ws3[4], hmain, "Result", "class + clinical\nsummary")
go(ax, xs3[3] + ws3[3], cy, xs3[4], cy)

fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nWrote figures to {OUT}")
