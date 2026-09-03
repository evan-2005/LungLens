"""
Fig. 1 - one representative radiograph per class.

Single column (3.4 in) at 400 dpi, greyscale, white ground, serif type, no
shadows or gradients. Each panel is one film from a DIFFERENT source dataset,
centre-cropped to square after the same BorderCrop(0.08) the served model
applies, so the reader sees what the classifier sees.

Panels were chosen against the radiological finding the paper's introduction
claims, and screened to exclude burned-in text, laterality markers, PORTABLE
stamps and visible support devices.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
BORDER = 0.08
KAG = r"C:\Users\evanl\.cache\kagglehub\datasets"

PANELS = [
    ("Normal",
     os.path.join(KAG, r"tawsifurrahman\tuberculosis-tb-chest-xray-dataset\versions\3\TB_Chest_Radiography_Database\Normal\Normal-2343.png"),
     "clear, well-aerated\nlung fields"),
    ("Pneumonia",
     os.path.join(KAG, r"pcbreviglieri\pneumonia-xray-images\versions\1\train\opacity\person22_bacteria_77.jpeg"),
     "left mid-lower zone\nconsolidation"),
    ("Tuberculosis",
     os.path.join(KAG, r"vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\images\tb1003.png"),
     "left upper-lobe\ncavitation"),
    ("Covid-19",
     os.path.join(KAG, r"tawsifurrahman\covid19-radiography-database\versions\5\COVID-19_Radiography_Dataset\COVID\images\COVID-1651.png"),
     "bilateral peripheral\nground-glass opacity"),
]

def prepare(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    dx, dy = int(round(w*BORDER)), int(round(h*BORDER))
    if w-2*dx > 0 and h-2*dy > 0:
        im = im.crop((dx, dy, w-dx, h-dy))
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    return np.array(im.resize((600, 600), Image.LANCZOS))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

# Panels are square, so the figure height is derived from the panel width
# rather than guessed: anything else letterboxes the films and opens gaps.
W_IN = 3.4
GAP_IN, MARGIN_IN = 0.028, 0.017
TITLE_IN, NOTE_IN = 0.150, 0.245
pw_in = (W_IN - 2*MARGIN_IN - 3*GAP_IN)/4
H_IN = pw_in + TITLE_IN + NOTE_IN

fig = plt.figure(figsize=(W_IN, H_IN), dpi=400)
gap, left = GAP_IN/W_IN, MARGIN_IN/W_IN
pw = pw_in/W_IN
bottom, ph = NOTE_IN/H_IN, pw_in/H_IN

for i, (name, path, note) in enumerate(PANELS):
    ax = fig.add_axes([left + i*(pw+gap), bottom, pw, ph])
    ax.imshow(prepare(path), cmap="gray", vmin=0, vmax=255,
              interpolation="lanczos", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color("0.25")
    ax.set_title(name, fontsize=7, pad=1.6)
    ax.text(0.5, -0.045, note, transform=ax.transAxes, fontsize=6,
            ha="center", va="top", linespacing=1.15, color="0.15")

out = os.path.join(OUT, "Fig1_class_samples.png")
fig.savefig(out, dpi=400, facecolor="white")
plt.close(fig)
# Flatten to RGB. Matplotlib writes RGBA even on an opaque white ground, and an
# alpha channel is an avoidable surprise in a LaTeX or Word submission pipeline.
_im = Image.open(out)
if _im.mode != "RGB":
    _bg = Image.new("RGB", _im.size, "white")
    _bg.paste(_im, mask=_im.split()[-1] if _im.mode == "RGBA" else None)
    _bg.save(out, dpi=(400, 400))

im = Image.open(out)
print("wrote", out, im.size, f"({im.size[0]/400:.2f} x {im.size[1]/400:.2f} in @400dpi)")
