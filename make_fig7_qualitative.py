"""
Fig. 7 - good and bad qualitative results.

Full width (7.16 in) at 400 dpi, white ground, serif type, no shadows or
gradients. Two groups: "Correctly localised" (green) and "Failure cases" (red).
Each case shows the input radiograph above the overlay the served app produces,
with the true label, the predicted label and what the overlay actually did.

Panels come straight from fig7_work/overlays/, which were written by
fig7_work/gen_overlays.py calling app.predict_image() - the same function the
Gradio button calls - so nothing here re-renders or re-thresholds the overlay.

Cases were chosen from the scored pool in fig7_work/localisation_table.csv
(245 test films: all 102 pneumonia-predicted-Normal errors, all 104 Montgomery
and Shenzhen test films, and 40 correct films stratified by class).
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OV = os.path.join(HERE, "fig7_work", "overlays")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

GREEN, RED = "#1B7F3B", "#B3261E"

GOOD = [
    ("235_COVID-3348",
     "true Covid-19\npred Covid-19 (0.99)",
     "outlines both upper zones;\n90% of region inside lung"),
    ("244_COVID-2706",
     "true Covid-19\npred Covid-19 (1.00)",
     "one compact focus on the\nleft mid-zone opacity"),
    ("167_CHNCXR_0520_1",
     "true Tuberculosis\npred Tuberculosis (0.99)",
     "covers the right mid-zone\nconsolidation; 81% in lung"),
]
BAD = [
    ("231_tb0366",
     "true Tuberculosis\npred Tuberculosis (0.85)",
     "right answer, wrong evidence:\n97% of region outside the lung"),
    ("192_CHNCXR_0658_1",
     "true Tuberculosis\npred Tuberculosis (1.00)",
     "diffuse over both lungs and\nchest wall; 86% of the image"),
    ("049_Lung_Opacity-4010",
     "true Pneumonia\npred Normal (1.00)",
     "clear lung fields; no region\nsurvives the overlay threshold"),
]

def square(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s))
    return np.array(im.resize((520, 520), Image.LANCZOS))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

W_IN = 7.16
MARGIN = 0.02          # outer margin, inches
LEFTPAD = 0.115        # strip on the left for the "input"/"overlay" row labels
COLGAP = 0.045         # gap between cases inside a group
GROUPGAP = 0.22        # gap between the two groups
ROWGAP = 0.035         # gap between the input row and the overlay row
HEADER = 0.24          # group title band
CAPTION = 0.50         # caption block under each case

pw = (W_IN - MARGIN - LEFTPAD - GROUPGAP - 4*COLGAP)/6
H_IN = HEADER + 2*pw + ROWGAP + CAPTION + MARGIN

fig = plt.figure(figsize=(W_IN, H_IN), dpi=400)
fx, fy = 1.0/W_IN, 1.0/H_IN

def place(x_in, y_in, w_in, h_in):
    return fig.add_axes([x_in*fx, y_in*fy, w_in*fx, h_in*fy])

y_over = MARGIN + CAPTION                 # overlay row bottom
y_in_  = y_over + pw + ROWGAP             # input row bottom
y_hdr  = y_in_ + pw + 0.045

groups = [(GOOD, GREEN, "Correctly localised", LEFTPAD),
          (BAD,  RED,   "Failure cases",
           LEFTPAD + 3*pw + 2*COLGAP + GROUPGAP)]

for cases, colour, title, x0 in groups:
    gw = 3*pw + 2*COLGAP
    fig.text((x0 + gw/2)*fx, (y_hdr + 0.055)*fy, title, fontsize=8,
             ha="center", va="bottom", color=colour)
    # thin rule under the group title, in the group's colour
    fig.add_artist(plt.Line2D([x0*fx, (x0+gw)*fx],
                              [(y_hdr + 0.035)*fy]*2,
                              lw=0.7, color=colour, transform=fig.transFigure))
    for j, (key, head, note) in enumerate(cases):
        x = x0 + j*(pw + COLGAP)
        for y, suffix in ((y_in_, "_orig"), (y_over, "_overlay")):
            ax = place(x, y, pw, pw)
            ax.imshow(square(os.path.join(OV, key + suffix + ".png")),
                      interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.9); sp.set_color(colour)
        fig.text((x + pw/2)*fx, (y_over - 0.035)*fy, head, fontsize=6,
                 ha="center", va="top", linespacing=1.2, color="0.05")
        fig.text((x + pw/2)*fx, (y_over - 0.035 - 0.155)*fy, note, fontsize=5.6,
                 ha="center", va="top", linespacing=1.2, color="0.3", style="italic")

# row labels on the far left of each image row
for y, lab in ((y_in_, "input"), (y_over, "overlay")):
    fig.text(0.30*LEFTPAD*fx, (y + pw/2)*fy, lab, fontsize=5.6, rotation=90,
             ha="left", va="center", color="0.45")

out = os.path.join(OUT, "Fig7_qualitative.png")
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
