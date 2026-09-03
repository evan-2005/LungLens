"""
Fig. 6 - the deployed LungLens interface.

Full width (7.16 in) at 400 dpi, white ground, serif captions, no shadows or
gradients. Panels come from fig7_work/ui_shots/, captured from the running app
by fig7_work/capture_ui.py (headless Chrome over CDP, light mode, one 800x1200
window for every shot, rendered at 2x device pixels and 3x for the two close-ups,
no browser chrome and no file paths or usernames in frame).

Two-by-two grid. Panel (b) is the largest: (a) is width-capped so that whatever
scale the rows settle at, (b) still covers more of the page.

Panels (c) and (d) are tight crops rather than whole windows. A whole-window
shot of the Uncertain state shrinks to about an inch across at this size and its
text stops being readable, which defeats the point of showing it; the crops put
the generated text at roughly 7 pt on the printed page.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "fig7_work", "ui_shots")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

PANELS = {
    "a": ("A_upload.png",  "(a) Upload state"),
    "b": ("B_result.png",  "(b) Confident correct prediction"),
    "c": ("C_summary.png", "(c) Clinical summary, close up"),
    "d": ("D_summary.png", "(d) Uncertain response"),
}
img = {k: Image.open(os.path.join(SHOTS, f)) for k, (f, _) in PANELS.items()}
asp = {k: im.size[0]/im.size[1] for k, im in img.items()}
for k in "abcd":
    print(f"  {PANELS[k][0]:18s} {img[k].size[0]}x{img[k].size[1]}  aspect {asp[k]:.3f}")

W_IN = 7.16
MARGIN, GAPX, GAPY = 0.03, 0.18, 0.24
TITLE = 0.155
ROW1_H, ROW2_H = 4.45, 1.95          # image heights of the two rows, inches

wB = asp["b"] * ROW1_H
areaB = wB * ROW1_H
# Cap (a) so it stays smaller than (b): w * w/asp_a < areaB  ->  w < sqrt(areaB*asp_a)
wA = min(0.95 * (areaB * asp["a"]) ** 0.5, (W_IN - 2*MARGIN - GAPX) - wB)
hA = wA / asp["a"]
wC, wD = asp["c"] * ROW2_H, asp["d"] * ROW2_H
H_IN = MARGIN + ROW2_H + TITLE + GAPY + ROW1_H + TITLE + MARGIN

print(f"  (a) {wA:.2f}x{hA:.2f}in  area {wA*hA:.1f}")
print(f"  (b) {wB:.2f}x{ROW1_H:.2f}in  area {areaB:.1f}   <- largest")
print(f"  (c) {wC:.2f}x{ROW2_H:.2f}in    (d) {wD:.2f}x{ROW2_H:.2f}in")
print(f"  figure {W_IN:.2f} x {H_IN:.2f} in")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
fig = plt.figure(figsize=(W_IN, H_IN), dpi=400)
fx, fy = 1.0/W_IN, 1.0/H_IN


def panel(key, x_in, y_in, w_in, h_in):
    ax = fig.add_axes([x_in*fx, y_in*fy, w_in*fx, h_in*fy])
    ax.imshow(img[key], interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("0.55")
    fig.text((x_in + w_in/2)*fx, (y_in + h_in + 0.030)*fy, PANELS[key][1],
             fontsize=7, ha="center", va="bottom", color="0.05")


row1_top = H_IN - MARGIN - TITLE
row1_x = (W_IN - (wA + GAPX + wB))/2
panel("a", row1_x, row1_top - hA, wA, hA)              # top-aligned with (b)
panel("b", row1_x + wA + GAPX, row1_top - ROW1_H, wB, ROW1_H)

row2_x = (W_IN - (wC + GAPX + wD))/2
panel("c", row2_x, MARGIN, wC, ROW2_H)
panel("d", row2_x + wC + GAPX, MARGIN, wD, ROW2_H)

out = os.path.join(OUT, "Fig6_interface.png")
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
