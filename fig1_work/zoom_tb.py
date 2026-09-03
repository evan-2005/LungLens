import os, sys, json, ast, csv
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
T = (r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified"
     r"\versions\1\tbx11k-simplified")
BORDER = 0.08
boxes = {}
with open(os.path.join(T, "data.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if r["image_type"] == "tb" and r["bbox"] not in ("", "none"):
            boxes.setdefault(r["fname"], []).append(ast.literal_eval(r["bbox"]))

cands = json.load(open(os.path.join(HERE, "tb_cands.json")))
names = sys.argv[2].split(",")
CELL = 460
sel = [c for c in cands if c["fname"] in names]
sel.sort(key=lambda c: names.index(c["fname"]))
cols = 4
rows = (len(sel)+cols-1)//cols
sheet = Image.new("RGB", (cols*CELL, rows*(CELL+22)), "white")
dd = ImageDraw.Draw(sheet)
for i, c in enumerate(sel):
    im = Image.open(c["path"]).convert("L")
    W, H = im.size
    dx, dy = int(round(W*BORDER)), int(round(H*BORDER))
    im = im.crop((dx, dy, W-dx, H-dy)).convert("RGB")
    cw, ch = im.size
    d = ImageDraw.Draw(im)
    for b in boxes[c["fname"]]:
        d.rectangle([b["xmin"]-dx, b["ymin"]-dy,
                     b["xmin"]+b["width"]-dx, b["ymin"]+b["height"]-dy],
                    outline=(255, 60, 60), width=3)
    im = im.resize((CELL, CELL), Image.LANCZOS)
    r_, c_ = divmod(i, cols)
    sheet.paste(im, (c_*CELL, r_*(CELL+22)))
    dd.text((c_*CELL+4, r_*(CELL+22)+CELL+5), c["fname"], fill="black")
sheet.save(sys.argv[1])
print("wrote", sys.argv[1])
