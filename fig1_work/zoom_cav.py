import os, sys, json, ast, csv
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
T = (r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified"
     r"\versions\1\tbx11k-simplified")
BORDER = 0.08
boxes = {}
with open(os.path.join(T, "data.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if r["image_type"] == "tb" and r["bbox"] not in ("", "none"):
            boxes.setdefault(r["fname"], []).append(ast.literal_eval(r["bbox"]))
rows = {r["fname"]: r for r in json.load(open(os.path.join(HERE, "tb_cavity.json")))}
sel = [rows[n] for n in sys.argv[2].split(",") if n in rows]
CELL, cols = 460, 4
nrows = (len(sel)+cols-1)//cols
sheet = Image.new("RGB", (cols*CELL, nrows*(CELL+22)), "white")
dd = ImageDraw.Draw(sheet)
for i, c in enumerate(sel):
    im = Image.open(c["path"]).convert("L")
    W, H = im.size
    dx, dy = int(round(W*BORDER)), int(round(H*BORDER))
    im = im.crop((dx, dy, W-dx, H-dy)).convert("RGB")
    d = ImageDraw.Draw(im)
    for b in boxes[c["fname"]]:
        d.rectangle([b["xmin"]-dx, b["ymin"]-dy, b["xmin"]+b["width"]-dx,
                     b["ymin"]+b["height"]-dy], outline=(255, 70, 70), width=3)
    if c["cavbox"]:
        x, y, w_, h_ = c["cavbox"]
        d.ellipse([x-dx-4, y-dy-4, x+w_-dx+4, y+h_-dy+4], outline=(60, 220, 90), width=3)
    im = im.resize((CELL, CELL), Image.LANCZOS)
    r_, c_ = divmod(i, cols)
    sheet.paste(im, (c_*CELL, r_*(CELL+22)))
    dd.text((c_*CELL+4, r_*(CELL+22)+CELL+5),
            f"{c['fname']} cav{c['cav']:.0f} test{int(c['in_test'])}", fill="black")
sheet.save(sys.argv[1]); print("wrote", sys.argv[1])
