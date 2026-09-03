import os, sys, json, csv
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
rows = list(csv.DictReader(open(os.path.join(ROOT,"fig7_work","test_manifest.csv"),encoding="utf-8")))
by = {}
for r in rows: by.setdefault(os.path.basename(r["path"]), []).append(r)
names = sys.argv[2].split(",")
CELL, cols = 640, 3
paths=[]
for n in names:
    cand = by.get(n)
    if cand: paths.append((cand[0]["path"], f"{n} [{cand[0]['source']}]"))
    else:
        p = os.path.join(r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified\versions\1\tbx11k-simplified\images", n)
        paths.append((p, f"{n} [tbx11k]"))
nrows=(len(paths)+cols-1)//cols
sh=Image.new("RGB",(cols*CELL,nrows*(CELL+20)),"white"); dd=ImageDraw.Draw(sh)
for i,(p,lab) in enumerate(paths):
    im=Image.open(p).convert("L"); W,H=im.size
    dx,dy=int(round(W*0.08)),int(round(H*0.08))
    im=im.crop((dx,dy,W-dx,H-dy)).resize((CELL,CELL),Image.LANCZOS).convert("RGB")
    r_,c_=divmod(i,cols); sh.paste(im,(c_*CELL,r_*(CELL+20)))
    dd.text((c_*CELL+4,r_*(CELL+20)+CELL+4),lab,fill="black")
sh.save(sys.argv[1]); print("wrote",sys.argv[1])
