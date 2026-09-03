import os, sys, csv
from PIL import Image, ImageDraw
rows=list(csv.DictReader(open("fig7_work/test_manifest.csv",encoding="utf-8")))
by={os.path.basename(r["path"]):r for r in rows}
names=sys.argv[2].split("|")
CELL=880
sh=Image.new("RGB",(CELL*2,(CELL+20)*2),"white"); dd=ImageDraw.Draw(sh)
for i,n in enumerate(names):
    p=by[n]["path"]
    im=Image.open(p).convert("L"); W,H=im.size
    dx,dy=int(round(W*0.08)),int(round(H*0.08))
    im=im.crop((dx,dy,W-dx,H-dy)).resize((CELL,CELL),Image.LANCZOS).convert("RGB")
    r,c=divmod(i,2); sh.paste(im,(c*CELL,r*(CELL+20)))
    dd.text((c*CELL+4,r*(CELL+20)+CELL+4),n,fill="black")
sh.save(sys.argv[1]); print("ok",sys.argv[1])
