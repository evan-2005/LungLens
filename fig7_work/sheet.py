"""Contact sheet of original|overlay pairs for visual selection."""
import os, sys, csv
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__)); OV=os.path.join(HERE,"overlays")
rows={r["key"]:r for r in csv.DictReader(open(os.path.join(HERE,"localisation_table.csv"),encoding="utf-8"))}
keys=sys.argv[2].split(",")
CELL=300; COLS=int(sys.argv[3]) if len(sys.argv)>3 else 4
def sq(im):
    w,h=im.size; s=min(w,h)
    return im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s)).resize((CELL,CELL),Image.LANCZOS)
n=len(keys); nr=(n+COLS-1)//COLS
sh=Image.new("RGB",(COLS*CELL*2, nr*(CELL+30)),"white"); d=ImageDraw.Draw(sh)
for i,k in enumerate(keys):
    r=rows[k]
    o=sq(Image.open(os.path.join(OV,f"{k}_orig.png")).convert("RGB"))
    v=sq(Image.open(os.path.join(OV,f"{k}_overlay.png")).convert("RGB"))
    rr,cc=divmod(i,COLS)
    sh.paste(o,(cc*CELL*2, rr*(CELL+30))); sh.paste(v,(cc*CELL*2+CELL, rr*(CELL+30)))
    lab=(f"{k} T={r['true']}/P={r['pred']} c={float(r['conf']):.2f}\n"
         f"inlung={r['in_lung_frac'][:4]} area={r['mask_area_frac'][:5]} side={r['side']}")
    d.text((cc*CELL*2+3, rr*(CELL+30)+CELL+2), lab, fill="black")
sh.save(sys.argv[1]); print("wrote",sys.argv[1])
