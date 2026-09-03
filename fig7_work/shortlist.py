import csv, os, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,"localisation_table.csv"),encoding="utf-8")))
def F(r,k,d=float("nan")):
    v=r.get(k,"");  return d if v in ("","nan") else float(v)
for r in rows:
    r["_il"]=F(r,"in_lung_frac"); r["_ar"]=F(r,"mask_area_frac")
    r["_cp"]=F(r,"compactness");  r["_sr"]=F(r,"share_right",0); r["_sl"]=F(r,"share_left",0)

def show(title, sel, n=12):
    print("\n=== "+title+f"  ({len(sel)} candidates) ===")
    for r in sel[:n]:
        print(f"  {r['key']:46s} {r['group'][:18]:18s} T={r['true'][:4]:4s} P={r['pred'][:4]:4s} "
              f"c={float(r['conf']):.2f} inlung={r['_il']:.2f} area={r['_ar']:.3f} "
              f"cmp={r['_cp']:.2f} side={r['side']:12s} dom={r['dominant']:12s} src={r['mask_source'][:4]}")

good=[r for r in rows if r["true"]==r["pred"] and r["_il"]>=0.80 and 0.015<=r["_ar"]<=0.14
      and r["_cp"]>=0.85 and r["side"]!="outside both" and r["outlined"]=="True"]
good.sort(key=lambda r:(-r["_il"], r["_ar"]))
show("GOOD: compact, inside lung, correct, outlined", good, 16)

b1=[r for r in rows if r["true"]==r["pred"] and r["_il"]<=0.30 and r["_ar"]>=0.02]
b1.sort(key=lambda r: r["_il"])
show("BAD-1: correct prediction, mask OUTSIDE lung tissue", b1, 16)

b2=[r for r in rows if r["_ar"]>=0.35 and r["_sr"]>0.15 and r["_sl"]>0.15]
b2.sort(key=lambda r:-r["_ar"])
show("BAD-2: diffuse, spans BOTH lungs", b2, 16)

b3=[r for r in rows if r["group"].startswith("pneu_as_normal")]
b3.sort(key=lambda r:-float(r["conf"]))
show("BAD-3: pneumonia predicted Normal (most confident first)", b3, 16)

b4=[r for r in rows if r["true"]==r["pred"] and r["_il"]>=0.6 and max(r["_sr"],r["_sl"])>=0.8]
b4.sort(key=lambda r:-max(r["_sr"],r["_sl"]))
show("BAD-4 pool: strongly one-sided masks (check against visible pathology)", b4, 16)
