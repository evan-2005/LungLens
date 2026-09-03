import csv, os, collections
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,"localisation_table.csv"),encoding="utf-8")))
def F(r,k):
    v=r.get(k,""); return float("nan") if v in ("","nan") else float(v)
for r in rows:
    r["_il"]=F(r,"in_lung_frac"); r["_ar"]=F(r,"mask_area_frac"); r["_cp"]=F(r,"compactness")
    r["_sr"]=F(r,"share_right"); r["_sl"]=F(r,"share_left")

def block(name, sub):
    n=len(sub)
    empty=[r for r in sub if r["side"]=="none"]
    nz=[r for r in sub if r["side"]!="none"]
    il=np.array([r["_il"] for r in nz]); ar=np.array([r["_ar"] for r in nz])
    out_maj = sum(1 for r in nz if r["_il"]<0.5)
    out_mostly = sum(1 for r in nz if r["_il"]<0.25)
    diffuse = sum(1 for r in nz if r["_ar"]>0.25)
    vdiffuse= sum(1 for r in nz if r["_ar"]>0.50)
    both = sum(1 for r in nz if r["_sr"]>0.15 and r["_sl"]>0.15)
    cent = collections.Counter(r["side"] for r in nz)
    print(f"\n--- {name}  (n={n}) ---")
    print(f"  no region above threshold      : {len(empty):3d} ({100*len(empty)/n:4.1f}%)")
    if not nz: return
    print(f"  in-lung fraction  median/mean  : {np.median(il):.3f} / {il.mean():.3f}")
    print(f"  majority outside lung (<50%)   : {out_maj:3d}/{len(nz)} ({100*out_maj/len(nz):4.1f}%)")
    print(f"  almost all outside lung (<25%) : {out_mostly:3d}/{len(nz)} ({100*out_mostly/len(nz):4.1f}%)")
    print(f"  mask area median/mean          : {np.median(ar):.3f} / {ar.mean():.3f} of the image")
    print(f"  diffuse  (>25% of image)       : {diffuse:3d}/{len(nz)} ({100*diffuse/len(nz):4.1f}%)")
    print(f"  very diffuse (>50% of image)   : {vdiffuse:3d}/{len(nz)} ({100*vdiffuse/len(nz):4.1f}%)")
    print(f"  spans BOTH lungs (>15% each)   : {both:3d}/{len(nz)} ({100*both/len(nz):4.1f}%)")
    print(f"  centroid  right/left/outside   : {cent['right lung']}/{cent['left lung']}/{cent['outside both']}")

print("="*74)
print("LOCALISATION SUMMARY - 245 scored test films")
print("Lung field: U-Net trained on the 21,165 COVID-19 Radiography Database lung")
print("masks (val Dice 0.9823), test-split films excluded from its training.")
print("Region: the app's own clean_region() applied to the app's own floored map,")
print("at the app's own threshold (0.50 U-Net mask / 0.45 Grad-CAM).")
print("="*74)
block("ALL", rows)
block("Pneumonia predicted Normal", [r for r in rows if r["group"].startswith("pneu_as_normal")])
block("Shenzhen TB (all test films)", [r for r in rows if "shenzhen_tb" in r["group"]])
block("Montgomery TB (all test films)", [r for r in rows if "montgomery_tb" in r["group"]])
block("Correct, random stratified 40", [r for r in rows if "correct_sample" in r["group"]])

print("\n--- Shenzhen/Montgomery split by label ---")
for src in ("shenzhen_tb","montgomery_tb"):
    sub=[r for r in rows if src in r["group"]]
    for lab in ("Normal","Tuberculosis"):
        s=[r for r in sub if r["true"]==lab]
        if not s: continue
        nz=[r for r in s if r["side"]!="none"]
        acc=sum(1 for r in s if r["true"]==r["pred"])/len(s)
        ar=np.array([r["_ar"] for r in nz]); il=np.array([r["_il"] for r in nz])
        print(f"  {src:14s} {lab:13s} n={len(s):3d} acc={100*acc:5.1f}%  "
              f"area med={np.median(ar):.3f}  in-lung med={np.median(il):.3f}")

print("\n--- correct-sample block by class ---")
for c in ("Normal","Pneumonia","Tuberculosis","Covid-19"):
    s=[r for r in rows if "correct_sample" in r["group"] and r["true"]==c]
    nz=[r for r in s if r["side"]!="none"]
    if not nz: 
        print(f"  {c:13s} n={len(s)}  all empty"); continue
    ar=np.array([r["_ar"] for r in nz]); il=np.array([r["_il"] for r in nz])
    print(f"  {c:13s} n={len(s):2d} empty={len(s)-len(nz)}  in-lung med={np.median(il):.3f}  "
          f"area med={np.median(ar):.3f}  best in-lung={il.max():.3f}")

print("\n--- overlay source actually used ---")
print("  ", collections.Counter(r["mask_source"] for r in rows))
print("  uncertain (top prob < 0.60):", sum(1 for r in rows if r["uncertain"]=="True"))

# where do the 102 pneumonia->Normal errors come from?
pn=[r for r in rows if r["group"].startswith("pneu_as_normal")]
print("\n--- provenance of the 102 Pneumonia->Normal errors ---")
print("  by source dataset:", collections.Counter(r["source"] for r in pn))
folder = collections.Counter(
    "Lung_Opacity" if "Lung_Opacity" in r["path"] else
    "Viral Pneumonia" if "Viral Pneumonia" in r["path"] else
    "pneu_ds/opacity" for r in pn)
print("  by class folder  :", dict(folder))
