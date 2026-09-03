import os, sys, csv, ast, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import artefact_score, load_cropped, contact_sheet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T = (r"C:\Users\evanl\.cache\kagglehub\datasets\vbookshelf\tbx11k-simplified"
     r"\versions\1\tbx11k-simplified")

test = {}
with open(os.path.join(ROOT, "fig7_work", "test_manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test[os.path.normcase(os.path.abspath(r["path"]))] = r

# group bboxes per file
boxes = {}
meta = {}
with open(os.path.join(T, "data.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if r["image_type"] != "tb":
            continue
        meta[r["fname"]] = r
        if r["bbox"] and r["bbox"] != "none":
            b = ast.literal_eval(r["bbox"])
            boxes.setdefault(r["fname"], []).append(b)

cands = []
for fname, bs in boxes.items():
    fp = os.path.join(T, "images", fname)
    if not os.path.exists(fp):
        continue
    key = os.path.normcase(os.path.abspath(fp))
    in_test = key in test
    H = float(meta[fname]["image_height"]); W = float(meta[fname]["image_width"])
    # centre of every lesion box, normalised
    cys = [(b["ymin"] + b["height"]/2)/H for b in bs]
    cxs = [(b["xmin"] + b["width"]/2)/W for b in bs]
    areas = [b["width"]*b["height"]/(H*W) for b in bs]
    if max(cys) > 0.42:            # every lesion must sit in the upper lung zone
        continue
    if len(bs) > 3:
        continue
    if max(areas) < 0.004:          # too tiny to read in a 3.4in figure
        continue
    if min(abs(cx-0.5) for cx in cxs) < 0.10:   # skip midline/mediastinal boxes
        continue
    a = artefact_score(load_cropped(fp))
    cands.append(dict(fname=fname, path=fp, in_test=in_test, nbox=len(bs),
                      cy=float(np.mean(cys)), cx=float(np.mean(cxs)),
                      area=float(max(areas)), tb_type=meta[fname]["tb_type"], **a))

cands.sort(key=lambda d: (not d["in_test"], d["n_blobs"], d["corner_bright"], -d["area"]))
print(f"{len(cands)} TB candidates with all lesions in the upper zone")
for c in cands[:36]:
    print(f"  {c['fname']:12s} test={int(c['in_test'])} nbox={c['nbox']} "
          f"cy={c['cy']:.2f} cx={c['cx']:.2f} area={c['area']:.4f} "
          f"blobs={c['n_blobs']} corner={c['corner_bright']:.4f} {c['tb_type']}")

json.dump(cands[:36], open(os.path.join(HERE, "tb_cands.json"), "w"), indent=1)
contact_sheet([(c["path"], f"{i}:{c['fname']} cy{c['cy']:.2f} b{c['n_blobs']}")
               for i, c in enumerate(cands[:36])],
              os.path.join(HERE, "sheet_tb.png"))
