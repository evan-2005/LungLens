"""
Mean CPU wall-clock inference time for the served path.

Times app.predict_image() itself - BorderCrop, U-Net forward, DenseNet forward,
Grad-CAM when the distilled mask is empty, heatmap blend and outline - which is
everything between the click and the rendered result, excluding Gradio's HTTP
transport and image encoding.

Eleven single-image requests; the first is discarded because it pays model
warm-up and lazy kernel init.
"""
import os, sys, csv, time, json, platform
import numpy as np
os.environ["LUNGLENS_SKIP_STARTUP"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import torch
from PIL import Image
import app

app.device = torch.device("cpu")
app.model_path = os.path.join(HERE, "chest_model_4class_HEAD.pth")
app.seg_model_path = os.path.join(ROOT, "chest_segmentation_model.pth")
app.seg_model, app.cls_model = app.load_models_from_disk()
app.seg_disease_model = True
print(f"device=cpu  torch={torch.__version__}  threads={torch.get_num_threads()}")
print(f"cpu={platform.processor()}")

rows = list(csv.DictReader(open(os.path.join(HERE, "test_manifest.csv"), encoding="utf-8")))
rng = np.random.default_rng(3)
N = int(os.environ.get("N_REQ", "11"))
picks = [rows[i] for i in rng.choice(len(rows), N, replace=False)]

ts, sizes, srcs = [], [], []
for i, r in enumerate(picks):
    im = Image.open(r["path"])
    im.load()                       # exclude file I/O from the timed section
    t0 = time.perf_counter()
    txt, _, _ = app.predict_image(im, app.AUTO_OVERLAY)
    dt = time.perf_counter() - t0
    ts.append(dt)
    src = "grad-cam" if "Grad-CAM" in txt else "u-net   "
    srcs.append(src.strip())
    sizes.append(im.size)
    print(f"  {'warm-up' if i == 0 else f'req {i:2d}'}: {dt*1000:7.1f} ms  {src}  "
          f"{im.size[0]}x{im.size[1]:<5} {os.path.basename(r['path'])[:30]}")

t = np.array(ts[1:])
print(f"\nmean over {t.size} requests (first excluded): {t.mean()*1000:.0f} ms  "
      f"(sd {t.std(ddof=1)*1000:.0f} ms, min {t.min()*1000:.0f}, max {t.max()*1000:.0f})")
print(f"warm-up request: {ts[0]*1000:.0f} ms")
px = np.array([w*h for w, h in sizes[1:]], float)
print(f"input pixels: median {np.median(px)/1e6:.2f} MP, range "
      f"{px.min()/1e6:.2f}-{px.max()/1e6:.2f} MP")
if len(t) > 3:
    print(f"corr(time, input pixels) = {np.corrcoef(t, px)[0,1]:.2f}")
first10 = t[:10]
print(f"first ten requests only:            {first10.mean()*1000:.0f} ms "
      f"(sd {first10.std(ddof=1)*1000:.0f} ms)")
for path in ("u-net", "grad-cam"):
    sel = np.array([x for x, s_ in zip(t, srcs[1:]) if s_ == path])
    if sel.size:
        print(f"  {path:9s} path: n={sel.size:2d}  mean {sel.mean()*1000:.0f} ms "
              f"(sd {sel.std(ddof=1)*1000 if sel.size > 1 else 0:.0f})")
json.dump({"n_requests": int(t.size),
           "first10_mean_ms": float(first10.mean()*1000),
           "first10_sd_ms": float(first10.std(ddof=1)*1000),
           "by_path": {p_: [float(x*1000) for x, s_ in zip(t, srcs[1:]) if s_ == p_]
                       for p_ in ("u-net", "grad-cam")},
           "mean_ms": float(t.mean()*1000), "sd_ms": float(t.std(ddof=1)*1000),
           "min_ms": float(t.min()*1000), "max_ms": float(t.max()*1000),
           "warmup_ms": float(ts[0]*1000), "threads": torch.get_num_threads(),
           "all_ms": [float(x*1000) for x in ts]},
          open(os.path.join(HERE, "cpu_timing.json"), "w"), indent=2)
