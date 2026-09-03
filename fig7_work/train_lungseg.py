"""
Train a lung-field U-Net used ONLY as a measuring instrument for Fig. 7's
localisation scoring. It is not part of LungLens and never touches the served
model.

Ground truth: the 21,165 lung masks shipped with the COVID-19 Radiography
Database. Images and masks are both BorderCrop(0.08)-ed before resizing, so the
predicted lung field lands in the same coordinate frame as the Grad-CAM map the
app produces (predict_image border-crops the PIL image before inference).

Test-split images are excluded from training so the instrument was never fitted
on a film it later measures.
"""
import os, sys, glob, random, csv
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")

SIZE = 224
BORDER = 0.08
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAD = (r"C:\Users\evanl\.cache\kagglehub\datasets\tawsifurrahman"
       r"\covid19-radiography-database\versions\5\COVID-19_Radiography_Dataset")

def border_crop(img):
    w, h = img.size
    dx, dy = int(round(w * BORDER)), int(round(h * BORDER))
    if w - 2*dx < 1 or h - 2*dy < 1:
        return img
    return img.crop((dx, dy, w-dx, h-dy))

# ── pairs, minus anything in the held-out test split ─────────────────────────
test_paths = set()
with open(os.path.join(HERE, "test_manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        test_paths.add(os.path.normcase(os.path.abspath(r["path"])))

pairs = []
for cls in ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]:
    for ip in sorted(glob.glob(os.path.join(RAD, cls, "images", "*.png"))):
        if os.path.normcase(os.path.abspath(ip)) in test_paths:
            continue
        mp = os.path.join(RAD, cls, "masks", os.path.basename(ip))
        if os.path.exists(mp):
            pairs.append((ip, mp))
random.Random(0).shuffle(pairs)
print(f"usable pairs (test-split excluded): {len(pairs)}", flush=True)

TRAIN_N, VAL_N = 9000, 1000
train_pairs, val_pairs = pairs[:TRAIN_N], pairs[TRAIN_N:TRAIN_N+VAL_N]

class LungDS(Dataset):
    def __init__(self, pairs, aug):
        self.pairs, self.aug = pairs, aug
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        im = border_crop(Image.open(ip).convert("L")).resize((SIZE, SIZE), Image.BILINEAR)
        mk = border_crop(Image.open(mp).convert("L")).resize((SIZE, SIZE), Image.NEAREST)
        x = torch.from_numpy(np.array(im, np.float32) / 255.0)[None]
        y = torch.from_numpy((np.array(mk, np.float32) > 127).astype(np.float32))[None]
        if self.aug and random.random() < 0.5:
            g = random.uniform(0.75, 1.35)
            x = x.clamp(0, 1) ** g
        return x, y

class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.b(x)

class UNet(nn.Module):
    def __init__(self, c=(32, 64, 128, 256)):
        super().__init__()
        self.d1, self.d2, self.d3 = Block(1, c[0]), Block(c[0], c[1]), Block(c[1], c[2])
        self.bt = Block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2); self.c3 = Block(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2); self.c2 = Block(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2); self.c1 = Block(c[1], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)
        self.p = nn.MaxPool2d(2)
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.p(d1)); d3 = self.d3(self.p(d2))
        b = self.bt(self.p(d3))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)

def dice(logits, y, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum((1, 2, 3))
    return ((2 * inter + eps) / (p.sum((1,2,3)) + y.sum((1,2,3)) + eps)).mean().item()

def main():
    tl = DataLoader(LungDS(train_pairs, True), batch_size=16, shuffle=True,
                    num_workers=4, persistent_workers=True, pin_memory=True)
    vl = DataLoader(LungDS(val_pairs, False), batch_size=16, shuffle=False, num_workers=2)
    net = UNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    bce = nn.BCEWithLogitsLoss()
    best = 0.0
    for ep in range(6):
        net.train()
        for i, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lo = net(x)
            p = torch.sigmoid(lo)
            soft = 1 - (2*(p*y).sum((1,2,3)) + 1) / (p.sum((1,2,3)) + y.sum((1,2,3)) + 1)
            loss = bce(lo, y) + soft.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 100 == 0:
                print(f"  ep{ep+1} batch {i}/{len(tl)} loss {loss.item():.4f}", flush=True)
        sched.step()
        net.eval(); ds = []
        with torch.no_grad():
            for x, y in vl:
                ds.append(dice(net(x.to(device)), y.to(device)))
        d = float(np.mean(ds))
        print(f"epoch {ep+1}/6  val Dice = {d:.4f}", flush=True)
        if d > best:
            best = d
            torch.save(net.state_dict(), os.path.join(HERE, "lungfield_unet.pth"))
    print(f"best val Dice = {best:.4f} -> lungfield_unet.pth", flush=True)

if __name__ == "__main__":
    main()
