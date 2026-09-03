"""Lung-field instrument shared by the Fig. 1 selection and the Fig. 7 scoring."""
import os
import numpy as np
import torch, torch.nn as nn
from PIL import Image
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE, BORDER = 224, 0.08
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.out = nn.Conv2d(c[0], 1, 1); self.p = nn.MaxPool2d(2)
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.p(d1)); d3 = self.d3(self.p(d2))
        b = self.bt(self.p(d3))
        x = self.c3(torch.cat([self.u3(b), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)

_net = None
def net():
    global _net
    if _net is None:
        _net = UNet()
        _net.load_state_dict(torch.load(os.path.join(HERE, "lungfield_unet.pth"),
                                        map_location=device, weights_only=True))
        _net.to(device).eval()
    return _net

def border_crop(img):
    w, h = img.size
    dx, dy = int(round(w*BORDER)), int(round(h*BORDER))
    if w-2*dx < 1 or h-2*dy < 1:
        return img
    return img.crop((dx, dy, w-dx, h-dy))

@torch.no_grad()
def lung_mask(pil_cropped, out_shape=None):
    """Binary lung field for an image ALREADY border-cropped. Returns HxW bool."""
    g = pil_cropped.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(g, np.float32)/255.0)[None, None].to(device)
    p = torch.sigmoid(net()(x))[0, 0].cpu().numpy()
    m = (p > 0.5).astype(np.uint8)
    if out_shape is not None and m.shape != tuple(out_shape[:2]):
        m = cv2.resize(m, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)

def split_lungs(mask):
    """Return (left_half_mask, right_half_mask) in IMAGE coordinates.

    On a PA film the patient's right lung is on the image's left, so
    image-left == anatomical right. Callers translate.
    """
    m = mask.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    if len(comps) < 2:
        # one blob (lungs touching): split at the mask's horizontal centroid
        xs = np.where(m.any(0))[0]
        mid = int(xs.mean()) if xs.size else m.shape[1]//2
        a = m.copy(); a[:, mid:] = 0
        b = m.copy(); b[:, :mid] = 0
        return a.astype(bool), b.astype(bool)
    c0, c1 = comps
    if cent[c0][0] > cent[c1][0]:
        c0, c1 = c1, c0
    return (lab == c0), (lab == c1)
