import os
import glob
import json
import random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import kagglehub
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (classification_report, f1_score, recall_score,
                             confusion_matrix)
import re
import threading

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
NUM_CLASSES = len(CLASSES)
model_path = "chest_model_4class.pth"
seg_model_path = "chest_segmentation_model.pth"
# Written at the end of every run so the UI can report the accuracy of the
# classifier checkpoint that is actually loaded, instead of an unverifiable
# claim. This describes chest_model_4class.pth (the served DenseNet).
metrics_path = "chest_classifier_metrics.json"
training_status = "Not Training"
training_logs = []
# Hybrid inference: the DenseNet checkpoint is a strong discriminative
# classifier, while the U-Net checkpoint provides segmentation masks but has a
# collapsed classification head (it outputs a near-constant prior). Each model
# does the job it is actually good at.
seg_model = None   # MultiTaskUNet: segmentation masks, fallback classifier
cls_model = None   # DenseNet-121: primary classifier, Grad-CAM fallback
# True only when the loaded seg_model is a fresh Grad-CAM-distilled DISEASE
# segmentation model (recorded in the metrics file). Older checkpoints trained
# on the brightness pseudo-mask trace anatomy, not disease, so inference must
# not prefer them; this flag gates that preference.
seg_disease_model = False
model_lock = threading.Lock()  # Guards loads/swaps of the global models.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# On CPU, leaving the thread count unset can let PyTorch oversubscribe cores and
# actually slow inference. Cap it at the physical core budget for steady latency.
if device.type == "cpu":
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))

# Shared configuration
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAL_SPLIT = 0.2
# Three-way patient-grouped split: hold out a test set that is used only for the
# final report, so the number shown is not the same set the checkpoint was
# selected on. Fractions are of the whole; the remainder is training data.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
EARLY_STOP_PATIENCE = 4         # epochs without a macro-F1 gain before stopping
SEG_LOSS_WEIGHT = 2.0
OVERLAY_THRESHOLD = 0.45        # Grad-CAM maps are relative, threshold after normalising
MASK_DISPLAY_THRESHOLD = 0.5    # U-Net masks are probabilities, threshold the raw sigmoid
CONFIDENCE_THRESHOLD = 0.60     # Below this top-class probability, report Uncertain
OVERLAY_MIN_PROB = 0.15         # Do not draw an overlay for a class this improbable
HEATMAP_FLOOR = 0.35            # Hide diffuse low activation so healthy areas stay clean
MIN_REGION_AREA_FRAC = 0.003    # Drop overlay specks smaller than 0.3% of the image
# Every Covid-19 image comes from RICORD, which burns annotations ("PORTABLE
# SEMI-ERECT", laterality markers) into the outer margin. Because that text is
# unique to one class's source, the classifier can shortcut on it instead of on
# lung pathology (visible as Grad-CAM lighting up an empty image corner). Cropping
# this fraction off each edge before resize denies the shortcut. Applied
# identically at train and inference time. Kept modest so the lung apices (where
# TB tends to show) survive.
BORDER_CROP_FRAC = 0.08
# Stage-2 U-Net training is memory-heavy at full resolution. Cap its batch so it
# fits alongside a live CUDA context on a small (6 GB) GPU; the classifier's
# larger batch would overflow VRAM and spill to shared system memory, which stalls
# training. See train_segmentation_head.
SEG_MAX_BATCH = 8
OVERLAY_COLOR = (0, 113, 227)  # RGB clinical blue used for the segmentation overlay
AUTO_OVERLAY = "Auto (predicted class)"

# Recommended run size for fine-tuning the DenseNet classifier. Use as much of
# the ~33k-image pool as the machine can afford: the minority classes (TB ~3%,
# Covid ~14%) only become well represented in the val/test folds at scale. Early
# stopping means over-requesting epochs is cheap, so aim high and let it stop.
RECOMMENDED_SAMPLES = 20000
RECOMMENDED_EPOCHS = 20
RECOMMENDED_BATCH = 16
RECOMMENDED_LR = 1e-4


def _normalize_transform():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


class BorderCrop:
    """Crop a fixed fraction off each edge of a PIL image.

    Defined at module level (not a lambda) so it stays picklable for DataLoader
    workers under Windows spawn. See BORDER_CROP_FRAC for why the crop exists.
    """
    def __init__(self, frac=BORDER_CROP_FRAC):
        self.frac = frac

    def __call__(self, img):
        w, h = img.size
        dx, dy = int(round(w * self.frac)), int(round(h * self.frac))
        # Guard against a degenerate crop on a tiny image.
        if w - 2 * dx < 1 or h - 2 * dy < 1:
            return img
        return img.crop((dx, dy, w - dx, h - dy))


def generate_pseudo_mask(gray_resized):
    """
    Build an anatomically informed pseudo-mask for the abnormal lung regions.

    Real per-pixel lung annotations are not available for these datasets, so the
    multi-task U-Net is supervised with a deterministic target derived from the
    image itself: contrast-equalise, isolate denser (brighter) tissue with an
    Otsu threshold, restrict it to an elliptical lung field, then clean the
    result with morphology. This is far closer to true opacity than the previous
    fixed-rectangle threshold and gives the segmentation head a learnable signal.
    Input and output are both float/uint8 arrays of shape (IMG_SIZE, IMG_SIZE).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_resized)

    _, thresh = cv2.threshold(equalized, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Elliptical lung field ROI instead of a hard rectangle.
    h, w = gray_resized.shape
    lung_roi = np.zeros_like(gray_resized)
    cv2.ellipse(lung_roi, (int(w * 0.50), int(h * 0.52)),
                (int(w * 0.38), int(h * 0.40)), 0, 0, 360, 255, -1)
    thresh = cv2.bitwise_and(thresh, lung_roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return np.clip(thresh / 255.0, 0.0, 1.0).astype(np.float32)

# Datasets
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except FileNotFoundError:
            print(f"WARNING: Image not found: {self.image_paths[idx]}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        except OSError as e:
            print(f"WARNING: Cannot open image {self.image_paths[idx]}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class SegmentationDataset(Dataset):
    """
    Generates pseudo-masks lazily inside __getitem__ so the DataLoader
    can parallelise the work instead of blocking the main thread.
    Uses a fast threshold fallback, Grad-CAM per-image is too slow on CPU.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WARNING: Image not found: {img_path}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        except OSError as e:
            print(f"WARNING: Cannot open image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)

        target_mask = np.zeros((NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        if label > 0:
            try:
                gray = np.array(image.convert("L").resize((IMG_SIZE, IMG_SIZE)))
                target_mask[label] = generate_pseudo_mask(gray)
            except cv2.error as e:
                print(f"WARNING: cv2 operation failed for {img_path}: {e}")
            except Exception as e:
                print(f"WARNING: Mask generation failed for {img_path}: {e}")

        return img_tensor, torch.tensor(target_mask, dtype=torch.float32), label


# Models
class CNNModel(nn.Module):
    def __init__(self, classCount, isTrained=True):
        super().__init__()
        self.cnnmodel = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if isTrained else None)
        self.cnnmodel.classifier = nn.Linear(
            self.cnnmodel.classifier.in_features, classCount)

    def forward(self, x):
        return self.cnnmodel(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout regularises the shallow classification head so it does not
        # overfit the majority classes. It has no parameters, so checkpoints
        # trained before this line still load with strict=True.
        self.cls_dropout = nn.Dropout(0.3)
        self.classifier  = nn.Linear(512, num_classes)
        self.up1      = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2      = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3      = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc    = nn.Conv2d(64, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        cls = self.classifier(self.cls_dropout(torch.flatten(self.avgpool(x4), 1)))
        d = self.conv_up1(torch.cat([self.up1(x4), x3], 1))
        d = self.conv_up2(torch.cat([self.up2(d),  x2], 1))
        d = self.conv_up3(torch.cat([self.up3(d),  x1], 1))
        return cls, self.sigmoid(self.outc(d))


# Losses
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1.0):
        flat_i = inputs.view(-1)
        flat_t = targets.view(-1)
        inter  = (flat_i * flat_t).sum()
        dice   = 1 - (2 * inter + smooth) / (flat_i.sum() + flat_t.sum() + smooth)
        bce    = nn.functional.binary_cross_entropy(inputs, targets)
        return bce + dice


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Mean Dice over only the (sample, channel) pairs that actually contain a
    region in either the prediction or the target. Averaging over *all* channels
    (the old behaviour) handed a free 1.0 to every empty channel, so a model
    predicting nothing still scored ~0.75 on a single-region-per-image target.
    Restricting to present channels makes the score reflect real overlap.
    """
    y_bin  = (y_pred > 0.5).float()
    inter  = (y_bin * y_true).sum(dim=(2, 3))
    union  = y_bin.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice   = (2.0 * inter + smooth) / (union + smooth)
    present = union > 0
    if present.any():
        result = dice[present].mean().item()
    else:
        # No region anywhere in pred or target: a correct empty prediction.
        result = 1.0
    if np.isnan(result) or np.isinf(result):
        return 0.0
    return result


# Training metrics persistence
def save_checkpoint_atomic(state_dict, path, retries=5, delay=1.5):
    """Save a state_dict durably: write to a temp file, then atomically replace
    the target. os.replace is atomic on Windows and never leaves a half-written
    checkpoint. Retries a few times because Windows Defender can briefly lock a
    freshly written .pth while it scans, which surfaces as "cannot be opened".
    Raises the last error only if every attempt fails.
    """
    import time
    tmp = f"{path}.tmp{os.getpid()}"
    last_err = None
    for attempt in range(retries):
        try:
            torch.save(state_dict, tmp)
            os.replace(tmp, path)
            return
        except (OSError, RuntimeError) as e:
            last_err = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(delay)
    raise last_err


def save_training_metrics(metrics):
    """Record how the saved checkpoint scored. Never raises, a failure to write
    the report must not fail an otherwise successful run."""
    try:
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return True
    except OSError as e:
        print(f"Warning: could not write {metrics_path}: {e}")
        return False


def load_training_metrics():
    """Return the recorded metrics for the checkpoint on disk, or None.

    Returns None when the file is missing, unreadable, or older than the
    checkpoint it claims to describe, so a stale report is never presented as
    the accuracy of a newer model.
    """
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read {metrics_path}: {e}")
        return None
    if not isinstance(metrics, dict) or "val_acc" not in metrics:
        return None
    if (os.path.exists(model_path)
            and os.path.getmtime(model_path) > os.path.getmtime(metrics_path) + 1):
        return None
    return metrics


# Grad-CAM
def get_gradcam_layer(model):
    """Get the final conv layer from DenseNet for GradCAM."""
    try:
        return model.cnnmodel.features.denseblock4.denselayer16.conv2
    except AttributeError:
        raise RuntimeError(
            f"Unsupported model architecture. Expected DenseNet, got {type(model).__name__}. "
            "Cannot extract GradCAM layer."
        )


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = self.features = None
        self.hook_handles = []
        fn_fwd = lambda m, i, o: setattr(self, 'features', o)
        fn_bwd = lambda m, gi, go: setattr(self, 'gradients', go[0])
        hook = (target_layer.register_full_backward_hook
                if hasattr(target_layer, 'register_full_backward_hook')
                else target_layer.register_backward_hook)
        self.hook_handles += [hook(fn_bwd),
                               target_layer.register_forward_hook(fn_fwd)]

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def generate_heatmap(self, input_tensor, class_idx):
        try:
            self.model.zero_grad()
            out = self.model(input_tensor)
            if class_idx >= out.shape[1]:
                raise ValueError(f"Invalid class index {class_idx} for model output shape {out.shape}")
            out[0, class_idx].backward(retain_graph=False)
            grads = self.gradients.detach().cpu().numpy()[0]
            feats = self.features.detach().cpu().numpy()[0]
            weights = grads.mean(axis=(1, 2))
            cam = (weights[:, None, None] * feats).sum(axis=0)
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        finally:
            self.remove_hooks()


# Data collection
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def get_dataset_paths():
    tb  = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    pn  = kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images")
    cov = kagglehub.dataset_download("raddar/ricord-covid19-xray-positive-tests")
    # Second source spanning Covid / Normal / Pneumonia (Lung Opacity + Viral
    # Pneumonia). Its whole point is decorrelation: because Covid now also comes
    # from here (not only RICORD) and this one source carries every disease, the
    # scanner/source signature stops being a valid shortcut for the label.
    radio = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
    # Shenzhen TB set: a SECOND source for Tuberculosis (the one class the
    # radiography set lacks), so TB too is no longer tied to a single dataset.
    shenzhen = kagglehub.dataset_download("raddar/tuberculosis-chest-xrays-shenzhen")
    return tb, pn, cov, radio, shenzhen


def _label_from_folder(rel_parts, folder_map):
    """
    Map an image to a class by the name of the class sub-folder it sits in,
    not by substring-matching the whole absolute path. The old approach only
    worked because the dataset directory names happened to contain the class
    words (e.g. every file under 'pneumonia-xray-images' matched 'pneumonia'),
    and it silently mislabels the moment a dataset is moved or renamed. Here the
    match is against the directory segments *relative to the dataset root*.
    Returns a class index or None if no folder matches.
    """
    for part in rel_parts:
        key = part.lower()
        if key in folder_map:
            return folder_map[key]
    return None


def _patient_group(source, filename):
    """
    Derive a stable patient/study id so all images of one patient stay on the
    same side of a train/val/test split. These datasets have many images per
    patient (RICORD Covid has up to 33), so an image-level split leaks the
    patient across sets and inflates validation scores. Each known filename
    scheme is handled; anything unrecognised falls back to its own filename so
    it is at worst treated as a distinct patient.
    """
    stem = os.path.splitext(filename)[0]
    m = re.search(r"person\d+", stem, re.IGNORECASE)          # pneumonia: person123_bacteria_4
    if m:
        return f"{source}:{m.group(0).lower()}"
    m = re.match(r"(\d+-\d+)_", stem)                          # RICORD covid: 419639-000025_...
    if m:
        return f"{source}:{m.group(1)}"
    m = re.match(r"(IM-\d+)", stem)                            # pneumonia normal: IM-0001-0001
    if m:
        return f"{source}:{m.group(1)}"
    return f"{source}:{stem}"


# Class folder -> label, per dataset. Note the pneumonia disease folder is
# literally named "opacity", which no substring rule for "pneumonia" would ever
# match; this is exactly why folder-relative labelling is required.
_TB_FOLDERS   = {"normal": 0, "tuberculosis": 2}
_PNEU_FOLDERS = {"normal": 0, "opacity": 1, "pneumonia": 1}
_CUSTOM_FOLDERS = {"normal": 0, "pneumonia": 1, "opacity": 1,
                   "tuberculosis": 2, "tb": 2, "covid": 3, "covid-19": 3}
# COVID-19 Radiography Database class folders. Lung Opacity and Viral Pneumonia
# both fold into the Pneumonia class (label 1), matching how the pneumonia set's
# "opacity" folder is treated.
_RADIO_FOLDERS = {"covid": 3, "normal": 0, "viral pneumonia": 1, "lung_opacity": 1}


def collect_dataset():
    """
    Walk every dataset and return one record per image as parallel lists:
    (paths, labels, groups, sources). `groups` feeds a patient-grouped split;
    `sources` lets us report per-source accuracy as a confounding probe. Covid
    and Pneumonia are drawn from two sources each (and the radiography set spans
    every disease), so source no longer perfectly predicts the label the way it
    did when each class came from a single dataset.
    """
    tb_base, pn_base, cov_base, radio_base, shenzhen_base = get_dataset_paths()
    paths, labels, groups, sources = [], [], [], []

    def add(f, label, source):
        paths.append(f)
        labels.append(label)
        sources.append(source)
        groups.append(_patient_group(source, os.path.basename(f)))

    for f in glob.glob(os.path.join(tb_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        rel = os.path.relpath(f, tb_base).split(os.sep)
        label = _label_from_folder(rel, _TB_FOLDERS)
        if label is not None:
            add(f, label, "tb_ds")

    for f in glob.glob(os.path.join(pn_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        rel = os.path.relpath(f, pn_base).split(os.sep)
        label = _label_from_folder(rel, _PNEU_FOLDERS)
        if label is not None:
            add(f, label, "pneu_ds")

    # RICORD is an all-positive Covid set. Guard against the non-image sidecar
    # files (.csv/.complete) with the extension filter, and require the image to
    # live under the study directory so a stray file in the root is not labelled.
    for f in glob.glob(os.path.join(cov_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        if "midrc" not in f.lower():
            continue
        add(f, 3, "covid_ds")

    # COVID-19 Radiography Database: {COVID,Lung_Opacity,Normal,Viral Pneumonia}/
    # {images,masks}/*.png. Take ONLY the X-rays under images/; the masks/ folder
    # holds lung segmentation masks that share the class-folder ancestor and would
    # otherwise be mislabelled as chest films.
    for f in glob.glob(os.path.join(radio_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        rel = os.path.relpath(f, radio_base).split(os.sep)
        rel_lower = [p.lower() for p in rel]
        if "images" not in rel_lower or "masks" in rel_lower:
            continue
        label = _label_from_folder(rel, _RADIO_FOLDERS)
        if label is not None:
            add(f, label, "radiography_db")

    # Shenzhen TB set encodes the label in the filename suffix, not in a folder:
    # CHNCXR_<id>_0 = normal, CHNCXR_<id>_1 = TB-positive.
    for f in glob.glob(os.path.join(shenzhen_base, "**", "*.png"), recursive=True):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = re.match(r"CHNCXR_\d+_([01])$", stem)
        if not m:
            continue
        add(f, 2 if m.group(1) == "1" else 0, "shenzhen_tb")

    custom_base = os.path.join(os.path.dirname(__file__), "custom_dataset")
    if os.path.exists(custom_base):
        for f in glob.glob(os.path.join(custom_base, "**", "*.*"), recursive=True):
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            rel = os.path.relpath(f, custom_base).split(os.sep)
            label = _label_from_folder(rel, _CUSTOM_FOLDERS)
            if label is not None:
                add(f, label, "custom")

    if not paths:
        print("WARNING: No images found in datasets!")
    return paths, labels, groups, sources


def collect_data():
    """Backward-compatible 2-tuple wrapper (paths, labels) for tooling that does
    not need group/source metadata (e.g. debug_training.py)."""
    paths, labels, _, _ = collect_dataset()
    combined = list(zip(paths, labels))
    if not combined:
        return [], []
    random.shuffle(combined)
    paths, labels = zip(*combined)
    return list(paths), list(labels)


# Training helpers
def stratified_subsample(paths, labels, groups, sources, num_samples, seed=42):
    """Take a class-balanced-in-proportion subsample instead of head-slicing a
    shuffled list, so the minority-class counts do not swing run to run."""
    n = len(paths)
    if num_samples >= n:
        return paths, labels, groups, sources
    idx = np.arange(n)
    # A stratified split gives a subset whose class proportions match the whole.
    keep_idx, _ = train_test_split(
        idx, train_size=num_samples, random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None)
    keep = set(keep_idx.tolist())
    sel = [i for i in range(n) if i in keep]
    return ([paths[i] for i in sel], [labels[i] for i in sel],
            [groups[i] for i in sel], [sources[i] for i in sel])


def _group_split(indices, labels, groups, test_size, seed):
    """Split an index array by group so no group spans both sides. Falls back to
    a plain stratified split if there is only one group per side is impossible."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    sub_labels = [labels[i] for i in indices]
    sub_groups = [groups[i] for i in indices]
    a_idx, b_idx = next(gss.split(indices, sub_labels, sub_groups))
    return ([indices[i] for i in a_idx], [indices[i] for i in b_idx])


def patient_grouped_split(paths, labels, groups, sources, seed=42):
    """
    Patient-grouped train/val/test split. Test is carved off first, then val
    from the remainder, always along group boundaries so a patient's images
    never appear in more than one split.
    """
    all_idx = list(range(len(paths)))
    trainval_idx, test_idx = _group_split(all_idx, labels, groups, TEST_FRACTION, seed)
    # val fraction is relative to the remaining trainval pool.
    val_rel = VAL_FRACTION / (1.0 - TEST_FRACTION)
    train_idx, val_idx = _group_split(trainval_idx, labels, groups, val_rel, seed)

    def gather(idx):
        return ([paths[i] for i in idx], [labels[i] for i in idx],
                [sources[i] for i in idx])
    return gather(train_idx), gather(val_idx), gather(test_idx)


def build_transforms():
    """
    Training augmentation is now geometric as well as photometric. Dropping the
    pseudo-mask target removed the alignment constraint that previously forbade
    rotation/flip/crop, so we can use them. RandomResizedCrop + rotation + flip +
    jitter + mild blur/noise attack the resolution and scanner cues that
    otherwise let the model tell the diseases apart by their source dataset
    rather than by pathology.
    """
    train_tf = transforms.Compose([
        BorderCrop(),
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ToTensor(),
        _normalize_transform(),
    ])
    eval_tf = transforms.Compose([
        BorderCrop(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        _normalize_transform(),
    ])
    return train_tf, eval_tf


def evaluate_classifier(net, loader):
    """Run a classifier over a loader and return (y_true, y_pred) as lists."""
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, target_labels in loader:
            inputs = inputs.to(device)
            logits = net(inputs)
            preds = logits.argmax(1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(target_labels.tolist())
    return y_true, y_pred


def per_source_accuracy(y_true, y_pred, sources):
    """
    Accuracy broken down by originating dataset. Because each disease comes from
    a single source, a large gap here (especially for Normal, which spans two
    sources) is a signal the model is keying on the scanner, not the pathology.
    """
    out = {}
    by_src = {}
    for t, p, s in zip(y_true, y_pred, sources):
        by_src.setdefault(s, []).append(t == p)
    for s, hits in by_src.items():
        out[s] = {"acc": 100.0 * sum(hits) / len(hits), "n": len(hits)}
    return out


# Segmentation via Grad-CAM distillation
def gradcam_soft_mask(cls_net, img_tensor, class_idx):
    """
    A soft disease-localisation mask in [0, 1] at IMG_SIZE, taken from the
    classifier's Grad-CAM for the given class. This is the segmentation *target*:
    it marks the lung region that actually drove the class score, which is what
    the pseudo-mask never did. Returns a float32 array (IMG_SIZE, IMG_SIZE).
    """
    layer = get_gradcam_layer(cls_net)
    cam = GradCAM(cls_net, layer)
    t = img_tensor.clone().detach().to(device).requires_grad_(True)
    heat = cam.generate_heatmap(t, class_idx)   # already normalised to [0,1], 224x224
    return heat.astype(np.float32)


def build_gradcam_targets(cls_net, paths, labels, eval_tf, cap, log):
    """
    Precompute Grad-CAM disease targets once (Normal -> all-zero mask), so the
    U-Net can then be trained over several epochs without paying the per-image
    backward pass every time. Bounded by `cap` because Grad-CAM on CPU is slow.
    The pool is shuffled before the cap is applied, otherwise the head would be
    dominated by whichever dataset was concatenated first and the targets would
    lack class variety.
    """
    order = list(range(len(paths)))
    random.Random(42).shuffle(order)
    order = order[:min(cap, len(paths))]
    n = len(order)
    imgs = torch.zeros(n, 3, IMG_SIZE, IMG_SIZE)
    masks = torch.zeros(n, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    cls_net.eval()
    made = 0
    for out_i, i in enumerate(order):
        try:
            image = Image.open(paths[i]).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        t = eval_tf(image).unsqueeze(0)
        imgs[out_i] = t[0]
        label = labels[i]
        if label > 0:
            heat = gradcam_soft_mask(cls_net, t, label)
            masks[out_i, label] = torch.from_numpy(heat)
        made += 1
        if made % max(1, n // 5) == 0:
            log(f"  Grad-CAM targets: {made}/{n}")
    return imgs, masks


def train_segmentation_head(cls_net, train_paths, train_labels, eval_tf,
                            seg_cap, seg_epochs, batch_size, num_workers, log):
    """
    Stage 2: distil the classifier's Grad-CAM into the MultiTaskUNet's
    segmentation output. The U-Net learns to reproduce the disease-localisation
    map in a single forward pass (no gradient needed at inference). Returns
    (trained_net, best_dice) or (None, 0.0) if it could not run.
    """
    log(f"Stage 2: building Grad-CAM segmentation targets (cap {seg_cap})...")
    imgs, masks = build_gradcam_targets(cls_net, train_paths, train_labels,
                                        eval_tf, seg_cap, log)
    if imgs.shape[0] < 8:
        log("Not enough images to train segmentation; skipping.")
        return None, 0.0

    ds = torch.utils.data.TensorDataset(imgs, masks)
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # Force workers=0 here: the seg data is already in-memory GPU-bound tensors,
    # so DataLoader workers add nothing, and spawning them alongside the live
    # CUDA context is exactly what triggers "CUDA error: unknown error" on Windows.
    seg_batch = min(batch_size, SEG_MAX_BATCH)
    tl = DataLoader(train_sub, batch_size=seg_batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_sub, batch_size=seg_batch, shuffle=False, num_workers=0)

    # The Grad-CAM targets are already built, so the classifier is no longer
    # needed on the GPU during U-Net training. Park it on the CPU (restored before
    # returning) and clear the cache so the U-Net has the VRAM to itself; otherwise
    # both models resident at once overflow a 6 GB card and training stalls.
    cls_home = next(cls_net.parameters()).device
    cls_net.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seg_net = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    seg_crit = DiceBCELoss()
    opt = optim.Adam(seg_net.parameters(), lr=1e-3)
    best_dice = 0.0
    best_state = None

    for ep in range(seg_epochs):
        seg_net.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            _, pred = seg_net(xb)
            loss = seg_crit(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_net.parameters(), 1.0)
            opt.step()

        seg_net.eval()
        dsum = k = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                _, pred = seg_net(xb)
                dsum += dice_coefficient(pred, yb)
                k += 1
        val_dice = dsum / max(1, k)
        log(f"  Seg epoch {ep+1}/{seg_epochs} | Val Dice: {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {kk: v.detach().cpu().clone() for kk, v in seg_net.state_dict().items()}

    if best_state is not None:
        seg_net.load_state_dict(best_state)
    # Restore the classifier to where the caller expects it (it publishes and
    # warms up eval_net on the GPU after this returns).
    cls_net.to(cls_home)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return seg_net, best_dice


# Training
def run_training_thread(num_samples, epochs, lr, batch_size, num_workers=0,
                        train_seg=True, seg_cap=800, seg_epochs=6):
    global training_status, training_logs, cls_model, seg_model, seg_disease_model
    training_status = "Training..."
    training_logs = []
    writer = None

    try:
        writer = SummaryWriter("runs/lunglens")
        writer.add_scalar("Training/started", 1, 0)
        writer.flush()
        training_logs.append("TensorBoard logging to runs/lunglens")
        training_logs.append("Collecting dataset (paths, labels, patient groups, sources)...")
        all_paths, all_labels, all_groups, all_sources = collect_dataset()
        training_logs.append(f"Total available images: {len(all_paths)}")
        if len(all_paths) < 20:
            raise RuntimeError("Not enough images found to train (need at least 20).")

        # Stratified subsample keeps class proportions stable across runs.
        all_paths, all_labels, all_groups, all_sources = stratified_subsample(
            all_paths, all_labels, all_groups, all_sources, num_samples)
        training_logs.append(f"Using {len(all_paths)} images after stratified subsample.")

        # Patient-grouped train/val/test split: no patient spans two splits, and
        # test is held out purely for the final report.
        (train_paths, train_labels, train_sources), \
            (val_paths, val_labels, val_sources), \
            (test_paths, test_labels, test_sources) = patient_grouped_split(
                all_paths, all_labels, all_groups, all_sources)
        training_logs.append(
            f"Split (patient-grouped) -> Train: {len(train_paths)} | "
            f"Val: {len(val_paths)} | Test: {len(test_paths)}")

        train_tf, eval_tf = build_transforms()

        # ChestXRayDataset returns (image, label): a pure classification target.
        # No pseudo-masks, no segmentation loss. The DenseNet trained here is the
        # model predict_image actually serves, and Grad-CAM supplies the overlay.
        train_ds = ChestXRayDataset(train_paths, train_labels, transform=train_tf)
        val_ds   = ChestXRayDataset(val_paths,   val_labels,   transform=eval_tf)
        test_ds  = ChestXRayDataset(test_paths,  test_labels,  transform=eval_tf)

        # pin_memory speeds the host->GPU copy. We deliberately do NOT use
        # persistent_workers: keeping the classifier's worker pool alive bleeds
        # those processes into the Stage-2 seg training and corrupts the CUDA
        # context there ("CUDA error: unknown error" on backward). Letting the
        # workers tear down after each epoch keeps the seg stage clean.
        dl_kw = dict(num_workers=num_workers,
                     pin_memory=(device.type == "cuda"))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kw)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kw)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **dl_kw)

        training_logs.append("Initializing DenseNet-121 classifier (ImageNet-pretrained)...")
        net = CNNModel(classCount=NUM_CLASSES, isTrained=True).to(device)

        # Inverse-frequency class weights so the minority classes (TB, Covid-19)
        # are not drowned out by Normal/Pneumonia. Normalised to mean 1.0 so the
        # loss scale is unchanged.
        counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
        inv = 1.0 / np.clip(counts, 1.0, None)
        class_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        training_logs.append(
            "Train class counts: "
            + ", ".join(f"{CLASSES[i]}={int(counts[i])}" for i in range(NUM_CLASSES)))

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        # Schedule and checkpoint-selection both key on validation macro-F1, the
        # right target under 7:1 class imbalance where raw accuracy is misleading.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)

        best_val_f1 = -1.0
        best_metrics = None
        epochs_since_improve = 0
        history = {"train_acc": [], "val_acc": [], "val_f1": []}

        for epoch in range(epochs):
            net.train()
            correct = total = 0
            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Training ({len(train_loader)} batches)...")
            for batch_idx, (inputs, target_labels) in enumerate(train_loader):
                try:
                    inputs = inputs.to(device)
                    target_labels = target_labels.to(device)

                    optimizer.zero_grad()
                    logits = net(inputs)
                    loss = criterion(logits, target_labels)

                    if torch.isnan(loss) or torch.isinf(loss):
                        training_logs.append(f"  WARNING Batch {batch_idx+1}: NaN/Inf loss")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

                    total += target_labels.size(0)
                    correct += logits.argmax(1).eq(target_labels).sum().item()
                    step = epoch * len(train_loader) + batch_idx + 1
                    if (batch_idx + 1) % max(1, len(train_loader) // 20) == 0:
                        writer.add_scalar("Loss/train_batch", loss.item(), step)
                        writer.add_scalar("Accuracy/train_running", 100.0 * correct / total, step)
                        writer.flush()
                    if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                        print(f"  Train batch {batch_idx+1}/{len(train_loader)}", flush=True)
                except Exception as e:
                    import traceback
                    training_logs.append(f"  ERROR in batch {batch_idx+1}: {e}")
                    print(f"Batch error: {e}\n{traceback.format_exc()}", flush=True)
                    continue

            train_acc = 100.0 * correct / total if total > 0 else 0.0

            # Validation: collect predictions so we can score macro-F1, not just
            # accuracy, and select the checkpoint on it.
            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Validation ({len(val_loader)} batches)...")
            y_true, y_pred = evaluate_classifier(net, val_loader)
            val_acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred)) if y_true else 0.0
            val_f1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0

            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]

            log = (f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | "
                   f"Val Acc: {val_acc:.2f}% | Val macro-F1: {val_f1:.2f}% | LR: {current_lr:.2e}")
            training_logs.append(log)
            print(log, flush=True)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)
            writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            writer.add_scalar("Accuracy/val", val_acc, epoch + 1)
            writer.add_scalar("F1/val_macro", val_f1, epoch + 1)
            writer.add_scalar("LR", current_lr, epoch + 1)
            writer.flush()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_since_improve = 0
                save_checkpoint_atomic(net.state_dict(), model_path)
                per_class_recall = recall_score(
                    y_true, y_pred, labels=list(range(NUM_CLASSES)),
                    average=None, zero_division=0).tolist()
                training_logs.append(
                    f"--> Saved best classifier (Val macro-F1: {best_val_f1:.2f}%)")
                # Captured at the saved epoch, since the last epoch is not
                # necessarily the one written to disk.
                best_metrics = {
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "train_acc": train_acc,
                    "epoch": epoch + 1,
                    "epochs_requested": epochs,
                    "val_per_class_recall": {CLASSES[i]: 100.0 * per_class_recall[i]
                                             for i in range(NUM_CLASSES)},
                    "train_samples": len(train_paths),
                    "val_samples": len(val_paths),
                    "test_samples": len(test_paths),
                    "batch_size": batch_size,
                    "lr": lr,
                    "class_counts": {CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)},
                    "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= EARLY_STOP_PATIENCE:
                    training_logs.append(
                        f"Early stopping: no macro-F1 gain for {EARLY_STOP_PATIENCE} epochs.")
                    if best_metrics is not None:
                        best_metrics["early_stopped"] = True
                    break

        # Reload the best checkpoint and report on the held-out TEST set, which
        # was never used for selection, so the headline number is honest.
        if os.path.exists(model_path):
            eval_net = CNNModel(classCount=NUM_CLASSES, isTrained=False)
            eval_net.load_state_dict(torch.load(model_path, map_location=device))
            eval_net.to(device)
        else:
            eval_net = net

        if test_paths and best_metrics is not None:
            t_true, t_pred = evaluate_classifier(eval_net, test_loader)
            test_acc = 100.0 * np.mean(np.array(t_true) == np.array(t_pred))
            test_f1 = 100.0 * f1_score(t_true, t_pred, average="macro", zero_division=0)
            report = classification_report(
                t_true, t_pred, labels=list(range(NUM_CLASSES)),
                target_names=CLASSES, zero_division=0)
            cm = confusion_matrix(t_true, t_pred, labels=list(range(NUM_CLASSES)))
            src_acc = per_source_accuracy(t_true, t_pred, test_sources)

            best_metrics.update({
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_per_source_acc": src_acc,
                "confusion_matrix": cm.tolist(),
            })
            # Both append (for the Training-tab log) and print (so a headless run
            # captures the held-out test + per-source confound probe in stdout).
            report_lines = [
                f"Held-out TEST | Acc: {test_acc:.2f}% | macro-F1: {test_f1:.2f}%",
                "Per-class report (test):\n" + report,
                "Per-source accuracy (test): "
                + ", ".join(f"{s}={v['acc']:.1f}% (n={v['n']})" for s, v in src_acc.items()),
                "Confusion matrix (rows=true, cols=pred; order "
                + ", ".join(CLASSES) + "):\n" + str(cm),
            ]
            for _line in report_lines:
                training_logs.append(_line)
                print(_line, flush=True)
            writer.add_scalar("Accuracy/test", test_acc, 0)
            writer.add_scalar("F1/test_macro", test_f1, 0)
            writer.flush()

        # Stage 2: distil the classifier's Grad-CAM into a disease-segmentation
        # U-Net. This is the segmentation model the Analysis tab uses; its target
        # is where the classifier actually looked, not image brightness.
        trained_seg = None
        if train_seg and best_metrics is not None:
            try:
                trained_seg, seg_dice = train_segmentation_head(
                    eval_net, train_paths, train_labels, eval_tf,
                    seg_cap, seg_epochs, batch_size, num_workers,
                    lambda m: (training_logs.append(m), print(m, flush=True)))
                if trained_seg is not None:
                    save_checkpoint_atomic(trained_seg.state_dict(), seg_model_path)
                    best_metrics["segmentation"] = "gradcam_distilled"
                    best_metrics["seg_val_dice"] = seg_dice
                    best_metrics["seg_train_images"] = min(seg_cap, len(train_paths))
                    training_logs.append(
                        f"--> Saved disease-segmentation U-Net (Val Dice: {seg_dice:.4f}).")
                    writer.add_scalar("Dice/seg_val", seg_dice, 0)
                    writer.flush()
            except Exception as e:
                import traceback
                training_logs.append(f"WARNING: segmentation stage failed: {e}")
                print(f"Seg stage failed: {e}\n{traceback.format_exc()}", flush=True)

        if best_metrics is not None and save_training_metrics(best_metrics):
            training_logs.append(f"Recorded checkpoint metrics to {metrics_path}.")

        # Plot on the Agg backend: pyplot's default GUI backend is not safe to
        # call from this worker thread and can hard-crash the process.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_acc"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs_range, history["train_acc"], "b-o", label="Train Acc")
        ax1.plot(epochs_range, history["val_acc"], "r-o", label="Val Acc")
        ax1.set_title("Classification Accuracy")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)"); ax1.legend(); ax1.grid(True)
        ax2.plot(epochs_range, history["val_f1"], "g-o", label="Val macro-F1")
        ax2.set_title("Validation Macro-F1")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 (%)"); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        plt.close(fig)
        training_logs.append("Graph saved as training_curves.png")
        writer.close()
        writer = None

        # Publish the reloaded best classifier (and the new segmentation U-Net,
        # if trained) for inference only after the run ends, under the lock, so
        # requests never see a half-trained model.
        eval_net.eval()
        warm_up_model(eval_net)
        if trained_seg is not None:
            trained_seg.eval()
            warm_up_model(trained_seg)
        with model_lock:
            cls_model = eval_net
            if trained_seg is not None:
                seg_model = trained_seg
                # From now on inference may prefer the U-Net disease mask.
                seg_disease_model = True

        training_status = "Training Finished"
        training_logs.append("Done.")

    except Exception as e:
        training_status = "Failed"
        import traceback
        error_trace = traceback.format_exc()
        training_logs.append(f"Error: {str(e)}")
        training_logs.append(error_trace)
        print(f"Training failed: {e}")
        print(error_trace)
    finally:
        if writer is not None:
            writer.close()


def _is_training_active():
    return training_status == "Training..."


def start_training(num_samples, epochs, lr, batch_size, num_workers,
                   train_seg, seg_cap, seg_epochs):
    global training_status

    # Returns (status_message, timer_update). The log-refresh Timer is only
    # activated while a run is actually in progress so it does not enqueue jobs
    # forever (an always-on Timer keeps the queue busy and leaves every other
    # component stuck in a perpetual "processing" state).
    def result(msg, active):
        return msg, gr.Timer(active=active)

    if training_status == "Training...":
        return result("Training is already in progress!", True)

    # Validate hyperparameters, lr is a free-form Number input and the others
    # can arrive as None/0 from the UI, which would crash the optimizer or loader.
    try:
        num_samples = int(num_samples)
        epochs      = int(epochs)
        batch_size  = int(batch_size)
        lr          = float(lr)
        num_workers = int(num_workers)
        train_seg   = bool(train_seg)
        seg_cap     = int(seg_cap)
        seg_epochs  = int(seg_epochs)
    except (TypeError, ValueError):
        return result("Invalid hyperparameters: please enter numeric values.", False)

    if num_samples < 1:
        return result("Dataset size must be at least 1.", False)
    if epochs < 1:
        return result("Epochs must be at least 1.", False)
    if batch_size < 1:
        return result("Batch size must be at least 1.", False)
    if not (0 < lr < 1):
        return result("Learning rate must be between 0 and 1 (e.g. 0.0001).", False)
    if num_workers < 0:
        return result("Data loader workers cannot be negative.", False)
    if train_seg and (seg_cap < 8 or seg_epochs < 1):
        return result("Segmentation needs cap >= 8 and epochs >= 1.", False)

    # Mark active synchronously so the first Timer tick does not race the thread.
    training_status = "Training..."
    threading.Thread(
        target=run_training_thread,
        args=(num_samples, epochs, lr, batch_size, num_workers),
        kwargs={"train_seg": train_seg, "seg_cap": seg_cap, "seg_epochs": seg_epochs},
    ).start()
    return result("Training started in background...", True)


def get_training_logs():
    # Third return value stops the Timer once the run reaches a terminal state.
    # The metrics box is re-read on every tick so it picks up the new scores as
    # soon as a run writes them, without needing a reload.
    return ("\n".join(training_logs), training_status,
            gr.Timer(active=_is_training_active()), build_metrics_html())


# Model loading / warm-up
def warm_up_model(m):
    """
    Run one dummy forward pass so PyTorch compiles its CPU convolution kernels
    now, at startup, instead of on the user's first click. This is the fix for
    the first-load freeze: without it the initial inference blocks for several
    seconds while oneDNN/MKL JIT-compiles, and Gradio just shows a spinner.
    """
    try:
        m.eval()
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        with torch.no_grad():
            m(dummy)
    except Exception as e:
        print(f"Warning: model warm-up failed (first inference may be slow): {e}")


def load_models_from_disk():
    """
    Load both checkpoints if present and warm them up. Returns
    (seg_model_or_None, cls_model_or_None). A failure loading one model does
    not prevent the other from loading.
    """
    seg = cls = None
    if os.path.exists(seg_model_path):
        try:
            seg = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES)
            seg.load_state_dict(torch.load(seg_model_path, map_location=device))
            seg.to(device)
            seg.eval()
            warm_up_model(seg)
        except Exception as e:
            print(f"Warning: could not load segmentation model: {e}")
            seg = None
    if os.path.exists(model_path):
        try:
            cls = CNNModel(classCount=NUM_CLASSES, isTrained=False)
            cls.load_state_dict(torch.load(model_path, map_location=device))
            cls.to(device)
            cls.eval()
            warm_up_model(cls)
        except Exception as e:
            print(f"Warning: could not load classifier model: {e}")
            cls = None
    return seg, cls


def build_heatmap(orig_np, mask, alpha=0.5):
    """
    Blend a translucent Grad-CAM heatmap over the full-resolution image.

    The mask (float in [0, 1]) is used both to colourise (JET) and as a
    per-pixel alpha, so only activated regions get colour and flat areas keep
    the original X-ray. This is a soft attention map, not a hard "finding"
    marker, so it never reads as a false lesion the way a filled block did.
    """
    orig_h, orig_w = orig_np.shape[:2]
    m = np.clip(mask, 0.0, 1.0).astype(np.float32)
    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Suppress diffuse low activation: rescale [floor, 1] to [0, 1] and zero the
    # rest, so only genuine attention gets coloured and healthy tissue keeps the
    # original grayscale instead of a full-image colour wash.
    m = np.where(m < HEATMAP_FLOOR, 0.0, (m - HEATMAP_FLOOR) / (1.0 - HEATMAP_FLOOR))

    heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)

    a = (alpha * m)[..., None]
    result = (orig_np.astype(np.float32) * (1.0 - a) + heat * a)
    return np.clip(result, 0, 255).astype(np.uint8)


def clean_region(mask, threshold, out_shape):
    """
    Threshold the mask on its RAW values (no min-max stretching, which forces a
    region even where the class is absent), remove speckle below
    MIN_REGION_AREA_FRAC, and resize to the original resolution. Returns a
    full-size binary region, or None if nothing survives.
    """
    binary = (mask > threshold).astype(np.uint8)
    if not binary.any():
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num, comp_map, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    min_area = MIN_REGION_AREA_FRAC * binary.size
    cleaned = np.zeros_like(binary)
    for comp in range(1, num):
        if stats[comp, cv2.CC_STAT_AREA] >= min_area:
            cleaned[comp_map == comp] = 1
    if not cleaned.any():
        return None

    orig_h, orig_w = out_shape[:2]
    return cv2.resize(cleaned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def outline_region(img, region):
    """Draw the region-of-interest outline on top of the (already heatmapped) image."""
    orig_h, orig_w = img.shape[:2]
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, round(min(orig_w, orig_h) / IMG_SIZE))
    cv2.drawContours(img, contours, -1, OVERLAY_COLOR, thickness)
    return img


# Inference
def predict_image(image, target_class_name):
    global seg_model, cls_model

    if seg_model is None and cls_model is None:
        # Serialise loads so a background training thread swapping the globals
        # cannot race a concurrent inference request. warm_up_model runs the
        # first forward pass at load time so the click does not block on JIT.
        try:
            with model_lock:
                if seg_model is None and cls_model is None:
                    seg_model, cls_model = load_models_from_disk()
        except Exception as e:
            gr.Warning(f"Failed to load model: {e}")
            return "", {}, gr.update(value=None)

    if seg_model is None and cls_model is None:
        gr.Warning("No trained model found. Please train a model first.")
        return "", {}, gr.update(value=None)

    if image is None:
        gr.Warning("Please upload a valid X-ray image.")
        return "", {}, gr.update(value=None)

    try:
        image = image.convert("RGB")
    except Exception as e:
        gr.Warning(f"Could not process image: {e}")
        return "", {}, gr.update(value=None)

    if image.getbbox() is None:
        gr.Warning("Uploaded image appears to be empty.")
        return "", {}, gr.update(value=None)

    # Match the training pipeline: strip the annotated outer margin so inference
    # sees the same framing the model was trained on. Crop the PIL image here
    # (not inside tf) so the displayed image and heatmap overlay align with it.
    image = BorderCrop()(image)

    try:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            _normalize_transform()])

        orig_np = np.array(image)

        # Classification comes from the DenseNet when available (the U-Net
        # head is only a fallback); masks come from the U-Net when available.
        pred_masks = None
        with torch.no_grad():
            t = tf(image).unsqueeze(0).to(device)
            seg_logits = None
            if seg_model is not None:
                seg_model.eval()
                seg_logits, pred_masks = seg_model(t)
            if cls_model is not None:
                cls_model.eval()
                cls_logits = cls_model(t)
            else:
                cls_logits = seg_logits
            probabilities = torch.softmax(cls_logits, 1)[0].cpu()

        if probabilities.shape[0] != len(CLASSES):
            raise RuntimeError(f"Probability shape mismatch: {probabilities.shape[0]} vs {len(CLASSES)}")
        top_idx  = int(torch.argmax(probabilities))
        top_cls  = CLASSES[top_idx]
        top_prob = float(probabilities[top_idx])

        # Native confidence bars (gr.Label), clearer than a markdown list.
        prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

        is_uncertain = top_prob < CONFIDENCE_THRESHOLD

        # Pick the class to visualise. Explicit dropdown choice wins. In auto
        # mode we normally visualise the prediction, but for a Normal (or
        # uncertain-Normal) result we instead show where the model assessed for
        # the most likely ABNORMAL class. Grad-CAM for "Normal" just lights up
        # central anatomy, which looks alarming and is not useful; showing the
        # lung-assessment map keeps the heatmap meaningful and never empty.
        is_auto = target_class_name not in CLASSES
        abnormal_probs = probabilities.clone()
        abnormal_probs[0] = -1.0
        top_abnormal_idx = int(torch.argmax(abnormal_probs))

        if not is_auto:
            viz_idx = CLASSES.index(target_class_name)
        elif top_idx != 0:
            viz_idx = top_idx
        else:
            viz_idx = top_abnormal_idx
        viz_prob = float(probabilities[viz_idx])
        viz_cls = CLASSES[viz_idx]

        # A confident, abnormal prediction is one where the visualised class is
        # the actual prediction; only then do we add the crisp region outline.
        is_confident_finding = (not is_uncertain and viz_idx == top_idx
                                and top_idx != 0 and viz_prob >= OVERLAY_MIN_PROB)

        # Compute the attention map. Grad-CAM from the DenseNet is preferred: it
        # marks the lung region that drove the score. The U-Net mask is only a
        # fallback (trained on brightness pseudo-masks, it tends to trace bone,
        # which is what made the old shading look wrong).
        mask = None
        threshold = MASK_DISPLAY_THRESHOLD
        mask_source = None
        # Prefer the U-Net disease mask when it is a fresh Grad-CAM-distilled
        # model: it reproduces the classifier's localisation in one forward pass
        # (no backward pass needed) and is the segmentation output the app is
        # meant to show. Fall back to live Grad-CAM, then to a stale U-Net mask.
        if (seg_disease_model and pred_masks is not None and viz_idx > 0):
            mask = pred_masks[0, viz_idx].cpu().numpy()
            threshold = MASK_DISPLAY_THRESHOLD
            mask_source = "segmentation"
        elif cls_model is not None:
            target_layer = get_gradcam_layer(cls_model)
            grad_cam     = GradCAM(cls_model, target_layer)
            grad_t       = tf(image).unsqueeze(0).to(device)
            grad_t.requires_grad_(True)
            mask = grad_cam.generate_heatmap(grad_t, viz_idx)
            threshold = OVERLAY_THRESHOLD
            mask_source = "gradcam"
        elif pred_masks is not None and viz_idx > 0:
            mask = pred_masks[0, viz_idx].cpu().numpy()
            mask_source = "segmentation"

        superimposed = orig_np
        has_region = False
        if mask is not None:
            superimposed = build_heatmap(orig_np, mask)
            if is_confident_finding:
                region = clean_region(mask, threshold, orig_np.shape)
                if region is not None:
                    superimposed = outline_region(superimposed, region)
                    has_region = True

        descriptions = {
            "Normal":       "No abnormal opacities detected in the lung fields.",
            "Pneumonia":    "Findings consistent with pneumonia. Warmer areas indicate possible consolidation.",
            "Tuberculosis": "Findings consistent with tuberculosis. Warmer areas indicate possible focal lesions or cavitation.",
            "Covid-19":     "Findings consistent with Covid-19. Warmer areas indicate possible bilateral ground-glass opacities.",
        }

        if is_uncertain:
            txt = (f"**Prediction:** Uncertain\n\n"
                   f"The top class is {top_cls} at {top_prob * 100:.1f}%, below the "
                   f"{CONFIDENCE_THRESHOLD * 100:.0f}% reporting threshold. Review the class "
                   f"confidence values; a repeat or higher quality image may help.")
        else:
            txt = (f"**Prediction:** {top_cls} ({top_prob * 100:.1f}% confidence)\n\n"
                   f"{descriptions[top_cls]}")

        # Explain what the heatmap represents for this specific case.
        if mask is None:
            txt += "\n\nNo attention map is available for this model."
        elif is_confident_finding:
            txt += (f"\n\nHeatmap: region driving the {viz_cls} prediction. "
                    f"The outline marks the most influential area.")
        elif is_auto and top_idx == 0:
            txt += (f"\n\nHeatmap: areas the model assessed for {viz_cls} "
                    f"(the next most likely class); none reached an abnormal level.")
        elif is_uncertain:
            txt += (f"\n\nHeatmap: areas the model weighed for {viz_cls}. "
                    f"Interpret with caution while the prediction is uncertain.")
        else:
            txt += f"\n\nHeatmap: model attention for {viz_cls} ({viz_prob * 100:.1f}%)."

        if mask is not None:
            txt += ("\n\n_Overlay source: disease-segmentation U-Net._"
                    if mask_source == "segmentation"
                    else "\n\n_Overlay source: Grad-CAM._")

        return txt, prob_dict, gr.update(value=Image.fromarray(superimposed))

    except Exception as e:
        gr.Warning(f"Diagnosis failed: {e}")
        return "", {}, gr.update(value=None)


# Startup model load. Loading + warming up here (rather than on the first click)
# is what removes the first-inference freeze described in the system plan.
# Skipped when LUNGLENS_SKIP_STARTUP=1 (headless training): a DataLoader worker
# re-imports this module on every Windows spawn, and warming up the models there
# would burn a GPU forward pass per worker per epoch for no benefit.
if os.environ.get("LUNGLENS_SKIP_STARTUP") == "1":
    seg_model = cls_model = None
    seg_disease_model = False
else:
    try:
        seg_model, cls_model = load_models_from_disk()
        # Only trust the on-disk seg model as a disease segmenter if the metrics
        # file says it was Grad-CAM-distilled; else it is the old anatomy tracer.
        _startup_metrics = load_training_metrics()
        seg_disease_model = bool(seg_model is not None and _startup_metrics
                                 and _startup_metrics.get("segmentation") == "gradcam_distilled")
        if seg_model is None and cls_model is None:
            print("No trained model found on disk. Train a model before inference.")
        else:
            loaded = [name for name, m in
                      [("U-Net segmentation", seg_model), ("DenseNet-121 classifier", cls_model)]
                      if m is not None]
            seg_kind = " (disease-distilled)" if seg_disease_model else ""
            print(f"Loaded and warmed up: {', '.join(loaded)}{seg_kind}.")
    except Exception as e:
        seg_model = cls_model = None
        seg_disease_model = False
        print(f"Warning: Could not load models at startup: {e}")


# UI
ll_css = """
:root {
    --ll-accent: #17557F;
    --ll-bg: #F2F4F6;
    --ll-surface: #FFFFFF;
    --ll-text: #1A2229;
    --ll-text-2: #55616C;
    --ll-muted: #7C8792;
    --ll-border: #D8DEE4;
}
.dark {
    --ll-accent: #6FA8CE;
    --ll-bg: #14181C;
    --ll-surface: #1B2127;
    --ll-text: #E4E9ED;
    --ll-text-2: #A9B4BD;
    --ll-muted: #76818B;
    --ll-border: #2C343C;
}

html, body, .gradio-container {
    background-color: var(--ll-bg) !important;
    font-family: "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif !important;
    min-height: 100vh !important; margin: 0 !important; padding: 0 !important;
}
.gradio-container { max-width: 1280px !important; width: 100% !important; margin: 0 auto !important; padding: 0 24px 40px !important; box-sizing: border-box !important; }

/* Header: plain title row with a hairline rule; the right side reports the
   loaded model, which is information, not decoration. */
.ll-header {
    display: flex; align-items: baseline; gap: 12px;
    padding: 18px 2px 12px;
    border-bottom: 1px solid var(--ll-border);
    margin-bottom: 18px;
}
.ll-header h1 { margin: 0; font-size: 16px; font-weight: 600; color: var(--ll-text); }
.ll-header .ll-sub { font-size: 13px; color: var(--ll-text-2); }
.ll-header .ll-status { margin-left: auto; font-size: 12px; color: var(--ll-muted); white-space: nowrap; }

/* Panels: flat surfaces, hairline border, small radius, no shadow */
.custom-panel {
    background: var(--ll-surface) !important;
    border: 1px solid var(--ll-border) !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    padding: 20px !important;
    margin-bottom: 14px !important;
}

.custom-panel h1, .custom-panel h2, .custom-panel h3, .custom-panel h4,
.custom-panel .gradio-markdown h3, .custom-panel .gradio-markdown strong {
    font-weight: 600 !important; letter-spacing: 0 !important; color: var(--ll-text) !important;
}
.custom-panel .gradio-markdown h3 { font-size: 14px !important; text-transform: none !important; }
.custom-panel p, .custom-panel label,
.custom-panel .gradio-markdown p, .custom-panel .gradio-markdown li {
    font-weight: 400 !important; color: var(--ll-text-2) !important;
}

/* Tabs: quiet text, accent only on the active tab */
.tab-nav button { color: var(--ll-text-2) !important; font-weight: 500 !important; }
.tab-nav button.selected { color: var(--ll-accent) !important; font-weight: 600 !important; }

/* Buttons: flat, 4px radius. Primary uses the accent; secondary stays neutral. */
.primary-btn {
    border-radius: 4px !important;
    background: var(--ll-accent) !important;
    color: #ffffff !important; font-weight: 600 !important; border: 1px solid var(--ll-accent) !important;
    padding: 10px 20px !important;
    box-shadow: none !important;
}
.primary-btn:hover { filter: brightness(1.08); }
.primary-btn:focus-visible { outline: 2px solid var(--ll-accent) !important; outline-offset: 2px !important; }
/* The dark palette uses a light accent, so the button label must be dark to
   keep readable contrast. */
.dark .primary-btn { color: #10181F !important; }
.secondary-btn {
    border-radius: 4px !important;
    background: var(--ll-surface) !important;
    color: var(--ll-text) !important; font-weight: 500 !important;
    border: 1px solid var(--ll-border) !important;
    padding: 10px 20px !important;
    box-shadow: none !important;
}
.secondary-btn:hover { border-color: var(--ll-muted) !important; }
.secondary-btn:focus-visible { outline: 2px solid var(--ll-accent) !important; outline-offset: 2px !important; }

/* Image wells: solid hairline instead of a dashed drop-zone */
.upload-zone .gradio-image, .upload-zone [data-testid="image"] {
    border: 1px solid var(--ll-border) !important; border-radius: 4px !important; background: transparent !important;
}
.upload-zone .gradio-image:focus-within { border-color: var(--ll-accent) !important; }

.conf-label, .conf-label .gr-label { border-radius: 4px !important; }

/* Model metrics box: quiet inset panel, numbers lead, labels follow. */
.ll-metrics {
    border: 1px solid var(--ll-border);
    border-radius: 4px;
    padding: 12px 14px;
    background: var(--ll-bg);
}
.ll-metrics-head {
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--ll-muted); margin-bottom: 8px;
}
.ll-metrics-head-2 { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--ll-border); }
.ll-metric { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.ll-metric-v { font-size: 20px; font-weight: 600; color: var(--ll-text); line-height: 1.2; }
.ll-metric-na { font-size: 14px; font-weight: 500; color: var(--ll-muted); }
.ll-metric-k { font-size: 12px; color: var(--ll-text-2); }
.ll-metrics-note {
    font-size: 12px !important; color: var(--ll-muted) !important;
    margin: 6px 0 0 !important; line-height: 1.5 !important;
}

.status-pill textarea, .status-pill input { font-weight: 600 !important; }

.disclaimer-note {
    font-size: 12px !important; color: var(--ll-muted) !important; text-align: left !important;
    margin-top: 8px !important; margin-bottom: 0 !important; line-height: 1.5 !important;
}

footer { display: none !important; }

/* ----- Mobile / responsive ----- */
@media (max-width: 768px) {
    .gradio-container { padding: 0 12px 28px !important; }

    .gradio-container .row,
    .gradio-container div[class*="row"] { flex-direction: column !important; flex-wrap: wrap !important; }
    .gradio-container .column,
    .gradio-container div[class*="column"] { min-width: 100% !important; }

    .custom-panel { padding: 14px !important; }

    .ll-header { flex-wrap: wrap; gap: 6px; }
    .ll-header .ll-status { margin-left: 0; flex-basis: 100%; }

    .upload-zone .gradio-image,
    .upload-zone [data-testid="image"] { min-height: 220px !important; }

    .primary-btn, .secondary-btn { width: 100% !important; padding: 12px 16px !important; }
}
"""


def build_header_html():
    """Header text reports which models are actually loaded."""
    parts = []
    if cls_model is not None:
        parts.append("DenseNet-121 classifier")
    if seg_model is not None:
        parts.append("U-Net segmentation")
    status = f"Models: {' + '.join(parts)}" if parts else "No model loaded"
    return f"""
<div class="ll-header">
  <h1>LungLens</h1>
  <span class="ll-sub">Chest X-ray classification and region overlay</span>
  <span class="ll-status">{status} &middot; Research use only</span>
</div>
"""


def build_metrics_html():
    """
    Status box for the served DenseNet classifier: its held-out test score (the
    honest number, since test is never used for checkpoint selection), per-class
    recall, and a per-source probe for the class/scanner confounding. When no
    metrics file matches the checkpoint on disk the score is reported as
    unrecorded rather than guessed.
    """
    metrics = load_training_metrics()
    if metrics is None:
        body = (
            "<div class='ll-metric'><span class='ll-metric-v ll-metric-na'>Not recorded</span>"
            "<span class='ll-metric-k'>Model accuracy</span></div>"
            "<p class='ll-metrics-note'>The classifier on disk predates metric tracking, "
            "or was trained elsewhere. Run a training pass to record its scores.</p>"
        )
    else:
        # Prefer the held-out test figures; fall back to validation for runs too
        # small to carve a test split.
        acc = metrics.get("test_acc", metrics.get("val_acc"))
        f1 = metrics.get("test_f1", metrics.get("val_f1"))
        scope = "held-out test" if "test_acc" in metrics else "validation"
        recalls = metrics.get("val_per_class_recall", {})
        recall_str = ", ".join(f"{k} {v:.0f}%" for k, v in recalls.items()) if recalls else ""
        src = metrics.get("test_per_source_acc", {})
        src_str = ", ".join(f"{s} {v['acc']:.0f}%" for s, v in src.items()) if src else ""

        rows = (
            f"<div class='ll-metric'><span class='ll-metric-v'>{acc:.1f}%</span>"
            f"<span class='ll-metric-k'>Accuracy ({scope})</span></div>"
        )
        if f1 is not None:
            rows += (
                f"<div class='ll-metric'><span class='ll-metric-v'>{f1:.1f}%</span>"
                f"<span class='ll-metric-k'>Macro-F1 &mdash; the number to watch under class imbalance</span></div>"
            )
        note = (
            f"<p class='ll-metrics-note'>DenseNet-121, best of "
            f"{metrics.get('epochs_requested', '?')} epochs (epoch {metrics.get('epoch', '?')}"
            f"{', early-stopped' if metrics.get('early_stopped') else ''}), "
            f"{metrics.get('train_samples', '?')} train / {metrics.get('val_samples', '?')} val / "
            f"{metrics.get('test_samples', '?')} test images, batch "
            f"{metrics.get('batch_size', '?')}, lr {metrics.get('lr', 0):.0e}. "
            f"Recorded {metrics.get('saved_at', 'unknown')}.</p>"
        )
        if recall_str:
            note += f"<p class='ll-metrics-note'>Per-class recall (val): {recall_str}.</p>"
        if src_str:
            note += (
                f"<p class='ll-metrics-note'>Accuracy by source dataset: {src_str}. "
                f"A large gap here suggests the model is keying on the scanner, not "
                f"the pathology.</p>"
            )
        if metrics.get("segmentation") == "gradcam_distilled":
            note += (
                f"<p class='ll-metrics-note'>Segmentation: disease U-Net distilled from "
                f"Grad-CAM (Dice {metrics.get('seg_val_dice', 0):.3f} vs its Grad-CAM target, "
                f"{metrics.get('seg_train_images', '?')} images). Drives the Analysis overlay.</p>"
            )
        else:
            note += (
                "<p class='ll-metrics-note'>Segmentation: none trained yet; the overlay "
                "uses live Grad-CAM. Train with segmentation enabled to add the disease U-Net.</p>"
            )
        body = rows + note

    return f"""
<div class="ll-metrics">
  <div class="ll-metrics-head">Current model &mdash; the served classifier</div>
  {body}
  <div class="ll-metrics-head ll-metrics-head-2">Recommended run</div>
  <p class="ll-metrics-note">
    {RECOMMENDED_SAMPLES} images &middot; {RECOMMENDED_EPOCHS} epochs &middot;
    batch {RECOMMENDED_BATCH} &middot; lr {RECOMMENDED_LR:.0e}.
    Use as much data as you can: the minority classes (TB, Covid-19) only become
    well represented in the val/test folds at scale. Early stopping means
    over-requesting epochs is safe. Expect several hours on CPU.
  </p>
</div>
"""


# Clears the input and all result components. Uses typed gr.update(value=None)
# so Gradio always receives a proper update dict for each component.
def reset_view():
    return (
        gr.update(value=None),   # input_img
        gr.update(value=None),   # output_heatmap
        gr.update(value=""),     # output_markdown
        gr.update(value=None),   # output_label
    )


# Gradio 6 requires Font objects here; plain strings crash launch() with
# AttributeError inside the built-in theme comparison.
ll_theme = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.Font("Segoe UI"), gr.themes.Font("system-ui"),
          gr.themes.Font("sans-serif")],
)

with gr.Blocks(title="LungLens", fill_width=True, theme=ll_theme) as demo:
    gr.HTML(build_header_html())

    with gr.Tab("Analysis"):
        # Single always-visible dashboard: input on the left, results on the
        # right. Container visibility is never toggled during an event, which is
        # what previously aborted the result stream and left the UI spinning.
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="custom-panel"):
                gr.Markdown("### Input")
                input_img = gr.Image(type="pil", label="", elem_classes="upload-zone", height=340)
                target_viz = gr.Dropdown(choices=[AUTO_OVERLAY] + CLASSES, value=AUTO_OVERLAY,
                                         label="Overlay class")
                gr.HTML("<p class='disclaimer-note'>The heatmap shows where the model focused "
                        "for the predicted class, or a specific class if one is selected. "
                        "For research use only; not a substitute for professional "
                        "medical diagnosis.</p>")
                with gr.Row():
                    predict_btn = gr.Button("Analyze", variant="primary", elem_classes="primary-btn")
                    reset_btn   = gr.Button("Clear", elem_classes="secondary-btn")

            with gr.Column(scale=1, elem_classes="custom-panel"):
                gr.Markdown("### Results")
                output_heatmap = gr.Image(label="Attention heatmap", value=None, height=340)
                output_label = gr.Label(label="Class confidence", num_top_classes=4,
                                        elem_classes="conf-label")
                output_markdown = gr.Markdown()

        predict_btn.click(fn=predict_image, inputs=[input_img, target_viz],
                          outputs=[output_markdown, output_label, output_heatmap])
        reset_btn.click(fn=reset_view, inputs=[],
                        outputs=[input_img, output_heatmap, output_markdown, output_label])

    with gr.Tab("Training"):
        with gr.Column(elem_classes="custom-panel"):
            gr.Markdown("### Train the classifier + disease segmentation\nStage 1 trains the "
                        "DenseNet-121 that serves predictions on the Analysis tab. Stage 2 "
                        "(optional) distils its Grad-CAM into a U-Net that segments the disease "
                        "region, which then drives the overlay. Status and logs update "
                        "automatically while a run is active.")
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples_slider = gr.Slider(200, 32000, value=RECOMMENDED_SAMPLES,
                                                   step=200, label="Dataset size (images to use)")
                    epochs_slider      = gr.Slider(1, 40, value=RECOMMENDED_EPOCHS, step=1,
                                                   label="Max epochs (early-stops on plateau)")
                    batch_size_slider  = gr.Slider(8, 64, value=RECOMMENDED_BATCH, step=8, label="Batch size")
                    lr_input           = gr.Number(value=RECOMMENDED_LR, label="Learning rate", precision=6)
                    workers_slider     = gr.Slider(0, 8, value=0, step=1,
                                                   label="Data loader workers (raise to speed up loading)")
                    seg_checkbox       = gr.Checkbox(value=True,
                                                     label="Stage 2: train disease segmentation (Grad-CAM distillation)")
                    seg_cap_slider     = gr.Slider(50, 3000, value=800, step=50,
                                                   label="Segmentation images (Grad-CAM targets; slow on CPU)")
                    seg_epochs_slider  = gr.Slider(1, 20, value=6, step=1, label="Segmentation epochs")
                    train_btn          = gr.Button("Start training", variant="primary", elem_classes="primary-btn")
                    status_box         = gr.Textbox(value=training_status, label="Status",
                                                    interactive=False, elem_classes="status-pill")
                    metrics_box        = gr.HTML(build_metrics_html())
                with gr.Column(scale=2):
                    log_box     = gr.Textbox(value="", label="Training log", interactive=False,
                                             lines=18, max_lines=30, autoscroll=True)
                    refresh_btn = gr.Button("Refresh", elem_classes="secondary-btn")

        # Auto-refresh logs/status while training runs. The Timer starts inactive
        # and is only switched on for the duration of a run; an always-on Timer
        # keeps the queue permanently busy and freezes the rest of the UI in a
        # "processing" state.
        log_timer = gr.Timer(2.0, active=False)
        log_timer.tick(fn=get_training_logs, inputs=[],
                       outputs=[log_box, status_box, log_timer, metrics_box])

        train_btn.click(fn=start_training,
                        inputs=[num_samples_slider, epochs_slider, lr_input,
                                batch_size_slider, workers_slider,
                                seg_checkbox, seg_cap_slider, seg_epochs_slider],
                        outputs=[status_box, log_timer])
        refresh_btn.click(fn=get_training_logs, inputs=[],
                          outputs=[log_box, status_box, log_timer, metrics_box])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="127.0.0.1", server_port=port, share=False, css=ll_css)
