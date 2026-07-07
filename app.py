import os
import glob
import random
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import threading

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
NUM_CLASSES = len(CLASSES)
model_path = "chest_model_4class.pth"
seg_model_path = "chest_segmentation_model.pth"
training_status = "Not Training"
training_logs = []
# Hybrid inference: the DenseNet checkpoint is a strong discriminative
# classifier, while the U-Net checkpoint provides segmentation masks but has a
# collapsed classification head (it outputs a near-constant prior). Each model
# does the job it is actually good at.
seg_model = None   # MultiTaskUNet: segmentation masks, fallback classifier
cls_model = None   # DenseNet-121: primary classifier, Grad-CAM fallback
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
SEG_LOSS_WEIGHT = 2.0
OVERLAY_THRESHOLD = 0.45        # Grad-CAM maps are relative, threshold after normalising
MASK_DISPLAY_THRESHOLD = 0.5    # U-Net masks are probabilities, threshold the raw sigmoid
CONFIDENCE_THRESHOLD = 0.60     # Below this top-class probability, report Uncertain
OVERLAY_MIN_PROB = 0.15         # Do not draw an overlay for a class this improbable
HEATMAP_FLOOR = 0.35            # Hide diffuse low activation so healthy areas stay clean
MIN_REGION_AREA_FRAC = 0.003    # Drop overlay specks smaller than 0.3% of the image
OVERLAY_COLOR = (0, 113, 227)  # RGB clinical blue used for the segmentation overlay
AUTO_OVERLAY = "Auto (predicted class)"


def _normalize_transform():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


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
    y_bin  = (y_pred > 0.5).float()
    inter  = (y_bin * y_true).sum(dim=(2, 3))
    union  = y_bin.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice   = (2.0 * inter + smooth) / (union + smooth)
    result = dice.mean().item()
    if torch.isnan(torch.tensor(result)) or torch.isinf(torch.tensor(result)):
        return 0.0
    return result


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
def get_dataset_paths():
    tb  = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    pn  = kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images")
    cov = kagglehub.dataset_download("raddar/ricord-covid19-xray-positive-tests")
    return tb, pn, cov

def collect_data():
    tb_base, pn_base, cov_base = get_dataset_paths()
    paths, labels = [], []

    for f in glob.glob(os.path.join(tb_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
        fl = f.lower()
        if 'normal' in fl:       paths.append(f); labels.append(0)
        elif 'tuberculosis' in fl or 'tb' in fl: paths.append(f); labels.append(2)

    for f in glob.glob(os.path.join(pn_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
        fl = f.lower()
        if 'normal' in fl:    paths.append(f); labels.append(0)
        elif 'pneumonia' in fl: paths.append(f); labels.append(1)

    for f in glob.glob(os.path.join(cov_base, "**", "*.*"), recursive=True):
        if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
        paths.append(f); labels.append(3)

    custom_base = os.path.join(os.path.dirname(__file__), "custom_dataset")
    if os.path.exists(custom_base):
        for f in glob.glob(os.path.join(custom_base, "**", "*.*"), recursive=True):
            if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
            fl = f.lower()
            if 'normal' in fl:      paths.append(f); labels.append(0)
            elif 'pneumonia' in fl: paths.append(f); labels.append(1)
            elif 'tb' in fl or 'tuberculosis' in fl: paths.append(f); labels.append(2)
            elif 'covid' in fl:     paths.append(f); labels.append(3)

    combined = list(zip(paths, labels))
    if not combined:
        print("WARNING: No images found in datasets!")
        return [], []
    random.shuffle(combined)
    paths, labels = zip(*combined)
    return list(paths), list(labels)


# Training
def run_training_thread(num_samples, epochs, lr, batch_size):
    global training_status, training_logs, seg_model
    training_status = "Training..."
    training_logs = []
    writer = None

    try:
        writer = SummaryWriter("runs/lunglens")
        writer.add_scalar("Training/started", 1, 0)
        writer.flush()
        training_logs.append("TensorBoard logging to runs/lunglens")
        training_logs.append("Collecting dataset paths...")
        image_paths, labels = collect_data()
        training_logs.append(f"Total available images: {len(image_paths)}")

        if num_samples < len(image_paths):
            image_paths = list(image_paths[:num_samples])
            labels      = list(labels[:num_samples])

        training_logs.append(f"Using {len(image_paths)} images for training/validation split.")

        stratify_labels = None
        if len(image_paths) >= 10 and len(set(labels)) > 1:
            try:
                label_counts = np.bincount(labels)
                if np.min(label_counts) >= 2:
                    stratify_labels = labels
                else:
                    training_logs.append("WARNING: Class imbalance detected, using random split")
            except Exception as e:
                training_logs.append(f"WARNING: Stratification check failed: {e}")

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=VAL_SPLIT, random_state=42, stratify=stratify_labels)

        training_logs.append(f"Train samples: {len(train_paths)} | Val samples: {len(val_paths)}")

        # Only photometric augmentation is applied: the pseudo-mask is computed
        # from the original image, so geometric transforms (rotation/flip) would
        # misalign the input and its segmentation target. Brightness/contrast
        # jitter changes pixel intensities without moving them, so it is safe.
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            _normalize_transform()])

        val_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            _normalize_transform()])

        # SegmentationDataset, masks built lazily in __getitem__, no pre-pass
        training_logs.append("Building datasets (lazy pseudo-mask generation)...")
        train_ds = SegmentationDataset(train_paths, train_labels, transform=train_tf)
        val_ds   = SegmentationDataset(val_paths,   val_labels,   transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        # Train on a local network; the global seg_model is only swapped after
        # the run finishes so inference never sees a half-trained model.
        training_logs.append("Initializing MultiTaskUNet...")
        net = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

        # Inverse-frequency class weights so the minority classes (typically
        # Covid-19) are not drowned out by the majority Normal/Pneumonia images.
        # Weights are normalised to mean 1.0 to keep the loss scale stable.
        counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
        inv = 1.0 / np.clip(counts, 1.0, None)
        class_weights = torch.tensor(
            inv / inv.mean(), dtype=torch.float32, device=device)
        training_logs.append(
            "Class counts: "
            + ", ".join(f"{CLASSES[i]}={int(counts[i])}" for i in range(NUM_CLASSES))
        )

        class_crit = nn.CrossEntropyLoss(weight=class_weights)
        seg_crit   = DiceBCELoss()
        optimizer  = optim.Adam(net.parameters(), lr=lr)
        # Reduce LR when validation Dice plateaus so the segmentation head keeps
        # improving instead of stalling at a fixed learning rate.
        scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2)

        best_val_dice = 0.0
        history = {
            "train_acc": [],
            "val_acc": [],
            "train_dice": [],
            "val_dice": [],
        }

        for epoch in range(epochs):
            net.train()
            run_loss = correct = total = 0
            total_dice = 0.0

            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Training ({len(train_loader)} batches)...")
            for batch_idx, (inputs, target_masks, target_labels) in enumerate(train_loader):
                try:
                    inputs        = inputs.to(device)
                    target_masks  = target_masks.to(device)
                    target_labels = target_labels.to(device)

                    optimizer.zero_grad()
                    cls_logits, pred_masks = net(inputs)
                    loss = class_crit(cls_logits, target_labels) + SEG_LOSS_WEIGHT * seg_crit(pred_masks, target_masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        training_logs.append(f"  WARNING Batch {batch_idx+1}: NaN/Inf loss detected: {loss.item()}")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()

                    run_loss += loss.item()
                    _, pred = cls_logits.max(1)
                    total   += target_labels.size(0)
                    correct += pred.eq(target_labels).sum().item()
                    total_dice += dice_coefficient(pred_masks, target_masks)
                    batch_step = epoch * len(train_loader) + batch_idx + 1
                    if (batch_idx + 1) % max(1, len(train_loader) // 20) == 0:
                        writer.add_scalar("Loss/train_batch", loss.item(), batch_step)
                        writer.add_scalar("Accuracy/train_running", 100.0 * correct / total, batch_step)
                        writer.add_scalar("Dice/train_running", total_dice / (batch_idx + 1), batch_step)
                        writer.flush()

                    if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                        print(f"  Train batch {batch_idx+1}/{len(train_loader)}", flush=True)
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    training_logs.append(f"  ERROR in batch {batch_idx+1}: {str(e)}")
                    training_logs.append(error_trace)
                    print(f"Batch error: {e}\n{error_trace}", flush=True)
                    continue

            train_acc  = 100.0 * correct / total if total > 0 else 0
            train_dice = total_dice / max(1, len(train_loader))

            net.eval()
            val_loss = val_correct = val_total = 0
            val_dice_sum = 0.0

            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Validation ({len(val_loader)} batches)...")
            with torch.no_grad():
                for batch_idx, (inputs, target_masks, target_labels) in enumerate(val_loader):
                    try:
                        inputs        = inputs.to(device)
                        target_masks  = target_masks.to(device)
                        target_labels = target_labels.to(device)

                        cls_logits, pred_masks = net(inputs)
                        loss = class_crit(cls_logits, target_labels) + SEG_LOSS_WEIGHT * seg_crit(pred_masks, target_masks)

                        if torch.isnan(loss) or torch.isinf(loss):
                            training_logs.append(f"  WARNING Val Batch {batch_idx+1}: NaN/Inf loss {loss.item()}")
                            continue

                        val_loss  += loss.item()
                        _, pred    = cls_logits.max(1)
                        val_total += target_labels.size(0)
                        val_correct += pred.eq(target_labels).sum().item()
                        dice_val = dice_coefficient(pred_masks, target_masks)
                        val_dice_sum += dice_val
                        val_batch_step = epoch * len(val_loader) + batch_idx + 1
                        if (batch_idx + 1) % max(1, len(val_loader) // 20) == 0:
                            writer.add_scalar("Loss/val_batch", loss.item(), val_batch_step)
                            writer.add_scalar("Accuracy/val_running", 100.0 * val_correct / val_total, val_batch_step)
                            writer.add_scalar("Dice/val_running", val_dice_sum / (batch_idx + 1), val_batch_step)
                            writer.flush()

                        if (batch_idx + 1) % max(1, len(val_loader) // 5) == 0:
                            print(f"  Val batch {batch_idx+1}/{len(val_loader)}", flush=True)
                    except Exception as e:
                        training_logs.append(f"  ERROR in val batch {batch_idx+1}: {str(e)}")
                        print(f"Val batch error: {e}", flush=True)

            val_acc  = 100.0 * val_correct / val_total if val_total > 0 else 0
            val_dice = val_dice_sum / max(1, len(val_loader))

            # Step the scheduler on validation Dice and report the active LR.
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_lr, epoch + 1)

            log = (f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | "
                   f"Train Dice: {train_dice:.4f} | Val Acc: {val_acc:.2f}% | "
                   f"Val Dice: {val_dice:.4f} | LR: {current_lr:.2e}")
            training_logs.append(log)
            print(log, flush=True)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            tensorboard_step = epoch + 1
            writer.add_scalar("Accuracy/train", train_acc, tensorboard_step)
            writer.add_scalar("Accuracy/val", val_acc, tensorboard_step)
            writer.add_scalar("Dice/train", train_dice, tensorboard_step)
            writer.add_scalar("Dice/val", val_dice, tensorboard_step)
            writer.flush()

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(net.state_dict(), seg_model_path)
                training_logs.append(f"--> Saved best model (Val Dice: {best_val_dice:.4f})")

        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_acc"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_acc"], "b-o", label="Train Acc")
        ax1.plot(epochs_range, history["val_acc"], "r-o", label="Val Acc")
        ax1.set_title("Classification Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs_range, history["train_dice"], "b-o", label="Train Dice")
        ax2.plot(epochs_range, history["val_dice"], "r-o", label="Val Dice")
        ax2.set_title("Segmentation Dice")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice Score")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        plt.show(block=False)
        plt.close(fig)
        print("Graph saved as training_curves.png", flush=True)
        training_logs.append("Graph saved as training_curves.png")
        writer.close()
        writer = None

        # Publish the best checkpoint for inference only after the run ends so
        # requests never see a half-trained model in train mode.
        publish = net
        if os.path.exists(seg_model_path):
            try:
                publish = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES)
                publish.load_state_dict(torch.load(seg_model_path, map_location=device))
                publish.to(device)
            except Exception as e:
                training_logs.append(f"WARNING: could not reload best checkpoint: {e}")
                publish = net
        publish.eval()
        warm_up_model(publish)
        with model_lock:
            seg_model = publish

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


def start_training(num_samples, epochs, lr, batch_size):
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

    # Mark active synchronously so the first Timer tick does not race the thread.
    training_status = "Training..."
    threading.Thread(target=run_training_thread,
                     args=(num_samples, epochs, lr, batch_size)).start()
    return result("Training started in background...", True)


def get_training_logs():
    # Third return value stops the Timer once the run reaches a terminal state.
    return "\n".join(training_logs), training_status, gr.Timer(active=_is_training_active())


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
        if cls_model is not None:
            target_layer = get_gradcam_layer(cls_model)
            grad_cam     = GradCAM(cls_model, target_layer)
            grad_t       = tf(image).unsqueeze(0).to(device)
            grad_t.requires_grad_(True)
            mask = grad_cam.generate_heatmap(grad_t, viz_idx)
            threshold = OVERLAY_THRESHOLD
        elif pred_masks is not None and viz_idx > 0:
            mask = pred_masks[0, viz_idx].cpu().numpy()

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

        return txt, prob_dict, gr.update(value=Image.fromarray(superimposed))

    except Exception as e:
        gr.Warning(f"Diagnosis failed: {e}")
        return "", {}, gr.update(value=None)


# Startup model load. Loading + warming up here (rather than on the first click)
# is what removes the first-inference freeze described in the system plan.
try:
    seg_model, cls_model = load_models_from_disk()
    if seg_model is None and cls_model is None:
        print("No trained model found on disk. Train a model before inference.")
    else:
        loaded = [name for name, m in
                  [("U-Net segmentation", seg_model), ("DenseNet-121 classifier", cls_model)]
                  if m is not None]
        print(f"Loaded and warmed up: {', '.join(loaded)}.")
except Exception as e:
    seg_model = cls_model = None
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
            gr.Markdown("### Train the multi-task U-Net\nSet parameters and start a run. "
                        "Status and logs update automatically while training is active.")
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples_slider = gr.Slider(100, 10000, value=1000, step=100, label="Dataset size")
                    epochs_slider      = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                    batch_size_slider  = gr.Slider(8, 64, value=16, step=8, label="Batch size")
                    lr_input           = gr.Number(value=0.0001, label="Learning rate", precision=6)
                    train_btn          = gr.Button("Start training", variant="primary", elem_classes="primary-btn")
                    status_box         = gr.Textbox(value=training_status, label="Status",
                                                    interactive=False, elem_classes="status-pill")
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
                       outputs=[log_box, status_box, log_timer])

        train_btn.click(fn=start_training,
                        inputs=[num_samples_slider, epochs_slider, lr_input, batch_size_slider],
                        outputs=[status_box, log_timer])
        refresh_btn.click(fn=get_training_logs, inputs=[],
                          outputs=[log_box, status_box, log_timer])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="127.0.0.1", server_port=port, share=False, css=ll_css)
