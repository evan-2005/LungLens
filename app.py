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
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared configuration
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAL_SPLIT = 0.2
SEG_LOSS_WEIGHT = 2.0
OVERLAY_THRESHOLD = 0.45
OVERLAY_COLOR = (0, 113, 227)  # RGB clinical blue used for the segmentation overlay


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
        self.classifier = nn.Linear(512, num_classes)
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
        cls = self.classifier(torch.flatten(self.avgpool(x4), 1))
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
    global training_status, training_logs, model
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

        training_logs.append("Initializing MultiTaskUNet...")
        model = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

        class_crit = nn.CrossEntropyLoss()
        seg_crit   = DiceBCELoss()
        optimizer  = optim.Adam(model.parameters(), lr=lr)
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
            model.train()
            run_loss = correct = total = 0
            total_dice = 0.0

            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Training ({len(train_loader)} batches)...")
            for batch_idx, (inputs, target_masks, target_labels) in enumerate(train_loader):
                try:
                    inputs        = inputs.to(device)
                    target_masks  = target_masks.to(device)
                    target_labels = target_labels.to(device)

                    optimizer.zero_grad()
                    cls_logits, pred_masks = model(inputs)
                    loss = class_crit(cls_logits, target_labels) + SEG_LOSS_WEIGHT * seg_crit(pred_masks, target_masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        training_logs.append(f"  WARNING Batch {batch_idx+1}: NaN/Inf loss detected: {loss.item()}")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            model.eval()
            val_loss = val_correct = val_total = 0
            val_dice_sum = 0.0

            training_logs.append(f"[Epoch {epoch+1}/{epochs}] Validation ({len(val_loader)} batches)...")
            with torch.no_grad():
                for batch_idx, (inputs, target_masks, target_labels) in enumerate(val_loader):
                    try:
                        inputs        = inputs.to(device)
                        target_masks  = target_masks.to(device)
                        target_labels = target_labels.to(device)

                        cls_logits, pred_masks = model(inputs)
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
                torch.save(model.state_dict(), seg_model_path)
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


# Inference
def predict_image(image, target_class_name):
    global model

    has_seg   = os.path.exists(seg_model_path)
    has_class = os.path.exists(model_path)

    if not has_seg and not has_class:
        gr.Warning("No trained model found. Please train a model first.")
        return "", {}, gr.update(value=None)

    # Load correct model type if not already loaded
    need_load = (model is None
                 or (has_seg   and not isinstance(model, MultiTaskUNet))
                 or (not has_seg and isinstance(model, MultiTaskUNet)))

    if need_load:
        try:
            if has_seg:
                model = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES)
                model.load_state_dict(torch.load(seg_model_path, map_location=device))
            else:
                model = CNNModel(classCount=NUM_CLASSES, isTrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            gr.Warning(f"Failed to load model: {e}")
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

        orig_img   = cv2.resize(np.array(image), (IMG_SIZE, IMG_SIZE))
        target_idx = CLASSES.index(target_class_name)

        model.eval()
        if isinstance(model, MultiTaskUNet):
            with torch.no_grad():
                t = tf(image).unsqueeze(0).to(device)
                cls_logits, pred_masks = model(t)
                probabilities = torch.softmax(cls_logits, 1)[0].cpu()
                mask = pred_masks[0, target_idx].cpu().numpy()
        else:
            with torch.no_grad():
                t = tf(image).unsqueeze(0).to(device)
                probabilities = torch.softmax(model(t), 1)[0].cpu()
            target_layer = get_gradcam_layer(model)
            grad_cam     = GradCAM(model, target_layer)
            grad_t       = tf(image).unsqueeze(0).to(device)
            grad_t.requires_grad_(True)
            mask = grad_cam.generate_heatmap(grad_t, target_idx)

        top_idx  = int(torch.argmax(probabilities))
        if top_idx < 0 or top_idx >= len(CLASSES):
            raise ValueError(f"Invalid prediction index {top_idx}, expected 0-{len(CLASSES)-1}")
        if probabilities.shape[0] != len(CLASSES):
            raise RuntimeError(f"Probability shape mismatch: {probabilities.shape[0]} vs {len(CLASSES)}")
        top_cls  = CLASSES[top_idx]
        probs    = [float(probabilities[i] * 100) for i in range(len(CLASSES))]

        # Native confidence bars (gr.Label), clearer than a markdown list.
        prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

        txt  = f"### Diagnostic Findings\n\nThe model found a **{probs[top_idx]:.1f}% probability of {top_cls}**.\n\n"
        descriptions = {
            "Normal":       "No significant abnormalities detected. The lungs appear clear.",
            "Pneumonia":    "Indications consistent with Pneumonia. The overlay highlights potential zones of consolidation.",
            "Tuberculosis": "Indications consistent with Tuberculosis. The overlay highlights potential focal lesions or cavities.",
            "Covid-19":     "Indications consistent with Covid-19. The overlay highlights potential bilateral ground-glass opacities.",
        }
        txt += descriptions[top_cls]

        # Build overlay
        superimposed = orig_img.copy()
        if target_idx > 0 and mask.max() > 0:
            # Normalize mask to [0, 1] to ensure highlights are visible even with low-confidence logits
            denom = mask.max() - mask.min()
            if denom > 0:
                mask = (mask - mask.min()) / denom
            else:
                mask = mask / mask.max()
            binary = (mask > OVERLAY_THRESHOLD).astype(np.uint8) * 255
            colored = np.zeros_like(orig_img)
            colored[mask > OVERLAY_THRESHOLD] = OVERLAY_COLOR
            cv2.addWeighted(colored, 0.4, superimposed, 1.0, 0, superimposed)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(superimposed, contours, -1, OVERLAY_COLOR, 2)

        return txt, prob_dict, gr.update(value=Image.fromarray(superimposed))

    except Exception as e:
        gr.Warning(f"Diagnosis failed: {e}")
        return "", {}, gr.update(value=None)


# Startup model load
if os.path.exists(seg_model_path):
    try:
        model = MultiTaskUNet(in_channels=3, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(seg_model_path, map_location=device))
        model.to(device); model.eval()
        print("Loaded MultiTaskUNet.")
    except Exception as e:
        print(f"Warning: Could not load U-Net: {e}")
elif os.path.exists(model_path):
    try:
        model = CNNModel(classCount=NUM_CLASSES, isTrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print("Loaded fallback CNN classifier.")
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}")


# UI
apple_css = """
:root {
    --ll-accent: #1F5C8B;
    --ll-accent-hi: #2A6FA3;
    --ll-bg: #EEF1F4;
    --ll-surface: #FFFFFF;
    --ll-text: #1B2733;
    --ll-text-2: #4A5A6A;
    --ll-muted: #8A97A4;
    --ll-border: rgba(20,40,60,0.10);
    --ll-radius: 12px;
}
.dark {
    --ll-accent: #4F9AD1;
    --ll-accent-hi: #62A9DC;
    --ll-bg: #11161C;
    --ll-surface: #1A222B;
    --ll-text: #E6ECF2;
    --ll-text-2: #B8C2CC;
    --ll-muted: #7E8A95;
    --ll-border: rgba(255,255,255,0.08);
}

html, body, .gradio-container {
    background-color: var(--ll-bg) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    min-height: 100vh !important; margin: 0 !important; padding: 0 !important;
}
.gradio-container { max-width: 1600px !important; width: 100% !important; margin: 0 auto !important; padding: 20px 32px 48px !important; box-sizing: border-box !important; }
@media (min-width: 1700px) { .gradio-container { max-width: 90vw !important; } }

/* Header */
.ll-header {
    display: flex; align-items: center; gap: 16px;
    background: var(--ll-surface);
    border: 1px solid var(--ll-border);
    border-top: 3px solid var(--ll-accent);
    border-radius: var(--ll-radius);
    padding: 20px 24px;
    margin: 4px 0 20px;
}
.ll-header .ll-mark {
    flex: 0 0 auto; width: 44px; height: 44px; border-radius: 10px;
    background: rgba(31,92,139,0.10); display: flex; align-items: center; justify-content: center;
}
.dark .ll-header .ll-mark { background: rgba(79,154,209,0.16); }
.ll-header .ll-mark svg { width: 26px; height: 26px; stroke: var(--ll-accent); }
.ll-header h1 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em; color: var(--ll-text); }
.ll-header p  { margin: 3px 0 0; font-size: 13.5px; color: var(--ll-text-2); }
.ll-header .ll-tag {
    margin-left: auto; align-self: flex-start;
    font-size: 11px; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;
    color: var(--ll-muted); border: 1px solid var(--ll-border); border-radius: 6px; padding: 4px 9px;
}

/* Panel card */
.custom-panel {
    background: var(--ll-surface) !important;
    border: 1px solid var(--ll-border) !important;
    border-radius: var(--ll-radius) !important;
    box-shadow: 0 1px 2px rgba(20,40,60,0.04) !important;
    padding: 22px !important;
    margin-bottom: 16px !important;
}

/* Typography (scoped to panels) */
.custom-panel h1, .custom-panel h2, .custom-panel h3, .custom-panel h4,
.custom-panel .gradio-markdown h1, .custom-panel .gradio-markdown h2,
.custom-panel .gradio-markdown h3, .custom-panel .gradio-markdown h4,
.custom-panel .gradio-markdown strong {
    font-weight: 650 !important; letter-spacing: -0.01em !important; color: var(--ll-text) !important;
}
.custom-panel p, .custom-panel label,
.custom-panel .gradio-markdown p, .custom-panel .gradio-markdown li {
    font-weight: 400 !important; color: var(--ll-text-2) !important;
}

/* Tab nav */
.tab-nav button { color: var(--ll-text-2) !important; font-weight: 600 !important; }
.tab-nav button.selected { color: var(--ll-accent) !important; }

/* Buttons */
.primary-btn {
    border-radius: 8px !important;
    background: var(--ll-accent) !important;
    color: #ffffff !important; font-weight: 600 !important; border: none !important;
    padding: 11px 22px !important;
    box-shadow: none !important;
    transition: background-color .15s ease !important;
}
.primary-btn:hover { background: var(--ll-accent-hi) !important; }
.primary-btn:focus-visible { outline: 2px solid var(--ll-accent-hi) !important; outline-offset: 2px !important; }

/* Upload zone */
.upload-zone .gradio-image, .upload-zone [data-testid="image"] {
    border: 1.5px dashed #C2CCD6 !important; border-radius: 10px !important; background: transparent !important;
    transition: border-color .15s ease, background .15s ease !important;
}
.dark .upload-zone .gradio-image { border-color: #3A4651 !important; }
.upload-zone .gradio-image:hover, .upload-zone .gradio-image:focus-within {
    border-color: var(--ll-accent) !important; background: rgba(31,92,139,0.03) !important;
}

/* Dropdown centring */
.viz-dropdown-row { justify-content: center !important; }
.viz-dropdown-row .gradio-dropdown { max-width: 340px !important; width: 100% !important; }

/* Confidence label bars */
.conf-label, .conf-label .gr-label { border-radius: 10px !important; }

/* Status field */
.status-pill textarea, .status-pill input { font-weight: 600 !important; }

/* Disclaimer */
.disclaimer-note {
    font-size: 12px !important; color: var(--ll-muted) !important; text-align: center !important;
    margin-top: 10px !important; margin-bottom: 0 !important; line-height: 1.5 !important; padding: 0 8px !important;
}

footer { display: none !important; }

/* Fade-in animation */
.fade-in { animation: fadeIn 0.4s ease both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
"""


HERO_HTML = """
<div class="ll-header">
  <span class="ll-mark">
    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 4v7"/>
      <path d="M8.5 8.5c0 4-2.5 5-3.5 7-0.8 1.6-0.5 4 1.5 4 1.8 0 2.5-1.4 2.5-3.2V11c0-1.4-1-2.5-2.5-2.5z"/>
      <path d="M15.5 8.5c0 4 2.5 5 3.5 7 0.8 1.6 0.5 4-1.5 4-1.8 0-2.5-1.4-2.5-3.2V11c0-1.4 1-2.5 2.5-2.5z"/>
    </svg>
  </span>
  <div>
    <h1>LungLens</h1>
    <p>Multi-task chest X-ray classification and region segmentation</p>
  </div>
  <span class="ll-tag">Research use only</span>
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


ll_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="LungLens", fill_width=True, theme=ll_theme) as demo:
    gr.HTML(HERO_HTML)

    with gr.Tab("Diagnostic Inference"):
        # Single always-visible dashboard: input on the left, results on the
        # right. Container visibility is never toggled during an event, which is
        # what previously aborted the result stream and left the UI spinning.
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_classes="custom-panel"):
                gr.Markdown("### Upload Scan")
                input_img = gr.Image(type="pil", label="", elem_classes="upload-zone", height=340)
                target_viz = gr.Dropdown(choices=CLASSES, value="Pneumonia",
                                         label="Target Visualization Class")
                gr.HTML("<p class='disclaimer-note'>The segmentation overlay highlights the detected region of interest "
                        "for the selected class, helping clinicians localise abnormalities. "
                        "This tool is not a substitute for professional medical diagnosis.</p>")
                with gr.Row():
                    predict_btn = gr.Button("Diagnose", variant="primary", elem_classes="primary-btn")
                    reset_btn   = gr.Button("Clear", elem_classes="primary-btn")

            with gr.Column(scale=1, elem_classes="custom-panel"):
                gr.Markdown("### Results")
                output_heatmap = gr.Image(label="Segmented Region of Interest", value=None, height=340)
                output_label = gr.Label(label="Class Confidence", num_top_classes=4,
                                        elem_classes="conf-label")
                output_markdown = gr.Markdown()

        predict_btn.click(fn=predict_image, inputs=[input_img, target_viz],
                          outputs=[output_markdown, output_label, output_heatmap])
        reset_btn.click(fn=reset_view, inputs=[],
                        outputs=[input_img, output_heatmap, output_markdown, output_label])

    with gr.Tab("Model Training"):
        with gr.Column(elem_classes="custom-panel"):
            gr.Markdown("### Train Segmentation Model (Multi-Task U-Net)\nAdjust parameters to fine-tune the model on your dataset. "
                        "Logs and status refresh automatically while training runs.")
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples_slider = gr.Slider(100, 10000, value=1000, step=100, label="Dataset Size")
                    epochs_slider      = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                    batch_size_slider  = gr.Slider(8, 64, value=16, step=8, label="Batch Size")
                    lr_input           = gr.Number(value=0.0001, label="Learning Rate", precision=6)
                    train_btn          = gr.Button("Start Training", variant="primary", elem_classes="primary-btn")
                    status_box         = gr.Textbox(value=training_status, label="Status",
                                                    interactive=False, elem_classes="status-pill")
                with gr.Column(scale=2):
                    log_box     = gr.Textbox(value="", label="Terminal Logs", interactive=False,
                                             lines=18, max_lines=30, autoscroll=True)
                    refresh_btn = gr.Button("Refresh Logs", elem_classes="primary-btn")

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
    demo.queue().launch(server_name="127.0.0.1", server_port=port, share=False, css=apple_css)
