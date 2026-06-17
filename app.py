import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import threading

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
model_path = "chest_model_4class.pth"
seg_model_path = "chest_segmentation_model.pth"
training_status = "Not Training"
training_logs = []
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Datasets ──────────────────────────────────────────────────────────────────

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
    Uses a fast threshold fallback — Grad-CAM per-image is too slow on CPU.
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

        target_mask = np.zeros((4, 224, 224), dtype=np.float32)
        if label > 0:
            try:
                gray = np.array(image.convert("L").resize((224, 224)))
                _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
                h, w = gray.shape
                lung_mask = np.zeros_like(gray)
                cv2.rectangle(lung_mask,
                              (int(w * 0.15), int(h * 0.15)),
                              (int(w * 0.85), int(h * 0.85)), 255, -1)
                thresh = cv2.bitwise_and(thresh, lung_mask)
                mask_values = (thresh / 255.0).astype(np.float32)
                target_mask[label] = np.clip(mask_values, 0.0, 1.0)
            except cv2.error as e:
                print(f"WARNING: cv2 operation failed for {img_path}: {e}")
            except Exception as e:
                print(f"WARNING: Mask generation failed for {img_path}: {e}")

        return img_tensor, torch.tensor(target_mask, dtype=torch.float32), label


# ── Models ────────────────────────────────────────────────────────────────────

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


# ── Losses ────────────────────────────────────────────────────────────────────

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


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

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


# ── Data collection ───────────────────────────────────────────────────────────

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


# ── Training ──────────────────────────────────────────────────────────────────

def run_training_thread(num_samples, epochs, lr, batch_size):
    global training_status, training_logs, model
    training_status = "Training..."
    training_logs = []

    try:
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
            image_paths, labels, test_size=0.2, random_state=42, stratify=stratify_labels)

        training_logs.append(f"Train samples: {len(train_paths)} | Val samples: {len(val_paths)}")

        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        # SegmentationDataset — masks built lazily in __getitem__, no pre-pass
        training_logs.append("Building datasets (lazy pseudo-mask generation)...")
        train_ds = SegmentationDataset(train_paths, train_labels, transform=train_tf)
        val_ds   = SegmentationDataset(val_paths,   val_labels,   transform=val_tf)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

        training_logs.append("Initializing MultiTaskUNet...")
        model = MultiTaskUNet(in_channels=3, num_classes=4).to(device)

        class_crit = nn.CrossEntropyLoss()
        seg_crit   = DiceBCELoss()
        optimizer  = optim.Adam(model.parameters(), lr=lr)

        best_val_dice = 0.0

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
                    loss = class_crit(cls_logits, target_labels) + 2.0 * seg_crit(pred_masks, target_masks)

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
                        loss = class_crit(cls_logits, target_labels) + 2.0 * seg_crit(pred_masks, target_masks)

                        if torch.isnan(loss) or torch.isinf(loss):
                            training_logs.append(f"  WARNING Val Batch {batch_idx+1}: NaN/Inf loss {loss.item()}")
                            continue

                        val_loss  += loss.item()
                        _, pred    = cls_logits.max(1)
                        val_total += target_labels.size(0)
                        val_correct += pred.eq(target_labels).sum().item()
                        dice_val = dice_coefficient(pred_masks, target_masks)
                        val_dice_sum += dice_val

                        if (batch_idx + 1) % max(1, len(val_loader) // 5) == 0:
                            print(f"  Val batch {batch_idx+1}/{len(val_loader)}", flush=True)
                    except Exception as e:
                        training_logs.append(f"  ERROR in val batch {batch_idx+1}: {str(e)}")
                        print(f"Val batch error: {e}", flush=True)

            val_acc  = 100.0 * val_correct / val_total if val_total > 0 else 0
            val_dice = val_dice_sum / max(1, len(val_loader))

            log = (f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | "
                   f"Train Dice: {train_dice:.4f} | Val Acc: {val_acc:.2f}% | Val Dice: {val_dice:.4f}")
            training_logs.append(log)
            print(log, flush=True)

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), seg_model_path)
                training_logs.append(f"--> Saved best model (Val Dice: {best_val_dice:.4f})")

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


def start_training(num_samples, epochs, lr, batch_size):
    global training_status
    if training_status == "Training...":
        return "Training is already in progress!"
    threading.Thread(target=run_training_thread,
                     args=(num_samples, epochs, lr, batch_size)).start()
    return "Training started in background..."

def get_training_logs():
    return "\n".join(training_logs), training_status


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(image, target_class_name):
    global model

    has_seg   = os.path.exists(seg_model_path)
    has_class = os.path.exists(model_path)

    if not has_seg and not has_class:
        gr.Warning("No trained model found. Please train a model first.")
        return "", None, gr.update(visible=True), gr.update(visible=False)

    # Load correct model type if not already loaded
    need_load = (model is None
                 or (has_seg   and not isinstance(model, MultiTaskUNet))
                 or (not has_seg and isinstance(model, MultiTaskUNet)))

    if need_load:
        try:
            if has_seg:
                model = MultiTaskUNet(in_channels=3, num_classes=4)
                model.load_state_dict(torch.load(seg_model_path, map_location=device))
            else:
                model = CNNModel(classCount=4, isTrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            gr.Warning(f"Failed to load model: {e}")
            return "", None, gr.update(visible=True), gr.update(visible=False)

    if image is None:
        gr.Warning("Please upload a valid X-ray image.")
        return "", None, gr.update(visible=True), gr.update(visible=False)

    try:
        image = image.convert("RGB")
    except Exception as e:
        gr.Warning(f"Could not process image: {e}")
        return "", None, gr.update(visible=True), gr.update(visible=False)

    if image.getbbox() is None:
        gr.Warning("Uploaded image appears to be empty.")
        return "", None, gr.update(visible=True), gr.update(visible=False)

    try:
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

        orig_img   = cv2.resize(np.array(image), (224, 224))
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

        txt  = f"### Diagnostic Findings\n\nThe model found a **{probs[top_idx]:.1f}% probability of {top_cls}**.\n\n"
        txt += f"- **Normal**: {probs[0]:.1f}%\n"
        txt += f"- **Pneumonia**: {probs[1]:.1f}%\n"
        txt += f"- **Tuberculosis**: {probs[2]:.1f}%\n"
        txt += f"- **Covid-19**: {probs[3]:.1f}%\n\n"
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
            binary = (mask > 0.45).astype(np.uint8) * 255
            colored = np.zeros_like(orig_img)
            colored[mask > 0.45] = (0, 113, 227)
            cv2.addWeighted(colored, 0.4, superimposed, 1.0, 0, superimposed)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(superimposed, contours, -1, (0, 113, 227), 2)

        return txt, Image.fromarray(superimposed), gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        gr.Warning(f"Diagnosis failed: {e}")
        return "", None, gr.update(visible=True), gr.update(visible=False)


# ── Startup model load ────────────────────────────────────────────────────────

if os.path.exists(seg_model_path):
    try:
        model = MultiTaskUNet(in_channels=3, num_classes=4)
        model.load_state_dict(torch.load(seg_model_path, map_location=device))
        model.to(device); model.eval()
        print("Loaded MultiTaskUNet.")
    except Exception as e:
        print(f"Warning: Could not load U-Net: {e}")
elif os.path.exists(model_path):
    try:
        model = CNNModel(classCount=4, isTrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print("Loaded fallback CNN classifier.")
    except Exception as e:
        print(f"Warning: Could not load classifier: {e}")


# ── UI ────────────────────────────────────────────────────────────────────────

apple_css = """
:root { --color-accent: #0071E3 !important; }
html, body, .gradio-container {
    background-color: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    min-height: 100vh !important; margin: 0 !important; padding: 0 !important;
}
.dark html, .dark body, .dark .gradio-container { background-color: #1C1C1E !important; }
.gradio-container { max-width: 100% !important; width: 100% !important; padding: 24px !important; box-sizing: border-box !important; }
.custom-panel {
    background: #FFFFFF !important; border: 1px solid rgba(0,0,0,0.05) !important;
    border-radius: 20px !important; box-shadow: 0 4px 24px rgba(0,0,0,0.02) !important;
    padding: 24px !important; margin-bottom: 16px !important;
}
.dark .custom-panel { background: #2C2C2E !important; border: 1px solid rgba(255,255,255,0.05) !important; }
h1,h2,h3,h4,.prose h1,.prose h2,.prose h3,.prose h4,.md h1,.md h2,.md h3,.md h4 {
    font-weight: 700 !important; letter-spacing: -0.015em !important; color: #1D1D1F !important;
}
.gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3, .gradio-markdown h4,
.gradio-markdown strong { color: #1D1D1F !important; }
.dark h1,.dark h2,.dark h3,.dark h4,
.dark .prose h1,.dark .prose h2,.dark .prose h3,.dark .prose h4,
.dark .md h1,.dark .md h2,.dark .md h3,.dark .md h4 { color: #F5F5F7 !important; }
p,label,.prose p,.md p,.gradio-markdown p { font-weight: 400 !important; color: #3a3a3c !important; }
.gradio-markdown li { color: #3a3a3c !important; }
.dark p,.dark label,.dark .prose p,.dark .md p,.dark .gradio-markdown p { color: #EBEBF5 !important; }
.dark .gradio-markdown h1,.dark .gradio-markdown h2,.dark .gradio-markdown h3,
.dark .gradio-markdown h4,.dark .gradio-markdown strong { color: #F5F5F7 !important; }
.dark .gradio-markdown p,.dark .gradio-markdown li { color: #EBEBF5 !important; }
.tab-nav button { color: #1D1D1F !important; }
.dark .tab-nav button { color: #F5F5F7 !important; }
.primary-btn {
    border-radius: 9999px !important; background-color: #0071E3 !important;
    color: white !important; font-weight: 600 !important; border: none !important;
    padding: 12px 24px !important; box-shadow: none !important;
    transition: transform 0.2s ease, background-color 0.2s ease !important;
}
.primary-btn:hover { background-color: #0077ED !important; transform: scale(1.02) !important; }
.upload-zone .gradio-image { border: 2px dashed #D2D2D7 !important; border-radius: 20px !important; background: transparent !important; }
.dark .upload-zone .gradio-image { border-color: #424245 !important; }
.upload-zone .gradio-image:hover, .upload-zone .gradio-image:focus-within { border-color: #0071E3 !important; }
.viz-dropdown-row { justify-content: center !important; }
.viz-dropdown-row .gradio-dropdown { max-width: 320px !important; width: 100% !important; }
.disclaimer-note {
    font-size: 12px !important; color: #86868B !important; text-align: center !important;
    margin-top: 6px !important; margin-bottom: 0 !important; line-height: 1.5 !important; padding: 0 8px !important;
}
footer { display: none !important; }
.fade-in { animation: fadeIn 0.8s cubic-bezier(0.16,1,0.3,1); }
@keyframes fadeIn { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
"""

def reset_view():
    return gr.update(visible=True), gr.update(visible=False), None, None

with gr.Blocks(title="LungLens", fill_width=True) as demo:
    gr.Markdown("# LungLens\nAdvanced Diagnostic Imaging")

    with gr.Tab("Diagnostic Inference"):
        with gr.Row(visible=True) as upload_view:
            gr.Column(scale=1)
            with gr.Column(scale=2, elem_classes="upload-zone fade-in custom-panel"):
                gr.Markdown("### Upload Scan")
                input_img = gr.Image(type="pil", label="", elem_classes="upload-zone")
                with gr.Row(elem_classes="viz-dropdown-row"):
                    target_viz = gr.Dropdown(choices=CLASSES, value="Pneumonia",
                                             label="Target Visualization Class")
                gr.HTML("<p class='disclaimer-note'>The segmentation overlay highlights the detected region of interest "
                        "for the selected class, helping clinicians localise abnormalities. "
                        "This tool is not a substitute for professional medical diagnosis.</p>")
                predict_btn = gr.Button("Diagnose", variant="primary", elem_classes="primary-btn")
            gr.Column(scale=1)

        with gr.Row(visible=False) as results_view:
            with gr.Column(scale=1, elem_classes="fade-in custom-panel"):
                output_heatmap = gr.Image(label="Segmented Region of Interest")
            with gr.Column(scale=1, elem_classes="fade-in custom-panel"):
                output_markdown = gr.Markdown()
                reset_btn = gr.Button("Analyze Another Scan", elem_classes="primary-btn")

        predict_btn.click(fn=predict_image, inputs=[input_img, target_viz],
                          outputs=[output_markdown, output_heatmap, upload_view, results_view])
        reset_btn.click(fn=reset_view, inputs=[],
                        outputs=[upload_view, results_view, input_img, output_heatmap])

    with gr.Tab("Model Training"):
        with gr.Column(elem_classes="custom-panel"):
            gr.Markdown("### Train Segmentation Model (Multi-Task U-Net)\nAdjust parameters to fine-tune the model on your dataset.")
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples_slider = gr.Slider(100, 10000, value=1000, step=100, label="Dataset Size")
                    epochs_slider      = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                    batch_size_slider  = gr.Slider(8, 64, value=16, step=8, label="Batch Size")
                    lr_input           = gr.Number(value=0.0001, label="Learning Rate", precision=6)
                    train_btn          = gr.Button("Start Training", elem_classes="primary-btn")
                    status_box         = gr.Textbox(value=training_status, label="Status", interactive=False)
                with gr.Column(scale=2):
                    log_box     = gr.Textbox(value="", label="Terminal Logs", interactive=False, lines=15, max_lines=30)
                    refresh_btn = gr.Button("Refresh Logs", elem_classes="primary-btn")

        train_btn.click(fn=start_training,
                        inputs=[num_samples_slider, epochs_slider, lr_input, batch_size_slider],
                        outputs=[status_box])
        refresh_btn.click(fn=get_training_logs, inputs=[], outputs=[log_box, status_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, css=apple_css)