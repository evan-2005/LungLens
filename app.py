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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import threading
import io

# Define classes
CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "Covid-19"]
model_path = "chest_model_4class.pth"
seg_model_path = "chest_segmentation_model.pth"
training_status = "Not Training"
training_logs = []
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class CNNModel(nn.Module):
    def __init__(self, classCount, isTrained=True):
        super(CNNModel, self).__init__()
        self.cnnmodel = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if isTrained else None)
        kernelCount = self.cnnmodel.classifier.in_features
        self.cnnmodel.classifier = nn.Linear(kernelCount, classCount)

    def forward(self, x):
        return self.cnnmodel(x)

# Multi-Task U-Net and Segmentation Utilities
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(MultiTaskUNet, self).__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Classification Head (Bottleneck features -> pool -> FC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Classification Branch
        class_features = self.avgpool(x4)
        class_features = torch.flatten(class_features, 1)
        class_logits = self.classifier(class_features)
        
        # Decoder
        x_dec = self.up1(x4)
        x_dec = torch.cat([x_dec, x3], dim=1)
        x_dec = self.conv_up1(x_dec)
        
        x_dec = self.up2(x_dec)
        x_dec = torch.cat([x_dec, x2], dim=1)
        x_dec = self.conv_up2(x_dec)
        
        x_dec = self.up3(x_dec)
        x_dec = torch.cat([x_dec, x1], dim=1)
        x_dec = self.conv_up3(x_dec)
        
        logits = self.outc(x_dec)
        masks = self.sigmoid(logits)
        
        return class_logits, masks

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred_bin = (y_pred > 0.5).float()
    intersection = (y_pred_bin * y_true).sum(dim=(2, 3))
    union = y_pred_bin.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, labels, class_model, device, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_model = class_model
        self.device = device
        self.transform = transform
        
        self.cam_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)
            
        target_mask = np.zeros((4, 224, 224), dtype=np.float32)
        
        if label > 0:
            pseudo_mask = None
            if self.class_model is not None:
                try:
                    target_layer = self.class_model.cnnmodel.features.denseblock4.denselayer16.conv2
                    grad_cam = GradCAM(self.class_model, target_layer)
                    
                    cam_img = Image.open(img_path).convert("RGB")
                    tensor_for_cam = self.cam_transform(cam_img).unsqueeze(0).to(self.device)
                    tensor_for_cam.requires_grad_(True)
                    
                    heatmap = grad_cam.generate_heatmap(tensor_for_cam, label)
                    grad_cam.remove_hooks()
                    
                    pseudo_mask = (heatmap > 0.45).astype(np.float32)
                except Exception as e:
                    pseudo_mask = None
            
            if pseudo_mask is None:
                try:
                    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    gray = cv2.resize(gray, (224, 224))
                    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
                    h, w = gray.shape
                    lung_mask = np.zeros_like(gray)
                    cv2.rectangle(lung_mask, (int(w * 0.15), int(h * 0.15)), (int(w * 0.85), int(h * 0.85)), 255, -1)
                    thresh = cv2.bitwise_and(thresh, lung_mask)
                    pseudo_mask = (thresh / 255.0).astype(np.float32)
                except Exception:
                    pseudo_mask = np.zeros((224, 224), dtype=np.float32)
            
            target_mask[label] = pseudo_mask
            
        target_mask_tensor = torch.tensor(target_mask, dtype=torch.float32)
        return img_tensor, target_mask_tensor, label

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        features = self.features.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

def get_dataset_paths():
    tuberculosis_path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    pneumonia_path = kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images")
    covid_path = kagglehub.dataset_download("raddar/ricord-covid19-xray-positive-tests")
    return tuberculosis_path, pneumonia_path, covid_path

def collect_data():
    tuberculosis_base, pneumonia_base, covid_base = get_dataset_paths()

    image_paths = []
    labels = []

    tuberculosis_files = glob.glob(os.path.join(tuberculosis_base, "**", "*.*"), recursive=True)
    for f in tuberculosis_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        f_lower = f.lower()
        if 'normal' in f_lower:
            image_paths.append(f); labels.append(0)
        elif 'tuberculosis' in f_lower or 'tb' in f_lower:
            image_paths.append(f); labels.append(2)

    pneumonia_files = glob.glob(os.path.join(pneumonia_base, "**", "*.*"), recursive=True)
    for f in pneumonia_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        f_lower = f.lower()
        if 'normal' in f_lower:
            image_paths.append(f); labels.append(0)
        elif 'pneumonia' in f_lower:
            image_paths.append(f); labels.append(1)

    covid_files = glob.glob(os.path.join(covid_base, "**", "*.*"), recursive=True)
    for f in covid_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        image_paths.append(f); labels.append(3)

    custom_base = os.path.join(os.path.dirname(__file__), "custom_dataset")
    if os.path.exists(custom_base):
        custom_files = glob.glob(os.path.join(custom_base, "**", "*.*"), recursive=True)
        for f in custom_files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            f_lower = f.lower()
            if 'normal' in f_lower:
                image_paths.append(f); labels.append(0)
            elif 'pneumonia' in f_lower:
                image_paths.append(f); labels.append(1)
            elif 'tb' in f_lower or 'tuberculosis' in f_lower:
                image_paths.append(f); labels.append(2)
            elif 'covid' in f_lower:
                image_paths.append(f); labels.append(3)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)
    return image_paths, labels

def run_training_thread(num_samples, epochs, lr, batch_size):
    global training_status, training_logs, model
    training_status = "Training..."
    training_logs = []

    try:
        training_logs.append("Collecting dataset paths...")
        image_paths, labels = collect_data()
        training_logs.append(f"Total available images: {len(image_paths)}")

        if num_samples < len(image_paths):
            image_paths = image_paths[:num_samples]
            labels = labels[:num_samples]

        training_logs.append(f"Using {len(image_paths)} images for training/validation split.")

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        training_logs.append(f"Train samples: {len(train_paths)} | Val samples: {len(val_paths)}")

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        class_model = None
        if os.path.exists(model_path):
            try:
                training_logs.append("Loading pre-trained classifier model for pseudo-mask generation...")
                class_model = CNNModel(classCount=4, isTrained=False)
                class_model.load_state_dict(torch.load(model_path, map_location=device))
                class_model.to(device)
                class_model.eval()
            except Exception as e:
                training_logs.append(f"Warning: Failed to load pre-trained classifier ({str(e)}). Falling back to threshold-based pseudo-masks.")
        else:
            training_logs.append("No pre-trained classifier model found. Falling back to threshold-based pseudo-masks.")

        train_dataset = SegmentationDataset(train_paths, train_labels, class_model, device, transform=train_transform)
        val_dataset = SegmentationDataset(val_paths, val_labels, class_model, device, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        training_logs.append("Initializing MultiTaskUNet...")
        model = MultiTaskUNet(in_channels=3, num_classes=4).to(device)

        class_criterion = nn.CrossEntropyLoss()
        seg_criterion = DiceBCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_dice = 0.0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            total_train_dice = 0.0

            for i, (inputs, target_masks, target_labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                target_masks = target_masks.to(device)
                target_labels = target_labels.to(device)

                optimizer.zero_grad()
                class_logits, pred_masks = model(inputs)
                
                loss_class = class_criterion(class_logits, target_labels)
                loss_seg = seg_criterion(pred_masks, target_masks)
                loss = loss_class + 2.0 * loss_seg
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = class_logits.max(1)
                total += target_labels.size(0)
                correct += predicted.eq(target_labels).sum().item()
                
                dice = dice_coefficient(pred_masks, target_masks)
                total_train_dice += dice

            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            train_dice = total_train_dice / len(train_loader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            total_val_dice = 0.0

            with torch.no_grad():
                for inputs, target_masks, target_labels in val_loader:
                    inputs = inputs.to(device)
                    target_masks = target_masks.to(device)
                    target_labels = target_labels.to(device)
                    
                    class_logits, pred_masks = model(inputs)
                    
                    loss_class = class_criterion(class_logits, target_labels)
                    loss_seg = seg_criterion(pred_masks, target_masks)
                    loss = loss_class + 2.0 * loss_seg
                    
                    val_loss += loss.item()
                    _, predicted = class_logits.max(1)
                    val_total += target_labels.size(0)
                    val_correct += predicted.eq(target_labels).sum().item()
                    
                    dice = dice_coefficient(pred_masks, target_masks)
                    total_val_dice += dice

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100.0 * val_correct / val_total
            epoch_val_dice = total_val_dice / len(val_loader)

            log_str = f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Dice: {train_dice:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}% | Val Dice: {epoch_val_dice:.4f}"
            training_logs.append(log_str)
            print(log_str)

            if epoch_val_dice > best_val_dice:
                best_val_dice = epoch_val_dice
                torch.save(model.state_dict(), seg_model_path)
                training_logs.append(f"--> Saved best model with validation Dice: {best_val_dice:.4f}")

        training_status = "Training Finished"
        training_logs.append("Evaluating final segmentation model on validation set...")

        if os.path.exists(seg_model_path):
            model.load_state_dict(torch.load(seg_model_path, map_location=device))
            model.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, _, target_labels in val_loader:
                inputs = inputs.to(device)
                class_logits, _ = model(inputs)
                _, predicted = class_logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target_labels.numpy())

        report = classification_report(all_targets, all_preds, target_names=CLASSES)
        training_logs.append("\nFinal Model Classification Report:\n" + report)

    except Exception as e:
        training_status = "Failed"
        training_logs.append(f"Error during training: {str(e)}")
        print(f"Error: {str(e)}")

def start_training(num_samples, epochs, lr, batch_size):
    global training_status
    if training_status == "Training...":
        return "Training is already in progress!"
    thread = threading.Thread(target=run_training_thread, args=(num_samples, epochs, lr, batch_size))
    thread.start()
    return "Training started in background..."

def get_training_logs():
    return "\n".join(training_logs), training_status

def predict_image(image, target_class_name):
    global model

    has_seg_model = os.path.exists(seg_model_path)
    has_class_model = os.path.exists(model_path)

    if not has_seg_model and not has_class_model:
        gr.Warning("No trained model found. Please train a model first on the Model Training tab.")
        return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

    if model is None or (has_seg_model and not isinstance(model, MultiTaskUNet)) or (not has_seg_model and isinstance(model, MultiTaskUNet)):
        try:
            if has_seg_model:
                model = MultiTaskUNet(in_channels=3, num_classes=4)
                model.load_state_dict(torch.load(seg_model_path, map_location=device))
                model.to(device)
                model.eval()
            else:
                model = CNNModel(classCount=4, isTrained=False)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
        except Exception as e:
            gr.Warning(f"Failed to load model: {str(e)}")
            return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

    if image is None:
        gr.Warning("Please upload a valid X-ray image.")
        return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

    try:
        image = image.convert("RGB")
    except Exception as e:
        gr.Warning(f"Could not process image: {str(e)}")
        return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

    if image.getbbox() is None:
        gr.Warning("Uploaded image appears to be empty. Please select a proper X-ray image.")
        return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        orig_img = np.array(image)
        orig_img = cv2.resize(orig_img, (224, 224))

        target_idx = CLASSES.index(target_class_name)

        if isinstance(model, MultiTaskUNet):
            model.eval()
            with torch.no_grad():
                infer_tensor = transform(image).unsqueeze(0).to(device)
                class_logits, pred_masks = model(infer_tensor)
                probabilities = torch.nn.functional.softmax(class_logits, dim=1)[0].cpu()
                mask = pred_masks[0, target_idx].cpu().numpy()
        else:
            model.eval()
            with torch.no_grad():
                infer_tensor = transform(image).unsqueeze(0).to(device)
                outputs_infer = model(infer_tensor)
                probabilities = torch.nn.functional.softmax(outputs_infer, dim=1)[0].cpu()

            target_layer = model.cnnmodel.features.denseblock4.denselayer16.conv2
            grad_cam = GradCAM(model, target_layer)
            grad_tensor = transform(image).unsqueeze(0).to(device)
            grad_tensor.requires_grad_(True)
            heatmap = grad_cam.generate_heatmap(grad_tensor, target_idx)
            grad_cam.remove_hooks()
            mask = heatmap

        top_class_idx = int(torch.argmax(probabilities).item())
        top_class = CLASSES[top_class_idx]
        top_prob      = probabilities[top_class_idx] * 100
        normal_prob   = probabilities[0] * 100
        pneumonia_prob = probabilities[1] * 100
        tuberculosis_prob = probabilities[2] * 100
        covid_prob    = probabilities[3] * 100

        findings_text  = f"### Diagnostic Findings\n\nThe model analyzed the chest X-ray and found a **{top_prob:.1f}% probability of {top_class}**.\n\n"
        findings_text += f"- **Normal**: {normal_prob:.1f}%\n"
        findings_text += f"- **Pneumonia**: {pneumonia_prob:.1f}%\n"
        findings_text += f"- **Tuberculosis**: {tuberculosis_prob:.1f}%\n"
        findings_text += f"- **Covid-19**: {covid_prob:.1f}%\n\n"

        if top_class == "Normal":
            findings_text += "No significant abnormalities detected in the provided scan. The lungs appear clear."
        elif top_class == "Pneumonia":
            findings_text += "The scan shows indications consistent with Pneumonia. Refer to the segmented overlay showing potential zones of fluid consolidation or infection."
        elif top_class == "Tuberculosis":
            findings_text += "The scan shows indications consistent with Tuberculosis. Refer to the segmented overlay showing potential zones of focal lesions or cavities."
        elif top_class == "Covid-19":
            findings_text += "The scan shows indications consistent with Covid-19. Refer to the segmented overlay showing potential bilateral ground-glass opacities."

        binary_mask = (mask > 0.45).astype(np.uint8) * 255
        superimposed = orig_img.copy()
        
        if target_idx > 0 and np.sum(binary_mask) > 0:
            color = (0, 113, 227) # Medical Blue
            mask_colored = np.zeros_like(orig_img)
            mask_colored[mask > 0.45] = color
            cv2.addWeighted(mask_colored, 0.4, superimposed, 1.0, 0, superimposed)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(superimposed, contours, -1, color, 2)

        superimposed_pil = Image.fromarray(superimposed)
        return findings_text, superimposed_pil, gr.update(visible=False), gr.update(visible=True)

    except Exception as e:
        gr.Warning(f"Diagnosis failed: {str(e)}")
        return gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)

# Initial load if file exists
if os.path.exists(seg_model_path):
    try:
        model = MultiTaskUNet(in_channels=3, num_classes=4)
        model.load_state_dict(torch.load(seg_model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Successfully loaded trained MultiTaskUNet on startup.")
    except Exception as e:
        print(f"Warning: Could not load U-Net model on startup: {e}")
elif os.path.exists(model_path):
    try:
        model = CNNModel(classCount=4, isTrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Successfully loaded fallback CNN classifier on startup.")
    except Exception as e:
        print(f"Warning: Could not load fallback classifier on startup: {e}")

apple_css = """
:root {
    --color-accent: #0071E3 !important;
    --color-accent-soft: rgba(0, 113, 227, 0.2) !important;
}
html, body, .gradio-container {
    background-color: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
}
.dark html, .dark body, .dark .gradio-container {
    background-color: #1C1C1E !important;
}
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 24px !important;
    box-sizing: border-box !important;
}
.custom-panel {
    background: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02) !important;
    padding: 24px !important;
    margin-bottom: 16px !important;
}
.dark .custom-panel {
    background: #2C2C2E !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.centered-card {
    margin: 0 auto !important;
    max-width: 600px !important;
    float: none !important;
}
h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    color: #1D1D1F !important;
}
.dark h1, .dark h2, .dark h3 {
    color: #F5F5F7 !important;
}
p, label {
    font-weight: 400 !important;
    color: #86868B !important;
}
.dark p, .dark label {
    color: #8D8D93 !important;
}
.primary-btn {
    border-radius: 9999px !important;
    background-color: #0071E3 !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 12px 24px !important;
    box-shadow: none !important;
    transition: transform 0.2s ease, background-color 0.2s ease !important;
}
.primary-btn:hover {
    background-color: #0077ED !important;
    transform: scale(1.02) !important;
}
.upload-zone .gradio-image {
    border: 2px dashed #D2D2D7 !important;
    border-radius: 20px !important;
    background: transparent !important;
}
.dark .upload-zone .gradio-image {
    border-color: #424245 !important;
}
.upload-zone .gradio-image:hover, .upload-zone .gradio-image:focus-within {
    border-color: #0071E3 !important;
}
/* Center the Target Visualization Class dropdown */
.viz-dropdown-row {
    justify-content: center !important;
}
.viz-dropdown-row .gradio-dropdown {
    max-width: 320px !important;
    width: 100% !important;
}
/* Disclaimer note styling */
.disclaimer-note {
    font-size: 12px !important;
    color: #86868B !important;
    text-align: center !important;
    margin-top: 6px !important;
    margin-bottom: 0 !important;
    line-height: 1.5 !important;
    padding: 0 8px !important;
}
footer {
    display: none !important;
}
.fade-in {
    animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

def reset_view():
    return gr.update(visible=True), gr.update(visible=False), None, None

with gr.Blocks(title="LungLens", fill_width=True) as demo:
    gr.Markdown(
        """
        # LungLens
        Advanced Diagnostic Imaging
        """
    )

    with gr.Tab("Diagnostic Inference"):
        with gr.Row(visible=True) as upload_view:
            gr.Column(scale=1)
            with gr.Column(scale=2, elem_classes="upload-zone fade-in custom-panel"):
                gr.Markdown("### Upload Scan")
                input_img = gr.Image(type="pil", label="", elem_classes="upload-zone")

                with gr.Row(elem_classes="viz-dropdown-row"):
                    target_viz = gr.Dropdown(
                        choices=CLASSES,
                        value="Pneumonia",
                        label="Target Visualization Class"
                    )

                gr.HTML(
                    "<p class='disclaimer-note'>The segmentation mask highlights the detected region of interest for the selected class, "
                    "helping clinicians localize abnormalities. This tool is not a substitute for professional medical diagnosis.</p>"
                )

                predict_btn = gr.Button("Diagnose", variant="primary", elem_classes="primary-btn")
            gr.Column(scale=1)

        with gr.Row(visible=False) as results_view:
            with gr.Column(scale=1, elem_classes="fade-in custom-panel"):
                output_heatmap = gr.Image(label="Segmented Region of Interest")

            with gr.Column(scale=1, elem_classes="fade-in custom-panel"):
                output_markdown = gr.Markdown()
                reset_btn = gr.Button("Analyze Another Scan", elem_classes="primary-btn")

        predict_btn.click(
            fn=predict_image,
            inputs=[input_img, target_viz],
            outputs=[output_markdown, output_heatmap, upload_view, results_view]
        )

        reset_btn.click(
            fn=reset_view,
            inputs=[],
            outputs=[upload_view, results_view, input_img, output_heatmap]
        )

    with gr.Tab("Model Training"):
        with gr.Column(elem_classes="custom-panel"):
            gr.Markdown(
                """
                ### Train Segmentation Model (Multi-Task U-Net)
                Adjust parameters to fine-tune the model on your dataset.
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    num_samples_slider = gr.Slider(
                        minimum=100, maximum=10000, value=1000, step=100,
                        label="Dataset Size"
                    )
                    epochs_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="Epochs"
                    )
                    batch_size_slider = gr.Slider(
                        minimum=8, maximum=64, value=16, step=8,
                        label="Batch Size"
                    )
                    lr_input = gr.Number(
                        value=0.0001, label="Learning Rate", precision=6
                    )
                    train_btn = gr.Button("Start Training", elem_classes="primary-btn")
                    status_box = gr.Textbox(value=training_status, label="Status", interactive=False)

                with gr.Column(scale=2):
                    log_box = gr.Textbox(
                        value="", label="Terminal Logs",
                        interactive=False, lines=15, max_lines=30
                    )
                    refresh_btn = gr.Button("Refresh Logs", elem_classes="primary-btn")

        train_btn.click(
            fn=start_training,
            inputs=[num_samples_slider, epochs_slider, lr_input, batch_size_slider],
            outputs=[status_box]
        )

        refresh_btn.click(
            fn=get_training_logs,
            inputs=[],
            outputs=[log_box, status_box]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, css=apple_css)