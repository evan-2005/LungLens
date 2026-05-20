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
CLASSES = ["Normal", "Pneumonia", "TB"]
model_path = "chest_model_3class.pth"
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
            # Fallback for corrupted images
            image = Image.new("RGB", (224, 224), (0,0,0))
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

        # Use register_full_backward_hook if available (newer torch), else fallback
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

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of features
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i]

        cam = np.maximum(cam, 0) # ReLU
        
        # Resize to match input image dimensions (224, 224)
        cam = cv2.resize(cam, (224, 224))
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam

def get_dataset_paths():
    tb_path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    pneumonia_path = kagglehub.dataset_download("pcbreviglieri/pneumonia-xray-images")
    return tb_path, pneumonia_path

def collect_data():
    tb_base, pneumonia_base = get_dataset_paths()
    
    image_paths = []
    labels = []

    # TB Dataset
    tb_files = glob.glob(os.path.join(tb_base, "**", "*.*"), recursive=True)
    for f in tb_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        f_lower = f.lower()
        if 'normal' in f_lower:
            image_paths.append(f)
            labels.append(0) # Normal
        elif 'tuberculosis' in f_lower or 'tb' in f_lower:
            image_paths.append(f)
            labels.append(2) # TB

    # Pneumonia Dataset
    pneumonia_files = glob.glob(os.path.join(pneumonia_base, "**", "*.*"), recursive=True)
    for f in pneumonia_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        f_lower = f.lower()
        if 'normal' in f_lower:
            image_paths.append(f)
            labels.append(0) # Normal
        elif 'pneumonia' in f_lower:
            image_paths.append(f)
            labels.append(1) # Pneumonia

    # Custom Local Dataset
    custom_base = os.path.join(os.path.dirname(__file__), "custom_dataset")
    if os.path.exists(custom_base):
        custom_files = glob.glob(os.path.join(custom_base, "**", "*.*"), recursive=True)
        for f in custom_files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            f_lower = f.lower()
            if 'normal' in f_lower:
                image_paths.append(f)
                labels.append(0)
            elif 'pneumonia' in f_lower:
                image_paths.append(f)
                labels.append(1)
            elif 'tb' in f_lower or 'tuberculosis' in f_lower:
                image_paths.append(f)
                labels.append(2)

    # Balance datasets slightly if needed, but for now just shuffle and return
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
        
        # Subset selection
        if num_samples < len(image_paths):
            image_paths = image_paths[:num_samples]
            labels = labels[:num_samples]
            
        training_logs.append(f"Using {len(image_paths)} images for training/validation split.")
        
        # Split train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        training_logs.append(f"Train samples: {len(train_paths)} | Val samples: {len(val_paths)}")
        
        # Image augmentations for training
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
        
        train_dataset = ChestXRayDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = ChestXRayDataset(val_paths, val_labels, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        training_logs.append("Initializing CNNModel (DenseNet121 pre-trained)...")
        model = CNNModel(classCount=3, isTrained=True).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            train_loss = running_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100.0 * val_correct / val_total
            
            log_str = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%"
            training_logs.append(log_str)
            print(log_str)
            
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                torch.save(model.state_dict(), model_path)
                training_logs.append(f"--> Saved best model with validation accuracy: {best_acc:.2f}%")
                
        training_status = "Training Finished"
        training_logs.append("Evaluating final model accuracy and metrics...")
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
                
        report = classification_report(all_targets, all_preds, target_names=CLASSES)
        training_logs.append("\nClassification Report:\n" + report)
        
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
    if model is None:
        if os.path.exists(model_path):
            model = CNNModel(classCount=3, isTrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        else:
            raise gr.Error("No trained model found. Please train a model first on the Training tab.")

    if image is None:
        raise gr.Error("Please upload an X-ray image.")

    # Force convert image to RGB (resolves grayscale channel broadcasting errors)
    image = image.convert("RGB")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process original image for visualization
    orig_img = np.array(image)
    orig_img = cv2.resize(orig_img, (224, 224))
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Setup Grad-CAM hook on DenseNet's last conv layer (conv2 of denselayer16 in denseblock4)
    target_layer = model.cnnmodel.features.denseblock4.denselayer16.conv2
    grad_cam = GradCAM(model, target_layer)
    
    # Prediction
    input_tensor.requires_grad = True
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].detach().cpu()
    
    top_class_idx = int(torch.argmax(probabilities).item())
    top_class = CLASSES[top_class_idx]
    top_prob = probabilities[top_class_idx] * 100
    
    normal_prob = probabilities[0] * 100
    pneumonia_prob = probabilities[1] * 100
    tb_prob = probabilities[2] * 100
    
    findings_text = f"### Findings\n\nThe model analyzed the chest X-ray and found a **{top_prob:.1f}% probability of {top_class}**.\n\n"
    findings_text += f"- **Normal**: {normal_prob:.1f}%\n"
    findings_text += f"- **Pneumonia**: {pneumonia_prob:.1f}%\n"
    findings_text += f"- **Tuberculosis (TB)**: {tb_prob:.1f}%\n\n"
    
    if top_class == "Normal":
        findings_text += "No significant abnormalities detected in the provided scan. The lungs appear clear."
    elif top_class == "Pneumonia":
        findings_text += "The scan shows indications consistent with Pneumonia. Please refer to the Grad-CAM heatmap to visualize areas of potential consolidation or infection."
    else:
        findings_text += "The scan shows indications consistent with Tuberculosis. Please refer to the Grad-CAM heatmap to visualize focal lesions or cavities."

    # Get index of target visualization class
    target_idx = CLASSES.index(target_class_name)
    
    # Generate Grad-CAM Heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor, target_idx)
    grad_cam.remove_hooks()
    
    # Create superimposed visualization
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed_img = heatmap_colored * 0.4 + orig_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
    return findings_text, superimposed_img, gr.update(visible=False), gr.update(visible=True)

# Initial load if file exists
if os.path.exists(model_path):
    model = CNNModel(classCount=3, isTrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

# Premium Gradio UI Design (LungLens Apple Aesthetic)
apple_css = """
:root {
    --color-accent: #0071E3 !important;
    --color-accent-soft: rgba(0, 113, 227, 0.2) !important;
}
body, .gradio-container {
    background-color: #F5F5F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
.dark body, .dark .gradio-container {
    background-color: #1C1C1E !important;
}
.gradio-container {
    max-width: 1200px !important;
}
.custom-panel {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.02) !important;
    padding: 24px !important;
    margin-bottom: 16px !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.dark .custom-panel {
    background: rgba(40, 40, 42, 0.7) !important;
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
/* Hide built-with-gradio footer */
footer {
    display: none !important;
}
/* Smooth transitions */
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

with gr.Blocks(title="LungLens") as demo:
    gr.Markdown(
        """
        # LungLens
        Advanced Diagnostic Imaging
        """
    )
    
    with gr.Tab("Diagnostic Inference"):
        # Upload State View
        with gr.Column(elem_classes="upload-zone fade-in custom-panel centered-card", visible=True) as upload_container:
            gr.Markdown("### Upload Scan")
            input_img = gr.Image(type="pil", label="", elem_classes="upload-zone")
            target_viz = gr.Dropdown(
                choices=CLASSES, 
                value="Pneumonia", 
                label="Target Visualization Class"
            )
            predict_btn = gr.Button("Diagnose", variant="primary", elem_classes="primary-btn")
        
        # Results State View
        with gr.Column(visible=False, elem_classes="fade-in custom-panel") as results_container:
            with gr.Row():
                with gr.Column(scale=1):
                    output_heatmap = gr.Image(label="Grad-CAM Analysis")
                
                with gr.Column(scale=1):
                    output_markdown = gr.Markdown()
                    reset_btn = gr.Button("Analyze Another Scan", elem_classes="primary-btn")

        predict_btn.click(
            fn=predict_image,
            inputs=[input_img, target_viz],
            outputs=[output_markdown, output_heatmap, upload_container, results_container]
        )
        
        reset_btn.click(
            fn=reset_view,
            inputs=[],
            outputs=[upload_container, results_container, input_img, output_heatmap]
        )
        
    with gr.Tab("Model Training"):
        with gr.Column(elem_classes="custom-panel"):
            gr.Markdown(
                """
                ### Train Classifier
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
        
        # Auto refresh helper in Gradio 6.0+ using gr.Timer
        timer = gr.Timer(value=3)
        timer.tick(
            fn=get_training_logs,
            inputs=[],
            outputs=[log_box, status_box]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, css=apple_css)

