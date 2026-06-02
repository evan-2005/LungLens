# LungLens

**LungLens** is an advanced, Apple-inspired clinical diagnostic web application that performs **joint semantic segmentation and classification** of chest X-rays across four categories: **Normal**, **Pneumonia**, **Tuberculosis**, and **Covid-19**.

The core model is a **Multi-Task U-Net** — a lightweight encoder-decoder architecture that simultaneously predicts diagnostic class probabilities *and* generates pixel-level segmentation masks that precisely outline detected pathology regions. The UI runs on [Gradio](https://www.gradio.app/) with a polished Apple-aesthetic design.

---

## 🌟 Key Features

- **Multi-Task U-Net Architecture** — A single forward pass produces both class probabilities and a 4-channel segmentation mask. No post-hoc saliency hacks.
- **Precise Lesion Segmentation** — Pathology regions (ground-glass opacities, consolidations, cavities) are overlaid as sharp, contoured Medical Blue masks directly on the scan.
- **Weakly-Supervised Training** — Requires **no pixel-level annotations**. Pseudo-masks are generated automatically from a pre-trained Grad-CAM classifier, or fall back to intensity-based thresholding.
- **Smart Fallback** — If a U-Net model hasn't been trained yet, the app automatically loads the legacy DenseNet-121 classifier and converts its Grad-CAM activation into a thresholded segmentation overlay so the app is usable immediately.
- **Built-in Training Dashboard** — Fine-tune the U-Net on Kaggle datasets entirely from the browser — no terminal required.
- **Human-Readable Findings** — Clinical-language summaries are generated alongside per-class probability scores.
- **Apple-Aesthetic UI** — SF Pro typography, glassmorphism panels, fade-in animations, and a strict monochrome + Medical Blue palette.

---

## 🏗️ Model Architecture

```
Input (3×224×224)
       │
  ┌────▼────┐
  │ Encoder │  DoubleConv × 4  (64 → 128 → 256 → 512 channels)
  └────┬────┘  MaxPool2d between each block
       │
  ┌────▼──────────────────────────────┐
  │ Bottleneck (512 channels)         │
  │  ├─ AdaptiveAvgPool → Linear(512,4) ──► Class Logits (4)
  │  └─ ConvTranspose2d × 3 + skip connections
  └───────────────────────────────────┘
       │
  ┌────▼────┐
  │ Decoder │  Upsample 28→56→112→224, skip connections from encoder
  └────┬────┘
       │
  Conv2d(64, 4, 1×1) → Sigmoid
       │
  Segmentation Masks (4×224×224)
```

**Loss function:**
```
Total Loss = CrossEntropyLoss(class) + 2.0 × DiceBCELoss(mask)
```
`DiceBCELoss = BCE(pred, target) + DiceLoss(pred, target)`

**Saved as:** `chest_segmentation_model.pth`

---

## 💾 Datasets

Datasets are downloaded automatically via [KaggleHub](https://github.com/Kaggle/kagglehub) on first training. No manual download needed.

| # | Dataset | Kaggle Source | Labels Used |
|---|---------|--------------|-------------|
| 1 | Tuberculosis Chest X-ray | [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | `normal`, `tuberculosis` |
| 2 | Pneumonia X-Ray Images | [pcbreviglieri/pneumonia-xray-images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images) | `normal`, `pneumonia` |
| 3 | COVID-19 Chest X-ray Positive Tests | [raddar/ricord-covid19-xray-positive-tests](https://www.kaggle.com/datasets/raddar/ricord-covid19-xray-positive-tests) | `covid-19` |

> Labels are inferred from file/folder names automatically — no CSV required.

---

## 🛠️ Setup & Installation

### Prerequisites
- **Python 3.8+**
- A [Kaggle API key](https://www.kaggle.com/docs/api) configured at `~/.kaggle/kaggle.json` (for dataset auto-download)
- *(Recommended)* An NVIDIA GPU with CUDA for practical training speeds

### 1. Clone the Repository
```bash
git clone https://github.com/evan-2005/LungLens.git
cd LungLens
```

### 2. Install Dependencies
```bash
pip install torch torchvision gradio opencv-python numpy scikit-learn kagglehub pillow matplotlib
```

For GPU (CUDA 11.8 example — check [pytorch.org](https://pytorch.org) for your version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Launch the App
```bash
python app.py
```

Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

On first launch, if `chest_segmentation_model.pth` exists it is loaded automatically. Otherwise, the app falls back to `chest_model_4class.pth` (legacy classifier) if present.

---

## 🚀 How to Use

### Diagnostic Inference
1. Go to the **Diagnostic Inference** tab.
2. Upload a chest X-ray (PNG, JPG, JPEG).
3. Select the **Target Visualization Class** — the class whose segmentation mask you want to inspect.
4. Click **Diagnose**.
5. Results panel shows:
   - **Segmented Region of Interest** — the original scan with a Medical Blue overlay + contour outline of the detected pathology region.
   - **Diagnostic Findings** — top predicted class, per-class probabilities, and a clinical description.
6. Click **Analyze Another Scan** to reset.

### Model Training
1. Go to the **Model Training** tab.
2. Adjust the hyperparameters (see guide below).
3. Click **Start Training** — training runs in a background thread.
4. Click **Refresh Logs** to poll epoch-by-epoch metrics.
5. The best model (highest validation Dice score) is saved as `chest_segmentation_model.pth` and loaded automatically when training finishes.

---

## 🎯 How to Train Effectively

### Understanding the Training Pipeline

Training uses **weakly-supervised pseudo-masks** — no pixel annotations are needed:

1. If `chest_model_4class.pth` (the legacy DenseNet-121 classifier) is present, **Grad-CAM heatmaps** are generated per image and thresholded at `0.45` to produce pseudo-masks. This gives better spatial priors.
2. If the classifier is not present, the fallback is **grayscale intensity thresholding** (pixels > 140 in the central 70% lung region).

**Tip:** For the best segmentation quality, train the classifier first (if you have the legacy model), then train the U-Net using it as a pseudo-mask teacher.

---

### Recommended Hyperparameter Settings

| Parameter | Slider Range | Recommended | Notes |
|-----------|-------------|-------------|-------|
| **Dataset Size** | 100 – 10,000 | **3,000 – 5,000** | More data = better generalisation. Start at 2,000 for a quick test run. |
| **Epochs** | 1 – 20 | **10 – 15** | Validation Dice typically plateaus around epoch 10–12. |
| **Batch Size** | 8 – 64 | **16** (CPU) / **32** (GPU) | Larger batches stabilise Dice loss. Reduce if you get OOM errors. |
| **Learning Rate** | (manual) | **0.0001** | Adam with 1e-4 is a safe default. Try 5e-5 for fine-tuning. |

### Quick-Start Recipe (CPU, ~30 min)
```
Dataset Size : 1000
Epochs       : 5
Batch Size   : 8
Learning Rate: 0.0001
```

### Quality Recipe (GPU, ~2–4 hours)
```
Dataset Size : 5000
Epochs       : 15
Batch Size   : 32
Learning Rate: 0.0001
```

### Reading the Training Logs

Each epoch prints:
```
Epoch 3/15 | Loss: 1.8432 | Train Acc: 72.50% | Train Dice: 0.3812 | Val Loss: 1.9210 | Val Acc: 68.75% | Val Dice: 0.3541
--> Saved best model with validation Dice: 0.3541
```

| Metric | What it means | Target |
|--------|--------------|--------|
| `Loss` | Combined classification + segmentation loss | Decreasing ✓ |
| `Train/Val Acc` | Classification accuracy on the 4 classes | > 70% is good |
| `Train/Val Dice` | Segmentation overlap score (0–1) | > 0.4 is reasonable with pseudo-masks |

> **Val Dice > Val Loss** is the primary checkpoint criterion — the model saved is the one with the highest validation Dice, not the lowest loss.

---

### Adding Your Own X-rays

Drop images into the `custom_dataset/` folder. Name files or parent folders to match disease labels:

```
custom_dataset/
├── normal_scan_001.jpg
├── pneumonia/
│   ├── case_a.png
│   └── case_b.png
├── tb_positive_01.jpg
└── covid_scan.jpg
```

Recognised keywords in filenames/folders: `normal`, `pneumonia`, `tuberculosis`, `tb`, `covid`

Then click **Start Training** — the custom images are merged automatically into the training split.

---

## 📁 Project Structure

```
LungLens/
├── app.py                        # Main Gradio app — model, training, inference
├── chest_model_4class.pth        # Legacy DenseNet-121 classifier (fallback)
├── chest_segmentation_model.pth  # Multi-Task U-Net (created after training)
├── custom_dataset/               # Drop your own X-rays here
├── test_gradcam.py               # Quick inference smoke test
├── model/
│   ├── CNNModel.py               # Multi-Task U-Net definition (standalone)
│   ├── DatasetGenerator.py       # Dataset with pseudo-mask generation
│   ├── TrainerTester.py          # Training/eval loops with DiceBCE loss
│   └── Main.py                   # Azure ML training entry point
├── data/
│   └── batch_download_zips.py
└── README.md
```

### Key Classes in `app.py`

| Class / Function | Purpose |
|-----------------|---------|
| `MultiTaskUNet` | The main segmentation model — encoder + decoder + classification head |
| `DoubleConv` | Shared Conv-BN-ReLU-Conv-BN-ReLU building block |
| `DiceBCELoss` | Combined Dice + Binary Cross Entropy segmentation loss |
| `dice_coefficient` | Binarised Dice metric for validation tracking |
| `SegmentationDataset` | Returns `(image, pseudo_mask, label)` — generates masks on-the-fly |
| `GradCAM` | Used as pseudo-mask teacher during training (legacy model only) |
| `predict_image` | Inference — runs U-Net or fallback classifier, renders overlay |
| `run_training_thread` | Background training thread with per-epoch logging |

---

## ⚠️ Disclaimer

LungLens is a **research and educational tool**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified radiologist or physician for medical decisions.
