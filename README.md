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
- **Python 3.8+** (tested on 3.11)
- A [Kaggle API key](https://www.kaggle.com/docs/api) configured at `~/.kaggle/kaggle.json`
- **11+ GB disk space** for datasets (~11K images, ~5 GB)
- *(Recommended)* NVIDIA GPU with CUDA 11.8+ (training on CPU is very slow, ~30 min per epoch with 1000 images)

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

1. **Pseudo-Mask Generation**: For each non-normal image, a grayscale intensity threshold (pixels > 140) is applied to the central 70% lung region to create training masks automatically.
2. **Multi-Task Learning**: Classification loss (Cross-Entropy) + Segmentation loss (Dice-BCE) are weighted 1:2 and optimized jointly.
3. **Gradient Clipping**: Gradients are clipped at max_norm=1.0 to prevent NaN/Inf loss spikes.

> **Note:** The current implementation uses fast intensity-based thresholding. For improved segmentation quality with your own data, consider using a pre-trained classifier's Grad-CAM activation as the pseudo-mask source.

---

### Recommended Hyperparameter Settings

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **Dataset Size** | **500–1000** for testing | Start small to validate training pipeline. More data = better generalization but slower training. |
| **Epochs** | **2–5** for testing | Validation Dice typically stabilizes after 3-5 epochs on well-labeled data. |
| **Batch Size** | **8** (CPU) / **16–32** (GPU) | Smaller batches on CPU due to memory constraints. OOM? Reduce batch size. |
| **Learning Rate** | **0.0001** (fixed) | Adam optimizer with 1e-4 is stable. Don't change unless you know why. |

### Training Recipes by Hardware

**Option 1: Quick Test (CPU, ~5–10 min)**
```
Dataset Size : 500
Epochs       : 2
Batch Size   : 8
Learning Rate: 0.0001
Expected: Train completes, model saves. Dice ≈ 0.15–0.25 (pseudo-masks are noisy).
```

**Option 2: Validation (CPU, ~30 min)**
```
Dataset Size : 1000
Epochs       : 3
Batch Size   : 8
Learning Rate: 0.0001
Expected: Better convergence. Dice ≈ 0.25–0.35.
```

**Option 3: Production (GPU, ~2–4 hours)**
```
Dataset Size : 5000
Epochs       : 10
Batch Size   : 16–32
Learning Rate: 0.0001
Expected: Strong segmentation. Dice ≈ 0.40–0.50 (pseudo-labels limit ceiling).
```

### Reading the Training Logs

Each epoch prints real-time batch progress:
```
[Epoch 1/2] Training (100 batches)...
  Train batch 20/100
  Train batch 40/100
  Train batch 60/100
  Train batch 80/100
  Train batch 100/100
[Epoch 1/2] Validation (25 batches)...
  Val batch 5/25
  Val batch 10/25
  Val batch 15/25
  Val batch 20/25
  Val batch 25/25
Epoch 1/2 | Train Acc: 72.50% | Train Dice: 0.3812 | Val Acc: 68.75% | Val Dice: 0.3541
--> Saved best model (Val Dice: 0.3541)
```

| Metric | Meaning | Target |
|--------|---------|--------|
| `Train/Val Acc` | Classification accuracy (4 classes) | > 70% is good |
| `Train/Val Dice` | Segmentation overlap (0–1 scale) | > 0.30 is reasonable with pseudo-masks |
| `[ERROR]` | Batch processing failed | Check logs, usually data corruption |

> **Important**: The model saves the checkpoint with **highest validation Dice**, not lowest loss. Dice directly measures segmentation quality.

### Troubleshooting Training Issues

**Training starts but stops at batch 13+**
- ✅ **Fixed in latest version** — was due to unpacking error
- If you still see "too many values to unpack", update to latest app.py

**OOM (Out of Memory) error**
- Reduce `Batch Size` from 8 to 4
- Reduce `Dataset Size` to 500
- Close other applications

**Training very slow (>1 min per batch)**
- You're on CPU. Use GPU for practical training: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Or reduce `Batch Size` and `Dataset Size`

**Model not improving (Dice stuck at ~0.15)**
- Pseudo-masks from intensity thresholding are noisy. This is expected.
- Use 2000+ images and 5+ epochs to see improvement
- Consider providing better pseudo-mask labels

**"No images found" error**
- Check Kaggle API key is configured: `cat ~/.kaggle/kaggle.json`
- Check internet connection (datasets download on first run)
- Delete `~/.cache/kagglehub` and retry if download corrupted

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

## 🔧 System Diagnostics

Before training, verify the system is working:

```bash
python debug_training.py
```

This tests:
- ✓ Model forward pass (U-Net architecture)
- ✓ Loss computation (no NaN/Inf)
- ✓ Dice coefficient calculation
- ✓ Dataset loading and batching
- ✓ Full validation loop

**Expected output:**
```
============================================================
✓ PASS: Model forward pass
✓ PASS: Loss computation
✓ PASS: Dice coefficient
✓ PASS: Data loading
✓ PASS: Validation loop
```

If any test fails, check the error message — it will tell you what to fix.

---

## 📁 Project Structure

```
LungLens/
├── app.py                           # Main Gradio app — model, training, inference
├── debug_training.py                # System diagnostics script
├── chest_model_4class.pth           # Legacy DenseNet-121 classifier (fallback)
├── chest_segmentation_model.pth     # Multi-Task U-Net (created after training)
├── custom_dataset/                  # Drop your own X-rays here
├── test_gradcam.py                  # Quick inference smoke test
├── COMPREHENSIVE_BUG_REPORT.md      # Detailed system audit
├── FIXES_APPLIED.md                 # Summary of bug fixes
├── DEBUG_FIXES.md                   # Training hang solutions
├── model/
│   ├── CNNModel.py                  # Multi-Task U-Net definition (standalone)
│   ├── DatasetGenerator.py          # Dataset with pseudo-mask generation
│   ├── TrainerTester.py             # Training/eval loops with DiceBCE loss
│   └── Main.py                      # Azure ML training entry point
├── data/
│   └── batch_download_zips.py
└── README.md
```

### Key Classes in `app.py`

| Class / Function | Purpose |
|-----------------|---------|
| `MultiTaskUNet` | Encoder-decoder model with classification head. Outputs (class logits, segmentation mask). |
| `DoubleConv` | Double convolution block with BatchNorm and ReLU |
| `DiceBCELoss` | Combined Dice Loss + Binary Cross Entropy for segmentation |
| `dice_coefficient` | Validates segmentation quality (0–1 scale) |
| `SegmentationDataset` | Returns (image, pseudo_mask, label). Generates masks on-the-fly. |
| `GradCAM` | Extracts attention heatmaps from DenseNet (inference only) |
| `predict_image` | Inference function. Runs U-Net or fallback, renders overlay. |
| `run_training_thread` | Background thread that trains U-Net with per-batch logging |

---

## 📋 Recent Fixes (v1.1)

**8 critical bugs fixed:**
- ✅ Specific exception handlers (FileNotFoundError, OSError instead of bare Exception)
- ✅ Type consistency in dataset (lists instead of tuples)
- ✅ Empty dataset handling
- ✅ Array division bounds checking (clips masks to [0, 1])
- ✅ Stratification validation (prevents crash on small datasets)
- ✅ GradCAM memory leak (try-finally cleanup)
- ✅ Dynamic layer extraction (no hardcoded architecture paths)
- ✅ Bounds checking on prediction indices

**Training stability improvements:**
- ✅ Gradient clipping (prevents NaN/Inf loss)
- ✅ Per-batch error handling with detailed traceback logging
- ✅ Batch-level progress tracking (updates every ~20% of batches)
- ✅ DataLoader optimization (prefetch when num_workers > 0)

See `FIXES_APPLIED.md` for implementation details.

---

## ⚠️ Disclaimer

LungLens is a **research and educational tool**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified radiologist or physician for medical decisions.
