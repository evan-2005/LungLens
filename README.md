# LungLens

LungLens is a local, CPU-friendly web application that classifies chest X-rays across four categories (Normal, Pneumonia, Tuberculosis, Covid-19) and shows a Grad-CAM attention heatmap explaining each prediction.

It runs a hybrid of two models. A **DenseNet-121** classifier makes the diagnostic call, and a **Multi-Task U-Net** provides segmentation and a fallback path. Classification comes from the DenseNet because it discriminates the four classes reliably; the visualization is a Grad-CAM heatmap over the region that drove the prediction. The interface is built with [Gradio](https://www.gradio.app/) using a restrained, professional clinical theme.

> Research and educational tool only. Not a certified medical device. See the Disclaimer at the end. For a candid log of the problems hit while building this and what is worth improving next, see [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

---

## Key Features

- **Accurate classification.** The DenseNet-121 classifier drives the prediction and separates the four classes cleanly on held-out images.
- **Uncertainty handling.** When the top class is below a confidence threshold (60 percent), the result is reported as Uncertain rather than a misleading confident label.
- **Grad-CAM attention heatmap.** Every result shows a translucent heatmap of where the model focused. For a confident abnormal finding, a crisp region outline is added on top. Low activation is suppressed so healthy tissue stays clean.
- **Sensible overlay for healthy scans.** On a Normal result the heatmap shows where the model assessed for the most likely abnormal class ("none abnormal"), instead of an alarming and unhelpful map over central anatomy.
- **No first-run freeze.** Both models load and run a warm-up pass at startup, so the first user inference is fast instead of stalling on CPU kernel compilation.
- **Multi-Task U-Net for segmentation.** A single U-Net forward pass produces class logits and a 4-channel segmentation mask, used as the fallback when the DenseNet is unavailable.
- **Weakly supervised U-Net training.** No pixel-level annotations are needed. Anatomically informed pseudo-masks are generated from each image (CLAHE contrast equalisation, Otsu thresholding, an elliptical lung-field region, and morphological cleanup).
- **Class-weighted, regularised training.** Inverse-frequency class weights counter dataset imbalance, and dropout on the classification head reduces overfitting to the majority classes.
- **In-browser training dashboard.** Train the U-Net on Kaggle datasets without a terminal. Logs and status refresh automatically while a run is in progress.
- **Learning-rate scheduling.** ReduceLROnPlateau lowers the learning rate when validation Dice stops improving.

---

## How Inference Works

Each request runs the hybrid pipeline:

```
Upload -> resize 224x224, ImageNet normalise
   |
   |-- DenseNet-121 --> softmax --> class probabilities   (the prediction)
   |
   |-- top prob < 60%? --> report "Uncertain"
   |
   |-- Grad-CAM on DenseNet for the visualised class --> attention map
   |        (Normal prediction -> map the most likely ABNORMAL class instead)
   |
   +-- render full-resolution heatmap
            + crisp region outline only for a confident abnormal finding
```

If the DenseNet checkpoint is missing, the U-Net's classifier head and its
segmentation mask are used as the fallback for both prediction and overlay.

### Multi-Task U-Net (segmentation and fallback)

```
Input (3 x 224 x 224)
        |
   [ Encoder ]    DoubleConv x 4   (64 -> 128 -> 256 -> 512 channels)
        |         MaxPool2d between each block
        |
   [ Bottleneck (512 channels) ]
        |-- AdaptiveAvgPool -> Dropout(0.3) -> Linear(512, 4) -> class logits (4)
        |-- ConvTranspose2d x 3 with skip connections
        |
   [ Decoder ]    Upsample 28 -> 56 -> 112 -> 224 with encoder skips
        |
   Conv2d(64, 4, 1x1) -> Sigmoid
        |
   Segmentation masks (4 x 224 x 224)
```

Loss during U-Net training:

```
Total Loss = CrossEntropyLoss(class, class_weights) + 2.0 * DiceBCELoss(mask)
DiceBCELoss = BCE(pred, target) + DiceLoss(pred, target)
```

`class_weights` are inverse-frequency weights normalised to mean 1.0. The best
checkpoint (highest validation Dice) is saved as `chest_segmentation_model.pth`.

> Note: training in the app trains the U-Net only. The DenseNet-121 classifier
> (`chest_model_4class.pth`) that drives predictions is a pre-trained checkpoint
> and is not retrained here.

---

## Datasets

Datasets are downloaded automatically via [KaggleHub](https://github.com/Kaggle/kagglehub) on first training. No manual download is required. Labels are inferred from file and folder names, so no CSV is needed.

| # | Dataset | Kaggle Source | Labels Used |
|---|---------|---------------|-------------|
| 1 | Tuberculosis Chest X-ray | [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | `normal`, `tuberculosis` |
| 2 | Pneumonia X-Ray Images | [pcbreviglieri/pneumonia-xray-images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images) | `normal`, `pneumonia` |
| 3 | COVID-19 Chest X-ray Positive Tests | [raddar/ricord-covid19-xray-positive-tests](https://www.kaggle.com/datasets/raddar/ricord-covid19-xray-positive-tests) | `covid-19` |

---

## Setup and Installation

### Prerequisites

- Python 3.8 or newer (tested on 3.11)
- A [Kaggle API key](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json` (required only for training)
- Roughly 11 GB of disk space for datasets (about 11,000 images)
- Optional but recommended: an NVIDIA GPU with CUDA. Training on CPU is slow.

### 1. Clone the repository

```bash
git clone https://github.com/evan-2005/LungLens.git
cd LungLens
```

### 2. Install dependencies

```bash
pip install torch torchvision gradio opencv-python numpy scikit-learn kagglehub pillow matplotlib
```

For a GPU build (CUDA 11.8 example; check [pytorch.org](https://pytorch.org) for your version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Launch the app

```bash
python app.py
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860). To use a different port, set the `PORT` environment variable, for example `PORT=8000 python app.py`.

On startup the app loads both `chest_model_4class.pth` (the DenseNet classifier) and `chest_segmentation_model.pth` (the U-Net) if present, and warms each one up. The header shows which models loaded.

---

## How to Use

### Analysis

1. Open the Analysis tab.
2. Upload a chest X-ray (PNG, JPG, or JPEG).
3. Leave Overlay class on "Auto (predicted class)" to visualise the prediction, or pick a specific class to inspect.
4. Click Analyze. The results panel shows:
   - Attention heatmap: the scan with a Grad-CAM heatmap, plus a region outline when the finding is a confident abnormality.
   - Class confidence: probability bars for all four classes.
   - A findings summary: the predicted class (or Uncertain) and a plain-language description of what the heatmap represents.
5. Click Clear to reset the inputs and outputs.

### Training

1. Open the Training tab.
2. Adjust the hyperparameters (see the guide below).
3. Click Start training. Training runs in a background thread.
4. Status and logs refresh automatically while the run is active. A manual Refresh button is also available.
5. The best U-Net by validation Dice is saved as `chest_segmentation_model.pth` and published for inference automatically when training finishes.

---

## How to Train Effectively

Training in the app improves the **U-Net** (segmentation quality and the fallback classifier). It does not change the DenseNet-121 that drives predictions in the Analysis tab.

### Step by step

1. Ensure your Kaggle API key is at `~/.kaggle/kaggle.json`. The datasets download automatically on the first run.
2. Open the Training tab and set Dataset size, Epochs, Batch size, and Learning rate (see the recommended values below).
3. Click Start training. The datasets are collected, split into train and validation (stratified when class counts allow), and pseudo-masks are generated on the fly.
4. Watch Status and the log panel. Each epoch reports train and validation accuracy and Dice, plus the current learning rate.
5. When the run finishes, the best checkpoint by validation Dice is saved to `chest_segmentation_model.pth` and published to the live model, so the Analysis tab uses it immediately (no restart needed).

### What happens under the hood

Training is weakly supervised, so no pixel annotations are needed:

1. **Pseudo-mask generation.** For each non-normal image, the grayscale scan is contrast-equalised (CLAHE), thresholded with Otsu, restricted to an elliptical lung-field region, and cleaned with morphological opening and closing. This produces an anatomically plausible target for the segmentation head.
2. **Aligned augmentation.** Only photometric augmentation (brightness and contrast jitter) is applied during training. Geometric transforms are omitted because the pseudo-mask is derived from the original image, so rotating or flipping the input would misalign it with its target.
3. **Class-weighted joint optimisation.** Classification loss (Cross-Entropy with inverse-frequency class weights) and segmentation loss (Dice-BCE) are combined with a 1:2 weighting and optimised together. Dropout on the classification head reduces overfitting to the majority classes.
4. **Stability.** Gradients are clipped at `max_norm = 1.0`, NaN and Inf batches are skipped, and ReduceLROnPlateau halves the learning rate when validation Dice plateaus.
5. **Safe publishing.** Training builds a local network and only swaps it into the live model after the run ends, so inference never sees a half-trained model.

### Recommended hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Dataset Size | 500 to 1000 for testing | Start small to validate the pipeline. More data generalises better but is slower. |
| Epochs | 2 to 5 for testing | Validation Dice usually stabilises within a few epochs. |
| Batch Size | 8 on CPU, 16 to 32 on GPU | Reduce if you hit out-of-memory errors. |
| Learning Rate | 0.0001 | A stable starting point for Adam. The scheduler lowers it automatically. |

Hyperparameters are validated before a run starts. Out-of-range values (for example a learning rate of 0 or a negative epoch count) are rejected with a clear message.

### Training recipes by hardware

Quick test (CPU):

```
Dataset Size : 500
Epochs       : 2
Batch Size   : 8
Learning Rate: 0.0001
```

Validation (CPU):

```
Dataset Size : 1000
Epochs       : 3
Batch Size   : 8
Learning Rate: 0.0001
```

Production (GPU):

```
Dataset Size : 5000
Epochs       : 10
Batch Size   : 16 to 32
Learning Rate: 0.0001
```

Pseudo-labels limit the achievable Dice ceiling, so expect modest segmentation scores even on long runs.

### Reading the training logs

Each epoch prints batch progress and a summary line:

```
[Epoch 1/2] Training (100 batches)...
  Train batch 20/100
  ...
[Epoch 1/2] Validation (25 batches)...
  Val batch 5/25
  ...
Epoch 1/2 | Train Acc: 72.50% | Train Dice: 0.3812 | Val Acc: 68.75% | Val Dice: 0.3541 | LR: 1.00e-04
--> Saved best model (Val Dice: 0.3541)
```

| Metric | Meaning | Target |
|--------|---------|--------|
| Train/Val Acc | Classification accuracy across 4 classes | Above 70 percent is good |
| Train/Val Dice | Segmentation overlap, 0 to 1 | Above 0.30 is reasonable with pseudo-masks |
| LR | Current learning rate | Drops when validation Dice plateaus |

The checkpoint with the highest validation Dice is saved, not the lowest loss, because Dice directly measures segmentation quality.

### TensorBoard

Training writes scalars (loss, accuracy, Dice, and learning rate) to `runs/`. View them with:

```bash
tensorboard --logdir runs
```

### Troubleshooting

**Out-of-memory error.** Reduce Batch Size, reduce Dataset Size, and close other applications.

**Very slow training.** You are likely on CPU. Install a CUDA build of PyTorch, or reduce Batch Size and Dataset Size.

**Dice stuck low.** Pseudo-masks are inherently noisy. Use more images and more epochs to improve.

**No images found.** Confirm the Kaggle key at `~/.kaggle/kaggle.json`, check your internet connection, and clear `~/.cache/kagglehub` if a download was interrupted.

---

## Adding Your Own X-rays

Drop images into a `custom_dataset/` folder. Name files or parent folders to match the disease labels:

```
custom_dataset/
  normal_scan_001.jpg
  pneumonia/
    case_a.png
    case_b.png
  tb_positive_01.jpg
  covid_scan.jpg
```

Recognised keywords in file and folder names: `normal`, `pneumonia`, `tuberculosis`, `tb`, `covid`. The custom images are merged automatically into the training split.

---

## System Diagnostics

Before training, you can verify the system with:

```bash
python debug_training.py
```

This checks the model forward pass, loss computation, Dice calculation, dataset loading, and the validation loop, then reports a pass or fail for each.

---

## Project Structure

```
LungLens/
  app.py                        Main Gradio app: model, training, inference, UI
  debug_training.py             System diagnostics script
  test_gradcam.py               Quick inference smoke test
  chest_model_4class.pth        DenseNet-121 classifier (fallback)
  chest_segmentation_model.pth  Multi-Task U-Net (loaded if present)
  custom_dataset/               Optional: drop your own X-rays here
  model/
    CNNModel.py                 Standalone Multi-Task U-Net definition
    DatasetGenerator.py         Dataset with pseudo-mask generation
    TrainerTester.py            Training and evaluation loops
    Main.py                     Azure ML training entry point
  data/
    batch_download_zips.py
  README.md
```

### Key components in `app.py`

| Class or function | Purpose |
|-------------------|---------|
| `CNNModel` | DenseNet-121 classifier. The primary prediction model. |
| `MultiTaskUNet` | Encoder-decoder returning (class logits, segmentation mask). Segmentation and fallback. |
| `DoubleConv` | Double convolution block with BatchNorm and ReLU. |
| `DiceBCELoss` | Combined Dice and Binary Cross-Entropy loss for segmentation. |
| `dice_coefficient` | Measures segmentation overlap, 0 to 1. |
| `generate_pseudo_mask` | Builds an anatomically informed pseudo-mask from a grayscale scan. |
| `SegmentationDataset` | Returns (image, pseudo-mask, label) with masks generated on the fly. |
| `GradCAM` | Extracts Grad-CAM attention heatmaps from the DenseNet. |
| `warm_up_model` | Runs a dummy forward pass so the first real inference is fast. |
| `load_models_from_disk` | Loads and warms up both checkpoints; returns (seg_model, cls_model). |
| `build_heatmap` | Blends a translucent, floor-suppressed Grad-CAM heatmap at full resolution. |
| `clean_region` / `outline_region` | Threshold, despeckle, and outline a confident region of interest. |
| `predict_image` | Inference: classify with the DenseNet, visualise with Grad-CAM, apply gating. |
| `run_training_thread` | Background U-Net training loop with per-batch logging and safe publishing. |

---

## Disclaimer

LungLens is a research and educational tool. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified radiologist or physician for medical decisions.
