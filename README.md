# LungLens — CSC 3014 Computer Vision (Part II)

**LungLens** is a deployed, CPU-capable four-class chest X-ray (CXR) web application built for the CSC 3014 Computer Vision project (Part II).
It classifies radiographs into **Normal**, **Pneumonia**, **Tuberculosis**, and **Covid-19**, explains each prediction with a **Grad-CAM heatmap**, generates a **disease-focused segmentation overlay** via a distilled U-Net, and accompanies every result with a **plain-language clinical summary**.

> **Research and educational tool only. Not a certified medical device.** Always consult a qualified radiologist or physician for medical decisions. See the Disclaimer at the end. For a candid log of issues and lessons learned see [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).

---

## Why LungLens? — Motivation

Respiratory disease is one of the largest preventable causes of death globally.
Tuberculosis was linked to ~1.3 million deaths in 2022, pneumonia remains the leading infectious cause of death in children under five, and COVID-19 has been associated with more than 7 million reported deaths.
The diagnostic burden falls hardest on imaging departments in low- and middle-income settings where radiologist access is limited.

A chest X-ray is cheap, fast, and widely available, but the radiographic findings for TB, pneumonia, and COVID-19 overlap heavily.
LungLens acts as a triage layer: it flags urgent cases, visualises *where* the model looked, and delivers a confidence-aware summary that can connect imaging to treatment in clinics with no on-site radiologist.

---

## Contributions (Part II)

This release corrects and extends the Part I prototype with four concrete changes:

1. **The network we train is the network that serves predictions.** In Part I, the trained U-Net never reached inference; the DenseNet was used instead. Here, Stage 1 fine-tunes the DenseNet-121 that actually drives the Analysis tab.
2. **Grad-CAM-distilled disease segmentation.** Stage 2 (optional) supervises the U-Net with the trained classifier's own Grad-CAM maps, so the overlay marks disease evidence rather than the anatomy-tracing brightness pseudo-masks used earlier.
3. **Dataset-source bias is measured, not assumed away.** Because each disease originally came from a single dataset, a model can learn scanner fingerprints instead of pathology. We counter this with multi-source class construction, border cropping, and a per-source accuracy probe that makes any remaining shortcut visible.
4. **An evaluation designed for class imbalance.** The headline metric is macro-F1 on a held-out test set (patient-grouped split). Per-class sensitivity/specificity, a confusion matrix, and a per-source accuracy breakdown are all reported.

## Dataset-Source Bias

### The problem

An early version learned a shortcut: because each disease originally came from a single dataset, the model could diagnose Covid-19 from the burned-in `PORTABLE SEMI-ERECT` watermark in RICORD scan corners rather than from the lungs. Grad-CAM overlays lit up image corners — empty of anatomy — for both COVID-19 and Normal predictions.

This is not a LungLens-specific failure. Zech et al. (PLOS Med. 2018) showed the same pattern across multiple pneumonia detectors: same-source accuracy overstates cross-source real-world performance. The per-source accuracy probe in the evaluation is reassuringly even, but it cannot fully settle the question because source and class remain partially confounded by construction.

### Mitigations applied

| Mitigation | Implementation |
|---|---|
| **Multi-source class construction** | Every class now spans at least two independent datasets; source fingerprints no longer predict the label |
| **8% border crop** | Applied identically at train and inference time, removing corner annotations and laterality markers |
| **Per-source accuracy probe** | Printed at the end of every training run; roughly even accuracy across sources indicates pathology was learned |

> The reported 96% accuracy should be read as an optimistic upper bound, not a clean held-out clinical result.

---

## Key Features

- **Trains the served classifier.** Stage 1 fine-tunes the DenseNet-121 that actually drives predictions on the Analysis tab, not a frozen checkpoint. The best model by validation macro-F1 is saved to disk and hot-swapped into the live app when the run finishes, no restart needed.
- **Grad-CAM-distilled disease segmentation.** Stage 2 (optional) trains the U-Net to reproduce the classifier's Grad-CAM localisation in a single forward pass, so the overlay marks where the classifier looked rather than tracing anatomy.
- **Multi-source, confound-aware data.** Five Kaggle datasets are merged so each class spans multiple sources; a per-source accuracy probe flags source-based shortcuts (see above).
- **Uncertainty handling.** When the top class is below a confidence threshold (60 percent), the result is reported as Uncertain rather than a misleading confident label.
- **Grad-CAM attention heatmap.** Every result shows a translucent heatmap of where the model focused. For a confident abnormal finding, a crisp region outline is added on top. Low activation is suppressed so healthy tissue stays clean.
- **Sensible overlay for healthy scans.** On a Normal result the heatmap shows where the model assessed for the most likely abnormal class ("none abnormal"), instead of an alarming and unhelpful map over central anatomy.
- **Honest evaluation.** A patient-grouped train/val/test split keeps every patient on one side of the split, and the headline accuracy is reported on a held-out test set that was never used for checkpoint selection.
- **Class-weighted, regularised training.** Inverse-frequency class weights counter dataset imbalance; geometric and photometric augmentation plus early stopping (on validation macro-F1) reduce overfitting.
- **Runs on CPU or GPU.** CUDA is auto-detected. Both models load and run a warm-up pass at startup so the first inference is fast.
- **In-browser or headless training.** Train from the Training tab (logs refresh live), or run `train_run.py` from the terminal for an unattended job that survives with no browser open.

---

## Results

Evaluated on a composite of five public datasets (~32,300 images), patient-grouped train/val/test split (70/15/15%).  
Numbers from `chest_classifier_metrics.json` for the current checkpoint (`chest_model_4class.pth`, saved 2026-07-22, early stopped at epoch 11/40).

| Metric | Value |
|---|---|
| **Test accuracy** | **96.19%** |
| **Test macro-F1** | **95.83%** |
| Val accuracy | 96.34% |
| Val macro-F1 | 95.70% |
| Test samples | 5,637 |
| Val samples | 5,630 |

### Per-class recall (validation)

| Class | Recall |
|---|---|
| Normal | 98.46% |
| Pneumonia | 93.49% |
| Tuberculosis | 94.27% |
| Covid-19 | 95.51% |

### Per-source accuracy (test)

Roughly even accuracy across sources is the positive signal that pathology — not scanner metadata — is driving predictions.

| Source | Accuracy | n |
|---|---|---|
| tb_ds (Tawsifurrahman TB) | 98.72% | 625 |
| pneu_ds (Breviglieri Pneumonia) | 97.94% | 922 |
| covid_ds (RICORD) | 98.20% | 111 |
| radiography_db (COVID-19 Radiography) | 94.61% | 3,156 |
| shenzhen_tb | 93.46% | 107 |
| montgomery_tb | 95.00% | 20 |
| tbx11k | 98.85% | 696 |

### Confusion matrix (test set)

|  | Pred Normal | Pred Pneumonia | Pred TB | Pred Covid-19 |
|---|---|---|---|---|
| **Normal** | 2,879 | 37 | 9 | 6 |
| **Pneumonia** | 110 | 1,657 | 5 | 10 |
| **Tuberculosis** | 11 | 1 | 264 | 3 |
| **Covid-19** | 15 | 7 | 1 | 622 |

Main confusion pairs: **Pneumonia ↔ Normal** (overlapping opacity patterns) and a small **Covid-19 → Normal** tail (PCR-positive RICORD scans with little visible abnormality).

> **Honesty note.** The served checkpoint was trained before this evaluation; the original train/val split cannot be fully verified. The numbers are an optimistic upper bound rather than a clean prospective result.

---

## How Inference Works

Each request runs the hybrid pipeline:

```
Upload -> 8% border crop, resize 224x224, ImageNet normalise
   |
   |-- DenseNet-121 --> softmax --> class probabilities   (the prediction)
   |
   |-- top prob < 60%? --> report "Uncertain"
   |
   |-- overlay source:
   |     Grad-CAM-distilled U-Net disease mask, if available   (one forward pass)
   |     else live Grad-CAM on the DenseNet for the visualised class
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

### Two-stage training

Training runs in two stages, both from the app or `train_run.py`:

**Stage 1, DenseNet-121 classifier (the served model).** Fine-tuned with
Cross-Entropy under inverse-frequency class weights (normalised to mean 1.0) to
counter imbalance. Selection, scheduling, and early stopping all key on
**validation macro-F1** (the right target under heavy class imbalance, where raw
accuracy is misleading). The best checkpoint is saved as `chest_model_4class.pth`,
and its metrics (including the held-out test report and per-source probe) go to
`chest_classifier_metrics.json`.

**Stage 2, Grad-CAM-distilled U-Net (optional).** The trained classifier's
Grad-CAM maps become segmentation targets, and the U-Net learns to reproduce them
in a single forward pass. Its loss is:

```
Total Loss = CrossEntropyLoss(class, class_weights) + 2.0 * DiceBCELoss(mask)
DiceBCELoss = BCE(pred, target) + DiceLoss(pred, target)
```

The best U-Net by validation Dice is saved as `chest_segmentation_model.pth`. On
the Analysis tab the app prefers this disease mask for the overlay when the
metrics file marks it Grad-CAM-distilled, and falls back to live Grad-CAM
otherwise.

> Note: unlike earlier versions, the DenseNet classifier that drives predictions
> **is** retrained here. Stage 1 is the model served on the Analysis tab.

---

## Datasets

Datasets are downloaded automatically via [KaggleHub](https://github.com/Kaggle/kagglehub) on first training. No manual download is required. Labels are inferred from file and folder names (and, for the Shenzhen set, the filename suffix), so no CSV is needed.

Five sources are merged, chosen so **every class comes from at least two independent datasets**. This is the core of the confound mitigation described above. About **32,900 images** in total.

| # | Dataset | Kaggle Source | Labels Used |
|---|---------|---------------|-------------|
| 1 | Tuberculosis Chest X-ray | [tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) | `normal`, `tuberculosis` |
| 2 | Pneumonia X-Ray Images | [pcbreviglieri/pneumonia-xray-images](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images) | `normal`, `pneumonia` |
| 3 | COVID-19 Chest X-ray Positive Tests (RICORD) | [raddar/ricord-covid19-xray-positive-tests](https://www.kaggle.com/datasets/raddar/ricord-covid19-xray-positive-tests) | `covid-19` |
| 4 | COVID-19 Radiography Database | [tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | `normal`, `pneumonia` (Lung Opacity + Viral Pneumonia), `covid-19` |
| 5 | Shenzhen Tuberculosis Chest X-rays | [raddar/tuberculosis-chest-xrays-shenzhen](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) | `normal`, `tuberculosis` |

Approximate class balance after merging: Normal ~15,600, Pneumonia ~11,600, Covid-19 ~4,600, Tuberculosis ~1,000. Public TB chest-X-ray data is scarce, so TB is the smallest class; inverse-frequency class weighting compensates in the loss.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or newer (tested on 3.11)
- A [Kaggle API key](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json` (required only for training)
- Roughly 15 GB of disk space for datasets (about 33,000 images across five sources)
- Optional but recommended: an NVIDIA GPU with CUDA. Training the classifier on the full dataset is slow on CPU.

### 1. Clone the repository

```bash
git clone https://github.com/evan-2005/LungLens.git
cd LungLens
```

### 2. Install dependencies

```bash
pip install torch torchvision gradio opencv-python numpy scikit-learn kagglehub pillow matplotlib
```

For a GPU build (CUDA 12.4 example; check [pytorch.org](https://pytorch.org) for the build matching your driver, and run `nvidia-smi` to see the max CUDA version it supports):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

A `2.x.x+cpu` PyTorch build has no CUDA support compiled in, so `torch.cuda.is_available()` stays `False` even with a GPU present. Reinstall a `+cuXXX` build to use the GPU.

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
2. Adjust the hyperparameters (see the guide below). Stage 2 (disease segmentation) can be toggled on or off.
3. Click Start training. Training runs in a background thread.
4. Status and logs refresh automatically while the run is active. A manual Refresh button is also available.
5. When the run finishes, the retrained DenseNet (`chest_model_4class.pth`) and, if Stage 2 ran, the U-Net (`chest_segmentation_model.pth`) are published to the live app automatically. The Analysis tab uses them immediately, no restart needed.

For a long unattended run, use the headless driver instead of the browser:

```bash
# num_samples epochs lr batch_size num_workers train_seg seg_cap seg_epochs
python train_run.py 32000 40 1e-4 32 4 1 3000 10 > train_full.log 2>&1
```

It writes the same logs to `train_full.log`; follow them with `tail -f train_full.log` (or `Get-Content train_full.log -Wait -Tail 30` on Windows PowerShell).

---

## How to Train Effectively

Training retrains the **DenseNet-121 classifier** that serves predictions, and optionally distils its Grad-CAM into the **disease-segmentation U-Net**.

### Step by step

1. Ensure your Kaggle API key is at `~/.kaggle/kaggle.json`. The datasets download automatically on the first run.
2. Set Dataset size, Epochs, Batch size, Learning rate, and (optionally) the Stage 2 segmentation controls (see the recommended values below).
3. Start training. The five datasets are collected, stratified-subsampled to the requested size, and split into train/val/test grouped by patient so no patient spans two splits.
4. Watch the log. Each epoch reports train accuracy, validation accuracy, validation macro-F1, and the current learning rate. The best model by validation macro-F1 is checkpointed as it improves.
5. When Stage 1 finishes, a held-out **test** report prints (per-class precision/recall, confusion matrix, and the **per-source accuracy probe**). If enabled, Stage 2 then trains the U-Net and both models are published to the live app.

### What happens under the hood

1. **Border crop + augmentation.** Every image is border-cropped 8%. Training adds geometric and photometric augmentation (random resized crop, flip, small rotation, brightness/contrast jitter, mild blur) to attack resolution and scanner cues that would otherwise let the model separate classes by their source dataset.
2. **Class-weighted loss.** Cross-Entropy with inverse-frequency class weights (normalised to mean 1.0) counters the imbalance, so minority classes (Tuberculosis, Covid-19) are not drowned out.
3. **Macro-F1 selection and scheduling.** Checkpoint selection, ReduceLROnPlateau, and early stopping (patience 4) all key on validation macro-F1, the honest target under a ~15:1 imbalance where accuracy misleads.
4. **Stability.** Gradients are clipped at `max_norm = 1.0`, NaN/Inf batches are skipped, and checkpoints are written atomically (temp file + rename, with retry) so a file-lock never corrupts a save.
5. **Safe publishing.** The best checkpoint is reloaded and only swapped into the live model after the run ends, so inference never sees a half-trained model.

### Recommended hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Dataset Size | 500 to 1000 for a quick test; 20,000+ for a real run | Minority classes (especially TB) only get well represented at scale. Max is the full ~32,000. |
| Max Epochs | 40 | Early stopping halts on a macro-F1 plateau, so over-requesting is cheap. |
| Batch Size | 8 on CPU, 16 to 32 on GPU | Reduce if you hit out-of-memory errors. |
| Learning Rate | 0.0001 | A stable starting point for Adam. The scheduler lowers it automatically. |
| Data loader workers | 0 on CPU, 4+ on GPU | Higher keeps the GPU fed; on Windows the app handles worker spawn safely. |
| Stage 2: segmentation images | 800 to 3000 | Grad-CAM targets to distil. More is slower but sharper. |
| Stage 2: segmentation epochs | 6 to 12 | Not early-stopped. |

Hyperparameters are validated before a run starts. Out-of-range values (for example a learning rate of 0 or a negative epoch count) are rejected with a clear message.

### Reading the per-source probe

Because each disease now spans multiple sources, the per-source accuracy line at the end of a run is the real test of whether the model learned pathology. Roughly **even accuracy across sources** is the good outcome; a big gap (one source far higher than the rest) is a sign the model is still leaning on a source fingerprint.

### Reading the training logs

Each epoch prints batch progress and a summary line:

```
[Epoch 1/40] Training (702 batches)...
  Train batch 140/702
  ...
Epoch 1/40 | Train Acc: 84.32% | Val Acc: 89.94% | Val macro-F1: 86.89% | LR: 1.00e-04
--> Saved best classifier (Val macro-F1: 86.89%)
...
Held-out TEST | Acc: <acc>% | macro-F1: <f1>%
Per-source accuracy (test): radiography_db=<acc>% (n=<count>), pneu_ds=<acc>% (n=<count>), ...
```

(The first epoch numbers above are from a real run; the test/per-source line shows the format printed once Stage 1 finishes.)

| Metric | Meaning | Target |
|--------|---------|--------|
| Train/Val Acc | Classification accuracy across 4 classes | Rises steadily; watch the train/val gap for overfitting |
| Val macro-F1 | Unweighted mean F1 across classes | The selection metric; fair to minority classes under imbalance |
| Held-out TEST | Scores on data never used for selection | The honest headline number |
| Per-source accuracy | Accuracy split by originating dataset | Should be roughly even across sources |
| LR | Current learning rate | Drops when validation macro-F1 plateaus |

The checkpoint with the highest validation macro-F1 is saved, not the lowest loss, because macro-F1 is the metric that stays honest under heavy class imbalance.

### TensorBoard

Training writes scalars (loss, accuracy, macro-F1, Dice, and learning rate) to `runs/`. View them with:

```bash
tensorboard --logdir runs
```

### Troubleshooting

**Out-of-memory error.** Reduce Batch Size, reduce Dataset Size, and close other applications.

**Very slow training.** You are likely on CPU, or `torch` is a `+cpu` build. Install a CUDA build of PyTorch (see Setup), and raise Data loader workers.

**Accuracy looks suspiciously perfect.** Check the per-source probe. A near-100% score concentrated in one source can mean the model is exploiting a source shortcut rather than pathology.

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
  app.py                        Main Gradio app: models, training, inference, UI
  train_run.py                  Headless training driver for unattended runs
  debug_training.py             System diagnostics script
  test_gradcam.py               Quick inference smoke test
  chest_model_4class.pth        DenseNet-121 classifier (the served model)
  chest_segmentation_model.pth  Multi-Task U-Net (loaded if present)
  chest_classifier_metrics.json Metrics for the saved checkpoint (test + per-source)
  custom_dataset/               Optional: drop your own X-rays here
  model/
    CNNModel.py                 Standalone Multi-Task U-Net definition
    DatasetGenerator.py         Dataset with pseudo-mask generation (legacy)
    TrainerTester.py            Training and evaluation loops (legacy)
    Main.py                     Azure ML training entry point
  data/
    batch_download_zips.py
  README.md
```

### Key components in `app.py`

| Class or function | Purpose |
|-------------------|---------|
| `CNNModel` | DenseNet-121 classifier. The primary prediction model, retrained in Stage 1. |
| `MultiTaskUNet` | Encoder-decoder returning (class logits, segmentation mask). Overlay and fallback. |
| `DoubleConv` | Double convolution block with BatchNorm and ReLU. |
| `DiceBCELoss` / `dice_coefficient` | Combined Dice+BCE loss and overlap metric for segmentation. |
| `BorderCrop` | Crops a fixed fraction off each edge to remove burned-in corner annotations. |
| `collect_dataset` | Merges the five Kaggle sources into (paths, labels, patient groups, sources). |
| `stratified_subsample` / `patient_grouped_split` | Class-balanced subsampling and a leak-free train/val/test split. |
| `GradCAM` | Extracts Grad-CAM attention heatmaps from the DenseNet. |
| `build_gradcam_targets` | Turns the classifier's Grad-CAM maps into Stage-2 segmentation targets. |
| `train_segmentation_head` | Distils Grad-CAM into the U-Net (Stage 2). |
| `save_checkpoint_atomic` | Writes a checkpoint via temp file + atomic rename, with retry. |
| `warm_up_model` / `load_models_from_disk` | Warm-up pass and startup loading of both checkpoints. |
| `build_heatmap` / `clean_region` / `outline_region` | Render the overlay and outline a confident region. |
| `predict_image` | Inference: classify with the DenseNet, overlay via distilled U-Net or Grad-CAM. |
| `run_training_thread` | Two-stage training loop with logging, held-out test, and safe publishing. |

---

## Limitations

1. **Unverified train/val split for the current checkpoint.** The figures should be read as an optimistic upper bound rather than a clean held-out clinical result.
2. **Distilled segmentation inherits the classifier's mistakes.** The U-Net is a fast disease locator, not an independent ground-truth segmenter; that would require pixel-level masks the public datasets do not supply.
3. **Label noise on the COVID boundary.** RICORD includes PCR-positive scans with little visible abnormality, adding noise that accounts for the 15 Covid-19 images predicted as Normal in the confusion matrix.
4. **Source–class confounding.** Even after mitigations, source and class are partially tied by construction. Only a genuinely multi-source dataset for every class would remove the confound.

---

## Future Work

- **Patient-grouped retraining from scratch** to give a verified held-out score.
- **Genuinely multi-source dataset per class** to break the remaining source–class confounding.
- **Class-weighted loss + oversampling** to improve TB and Covid-19 minority representation further.
- **Grad-CAM++** for sharper overlays (Chattopadhay et al., WACV 2018).
- **Prospective radiologist validation** — the standard step before any clinical use.

---

## Key References

The following works directly inform the architecture, dataset choices, and evaluation design:

- Rajpurkar et al., *CheXNet* (arXiv:1711.05225)
- Huang et al., *Densely Connected CNNs / DenseNet* (CVPR 2017)
- KC et al., *Evaluation of deep learning approaches for COVID-19 CXR* (SIVP 2021)
- Selvaraju et al., *Grad-CAM* (ICCV 2017)
- Ronneberger et al., *U-Net* (MICCAI 2015)
- Gundel et al., *Multi-task learning for CXR abnormality classification* (arXiv:1905.06362)
- Zech et al., *Variable generalisation of pneumonia detection* (PLOS Med. 2018)
- Apostolopoulos & Mpesiana, *Covid-19 detection via transfer learning* (Phys. Eng. Sci. Med. 2020)
- Minaee et al., *Deep-COVID* (Med. Image Anal. 2020)

Full reference list is in the CSC 3014 Literature Review Part II document.

---

## Disclaimer

LungLens is a research and educational tool developed for CSC 3014 Computer Vision. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis. Always consult a qualified radiologist or physician for medical decisions.
