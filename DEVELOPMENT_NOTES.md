# LungLens Development Notes

Difficulties faced while building and debugging LungLens, how each was resolved, and
what is worth improving in future iterations. This is a working log, not marketing
copy; the limitations section is deliberately honest.

## 1. Difficulties faced and how they were overcome

### 1.1 First inference froze the whole app
**Problem.** The first image submitted after startup hung on a spinner for a long time,
while later submissions were fast. It looked like a deadlock but was not.

**Cause.** The model was loaded at startup but never exercised. On CPU, PyTorch
JIT-compiles its oneDNN/MKL convolution kernels on the *first* forward pass. For a
U-Net that first pass takes several seconds, and Gradio's queue showed only a spinner
during it. Once the kernels were compiled, every later call reused them.

**Fix.** Added a `warm_up_model()` helper that runs one dummy forward pass
(`torch.zeros(1, 3, 224, 224)`) at load time, and a `load_models_from_disk()` loader
that both loads and warms up. The JIT cost now happens during startup instead of on the
user's first click. Measured result: first user inference dropped from a multi-second
freeze to about 0.3 seconds.

### 1.2 The model effectively always predicted "Normal"
**Problem.** Almost every image came back as Normal at roughly 42 percent, with
Pneumonia close behind at 41 percent, regardless of the actual scan.

**Cause.** Inference used the multi-task U-Net's classification head. That head is a
single linear layer on top of the segmentation encoder and it had collapsed to a
near-constant prior, so it could not discriminate between classes. Meanwhile a separate,
genuinely accurate DenseNet-121 checkpoint sat unused in the same folder.

**Fix.** Switched to a hybrid pipeline: the DenseNet drives classification, the U-Net is
kept only as a segmentation and fallback source. Both load and warm up at startup.
Directly tested against cached Kaggle X-rays, the DenseNet separates classes cleanly
(Normal 0.92 to 1.00, Pneumonia 0.89 to 1.00, TB 0.70 to 0.98). Also added a 60 percent
confidence threshold so a genuinely ambiguous scan reports "Uncertain" instead of a
false "Normal."

### 1.3 The segmentation overlay traced bone, not disease
**Problem.** The shaded region followed the spine, ribs, and mediastinum rather than the
pathology, so the highlight rarely matched where a clinician would look.

**Cause.** There are no real per-pixel lung annotations for these datasets, so the U-Net
was supervised with a brightness-threshold pseudo-mask (Otsu on intensity). Bone is the
brightest thing in an X-ray, so the model learned "bright equals highlight." On top of
that, the display code min-max stretched the mask before thresholding, which guarantees
some pixel crosses the threshold even when the class is absent, producing phantom
regions.

**Fix.** Replaced the U-Net mask as the primary overlay with Grad-CAM computed on the
accurate DenseNet, which marks the region that actually drove the prediction. Removed the
min-max stretch (threshold on raw values), added morphological opening and
connected-component speckle removal, and rendered everything at the original image
resolution instead of a downscaled 224x224 copy. On the test pneumonia image the overlay
now lands on the lower-right lobe consolidation rather than the spine.

### 1.4 Overlays appeared for classes the model had no evidence for
**Problem.** Forcing a class such as Covid-19 on a clearly Normal scan still drew a
confident-looking blue region.

**Cause.** Grad-CAM always produces some peak, even for a class at 0.2 percent
probability, so blindly rendering it created misleading shading.

**Fix.** Gated the crisp region outline behind three conditions: the class is abnormal,
the prediction is not uncertain, and the class probability clears a minimum floor. The
forced-Covid case now says "little evidence for Covid-19 (0.2%)" and draws no outline.

### 1.5 Over-correction: the heatmap disappeared entirely
**Problem.** After the gating in 1.4, healthy scans showed the bare X-ray with nothing
drawn, so it looked like the visualization feature had been removed.

**Cause.** The gate skipped drawing anything when the prediction was Normal, uncertain,
or low-probability, and Normal is the common case on a healthy scan.

**Fix.** Split the visualization into two layers. A translucent Grad-CAM heatmap is now
always shown so the panel is never empty, and the crisp outline is added only for a
confident abnormal finding. A `HEATMAP_FLOOR` suppresses diffuse low activation so
healthy tissue stays grayscale instead of getting a full-image color wash.

### 1.6 Grad-CAM for "Normal" looked alarming
**Problem.** Showing the heatmap for the Normal class lit up the central mediastinum in a
large red blob, which is not clinically useful and looks like an emergency on a healthy
scan.

**Cause.** The DenseNet decides "normal" partly from central anatomy, so its attention for
that class is central and diffuse.

**Fix.** On an auto Normal result, the heatmap instead shows where the model assessed for
the most likely abnormal class, labeled "areas assessed, none abnormal." The map stays
over the lung fields and stays meaningful, and abnormal predictions still get the hot
heatmap plus outline.

### 1.7 Class imbalance across the datasets
**Problem.** The four sources are very uneven in size (the RICORD Covid set is far smaller
than the Normal and Pneumonia sets), which biases training toward the majority classes.

**Fix.** Added inverse-frequency class weights to the training loss, normalized to mean
1.0 so the loss scale is stable. Verified the minority classes receive proportionally
higher weight. Also added dropout to the U-Net classification head; because dropout has
no parameters, existing checkpoints still load with strict matching.

### 1.8 Windows-specific breakages
**Problem.** Two failures showed up only on Windows.
- The debug script crashed with `UnicodeEncodeError` when printing check and cross
  glyphs, because the default console codepage is cp1252.
- The app crashed on launch after a UI change with
  `AttributeError: 'str' object has no attribute 'name'`.

**Cause and fix.**
- Reconfigured the debug script's stdout to UTF-8 so the status glyphs print.
- Gradio 6 requires `Font` objects in the theme, not plain strings; wrapping the font
  names in `gr.themes.Font(...)` fixed the launch crash.

### 1.9 Thread safety around the shared model
**Problem.** Training runs in a background thread and swaps the global model, which could
race a concurrent inference request or expose a half-trained model.

**Fix.** Added a lock around model loads and swaps. Training now builds a local network
and only publishes the best checkpoint to the live model after the run finishes, so
inference never sees a model that is mid-training or in train mode.

## 2. Verification approach

Fixes were checked against real chest X-rays cached locally from the Kaggle datasets, not
just synthetic tensors:
- Timed the first inference after warm-up to confirm the freeze was gone.
- Ran the DenseNet directly on Normal, Pneumonia, and TB images to confirm it
  discriminates.
- Rendered and visually inspected overlays for pneumonia (hot region plus outline on the
  consolidation), normal (lung-focused assessment map), a forced improbable class
  (suppressed outline), and pure noise.
- Booted the Gradio UI and confirmed the panel labels, disclaimer text, and dropdown
  default were wired correctly.

## 3. Known limitations

- **No out-of-distribution rejection.** A non-X-ray or pure noise still gets classified
  into one of the four classes (noise was classified as TB at over 80 percent). The
  confidence threshold only catches low-confidence cases, not confidently wrong ones.
- **Segmentation masks are pseudo-labels.** Without real lung or lesion annotations the
  U-Net cannot learn true opacity boundaries; the Grad-CAM heatmap is an attention map,
  not a precise lesion mask.
- **Grad-CAM is coarse.** It comes from a low-resolution feature map upsampled to full
  size, so it localizes a region, not a sharp boundary.
- **The Normal-case heatmap is broad.** Showing "areas assessed" over the lung fields is
  honest but not a tight localization, because there is no lesion to localize.
- **CPU-only latency.** Running two models per request is fine on a laptop but not
  optimized; there is no batching or quantization yet.

## 4. Ideas for future iterations

### Accuracy and trust
- Add an out-of-distribution or non-chest-X-ray detector (for example an energy or
  softmax-entropy gate, or a lightweight binary "is this a chest X-ray" check) so noise
  and unrelated images are rejected rather than confidently misclassified.
- Calibrate the classifier (temperature scaling) so the reported percentages reflect true
  probabilities, then set the uncertainty threshold from calibrated confidence.
- Report a short differential (top two classes with a margin) instead of a single label
  when the two leading classes are close.

### Better localization
- Train real lung-field segmentation using a public labeled set (for example the
  Montgomery and Shenzhen masks) so the overlay is constrained to the lungs and cannot
  wander onto bone.
- Consider Grad-CAM++ or higher-resolution CAM variants, or a small supervised
  segmentation head, for tighter regions.
- Restrict any displayed region to the lung fields with an anatomical mask as a safety net.

### Robustness and engineering
- Add automated tests (pytest) for the inference pipeline: prediction shape, overlay
  gating decisions, and heatmap rendering, with a few fixed sample images checked into a
  small test asset folder.
- Add structured logging (the standard `logging` module) with per-request inference time,
  replacing scattered prints.
- Validate uploaded images more strictly (aspect ratio, single channel expected, size
  bounds) and give clearer guidance on rejection.

### Performance
- Quantize the models to INT8 or export to ONNX Runtime for faster CPU inference.
- Cache the last result and avoid recomputing the classifier forward pass when only the
  overlay class changes.
- Optionally run only the DenseNet by default and compute Grad-CAM lazily, loading the
  U-Net only if segmentation is explicitly requested.

### Product and UX
- Let the user toggle the heatmap on and off and adjust its opacity.
- Offer a side-by-side view (original next to heatmap) instead of a single blended image.
- Add an export button for the annotated image and a short text report.
- Make the disclaimer and "research use only" framing persistent and unmissable, since the
  tool can produce confident-looking but incorrect results.

### Data and training
- Grow and balance the datasets, especially Covid-19, and track a proper held-out test set
  with per-class precision and recall rather than only accuracy and Dice.
- Add k-fold or at least a fixed, documented train/validation/test split so results are
  reproducible.
- Log training runs (already writing TensorBoard scalars) and keep a short changelog of
  checkpoint versions and their metrics.
