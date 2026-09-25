# E1: lung-constrained attention training (prototype)

Trains the LungLens DenseNet-121 with an extra loss that penalises class-activation energy **outside the lung field**. It targets the measured failure that 55% of served heatmaps put most of their region outside the lungs. The method follows "Right for the Right Reasons" (Ross et al., 2017) and GAIN (Li et al., 2018). Rationale and paper placement are in `paper/PAPER_PLAN.md`, Section 6.2.

## Files

| File | Purpose |
| --- | --- |
| `attention_loss.py` | CAM from the final DenseNet features, lung masks from the augmented batch, outside-lung fraction loss |
| `train_lungattn.py` | Training driver, identical to `app.run_training_thread` plus `lambda x L_attn`; writes to `--out` only |
| `eval_lungattn.py` | Classification metrics plus where the **served** Grad-CAM lands, against ground-truth and predicted lung masks |
| `split_manifest.py` | Freezes the paper's train/val/test split so it is identical on any OS |
| `splits/split_s32000_seed42.csv` | That frozen split (22,419 / 4,822 / 4,759), exported from the laptop |
| `summarise_runs.py` | One Markdown comparison table from all evaluation outputs |
| `aws/` | Cloud setup and sweep scripts; **start with `aws/README.md`** |
| `tests/` | Unit tests (stdlib `unittest`) |

## Loss

```
CAM_c   = ReLU( sum_k w_ck * F_k )          # final features, GT class c
M       = dilate( lungUNet(augmented image) > 0.5 , 7 px )
L_attn  = sum(CAM_c * (1 - M)) / sum(CAM_c)  # averaged over images with lung area in [10%, 70%]
L       = weighted CE + lambda * L_attn      # lambda = 0 for --warmup-epochs
```

## Run

The full experiment is meant to run on AWS: see [aws/README.md](aws/README.md), which does all of the steps below with one script. Always pass `--split-manifest lung_attention/splits/split_s32000_seed42.csv` to the train and eval scripts; without it the split is re-derived, and a different operating system gives a different split.

To run by hand from the repository root:

```bash
python -m unittest discover -s lung_attention/tests -t .
```

```bash
python -m lung_attention.train_lungattn --lam 0 --split-manifest lung_attention/splits/split_s32000_seed42.csv --out runs/lungattn/lam0
```

```bash
python -m lung_attention.train_lungattn --lam 1.0 --split-manifest lung_attention/splits/split_s32000_seed42.csv --out runs/lungattn/lam1
```

```bash
python -m lung_attention.eval_lungattn --ckpt runs/lungattn/lam1/classifier.pth --split val --split-manifest lung_attention/splits/split_s32000_seed42.csv --out runs/lungattn/lam1/eval_val.json
```

Defaults match the served recipe (32,000 samples, Adam 1e-4, batch 32, early stopping on validation macro-F1). The defaults fit an A10G (AWS `g5.xlarge`); on a small GPU such as a 6 GB laptop card use `--batch 16 --amp`, for smoke tests only. Choose lambda on `--split val`; run `--split test` once per final model.

## Evaluation rules

- **Baseline:** evaluate the served checkpoint as committed in git (`git show HEAD:chest_model_4class.pth`; `aws/run_e1.sh` does this). On the laptop the working-tree `chest_model_4class.pth` has uncommitted changes and must not be used; `fig7_work/chest_model_4class_HEAD.pth` is a byte-identical copy of the committed one. Also run `--lam 0` through this script as the like-for-like control.
- **Circularity:** the lung U-Net that supervises training also produces the `heatmap_pred_lung` numbers, so those flatter E1. The headline in-lung numbers must come from `heatmap_gt_lung` (Radiography Database ground-truth masks).
- **What is scored:** the app's own Grad-CAM at `denseblock4.denselayer16.conv2`, floored and thresholded exactly as the UI does. It is not the final-layer CAM the loss uses.
- **Known limitation:** the lung U-Net was trained only on Radiography Database films. `lung_mask_valid_frac` in the training log shows how often the penalty was skipped because a predicted lung mask was implausible; report it per source before trusting E1 on Shenzhen, Montgomery or the paediatric set.
