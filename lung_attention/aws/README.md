# Running E1 on AWS

Everything the experiment needs runs on one AWS GPU instance. Your laptop is only used once, to push the code and upload one weights file.

## Why this is not just "copy the code and run it"

Two things would silently change the results on a cloud machine, and both are handled:

1. **The split.** The code sorts file paths before building the train/val/test split. Windows and Linux sort paths differently, so a Linux machine would pick a different 32,000-image subsample from the one behind the paper. The split is therefore frozen in `lung_attention/splits/split_s32000_seed42.csv`, exported on the laptop, and checked by fingerprint on AWS. Its counts (22,419 / 4,822 / 4,759) match the served model's recorded split exactly.
2. **Gitignored files.** The baseline checkpoint is committed, so a fresh clone gets it. The lung-field segmenter (`fig7_work/lungfield_unet.pth`, 7.7 MB) is not, so it has to be uploaded once.

## Instance

| Choice | Recommendation |
| --- | --- |
| Instance type | `g5.xlarge` (1x A10G 24 GB, 4 vCPU, 16 GB RAM), the same GPU as the original training. `g6.xlarge` (L4) also works. |
| AMI | AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" or the 24.04 version. It only needs the NVIDIA driver; the script builds its own Python 3.11 environment. |
| Disk | 100 GB gp3 root volume (the eight datasets plus caches) |
| Purchase | On-demand is simplest. Spot is cheaper but can be interrupted; the run script skips finished runs, but an interrupted training run restarts from zero. |
| Region | Any region with g5 capacity; put the S3 bucket in the same region. |

Check current pricing in the AWS console before launching. Stopping the instance stops GPU charges, but the disk (EBS) is still billed until the instance is terminated.

## Steps

**1. On the laptop: commit and push the branch** so the instance can clone it (the code, the split manifest and the committed baseline checkpoint).

**2. On the laptop: upload the lung segmenter weights.** Either to S3:

```bash
aws s3 cp fig7_work/lungfield_unet.pth s3://YOUR_BUCKET/lunglens/lungfield_unet.pth
```

or straight to the instance after step 3 with `scp`.

**3. Launch the instance and SSH in**, then clone:

```bash
git clone -b paper/localisation-faithfulness https://github.com/evan-2005/LungLens.git && cd LungLens
```

**4. Set credentials for this session only** (never commit them). Kaggle API token from kaggle.com, Account, Create New Token:

```bash
export KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key LUNGFIELD_S3=s3://YOUR_BUCKET/lunglens/lungfield_unet.pth
```

The instance needs an IAM role with read/write access to that bucket for the S3 steps.

**5. Set up** (GPU check, Python 3.11 with the laptop's exact package versions, dataset download, split verification, unit tests):

```bash
bash lung_attention/aws/setup_instance.sh
```

If the split check fails, a Kaggle dataset has changed since the paper was written. Stop there: results would not be comparable.

**6. Run the sweep inside tmux** so it keeps going if SSH drops:

```bash
tmux new -s e1 'RESULTS_S3=s3://YOUR_BUCKET/lunglens/e1 bash lung_attention/aws/run_e1.sh'
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t e1`. It evaluates the baseline, then trains and evaluates lambda = 0, 0.5, 1 and 2, and writes `runs/lungattn/summary_val.md`. Add `SHUTDOWN_WHEN_DONE=1` to power off at the end and `AMP=1` to run in mixed precision.

**7. Pick lambda from the validation table, then score the test split once:**

```bash
FINAL_TEST=1 bash lung_attention/aws/run_e1.sh
```

Finished training runs are skipped, so this only adds the two test evaluations.

**8. Get the results back to the laptop:**

```bash
aws s3 sync s3://YOUR_BUCKET/lunglens/e1 runs/lungattn
```

Checkpoints (`*.pth`) are not synced by default; copy the chosen one explicitly if you want to serve it.

**9. Terminate the instance** when finished, so the disk stops being billed.

## Time estimate

Not measured yet. The original classifier run stopped at epoch 18 on an A10G. Expect each lambda to take a similar number of epochs, plus a few minutes per evaluation (one Grad-CAM per validation image). Time the lambda = 0 run first and scale the sweep from that.
