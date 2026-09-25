#!/usr/bin/env bash
# One-time setup of an AWS GPU instance for the E1 experiments.
# Run ON the instance, from the repository root, after cloning:
#
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...            # your Kaggle API token
#   export LUNGFIELD_S3=s3://YOUR_BUCKET/lunglens/lungfield_unet.pth   # optional
#   bash lung_attention/aws/setup_instance.sh
#
# Safe to re-run: every step is idempotent.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
VENV="${VENV:-$HOME/lunglens-venv}"
LUNG_WEIGHTS="$REPO_ROOT/fig7_work/lungfield_unet.pth"
MANIFEST="$REPO_ROOT/lung_attention/splits/split_s32000_seed42.csv"

echo "== 1/6 GPU check"
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
  echo "No NVIDIA driver found. Use a GPU instance (g5/g6) with an AWS Deep Learning Base GPU AMI." >&2
  exit 1
fi

echo "== 2/6 Python 3.11 environment (matches the laptop)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r lung_attention/aws/requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== 3/6 Kaggle credentials"
if [ -z "${KAGGLE_USERNAME:-}" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "WARNING: no Kaggle credentials. Set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json if a download fails." >&2
fi

echo "== 4/6 Lung-field segmenter weights (gitignored, so not in the clone)"
if [ ! -f "$LUNG_WEIGHTS" ]; then
  if [ -n "${LUNGFIELD_S3:-}" ]; then
    aws s3 cp "$LUNGFIELD_S3" "$LUNG_WEIGHTS"
  else
    echo "Missing $LUNG_WEIGHTS. Upload it (scp or S3) or set LUNGFIELD_S3, then re-run." >&2
    exit 1
  fi
fi

echo "== 5/6 Download the eight datasets and verify the frozen split"
LUNGLENS_SKIP_STARTUP=1 python -c "import app; app.get_dataset_paths()"
python -m lung_attention.split_manifest verify --manifest "$MANIFEST"

echo "== 6/6 Unit tests"
python -m unittest discover -s lung_attention/tests -t .

echo "Setup complete. Start the experiments inside tmux:"
echo "  tmux new -s e1 'bash lung_attention/aws/run_e1.sh'"
