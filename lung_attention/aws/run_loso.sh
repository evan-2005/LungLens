#!/usr/bin/env bash
# Leave-one-source-out (#9): for each source, train on the other sources and
# test on all of the held-out one. Run inside tmux after setup_instance.sh:
#
#   tmux new -s loso 'bash lung_attention/aws/run_loso.sh'
#
# Settings (environment variables, all optional):
#   HOLDOUTS="tb_ds pneu_ds covid_ds radiography_db shenzhen_tb montgomery_tb tbx11k"
#   LAMBDAS="0 1"          0 = the baseline recipe; 1 = lung-constrained attention (E1)
#   DROP_LUNG_OPACITY=0    set 1 to also apply the #7 label fix
#   BATCH=32 WORKERS=4 EPOCHS=40 AMP=0 RESULTS_S3= SHUTDOWN_WHEN_DONE=0
#
# Holding out radiography_db leaves ~9.8k training images and tests on ~18k:
# expect that run to be the slowest to train and the hardest to generalise.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${VENV:-$HOME/lunglens-venv}/bin/activate"

HOLDOUTS="${HOLDOUTS:-tb_ds pneu_ds covid_ds radiography_db shenzhen_tb montgomery_tb tbx11k}"
LAMBDAS="${LAMBDAS:-0 1}"
BATCH="${BATCH:-32}"; WORKERS="${WORKERS:-4}"; EPOCHS="${EPOCHS:-40}"
EXTRA=()
[ "${AMP:-0}" = "1" ] && EXTRA+=(--amp)
SPLIT_ARGS=(--split-manifest lung_attention/splits/split_s32000_seed42.csv)
SUFFIX=""
if [ "${DROP_LUNG_OPACITY:-0}" = "1" ]; then SPLIT_ARGS+=(--drop-lung-opacity); SUFFIX="_noLO"; fi
OUT="runs/loso$SUFFIX"
mkdir -p "$OUT"

finish() {
  if [ -n "${RESULTS_S3:-}" ]; then
    aws s3 sync "$OUT" "$RESULTS_S3/loso$SUFFIX" --exclude "*.pth" --only-show-errors || \
      echo "WARNING: S3 sync failed; results remain in $OUT" >&2
  fi
  if [ "${SHUTDOWN_WHEN_DONE:-0}" = "1" ]; then sleep 60 && sudo shutdown -h now; fi
}
trap finish EXIT

for SRC in $HOLDOUTS; do
  for LAM in $LAMBDAS; do
    RUN="$OUT/$SRC/lam$LAM"
    mkdir -p "$RUN"
    if [ ! -f "$RUN/train_metrics.json" ]; then
      echo "== hold out $SRC, lambda=$LAM: training"
      python -m lung_attention.train_lungattn --lam "$LAM" "${SPLIT_ARGS[@]}" \
        --holdout-source "$SRC" --batch "$BATCH" --workers "$WORKERS" --epochs "$EPOCHS" \
        "${EXTRA[@]}" --out "$RUN" 2>&1 | tee "$RUN/train.log"
    fi
    if [ ! -f "$RUN/eval_test.json" ]; then
      echo "== hold out $SRC, lambda=$LAM: scoring the held-out source"
      python -m lung_attention.eval_lungattn --ckpt "$RUN/classifier.pth" --split test \
        "${SPLIT_ARGS[@]}" --holdout-source "$SRC" --out "$RUN/eval_test.json" \
        2>&1 | tee "$RUN/eval_test.log"
    fi
  done
done

python -m lung_attention.summarise_runs --runs "$OUT" --split test --out "$OUT/summary_test.md"
echo "Done. Table: $OUT/summary_test.md"
