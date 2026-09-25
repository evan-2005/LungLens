#!/usr/bin/env bash
# Run the E1 lambda sweep on an AWS GPU instance, evaluate every model on the
# validation split, and write one comparison table. Run inside tmux so it
# survives an SSH disconnect:
#
#   tmux new -s e1 'bash lung_attention/aws/run_e1.sh'
#
# Settings (environment variables, all optional):
#   LAMBDAS="0 0.5 1 2"        attention-loss weights; 0 is the control run
#   BATCH=32 WORKERS=4 EPOCHS=40
#   AMP=0                      set 1 for mixed precision (faster, less memory)
#   RESULTS_S3=s3://bucket/lunglens/e1   copy results after every step
#   FINAL_TEST=""              a lambda (e.g. 1) to ALSO score on the test split;
#                              choose it from the validation table first
#   SHUTDOWN_WHEN_DONE=0       set 1 to power the instance off at the end
#
# Finished runs are skipped on re-run, so an interrupted sweep can be restarted.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${VENV:-$HOME/lunglens-venv}/bin/activate"

LAMBDAS="${LAMBDAS:-0 0.5 1 2}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-40}"
AMP_FLAG=""
[ "${AMP:-0}" = "1" ] && AMP_FLAG="--amp"
MANIFEST="lung_attention/splits/split_s32000_seed42.csv"
OUT="runs/lungattn"
mkdir -p "$OUT"

sync_results() {
  if [ -n "${RESULTS_S3:-}" ]; then
    aws s3 sync "$OUT" "$RESULTS_S3" --exclude "*.pth" --only-show-errors || \
      echo "WARNING: S3 sync failed; results are still on this instance in $OUT" >&2
  fi
}

finish() {
  sync_results
  if [ "${SHUTDOWN_WHEN_DONE:-0}" = "1" ]; then
    echo "Shutting down in 60 s (Ctrl-C to cancel)."
    sleep 60 && sudo shutdown -h now
  fi
}
trap finish EXIT

evaluate() {  # $1 = checkpoint, $2 = run dir, $3 = split
  local target="$2/eval_$3.json"
  [ -f "$target" ] && { echo "skip eval: $target exists"; return; }
  python -m lung_attention.eval_lungattn --ckpt "$1" --split "$3" \
    --split-manifest "$MANIFEST" --out "$target" 2>&1 | tee "$2/eval_$3.log"
}

echo "== Baseline: the served checkpoint as committed in git"
mkdir -p "$OUT/baseline"
git show HEAD:chest_model_4class.pth > "$OUT/baseline/classifier.pth"
evaluate "$OUT/baseline/classifier.pth" "$OUT/baseline" val
sync_results

for LAM in $LAMBDAS; do
  RUN="$OUT/lam$LAM"
  mkdir -p "$RUN"
  if [ -f "$RUN/train_metrics.json" ]; then
    echo "== lambda=$LAM already trained, skipping"
  else
    echo "== Training lambda=$LAM"
    python -m lung_attention.train_lungattn --lam "$LAM" --split-manifest "$MANIFEST" \
      --batch "$BATCH" --workers "$WORKERS" --epochs "$EPOCHS" $AMP_FLAG --out "$RUN" \
      2>&1 | tee "$RUN/train.log"
  fi
  evaluate "$RUN/classifier.pth" "$RUN" val
  sync_results
done

python -m lung_attention.summarise_runs --split val --out "$OUT/summary_val.md"

if [ -n "${FINAL_TEST:-}" ]; then
  echo "== Test split, once: baseline and lambda=$FINAL_TEST"
  evaluate "$OUT/baseline/classifier.pth" "$OUT/baseline" test
  evaluate "$OUT/lam$FINAL_TEST/classifier.pth" "$OUT/lam$FINAL_TEST" test
  python -m lung_attention.summarise_runs --split test --out "$OUT/summary_test.md"
fi
echo "Done. Tables: $OUT/summary_val.md ${FINAL_TEST:+and $OUT/summary_test.md}"
