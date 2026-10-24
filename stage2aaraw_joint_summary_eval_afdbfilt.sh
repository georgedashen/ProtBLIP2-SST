#!/usr/bin/env bash
# Zero-shot eval: Galactica joint prefix [START_AMINO]seq[END_AMINO] Summary Protein:
# on SwissProtV3 test (_afdbfilt). LLM frozen (llm_tune=freeze).
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

FILENAME="${FILENAME:-stage2_aaraw_joint_summary_galactica_zero_shot_afdbfilt_eval}"

python stage2_AAraw_joint.py \
  --mode eval \
  --devices '0,1' \
  --strategy deepspeed \
  --filename "${FILENAME}" \
  --suffix _afdbfilt \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --inference_batch_size 4 \
  --llm_tune freeze \
  --llm_name facebook/galactica-1.3b \
  --caption_eval_epoch 1 \
  --check_val_every_n_epoch 1 \
  --precision bf16-mixed \
  --num_workers 2 \
  --enable_flash \
  --use_tb_logger

LOG_ROOT="all_checkpoints/${FILENAME}"
echo ""
echo "Done. Outputs:"
echo "  Predictions: ${LOG_ROOT}/lightning_logs/version_*/dataset0_predictions.txt"
echo "  Metrics:     ${LOG_ROOT}/lightning_logs/version_*/metrics.csv and ${LOG_ROOT}/tb_logs/"
