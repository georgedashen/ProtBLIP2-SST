#!/usr/bin/env bash
# Eval a finetuned stage2_aaraw_presuffix checkpoint on SwissProtV3 test (_afdbfilt).
#
# Usage:
#   bash stage2aaraw_presuffix_eval_finetuned_afdbfilt.sh
#   DEEPSPEED_CKPT=last.ckpt bash stage2aaraw_presuffix_eval_finetuned_afdbfilt.sh
#   CKPT_DIR=all_checkpoints/stage2_aaraw_presuffix_afdbfilt DEEPSPEED_CKPT=epoch=09.ckpt \
#     bash stage2aaraw_presuffix_eval_finetuned_afdbfilt.sh
set -euo pipefail
cd "$(dirname "$0")"

CKPT_DIR="${CKPT_DIR:-all_checkpoints/stage2_aaraw_presuffix_afdbfilt}"
DEEPSPEED_CKPT="${DEEPSPEED_CKPT:-last.ckpt}"
FP32_CKPT="${FP32_CKPT:-${CKPT_DIR}/stage2_aaraw_presuffix_afdbfilt_fp32.ckpt}"
EVAL_TAG="${EVAL_TAG:-eval}"
FILENAME="${FILENAME:-stage2_aaraw_presuffix_afdbfilt_${EVAL_TAG}}"

if [[ ! -f "${FP32_CKPT}" ]]; then
  if [[ ! -d "${CKPT_DIR}/${DEEPSPEED_CKPT}" ]]; then
    echo "Error: checkpoint not found: ${CKPT_DIR}/${DEEPSPEED_CKPT}" >&2
    echo "Available:" >&2
    ls -1d "${CKPT_DIR}"/*.ckpt 2>/dev/null | xargs -n1 basename >&2 || true
    exit 1
  fi
  echo "Converting DeepSpeed checkpoint ${CKPT_DIR}/${DEEPSPEED_CKPT} -> ${FP32_CKPT}"
  python convert.py \
    --input "${CKPT_DIR}/${DEEPSPEED_CKPT}" \
    --output "${FP32_CKPT}"
fi

if [[ ! -f "${FP32_CKPT}" ]]; then
  echo "Error: FP32 checkpoint missing: ${FP32_CKPT}" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python stage2_AAraw_presuffix.py \
  --mode eval \
  --devices '0,1' \
  --strategy deepspeed \
  --filename "${FILENAME}" \
  --init_checkpoint "${FP32_CKPT}" \
  --suffix _afdbfilt \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --inference_batch_size 4 \
  --llm_tune mid_lora \
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
echo "  Metrics:     ${LOG_ROOT}/metrics.csv and ${LOG_ROOT}/tb_logs/"
