#!/usr/bin/env bash
# Eval stage2 PLM8 SaProt 35M (8-token prefix, gentime-safe caption eval).
# Writes to a separate dir; does not overwrite full-length SaProt PLM results.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${HOME}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_DIR="${CKPT_DIR:-all_checkpoints/stage2_plm_saprot_35m_8tok_saprot}"
DEEPSPEED_CKPT="${DEEPSPEED_CKPT:-epoch=09.ckpt}"
FP32_CKPT="${FP32_CKPT:-${CKPT_DIR}/SaProt_35M_plm8_stage2_fp32.ckpt}"
PLM="../SaProt/weights/PLMs/SaProt_35M_AF2"
EVAL_FILENAME="${EVAL_FILENAME:-stage2_plm_saprot_35m_8tok_saprot_eval}"
LOG="${LOG:-stage2_plm_saprot_35m_8tok_eval.log}"

if [[ ! -f "${FP32_CKPT}" ]]; then
  echo "Converting ${CKPT_DIR}/${DEEPSPEED_CKPT} -> ${FP32_CKPT}"
  python convert.py --input "${CKPT_DIR}/${DEEPSPEED_CKPT}" --output "${FP32_CKPT}"
fi

echo "========== eval stage2 PLM8 SaProt 35M -> ${LOG} =========="
python ablations/stage2_pLM8.py \
  --mode eval \
  --devices "${DEVICES:-0,1}" \
  --strategy deepspeed \
  --filename "${EVAL_FILENAME}" \
  --init_checkpoint "${FP32_CKPT}" \
  --plm_model "${PLM}" \
  --llm_name facebook/galactica-1.3b \
  --suffix _saprot \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --num_prefix_tokens 8 \
  --llm_tune mid_lora \
  --enable_flash \
  --use_tb_logger \
  --caption_eval_epoch 1 \
  --gen_max_time_sec "${GEN_MAX_TIME_SEC:-600}" \
  > "${LOG}" 2>&1

echo "Finished. Log: ${LOG}"
echo "Predictions: all_checkpoints/${EVAL_FILENAME}/lightning_logs/version_0/dataset0_predictions.txt"
echo "Original full-seq (unchanged): all_checkpoints/stage2_plm_saprot_35m_saprot/lightning_logs/"
