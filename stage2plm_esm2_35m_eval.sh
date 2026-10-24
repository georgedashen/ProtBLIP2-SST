#!/usr/bin/env bash
# Eval-only for stage2 PLM ESM2 35M (_afdbfilt) with per-sample generation timeout.
# Writes to a separate run dir so training/original lightning_logs are not overwritten.
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${HOME}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_DIR="all_checkpoints/stage2_plm_esm2_35m_afdbfilt"
DEEPSPEED_CKPT="${DEEPSPEED_CKPT:-epoch=09.ckpt}"
FP32_CKPT="${FP32_CKPT:-${CKPT_DIR}/ESM2_35M_stage2_fp32.ckpt}"
PLM="fredzzp/esm2_t12_35M_UR50D"
# Distinct filename -> all_checkpoints/stage2_plm_esm2_35m_afdbfilt_eval_gentime/
EVAL_FILENAME="${EVAL_FILENAME:-stage2_plm_esm2_35m_afdbfilt_eval_gentime}"
LOG="${LOG:-stage2_plm_esm2_35m_eval_gentime.log}"

if [[ ! -f "${FP32_CKPT}" ]]; then
  echo "Converting ${CKPT_DIR}/${DEEPSPEED_CKPT} -> ${FP32_CKPT}"
  python convert.py --input "${CKPT_DIR}/${DEEPSPEED_CKPT}" --output "${FP32_CKPT}"
fi

echo "========== eval stage2 PLM ESM2 35M (gentime pipeline) -> ${LOG} =========="
echo "Checkpoint: ${FP32_CKPT}"
echo "Output dir: all_checkpoints/${EVAL_FILENAME}/"
python stage2_pLM.py \
  --mode eval \
  --devices "${DEVICES:-0,1}" \
  --strategy deepspeed \
  --filename "${EVAL_FILENAME}" \
  --init_checkpoint "${FP32_CKPT}" \
  --plm_model "${PLM}" \
  --llm_name facebook/galactica-1.3b \
  --suffix _afdbfilt \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --llm_tune mid_lora \
  --enable_flash \
  --use_tb_logger \
  --caption_eval_epoch 1 \
  --gen_max_time_sec "${GEN_MAX_TIME_SEC:-600}" \
  > "${LOG}" 2>&1

echo "Finished. Log: ${LOG}"
echo "Predictions: all_checkpoints/${EVAL_FILENAME}/lightning_logs/version_0/dataset0_predictions.txt"
echo "Metrics:     all_checkpoints/${EVAL_FILENAME}/lightning_logs/version_0/metrics.csv"
echo "Original (unchanged): ${CKPT_DIR}/lightning_logs/version_0/"
