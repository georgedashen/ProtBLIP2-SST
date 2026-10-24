#!/usr/bin/env bash
# Train stage2 PLM ESM2 35M with L->8 prefix compression (no Q-Former).
# Mirrors stage2plm_run.sh / stage2_reproduce.sh (35M _afdbfilt block).
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${HOME}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG="${LOG:-stage2_plm_esm2_35m_8tok.log}"

echo "========== train stage2 PLM8 ESM2 35M -> ${LOG} =========="
python stage2_pLM8.py \
  --devices "${DEVICES:-0,1}" \
  --mode train \
  --strategy deepspeed \
  --filename stage2_plm_esm2_35m_8tok_afdbfilt \
  --plm_model fredzzp/esm2_t12_35M_UR50D \
  --llm_name facebook/galactica-1.3b \
  --suffix _afdbfilt \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --num_prefix_tokens 8 \
  --llm_tune mid_lora \
  --enable_flash \
  --use_tb_logger \
  --save_every_n_epochs 10 \
  --max_epochs 10 \
  > "${LOG}" 2>&1

echo "Finished. Log: ${LOG}"
echo "Checkpoints: all_checkpoints/stage2_plm_esm2_35m_8tok_afdbfilt/"
