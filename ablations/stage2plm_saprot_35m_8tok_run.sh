#!/usr/bin/env bash
# Train stage2 PLM8 SaProt 35M: compress full PLM hidden states to 8 Galactica prefix tokens.
# Mirrors stage2plm_saprot_35m_run.sh (full-seq) and stage2plm_esm2_35m_8tok_run.sh.
#
#   SaProt [B,L,480] -> plm_proj -> [B,L,2048] -> seq_compress -> [B,8,2048] -> Galactica
#   Dataset: SwissProtV3 _saprot (same splits as stage2_plm_saprot_35m_saprot)
#   Does not overwrite all_checkpoints/stage2_plm_saprot_35m_saprot/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${HOME}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PLM="../SaProt/weights/PLMs/SaProt_35M_AF2"
LOG="${LOG:-stage2_plm_saprot_35m_8tok.log}"

echo "========== train stage2 PLM8 SaProt 35M (_saprot) -> ${LOG} =========="
python ablations/stage2_pLM8.py \
  --devices "${DEVICES:-0,1}" \
  --mode train \
  --strategy deepspeed \
  --filename stage2_plm_saprot_35m_8tok_saprot \
  --plm_model "${PLM}" \
  --llm_name facebook/galactica-1.3b \
  --suffix _saprot \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --num_prefix_tokens 8 \
  --llm_tune mid_lora \
  --enable_flash \
  --use_tb_logger \
  --save_every_n_epochs 10 \
  --max_epochs 10 \
  --gen_max_time_sec "${GEN_MAX_TIME_SEC:-600}" \
  > "${LOG}" 2>&1

echo "Finished. Log: ${LOG}"
echo "Checkpoints: all_checkpoints/stage2_plm_saprot_35m_8tok_saprot/"
echo "Metrics: all_checkpoints/stage2_plm_saprot_35m_8tok_saprot/lightning_logs/version_0/metrics.csv"
