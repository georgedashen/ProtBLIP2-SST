#!/usr/bin/env bash
# Stage-2 PLM-direct captioning: SaProt 35M on SwissProtV3 _saprot splits.
# Mirrors stage2plm_run.sh / all_checkpoints/stage2_plm_esm2_35m_afdbfilt:
#   - no Q-Former; full PLM hidden states -> Galactica prefix
#   - Galactica-1.3b + mid_lora, batch_size 4, 10 epochs, bf16-mixed
#   - checkpoints/logs -> all_checkpoints/stage2_plm_saprot_35m_saprot/
#
# Optional: load stage-1 SaProt weights (commented; ESM reference run did not use stage1).
#   --stage1_path all_checkpoints/stage1_sa/SaProt_35M_fp32.ckpt
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${HOME}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PLM="../SaProt/weights/PLMs/SaProt_35M_AF2"
LOG="${LOG:-stage2_plm_saprot_35m_saprot.log}"

echo "========== stage2 PLM SaProt 35M (_saprot) -> ${LOG} =========="
python stage2_pLM.py \
  --devices "${DEVICES:-0,1}" \
  --mode train \
  --strategy deepspeed \
  --filename stage2_plm_saprot_35m_saprot \
  --plm_model "${PLM}" \
  --llm_name facebook/galactica-1.3b \
  --suffix _saprot \
  --root data/SwissProtV3 \
  --batch_size 4 \
  --llm_tune mid_lora \
  --enable_flash \
  --use_tb_logger \
  --save_every_n_epochs 10 \
  --gen_max_time_sec "${GEN_MAX_TIME_SEC:-600}" \
  > "${LOG}" 2>&1

echo "Finished. Log: ${LOG}"
echo "Checkpoints: all_checkpoints/stage2_plm_saprot_35m_saprot/"
echo "Metrics: all_checkpoints/stage2_plm_saprot_35m_saprot/lightning_logs/version_0/metrics.csv"
