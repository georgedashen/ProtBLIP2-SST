#!/usr/bin/env bash
# Stage1 SaProt 35M ablation: PTM and LM losses (Table 2 style).
# Baseline (ptm=1, lm=1) already exists: all_checkpoints/stage1_sa/ (TB version_6).
#
# Runs sequentially on 2 GPUs. Each job ~same cost as full stage1_sa 35M training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false

PLM="../SaProt/weights/PLMs/SaProt_35M_AF2"
COMMON=(
  --strategy deepspeed --devices '0,1' --mode train
  --num_query_token 8 --num_workers 8
  --plm_name "${PLM}" --plm_tune freeze
  --save_every_n_epochs 10 --batch_size 32
  --precision bf16-mixed --match_batch_size 64
  --max_epochs 50 --use_tb_logger
  --suffix _saprot
  --retrieval_eval_epoch 10
  --check_val_every_n_epoch 1
  --seed 42
)

run_one() {
  local tag="$1" ptm="$2" lm="$3" log="$4"
  echo "========== ${tag} (ptm=${ptm}, lm=${lm}) -> ${log} =========="
  python stage1sa.py "${COMMON[@]}" \
    --filename "stage1_sa_abl35m_${tag}" \
    --ptm "${ptm}" --lm "${lm}" \
    > "${log}" 2>&1
  echo "Finished ${tag}. Log: ${log}"
}

# Skip full if baseline checkpoint already trained (stage1_sa / epoch=49.ckpt).
if [[ "${SKIP_FULL:-1}" == "1" ]] && [[ -d all_checkpoints/stage1_sa/epoch=49.ckpt ]]; then
  echo "Skipping full baseline (stage1_sa epoch=49 exists). Set SKIP_FULL=0 to retrain."
else
  run_one "full" 1 1 "sa_abl35m_full.log"
fi

run_one "noptm" 0 1 "sa_abl35m_noptm.log"
run_one "nolm"  1 0 "sa_abl35m_nolm.log"

echo "All ablation runs complete."
echo "Extract metrics: python read_results.py --path all_checkpoints/stage1_sa_abl35m_noptm/tb_logs/lightning_logs/version_0"
