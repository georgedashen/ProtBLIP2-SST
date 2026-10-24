#!/usr/bin/env bash
# Resume stage1_pt5_afdbfilt from epoch=39.ckpt (continues at epoch 40).
# Hyperparameters must match version_7 in tb_logs (see hparams.yaml).
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_PATH="${CKPT_PATH:-all_checkpoints/stage1_pt5_afdbfilt/epoch=39.ckpt}"

echo "Resuming stage1_pt5_afdbfilt from ${CKPT_PATH}"

python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_pt5_afdbfilt --num_query_token 8 --num_workers 2 \
  --plm_name "Rostlab/ProstT5" --save_every_n_epochs 10 --batch_size 32 \
  --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 \
  --use_tb_logger --suffix _afdbfilt \
  --plm_embed_cache_workers_per_gpu 2 --plm_embed_cache_batch_size 32 \
  --ckpt_path "${CKPT_PATH}" \
  > pt5_afdbfilt_cached_resume.log 2>&1

echo "Resume training finished. Log: pt5_afdbfilt_cached_resume.log"
