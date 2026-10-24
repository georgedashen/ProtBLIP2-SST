#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
# Use local HF cache when available (ProstT5, BiomedBERT are cached on this machine)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting stage1pt5 ProstT5 training with aldbfilt cached"

# Run sequentially to avoid GPU/checkpoint collisions. Distinct filenames per suffix.
python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_pt5_afdbfilt --num_query_token 8 --num_workers 2 \
  --plm_name "Rostlab/ProstT5" --save_every_n_epochs 10 --batch_size 32 \
  --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 \
  --use_tb_logger --suffix _afdbfilt \
  --plm_embed_cache_workers_per_gpu 2 --plm_embed_cache_batch_size 32 > pt5_afdbfilt_cached.log 2>&1

echo "Starting stage1pt5 ProstT5 training with saprot 3Dimask cached"

python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_pt5_saprot_3Dimask --num_query_token 8 --num_workers 2 \
  --plm_name "Rostlab/ProstT5" --save_every_n_epochs 10 --batch_size 32 \
  --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 \
  --use_tb_logger --suffix _saprot_3Dimask \
  --plm_embed_cache_workers_per_gpu 2 --plm_embed_cache_batch_size 32 > pt5_saprot_3Dimask_cached.log 2>&1

echo "All training processes have completed."