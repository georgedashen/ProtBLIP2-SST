#!/usr/bin/env bash
# Finetune raw AA letters on SwissProtV3 _afdbfilt
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python stage2_AAraw.py --devices 0,1 --mode train --strategy deepspeed \
  --filename stage2_aaraw_afdbfilt \
  --suffix _afdbfilt \
  --root data/SwissProtV3 --batch_size 4 --llm_tune mid_lora --enable_flash \
  --use_tb_logger --save_every_n_epochs 10
