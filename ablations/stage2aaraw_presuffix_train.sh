#!/usr/bin/env bash
# Finetune Galactica on SwissProtV3 _afdbfilt with [START_AMINO]/[END_AMINO] AA prefix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python ablations/stage2_AAraw_presuffix.py --devices 0,1 --mode train --strategy deepspeed \
  --filename stage2_aaraw_presuffix_afdbfilt \
  --suffix _afdbfilt \
  --root data/SwissProtV3 --batch_size 4 --llm_tune mid_lora --enable_flash \
  --use_tb_logger --save_every_n_epochs 10 --max_epochs 10
