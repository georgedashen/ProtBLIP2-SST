#!/usr/bin/env bash
# Finetune on SwissProtV3 _afdbfilt, then evaluate the finetuned checkpoint on test.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash stage2aaraw_presuffix_train.sh
bash stage2aaraw_presuffix_eval_finetuned_afdbfilt.sh
