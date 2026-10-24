#!/usr/bin/env bash
# Re-run stage1 retrieval eval on a finished checkpoint.
# Use --disable_cached_plm_embeds for immediate correct eval (slow), or build
# *_match.pt caches first when using cached embeddings.
set -euo pipefail
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_PATH="${CKPT_PATH:-all_checkpoints/stage1_pt5_afdbfilt/epoch=49.ckpt}"
DISABLE_CACHE="${DISABLE_CACHE:-1}"  # 1 = live PLM forward (correct for match eval)

EXTRA=()
if [[ "${DISABLE_CACHE}" == "1" ]]; then
  EXTRA+=(--disable_cached_plm_embeds)
fi

python stage1pt5.py --strategy deepspeed --devices '0,1' --mode eval \
  --filename stage1_pt5_afdbfilt --num_query_token 8 --num_workers 2 \
  --plm_name "Rostlab/ProstT5" --save_every_n_epochs 10 --batch_size 32 \
  --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 \
  --use_tb_logger --suffix _afdbfilt \
  --ckpt_path "${CKPT_PATH}" \
  "${EXTRA[@]}" \
  > pt5_afdbfilt_retrieval_eval.log 2>&1

echo "Done. See pt5_afdbfilt_retrieval_eval.log and tb_logs for metrics."
