#!/bin/bash
# SaProt stage2 protein captioning
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Checkpoints/logs -> all_checkpoints/<filename>/ (same layout as stage2.py)
# With --use_tb_logger: metrics.csv + predictions under lightning_logs/, TB events under tb_logs/
# python stage2sa.py \
#   --strategy deepspeed \
#   --devices '0,1' \
#   --stage1_path all_checkpoints/stage1_sa/SaProt_650M_fp32.ckpt \
#   --max_epochs 10 \
#   --batch_size 32 \
#   --precision 'bf16-mixed' \
#   --llm_tune mid_lora \
#   --num_workers 8 \
#   --use_tb_logger \
#   --enable_flash \
#   --filename stage2_sa_650m_protein_captioning \
#   --save_every_n_epochs 10 \
#   --num_query_token 8  \
#   --plm_model "../SaProt/weights/PLMs/SaProt_650M_AF2" \
#   --llm_name "facebook/galactica-1.3b" \
#   --root data/SwissProtV3 \
#   --suffix _saprot

python stage2sa.py \
  --strategy deepspeed \
  --devices '0,1' \
  --stage1_path all_checkpoints/stage1_sa/SaProt_35M_fp32.ckpt \
  --max_epochs 10 \
  --batch_size 32 \
  --precision 'bf16-mixed' \
  --llm_tune mid_lora \
  --num_workers 8 \
  --use_tb_logger \
  --enable_flash \
  --filename stage2_sa_35m_protein_captioning \
  --save_every_n_epochs 10 \
  --num_query_token 8  \
  --plm_model "../SaProt/weights/PLMs/SaProt_35M_AF2" \
  --llm_name "facebook/galactica-1.3b" \
  --root data/SwissProtV3 \
  --suffix _saprot