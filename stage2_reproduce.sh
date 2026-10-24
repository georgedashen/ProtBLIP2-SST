# this is Esm2_150M
# python stage2.py \
#     --devices '0,1' \
#     --mode eval \
#     --strategy deepspeed \
#     --filename protein_captioning_swiss_dataset_eval \
#     --caption_eval_epoch 1 \
#     --num_query_token 8  \
#     --save_every_n_epochs 10 \
#     --batch_size 32 \
#     --precision 'bf16-mixed' \
#     --num_workers 8 \
#     --llm_tune mid_lora \
#     --enable_flash \
#     --root 'data/SwissProtV3' \
#     --init_checkpoint 'all_checkpoints/swiss_ft.ckpt'

# Use --suffix _afdbfilt to match stage1 (fa_run.sh) and SaProt test order (9239 samples).
# Without suffix, stage2 uses test_set.json (10000) and indices do not align with _saprot.

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python stage2.py \
    --devices '0,1' \
    --mode train \
    --strategy deepspeed \
    --filename prott3_650m_afdbfilt_protein_captioning \
    --plm_model "fredzzp/esm2_t33_650M_UR50D" \
    --num_query_token 8  \
    --max_epochs 10 \
    --save_every_n_epochs 10 \
    --batch_size 32 \
    --precision 'bf16-mixed' \
    --num_workers 8 \
    --use_tb_logger \
    --llm_tune mid_lora \
    --enable_flash \
    --root 'data/SwissProtV3' \
    --suffix _afdbfilt \
    --stage1_path 'all_checkpoints/stage1_650mfa/ProtT3_650M_afdbfilt_fp32.ckpt'

python stage2.py \
    --devices '0,1' \
    --mode train \
    --strategy deepspeed \
    --filename prott3_35m_afdbfilt_protein_captioning \
    --plm_model "fredzzp/esm2_t12_35M_UR50D" \
    --num_query_token 8  \
    --max_epochs 10 \
    --save_every_n_epochs 10 \
    --batch_size 32 \
    --precision 'bf16-mixed' \
    --num_workers 8 \
    --use_tb_logger \
    --llm_tune mid_lora \
    --enable_flash \
    --root 'data/SwissProtV3' \
    --suffix _afdbfilt \
    --stage1_path 'all_checkpoints/stage1_35mfa/ProtT3_35M_afdbfilt_fp32.ckpt'