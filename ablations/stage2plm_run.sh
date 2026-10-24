export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python ablations/stage2_pLM.py --devices 0,1 --mode train --strategy deepspeed \
  --filename stage2_plm_esm2_35m_afdbfilt \
  --plm_model fredzzp/esm2_t12_35M_UR50D --suffix _afdbfilt \
  --root data/SwissProtV3 --batch_size 4 --llm_tune mid_lora --enable_flash \
  --use_tb_logger --save_every_n_epochs 10