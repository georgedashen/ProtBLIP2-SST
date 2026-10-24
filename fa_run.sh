python stage1.py --mode train --filename stage1_650mfa --num_query_token 8 --plm_name "fredzzp/esm2_t33_650M_UR50D" --strategy deepspeed --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --seed 42 --match_batch_size 64 --use_tb_logger --max_epochs 50 --suffix _afdbfilt &
python stage1.py --mode train --filename stage1_650mfa --num_query_token 8 --plm_name "fredzzp/esm2_t33_650M_UR50D" --strategy deepspeed --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --num_workers 8 --seed 42 --match_batch_size 64 --use_tb_logger --max_epochs 50 &

wait
echo "All training processes have completed."