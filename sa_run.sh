python stage1sa.py --strategy deepspeed --devices '0,1' --mode train --filename stage1_sa --num_query_token 8 --num_workers 8 --plm_name "../SaProt/weights/PLMs/SaProt_650M_AF2" --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --match_batch_size 64 --use_tb_logger --max_epochs 50 &

python stage1sa.py --strategy deepspeed --devices '0,1' --mode train --filename stage1_sa --num_query_token 8 --num_workers 8 --plm_name "../SaProt/weights/PLMs/SaProt_35M_AF2"  --save_every_n_epochs 10 --batch_size 32 --precision 'bf16-mixed' --match_batch_size 64 --use_tb_logger --max_epochs 50 &

# python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train --filename stage1_pt5 --num_query_token 8 --num_workers 8 --plm_name "Rostlab/ProstT5" --save_every_n_epoch 10 --batch_size 32 --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 --use_tb_logger > pt5.log &

# python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train --filename stage1_pt5 --num_query_token 8 --num_workers 8 --plm_name "Rostlab/ProstT5" --save_every_n_epoch 10 --batch_size 32 --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 --use_tb_logger  --suffix _afdbfilt > pt5filt.log &

# python stage1pt5.py --strategy deepspeed --devices '0,1' --mode train --filename stage1_pt5 --num_query_token 8 --num_workers 8 --plm_name "Rostlab/ProstT5" --save_every_n_epoch 10 --batch_size 32 --precision 'bf16-mixed' --match_batch_size 64 --max_epochs 50 --use_tb_logger  --suffix _saprot_3Dimask > pt53Di.log &

wait

echo "All training processes have completed."
