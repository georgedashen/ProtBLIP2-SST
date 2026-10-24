import os
try:
    from model.opt_flash_attention import replace_opt_attn_with_flash_attn
except (ModuleNotFoundError, ImportError):
    def replace_opt_attn_with_flash_attn():
        print("Warning: flash attention for OPT is unavailable; continuing without it.")
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from data_provider.stage2_dm import Stage2DM, Stage2MixDM
from data_provider.prot_qa_dm import ProtQADM
from model.blip2_stage2_sa import Blip2Stage2
from model.dist_funs import MyDeepSpeedStrategy
from model.hf_local import configure_hf_offline_if_cached
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
torch.set_float32_matmul_precision('medium')


def main(args):
    pl.seed_everything(args.seed)
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args, map_location='cpu')
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)
    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.mix_dataset:
        dm = Stage2MixDM(args.root, args)
    else:
        if args.root.find('PDBDataset') >= 0:
            dm = ProtQADM(args.root, args)
        else:
            dm = Stage2DM(args.root, args)
    dm.init_tokenizer(model.blip2.llm_tokenizer, model.blip2.plm_tokenizer)
    if hasattr(dm, 'prepare_plm_embed_cache'):
        dm.prepare_plm_embed_cache(model.blip2.plm, is_saprot=True)

    if args.test:
        args.caption_eval_epoch = 1
        args.log_every_n_steps = 10
        limit_train_batches = min(5, max(1, 100 // max(args.batch_size, 1)))
        limit_val_batches = 2
        num_sanity_val_steps = 0
    else:
        limit_train_batches = None
        limit_val_batches = None
        num_sanity_val_steps = 2

    log_dir = Path(f'all_checkpoints/{args.filename}')
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints and logs -> {log_dir}")

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath=str(log_dir),
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_last=True, 
                                         save_top_k=-1,
                                         save_on_train_epoch_end=True))
    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            find_unused_parameters = (not args.ptm) or (not args.lm)
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
        elif args.strategy == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            raise NotImplementedError()
    else:
        strategy = None
        if ',' in args.devices:
            args.devices = [int(d) for d in args.devices.split(',')]
        else:
            args.devices = [int(args.devices)]
    if args.use_wandb_logger:
        (log_dir / 'wandb').mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=str(log_dir))
    elif args.use_tb_logger:
        tb_dir = log_dir / 'tb_logs'
        tb_dir.mkdir(parents=True, exist_ok=True)
        # CSV + TB: metrics.csv under lightning_logs/ (same as stage2.py layout when using CSV only)
        logger = [
            CSVLogger(save_dir=str(log_dir)),
            TensorBoardLogger(save_dir=str(tb_dir)),
        ]
        # tensorboard --logdir all_checkpoints/<filename>/tb_logs
    else:
        logger = CSVLogger(save_dir=str(log_dir))
    trainer_kwargs = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
    )
    if strategy is not None:
        trainer_kwargs['strategy'] = strategy
    trainer_kwargs['num_sanity_val_steps'] = num_sanity_val_steps
    if limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = limit_train_batches
        trainer_kwargs['limit_val_batches'] = limit_val_batches
    trainer = Trainer(**trainer_kwargs)
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_sa_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy', type=str, default='deepspeed')
    parser.add_argument('--test', type=int, default=0, help='for debugging, use a smaller dataset')
    parser.add_argument('--suffix', type=str, default='_saprot', help='custom dataset suffix')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--enable_flash', action='store_true', default=False)
    parser.add_argument('--disable_flash', action='store_true', default=False)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser.add_argument('--use_tb_logger', action='store_true', default=False)
    parser.add_argument('--mix_dataset', action='store_true', default=False)
    parser = Blip2Stage2.add_model_specific_args(parser)
    parser = Stage2DM.add_model_specific_args(parser)
    args = parser.parse_args()
    configure_hf_offline_if_cached(args.plm_model, args.llm_name, args.bert_name)

    args.enable_flash = args.enable_flash or not args.disable_flash
    if args.enable_flash:
        replace_opt_attn_with_flash_attn()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


if __name__ == '__main__':
    main(get_args())
