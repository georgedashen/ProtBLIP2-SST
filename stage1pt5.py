import os
import argparse
import math
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from model.hf_local import configure_hf_offline_if_cached
from model.blip2_stage1_t5 import Blip2Stage1
from data_provider.stage1_dmpt5 import Stage1DM, Stage1MixDM
from model.dist_funs import MyDeepSpeedStrategy
from pathlib import Path


RESUME_HPARAM_KEYS = (
    "max_epochs", "scheduler", "init_lr", "min_lr", "warmup_lr",
    "warmup_steps", "weight_decay", "batch_size", "plm_name", "bert_name",
)


def _read_deepspeed_ckpt_meta(ckpt_path: str) -> dict:
    ckpt_dir = Path(ckpt_path)
    if ckpt_dir.is_dir():
        state_path = ckpt_dir / "checkpoint" / "mp_rank_00_model_states.pt"
    else:
        state_path = ckpt_dir
    if not state_path.exists():
        raise FileNotFoundError(f"Cannot find DeepSpeed checkpoint state at {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    return {
        "epoch": state.get("epoch"),
        "global_step": state.get("global_step"),
        "hyper_parameters": state.get("hyper_parameters") or {},
    }


def _expected_cosine_lr(epoch: int, max_epochs: int, init_lr: float, min_lr: float) -> float:
    return (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epochs)) + min_lr


def _validate_resume_hparams(args, ckpt_hparams: dict) -> None:
    mismatches = []
    for key in RESUME_HPARAM_KEYS:
        if key not in ckpt_hparams:
            continue
        ckpt_val = ckpt_hparams[key]
        arg_val = getattr(args, key, None)
        if ckpt_val != arg_val:
            mismatches.append(f"  {key}: checkpoint={ckpt_val!r}, args={arg_val!r}")
    if mismatches:
        print("WARNING: resume hyperparameter mismatch (optimizer/LR schedule may be wrong):")
        print("\n".join(mismatches))


def _print_resume_info(args, ckpt_path: str) -> None:
    meta = _read_deepspeed_ckpt_meta(ckpt_path)
    epoch = meta["epoch"]
    global_step = meta["global_step"]
    next_epoch = None if epoch is None else epoch + 1
    print(f"Resuming from checkpoint: {ckpt_path}")
    print(f"  saved at end of epoch {epoch} (0-indexed), global_step={global_step}")
    print(f"  training will continue from epoch {next_epoch}")
    if args.scheduler == "linear_warmup_cosine_lr" and next_epoch is not None:
        lr = _expected_cosine_lr(next_epoch, args.max_epochs, args.init_lr, args.min_lr)
        print(f"  expected LR at first step of epoch {next_epoch}: {lr:.6e}")
    elif args.scheduler == "linear_warmup_step_lr" and next_epoch is not None:
        lr = max(args.min_lr, args.init_lr * (args.lr_decay_rate ** next_epoch))
        print(f"  expected LR at first step of epoch {next_epoch}: {lr:.6e}")
    _validate_resume_hparams(args, meta["hyper_parameters"])


os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)




def main(args):
    configure_hf_offline_if_cached(args.plm_name, args.bert_name)
    pl.seed_everything(args.seed)

    if args.ckpt_path and args.init_checkpoint:
        raise ValueError("Use either --ckpt_path (full resume) or --init_checkpoint (weights-only restart), not both.")

    if args.ckpt_path:
        _print_resume_info(args, args.ckpt_path)

    # model
    if args.init_checkpoint:
        print(f"loading model weights only from {args.init_checkpoint} (restarts at epoch 0)")
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, strict=False)
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    if args.mix_dataset:
        dm = Stage1MixDM(args.num_workers, args.batch_size, args.root, args)
        dm.init_tokenizer(model.blip2qformer.tokenizer, model.blip2qformer.plm_tokenizer)
        dm.prepare_plm_embed_cache(model.blip2qformer.plm, is_saprot=False)
        model.swiss_val_match_loader, model.swiss_test_match_loader = dm.swiss_match_dataloader()
        model.onto_val_match_loader, model.onto_test_match_loader = dm.onto_match_dataloader()
    else:
        dm = Stage1DM(args.num_workers, args.batch_size, args.root, args)
        dm.init_tokenizer(model.blip2qformer.tokenizer, model.blip2qformer.plm_tokenizer)
        dm.prepare_plm_embed_cache(model.blip2qformer.plm, is_saprot=False)
        model.val_match_loader, model.test_match_loader = dm.match_dataloader()

    if args.test:
        args.log_every_n_steps = 10
    
    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1,
                                         save_last=True,
                                         save_on_train_epoch_end=True))
    
    
    if len(args.devices.split(',')) > 1:
        if args.strategy == 'ddp':
            # find_unused_parameters = (not args.ptm) or (not args.lm)
            # strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
            strategy = 'ddp'
            pass
        elif args.strategy == 'deepspeed':
            # strategy = strategies.DeepSpeedStrategy(stage=2)
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            raise NotImplementedError()
        # strategy = strategies.FSDPStrategy()
    else:
        os.environ['MASTER_ADDR'] = 'localhost'  # Address of the master node (can be any node)
        os.environ['MASTER_PORT'] = args.port  # Port for communication
        os.environ['WORLD_SIZE'] = '1'  # Total number of processes
        os.environ['RANK'] = '0'  # Rank of the current process (0, 1, 2, or 3 in this example)
        torch.distributed.init_process_group('nccl',init_method='env://')
        args.devices = [0]
        strategy = None
        #strategy = strategies.SingleDeviceStrategy(device='cuda:0')
    if args.use_wandb_logger:
        Path(f'./all_checkpoints/{args.filename}/wandb').mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=f'./all_checkpoints/{args.filename}/')
    elif args.use_tb_logger:
        Path(f'./all_checkpoints/{args.filename}/tb_logs').mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(save_dir=f'./all_checkpoints/{args.filename}/tb_logs')
        # !tensorboard --logdir ./all_checkpoints/<your_filename>/tb_logs
    else:
        logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    
    trainer_args = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        # limit_val_batches=2,
    )
    if strategy is not None:
        trainer_args['strategy'] = strategy

    trainer = Trainer(**trainer_args)

    if args.mode == 'train':
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path or None)
    elif args.mode == 'eval':
        model.force_retrieval_eval = True
        trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt_path or None)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage1_pt5")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy', type=str, default='deepspeed')
    parser.add_argument('--port', type=str, default='12358')
    parser.add_argument('--test', type=int, default=0, help='for debugging, use a smaller dataset')
    parser.add_argument('--suffix', type=str, default='', help='custom dataset')
    
    ## trainer arguments
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--use_wandb_logger', action='store_true', default=False)
    parser.add_argument('--use_tb_logger', action='store_true', default=False)
    parser.add_argument('--mix_dataset', action='store_true', default=False)
    parser.add_argument(
        '--ckpt_path', type=str, default='',
        help='Full training resume (weights, optimizer, epoch, global_step). '
             'Use DeepSpeed checkpoint dir, e.g. all_checkpoints/stage1_pt5_afdbfilt/epoch=39.ckpt',
    )
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

