#!/usr/bin/env python3
"""
Stage-2 training/eval with Galactica [START_AMINO]/[END_AMINO] AA prefix (no PLM, no Q-Former).

Compared with stage2_AAraw.py (space-separated letters):
  - stage2_AAraw_presuffix.py: [START_AMINO]seq[END_AMINO] + escape_custom_split_sequence

Example (finetune):
  bash stage2aaraw_presuffix_train.sh

Example (zero-shot Galactica eval):
  bash stage2aaraw_presuffix_eval_galactica_afdbfilt.sh

Example (finetuned checkpoint eval):
  bash stage2aaraw_presuffix_eval_finetuned_afdbfilt.sh
"""
import os
try:
    from model.opt_flash_attention import replace_opt_attn_with_flash_attn
except ModuleNotFoundError:
    pass
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from data_provider.stage2_aaraw_presuffix_dm import Stage2AArawPresuffixDM
from model.blip2_stage2_aaraw_presuffix import Blip2Stage2AArawPresuffix
from model.dist_funs import MyDeepSpeedStrategy
from model.hf_local import configure_hf_offline_if_cached
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
torch.set_float32_matmul_precision("medium")


def main(args):
    pl.seed_everything(args.seed)

    if args.init_checkpoint:
        model = Blip2Stage2AArawPresuffix.load_from_checkpoint(
            args.init_checkpoint, strict=False, args=args, map_location="cpu"
        )
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    else:
        model = Blip2Stage2AArawPresuffix(args)
    print("total params:", sum(p.numel() for p in model.parameters()))
    print(
        "trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    dm = Stage2AArawPresuffixDM(args.root, args)
    dm.init_tokenizer(model.blip2.llm_tokenizer)

    log_dir = Path(f"all_checkpoints/{args.filename}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints and logs -> {log_dir}")

    callbacks = [
        plc.ModelCheckpoint(
            dirpath=str(log_dir),
            filename="{epoch:02d}",
            every_n_epochs=args.save_every_n_epochs,
            save_last=True,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        )
    ]

    if len(args.devices.split(",")) > 1:
        if args.strategy == "ddp":
            strategy = strategies.DDPStrategy(start_method="spawn", find_unused_parameters=False)
        elif args.strategy == "deepspeed":
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            raise NotImplementedError()
    else:
        strategy = None
        if "," in args.devices:
            args.devices = [int(d) for d in args.devices.split(",")]
        else:
            args.devices = [int(args.devices)]

    if args.use_wandb_logger:
        (log_dir / "wandb").mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(project=args.filename, save_dir=str(log_dir))
    elif args.use_tb_logger:
        tb_dir = log_dir / "tb_logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        logger = [CSVLogger(save_dir=str(log_dir)), TensorBoardLogger(save_dir=str(tb_dir))]
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
        trainer_kwargs["strategy"] = strategy
    trainer = Trainer(**trainer_kwargs)

    if args.mode == "train":
        trainer.fit(model, datamodule=dm)
    elif args.mode == "eval":
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


def get_args():
    parser = argparse.ArgumentParser(description="Stage-2 Galactica AA presuffix captioning")
    parser.add_argument("--filename", type=str, default="stage2_aaraw_presuffix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--strategy", type=str, default="deepspeed")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--use_wandb_logger", action="store_true", default=False)
    parser.add_argument("--use_tb_logger", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="", help="dataset suffix, e.g. _afdbfilt")
    parser = Blip2Stage2AArawPresuffix.add_model_specific_args(parser)
    parser = Stage2AArawPresuffixDM.add_model_specific_args(parser)
    args = parser.parse_args()
    configure_hf_offline_if_cached(args.llm_name)

    if args.enable_flash:
        replace_opt_attn_with_flash_attn()

    print("=========================================")
    print("Stage-2 Galactica [START_AMINO]/[END_AMINO] AA prefix")
    print(f"  LLM: {args.llm_name}")
    print(f"  Galactica max positions: {args.max_llm_seq_len}")
    print(f"  prot_max_len (AA tokens): {args.prot_max_len}")
    print(f"  text_max_len (caption tokens): {args.text_max_len}")
    print(f"  max_inference_len (new tokens): {args.max_inference_len}")
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
