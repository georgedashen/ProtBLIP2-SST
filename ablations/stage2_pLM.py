#!/usr/bin/env python3
"""
Stage-2 training with direct PLM hidden states as Galactica prefix (no Q-Former).

Compared with stage2.py:
  - stage2.py:  protein -> Q-Former (8 queries) -> opt_proj -> Galactica prefix (~8 tokens)
  - stage2_pLM.py: protein -> plm_proj -> Galactica prefix (up to prot_max_len tokens)

Galactica-1.3B limits:
  - max_position_embeddings = 2048 (total input positions per forward/generate step)
  - Default text_max_len=128 keeps train sequences around 1024+80 ~ 1104 (< 2048)
  - AA sequences: p99 ~ 938, only 1 train sample > 1024 aa -> prot_max_len=1024 is sufficient
  - Inference: prefix ~1029 tokens; use --max_inference_len as NEW tokens (default 512)

PLM examples (default: fredzzp/esm2_t12_35M_UR50D):
  # ESM2 35M
  python stage2_pLM.py --devices 0,1 --plm_model fredzzp/esm2_t12_35M_UR50D ...

  # ProstT5 (stage1pt5-style --plm_name)
  python stage2_pLM.py --devices 0,1 --plm_name Rostlab/ProstT5 --suffix _afdbfilt \\
    --enable_flash --filename stage2_plm_pt5_afdbfilt ...
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
from data_provider.stage2_dm import Stage2DM, Stage2MixDM
from data_provider.prot_qa_dm import ProtQADM
from model.blip2_stage2_plm import Blip2Stage2PLM
from model.dist_funs import MyDeepSpeedStrategy
from model.plm_encode import is_saprot_plm
from model.hf_local import configure_hf_offline_if_cached
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
torch.set_float32_matmul_precision("medium")


def resolve_plm_args(args):
    """Unify --plm_name (stage1) and --plm_model (stage2) for cache paths and model init."""
    if getattr(args, "plm_name", ""):
        args.plm_model = args.plm_name
    args.plm_name = args.plm_model
    return args


def main(args):
    pl.seed_everything(args.seed)

    if args.init_checkpoint:
        model = Blip2Stage2PLM.load_from_checkpoint(
            args.init_checkpoint, strict=False, args=args, map_location="cpu"
        )
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    else:
        model = Blip2Stage2PLM(args)
        if args.stage1_path:
            model.load_plm_from_stage1_checkpoint(args.stage1_path)
            print(f"loaded PLM weights from stage1 checkpoint {args.stage1_path}")
    print("total params:", sum(p.numel() for p in model.parameters()))
    print(
        "trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    if args.mix_dataset:
        dm = Stage2MixDM(args.root, args)
    else:
        if args.root.find("PDBDataset") >= 0:
            dm = ProtQADM(args.root, args)
        else:
            dm = Stage2DM(args.root, args)
    dm.init_tokenizer(model.blip2.llm_tokenizer, model.blip2.plm_tokenizer)
    if hasattr(dm, "prepare_plm_embed_cache"):
        dm.prepare_plm_embed_cache(
            model.blip2.plm, is_saprot=is_saprot_plm(args.plm_model)
        )

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
    parser = argparse.ArgumentParser(description="Stage-2 PLM-direct captioning")
    parser.add_argument("--filename", type=str, default="stage2_plm")
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
    parser.add_argument("--mix_dataset", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="", help="dataset suffix, e.g. _afdbfilt")
    parser = Blip2Stage2PLM.add_model_specific_args(parser)
    parser = Stage2DM.add_model_specific_args(parser)
    args = parser.parse_args()
    args = resolve_plm_args(args)
    configure_hf_offline_if_cached(args.plm_model, args.llm_name)

    if args.enable_flash:
        replace_opt_attn_with_flash_attn()

    print("=========================================")
    print("Stage-2 PLM-direct (no Q-Former)")
    print(f"  PLM: {args.plm_model}")
    print(f"  Galactica max positions: {args.max_llm_seq_len}")
    print(f"  prot_max_len (PLM tokens): {args.prot_max_len}")
    print(f"  text_max_len (caption tokens): {args.text_max_len}")
    print(f"  max_inference_len (new tokens): {args.max_inference_len}")
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    print("=========================================")
    return args


if __name__ == "__main__":
    main(get_args())
