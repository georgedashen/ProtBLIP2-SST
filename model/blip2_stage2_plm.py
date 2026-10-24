"""Lightning wrapper for stage-2 PLM-direct captioning (no Q-Former)."""
import os
import json
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from typing import Any, Dict

from model.blip2_opt_plm import Blip2OPT_PLM, DEFAULT_PLM_MODEL, GALACTICA_DEFAULT_MAX_POS
from model.blip2_stage2 import (
    _resolve_log_dir,
    evaluate_exact_match,
)
from model.help_funcs import caption_evaluate, flatten_caption_field, AttrDict
from model.caption_gen_utils import generate_captions_safe
try:
    from model.opt_flash_attention import (
        replace_opt_attn_with_flash_attn,
        replace_opt_attn_with_original_attn,
    )
except ModuleNotFoundError:
    pass


class Blip2Stage2PLM(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        to_be_removed = []
        for key in checkpoint["state_dict"]:
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint["state_dict"].pop(key)

    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_inference_len = args.max_inference_len
        self.min_inference_len = args.min_inference_len
        self.llm_tune = args.llm_tune
        self.enable_flash = args.enable_flash
        self.gen_max_time_sec = float(getattr(args, "gen_max_time_sec", 0.0))

        if args.llm_name.find("galactica") < 0:
            raise NotImplementedError("Blip2Stage2PLM currently supports Galactica/OPT only")
        self.blip2 = self._build_blip2(args)
        self.save_hyperparameters(args)

    def _build_blip2(self, args):
        return Blip2OPT_PLM(
            args.plm_model,
            args.plm_tune,
            args.llm_name,
            args.llm_tune,
            args.peft_dir,
            args,
        )

    def load_plm_from_stage1_checkpoint(self, path: str):
        """Optionally load frozen PLM + LayerNorm weights from a stage-1 checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        plm_sd, ln_sd = {}, {}
        for key, value in state_dict.items():
            if "blip2qformer.plm." in key:
                plm_sd[key.split("blip2qformer.plm.", 1)[1]] = value
            elif "blip2qformer.ln_layer." in key:
                ln_sd[key.split("blip2qformer.ln_layer.", 1)[1]] = value
            elif key.startswith("blip2.plm."):
                plm_sd[key.split("blip2.plm.", 1)[1]] = value
            elif key.startswith("blip2.ln_layer."):
                ln_sd[key.split("blip2.ln_layer.", 1)[1]] = value
        if plm_sd:
            self.blip2.plm.load_state_dict(plm_sd, strict=False)
        if ln_sd:
            self.blip2.ln_layer.load_state_dict(ln_sd, strict=False)
        return self

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == "linear_warmup_cosine_lr":
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                self.args.max_epochs,
                self.args.min_lr,
                self.args.init_lr,
                warmup_steps,
                self.args.warmup_lr,
            )
        elif self.args.scheduler == "linear_warmup_step_lr":
            self.scheduler = LinearWarmupStepLRScheduler(
                optimizer,
                self.args.max_epochs,
                self.args.min_lr,
                self.args.init_lr,
                self.args.lr_decay_rate,
                self.args.warmup_lr,
                warmup_steps,
            )
        elif self.args.scheduler == "None":
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def save_predictions(self, predictions, targets, q_types=None, log_prefix=""):
        assert len(predictions) == len(targets)
        name = f"{log_prefix}_predictions.txt" if log_prefix else "predictions.txt"
        log_dir = _resolve_log_dir(self.trainer, self.logger)
        with open(os.path.join(log_dir, name), "w", encoding="utf8") as f:
            if q_types is not None:
                for p, t, q in zip(predictions, targets, q_types):
                    f.write(json.dumps({"prediction": p, "target": t, "q_type": q}, ensure_ascii=True) + "\n")
            else:
                for p, t in zip(predictions, targets):
                    f.write(json.dumps({"prediction": p, "target": t}, ensure_ascii=True) + "\n")

    def on_validation_epoch_start(self) -> None:
        if self.enable_flash:
            replace_opt_attn_with_original_attn()
        self.saved_dict_list = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if (dataloader_idx % 2) == 0:
            text_batch = batch[-1]
            batch_size = text_batch.input_ids.shape[0]
            loss = self.blip2(batch)
            self.log(f"dataloader{dataloader_idx}/val loss", float(loss), batch_size=batch_size, sync_dist=True)
        elif (dataloader_idx % 2) == 1:
            if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
                return
            prot_batch, prompt_batch, target_dict = batch
            predictions = generate_captions_safe(
                self.blip2,
                prot_batch,
                prompt_batch,
                target_dict["targets"],
                target_dict.get("indices", list(range(len(target_dict["targets"])))),
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_new_tokens=self.max_inference_len,
                min_length=self.min_inference_len,
                gen_max_time_sec=self.gen_max_time_sec,
                rank=self.global_rank,
            )
            target_dict["predictions"] = predictions
            self.saved_dict_list.append(target_dict)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def gather_dict_results(self, dict_list):
        if dist.is_available() and dist.is_initialized():
            list_of_dict_list = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(list_of_dict_list, dict_list)
            dict_list = [i for ii in list_of_dict_list for i in ii]
        else:
            dict_list = list(dict_list)
        keys = dict_list[0].keys()
        gathered_dict = {key: [i for d in dict_list for i in d[key]] for key in keys}
        dict_list = []
        for i in range(len(gathered_dict["predictions"])):
            dict_list.append({k: gathered_dict[k][i] for k in keys})
        return dict_list

    def save_results(self, dict_list, log_prefix=""):
        name = f"{log_prefix}_predictions.txt" if log_prefix else "predictions.txt"
        log_dir = _resolve_log_dir(self.trainer, self.logger)
        with open(os.path.join(log_dir, name), "w", encoding="utf8") as f:
            for d in dict_list:
                f.write(json.dumps(d, ensure_ascii=True) + "\n")

    def on_validation_epoch_end(self):
        if self.enable_flash:
            replace_opt_attn_with_flash_attn()
        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return
        if not self.saved_dict_list:
            return
        result_list = self.gather_dict_results(self.saved_dict_list)
        self.saved_dict_list = []
        if self.global_rank == 0:
            self.save_results(result_list, "dataset0")
            all_predictions = flatten_caption_field(result_list, "predictions")
            all_targets = flatten_caption_field(result_list, "targets")
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = caption_evaluate(
                all_predictions, all_targets, self.blip2.llm_tokenizer, self.max_inference_len
            )
            acc = evaluate_exact_match(all_predictions, all_targets)
            log_prefix = "dataset0"
            self.log(f"{log_prefix}/acc", acc, sync_dist=False)
            self.log(f"{log_prefix}/bleu2", bleu2, sync_dist=False)
            self.log(f"{log_prefix}/bleu4", bleu4, sync_dist=False)
            self.log(f"{log_prefix}/rouge_1", rouge_1, sync_dist=False)
            self.log(f"{log_prefix}/rouge_2", rouge_2, sync_dist=False)
            self.log(f"{log_prefix}/rouge_l", rouge_l, sync_dist=False)
            self.log(f"{log_prefix}/meteor_score", meteor_score, sync_dist=False)

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = batch[-1].input_ids.size(0)
        loss = self.blip2(batch)
        self.log("loss", float(loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=batch_size, sync_dist=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ProtBlip2PLM")
        parser.add_argument("--save_every_n_epochs", type=int, default=0)
        parser.add_argument("--llm_name", type=str, default="facebook/galactica-1.3b")
        parser.add_argument("--num_beams", type=int, default=5)
        parser.add_argument("--do_sample", action="store_true", default=False)
        parser.add_argument(
            "--max_inference_len",
            type=int,
            default=512,
            help="max NEW tokens to generate at inference (prefix is ~1024 prot + prompt)",
        )
        parser.add_argument("--min_inference_len", type=int, default=1)
        parser.add_argument("--llm_tune", type=str, default="mid_lora")
        parser.add_argument("--peft_config", type=str, default="")
        parser.add_argument("--peft_dir", type=str, default="")
        parser.add_argument(
            "--max_llm_seq_len",
            type=int,
            default=GALACTICA_DEFAULT_MAX_POS,
            help="Galactica context window; prot prefix is trimmed if prot+text exceeds this",
        )
        parser.add_argument(
            "--plm_model",
            type=str,
            default=DEFAULT_PLM_MODEL,
            help="protein LM, e.g. fredzzp/esm2_t12_35M_UR50D or Rostlab/ProstT5",
        )
        parser.add_argument(
            "--plm_name",
            type=str,
            default="",
            help="alias for --plm_model (stage1-style, e.g. Rostlab/ProstT5)",
        )
        parser.add_argument("--plm_tune", type=str, default="freeze")
        parser.add_argument("--enable_flash", action="store_true", default=False)
        parser.add_argument("--disable_flash", action="store_false", dest="enable_flash")
        parser.add_argument(
            "--use_triton_flash",
            action="store_true",
            default=False,
            help="Triton flash attention for ProstT5 (requires patched flash_attn_triton)",
        )
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=16)
        parser.add_argument("--lora_dropout", type=float, default=0.1)
        parser.add_argument("--enbale_gradient_checkpointing", action="store_true", default=False)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--min_lr", type=float, default=1e-5)
        parser.add_argument("--warmup_lr", type=float, default=1e-6)
        parser.add_argument("--warmup_steps", type=int, default=1000)
        parser.add_argument("--lr_decay_rate", type=float, default=0.9)
        parser.add_argument("--scheduler", type=str, default="linear_warmup_cosine_lr")
        parser.add_argument(
            "--stage1_path",
            type=str,
            default="",
            help="optional: load frozen PLM+LayerNorm from stage-1 checkpoint (Q-Former weights ignored)",
        )
        parser.add_argument("--init_checkpoint", type=str, default="")
        parser.add_argument("--caption_eval_epoch", type=int, default=10)
        parser.add_argument(
            "--gen_max_time_sec",
            type=float,
            default=600.0,
            help="Per-sample generation time limit (seconds). Uses HF max_time; partial "
            "output is kept, empty string if nothing generated. Set 0 to disable. "
            "Prevents NCCL timeout when one sample hangs in beam search.",
        )
        return parent_parser
