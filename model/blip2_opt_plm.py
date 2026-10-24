"""
Stage-2 variant: feed full PLM hidden states [B, L, H] into Galactica/OPT
instead of compressing through a Q-Former (8 query tokens).

Galactica-1.3B (OPT) limits (from HF config):
  - max_position_embeddings: 2048  (hard context window)
  - hidden_size: 2048

SwissProtV3 + text_max_len=128:
  - typical train seq: 1024 (prot) + ~80 (text) ~ 1104  (< 2048)
  - with text_max_len=128, prot rarely needs extra truncation at 2048 cap
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import disabled_train
from model.blip2 import Blip2Base
from model.help_funcs import hf_enable_gradient_checkpointing
from model.plm_encode import encode_prot_hidden, is_saprot_plm
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
from transformers import AutoTokenizer, OPTForCausalLM, EsmTokenizer, EsmModel, T5Tokenizer, T5EncoderModel
from model.esm_flash_attention import replace_esm_attn_with_flash_attn
from model.t5_flash_attention import replace_t5_attn_with_flash_attn
from model.hf_local import hf_from_pretrained_kwargs

GALACTICA_DEFAULT_MAX_POS = 2048
DEFAULT_PLM_MODEL = "fredzzp/esm2_t12_35M_UR50D"


def fit_prot_hidden_to_llm_budget(
    prot_embeds: torch.Tensor,
    prot_mask: torch.Tensor,
    text_seq_len: int,
    max_llm_seq_len: int = GALACTICA_DEFAULT_MAX_POS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim protein tokens from the right so prot_len + text_len <= max_llm_seq_len."""
    prot_len = prot_embeds.shape[1]
    max_prot = int(max_llm_seq_len) - int(text_seq_len)
    if max_prot < 1:
        raise ValueError(
            f"text sequence length ({text_seq_len}) exceeds max_llm_seq_len ({max_llm_seq_len})"
        )
    if prot_len <= max_prot:
        return prot_embeds, prot_mask
    return prot_embeds[:, :max_prot, :], prot_mask[:, :max_prot]


class Blip2OPT_PLM(Blip2Base):
    """BLIP-2 stage-2 decoder with direct PLM hidden-state prefix (no Q-Former)."""

    def __init__(
        self,
        plm_model: str = DEFAULT_PLM_MODEL,
        plm_tune: str = "freeze",
        llm_name: str = "facebook/galactica-1.3b",
        llm_tune: str = "freeze",
        peft_dir: str = "",
        args=None,
    ):
        super().__init__()
        self.args = args
        self.max_llm_seq_len = int(getattr(args, "max_llm_seq_len", GALACTICA_DEFAULT_MAX_POS))
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing

        self.is_saprot = is_saprot_plm(plm_model)
        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder_plm(plm_model)
        self.plm_tune = plm_tune
        if plm_tune == "freeze":
            for param in self.plm.parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm encoder")
        elif plm_tune == "lora":
            lora_config = DeltaLoraConfig(
                args.lora_r,
                args.lora_alpha,
                args.lora_dropout,
                modified_modules=["query", "value"],
            )
            self.delta = LoraModel.from_config(lora_config, self.plm)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError(f"unsupported plm_tune: {plm_tune}")

        self.llm_model, self.llm_tokenizer = self._load_llm(
            llm_name,
            enable_gradient_checkpointing=self.enbale_gradient_checkpointing,
        )
        self._configure_llm_tune(llm_tune, args)

        self.eos_token_id = self.llm_tokenizer("\n", add_special_tokens=False).input_ids[0]
        self.plm_proj = nn.Linear(self.plm.num_features, self.llm_model.config.hidden_size)

    def init_protein_encoder_plm(self, plm_name: str, load_4bit: bool = False):
        """Load ESM-2 (facebook/*, fredzzp/*) or ProstT5 (Rostlab/*)."""
        enable_flash = bool(getattr(self.args, "enable_flash", False))
        use_triton_flash = bool(getattr(self.args, "use_triton_flash", False))
        plm_lower = (plm_name or "").lower()

        if plm_name.startswith("Rostlab") or ("prost" in plm_lower and "t5" in plm_lower):
            hf_kw = hf_from_pretrained_kwargs(plm_name)
            plm_tokenizer = T5Tokenizer.from_pretrained(plm_name, do_lower_case=False, **hf_kw)
            if enable_flash:
                replace_t5_attn_with_flash_attn(use_triton=use_triton_flash)
            plm = T5EncoderModel.from_pretrained(plm_name, torch_dtype=torch.bfloat16, **hf_kw)
        elif "esm" in plm_lower:
            hf_kw = hf_from_pretrained_kwargs(plm_name)
            plm_tokenizer = EsmTokenizer.from_pretrained(plm_name, trust_remote_code=True, **hf_kw)
            plm = EsmModel.from_pretrained(
                plm_name,
                add_pooling_layer=False,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **hf_kw,
            )
            replace_esm_attn_with_flash_attn()
        elif is_saprot_plm(plm_name):
            saprot_root = str(Path.home())
            if saprot_root not in sys.path:
                sys.path.insert(0, saprot_root)
            from SaProt.model.saprot.base import SaprotBaseModel

            plm_tokenizer = EsmTokenizer.from_pretrained(
                plm_name, trust_remote_code=True, **hf_from_pretrained_kwargs(plm_name)
            )
            plm = SaprotBaseModel(task="base", config_path=plm_name, load_pretrained=True)
            plm = plm.to(torch.bfloat16)
            replace_esm_attn_with_flash_attn()
            if "650M" in plm_name:
                plm.num_features = 1280
            elif "35M" in plm_name:
                plm.num_features = 480
            else:
                plm.num_features = plm.model.esm.config.hidden_size
            ln_layer = nn.LayerNorm(plm.num_features)
            return plm_tokenizer, plm, ln_layer
        else:
            return super().init_protein_encoder(plm_name, load_4bit=load_4bit)

        hidden_dim = getattr(plm.config, "hidden_size", None) or plm.config.d_model
        plm.num_features = hidden_dim
        ln_layer = nn.LayerNorm(hidden_dim)
        return plm_tokenizer, plm, ln_layer

    def _load_llm(self, llm_model: str, enable_gradient_checkpointing: bool = True):
        hf_kw = hf_from_pretrained_kwargs(llm_model)
        llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, use_fast=False, padding_side="right", **hf_kw
        )
        llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16, **hf_kw)
        llm_model.resize_token_embeddings(len(llm_tokenizer))
        if enable_gradient_checkpointing:
            llm_model = hf_enable_gradient_checkpointing(llm_model)
        return llm_model, llm_tokenizer

    def _configure_llm_tune(self, llm_tune: str, args):
        if llm_tune == "freeze":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        elif llm_tune == "full":
            for param in self.llm_model.parameters():
                param.requires_grad = True
        elif llm_tune == "lora":
            lora_config = DeltaLoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout)
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif llm_tune == "mid_lora":
            lora_config = DeltaLoraConfig(
                args.lora_r,
                args.lora_alpha,
                args.lora_dropout,
                modified_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            )
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError(f"unsupported llm_tune: {llm_tune}")

    def _encode_and_project_prot(self, prot_batch):
        prot_embeds, prot_mask = encode_prot_hidden(
            self.plm, self.ln_layer, prot_batch, self.plm_tune, is_saprot=self.is_saprot
        )
        prot_tokens = self.plm_proj(prot_embeds)
        return prot_tokens, prot_mask

    def forward(self, batch):
        prot_batch, text_batch = batch
        prot_tokens, prot_mask = self._encode_and_project_prot(prot_batch)
        device = prot_tokens.device

        text_seq_len = text_batch.input_ids.shape[1]
        prot_tokens, prot_mask = fit_prot_hidden_to_llm_budget(
            prot_tokens, prot_mask, text_seq_len, self.max_llm_seq_len
        )

        prot_empty_targets = torch.full(
            prot_tokens.shape[:2], -100, dtype=torch.long, device=device
        )
        targets = text_batch.input_ids.masked_fill(
            text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill(text_batch.token_type_ids == 0, -100)
        targets = torch.cat([prot_empty_targets, targets], dim=1)

        text_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, text_embeds), dim=1)
        attention_mask = torch.cat([prot_mask, text_batch.attention_mask], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample: bool = False,
        num_beams: int = 5,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_captions: int = 1,
        temperature: float = 1.0,
        max_time: Optional[float] = None,
    ):
        prot_batch = samples["prot_batch"]
        prompt_batch = samples["prompt_batch"]

        prot_tokens, prot_mask = self._encode_and_project_prot(prot_batch)
        text_seq_len = prompt_batch.input_ids.shape[1]
        prot_tokens, prot_mask = fit_prot_hidden_to_llm_budget(
            prot_tokens, prot_mask, text_seq_len, self.max_llm_seq_len
        )

        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, prompt_embeds), dim=1)
        attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask], dim=1)

        prefix_len = inputs_embeds.shape[1]
        if max_new_tokens is None:
            # Back-compat: treat max_length as *new* tokens when prefix is long.
            max_new_tokens = max_length if max_length is not None else 128
        max_new_tokens = min(
            int(max_new_tokens),
            self.max_llm_seq_len - prefix_len,
        )
        if max_new_tokens < 1:
            raise ValueError(
                f"prefix length {prefix_len} leaves no room in max_llm_seq_len={self.max_llm_seq_len}"
            )

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            use_cache=True,
        )
        if max_time is not None and max_time > 0:
            gen_kwargs["max_time"] = float(max_time)
        outputs = self.llm_model.generate(**gen_kwargs)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in output_text]
