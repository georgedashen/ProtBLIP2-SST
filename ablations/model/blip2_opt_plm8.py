"""
Stage-2 PLM variant: compress ESM hidden states [B, L, H_plm] to K LLM prefix tokens
[B, K, H_llm] with a lightweight projection head (no Q-Former).

Pipeline (vs blip2_opt_plm.py full-length prefix):
  - blip2_opt_plm.py:  encode -> plm_proj (per token) -> Galactica prefix (~L tokens)
  - blip2_opt_plm8.py: encode -> plm_proj (per token) -> seq_compress (L->K) -> Galactica prefix (K tokens)

ProtT3 stage2 uses Q-Former(8); this ablation keeps raw ESM embeddings + a small cross-attn compressor.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import disabled_train

from model.blip2 import Blip2Base
from model.blip2_opt_plm import (
    DEFAULT_PLM_MODEL,
    GALACTICA_DEFAULT_MAX_POS,
    Blip2OPT_PLM,
    fit_prot_hidden_to_llm_budget,
)
from model.help_funcs import hf_enable_gradient_checkpointing
from model.hf_local import hf_from_pretrained_kwargs
from model.plm_encode import encode_prot_hidden
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
from transformers import AutoTokenizer, OPTForCausalLM


class PLMSeqToKCompressor(nn.Module):
    """Project a variable-length PLM sequence [B, L, D] to K prefix tokens [B, K, D]."""

    def __init__(self, hidden_dim: int, num_prefix_tokens: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_queries = nn.Parameter(
            torch.empty(1, num_prefix_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.prefix_queries, std=0.02)
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, prot_tokens: torch.Tensor, prot_mask: torch.Tensor) -> torch.Tensor:
        batch_size = prot_tokens.shape[0]
        queries = self.prefix_queries.expand(batch_size, -1, -1)
        key_padding_mask = prot_mask == 0
        compressed, _ = self.cross_attn(
            queries,
            prot_tokens,
            prot_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(compressed)


class Blip2OPT_PLM8(Blip2OPT_PLM):
    """Frozen PLM + per-token plm_proj + K-token compressor + Galactica."""

    def __init__(
        self,
        plm_model: str = DEFAULT_PLM_MODEL,
        plm_tune: str = "freeze",
        llm_name: str = "facebook/galactica-1.3b",
        llm_tune: str = "freeze",
        peft_dir: str = "",
        num_prefix_tokens: int = 8,
        args=None,
    ):
        Blip2Base.__init__(self)
        self.args = args
        self.max_llm_seq_len = int(getattr(args, "max_llm_seq_len", GALACTICA_DEFAULT_MAX_POS))
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing
        self.num_prefix_tokens = num_prefix_tokens

        self.is_saprot = self._is_saprot_plm(plm_model)
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

        llm_dim = self.llm_model.config.hidden_size
        self.plm_proj = nn.Linear(self.plm.num_features, llm_dim)
        attn_heads = min(8, llm_dim)
        while llm_dim % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1
        self.seq_compress = PLMSeqToKCompressor(
            llm_dim, num_prefix_tokens=num_prefix_tokens, num_heads=attn_heads
        )

        self.eos_token_id = self.llm_tokenizer("\n", add_special_tokens=False).input_ids[0]

    @staticmethod
    def _is_saprot_plm(plm_model: str) -> bool:
        from model.plm_encode import is_saprot_plm

        return is_saprot_plm(plm_model)

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

    def _encode_and_project_prot(self, prot_batch):
        prot_embeds, prot_mask = encode_prot_hidden(
            self.plm, self.ln_layer, prot_batch, self.plm_tune, is_saprot=self.is_saprot
        )
        prot_tokens = self.plm_proj(prot_embeds)
        prot_tokens = self.seq_compress(prot_tokens, prot_mask)
        prot_mask = torch.ones(
            prot_tokens.shape[:2], dtype=prot_mask.dtype, device=prot_tokens.device
        )
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
            max_new_tokens = max_length if max_length is not None else 128
        max_new_tokens = min(int(max_new_tokens), self.max_llm_seq_len - prefix_len)
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
