"""
Stage-2 variant: feed amino-acid sequences with Galactica [START_AMINO]/[END_AMINO] markers.

Uses escape_custom_split_sequence for one token per residue inside the markers.
Prefix embeddings come from the LLM embedding table (no PLM).
"""
from typing import Optional

import torch
from model.blip2 import Blip2Base
from model.blip2_opt_plm import GALACTICA_DEFAULT_MAX_POS, fit_prot_hidden_to_llm_budget
from model.help_funcs import hf_enable_gradient_checkpointing
from opendelta import LoraModel
from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
from model.hf_local import hf_from_pretrained_kwargs
from transformers import AutoTokenizer, OPTForCausalLM


class Blip2OPT_AArawPresuffix(Blip2Base):
    """BLIP-2 stage-2 decoder with Galactica AA presuffix prefix tokens."""

    def __init__(
        self,
        llm_name: str = "facebook/galactica-1.3b",
        llm_tune: str = "freeze",
        peft_dir: str = "",
        args=None,
    ):
        super().__init__()
        self.args = args
        self.max_llm_seq_len = int(getattr(args, "max_llm_seq_len", GALACTICA_DEFAULT_MAX_POS))
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing

        self.llm_model, self.llm_tokenizer = self._load_llm(
            llm_name,
            enable_gradient_checkpointing=self.enbale_gradient_checkpointing,
        )
        self._configure_llm_tune(llm_tune, args)
        self.eos_token_id = self.llm_tokenizer("\n", add_special_tokens=False).input_ids[0]

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

    def _encode_prot(self, prot_batch):
        input_ids = prot_batch["input_ids"]
        attention_mask = prot_batch["attention_mask"]
        prot_embeds = self.llm_model.get_input_embeddings()(input_ids)
        return prot_embeds, attention_mask

    def forward(self, batch):
        prot_batch, text_batch = batch
        prot_tokens, prot_mask = self._encode_prot(prot_batch)
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
    ):
        prot_batch = samples["prot_batch"]
        prompt_batch = samples["prompt_batch"]

        prot_tokens, prot_mask = self._encode_prot(prot_batch)
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
        max_new_tokens = min(
            int(max_new_tokens),
            self.max_llm_seq_len - prefix_len,
        )
        if max_new_tokens < 1:
            raise ValueError(
                f"prefix length {prefix_len} leaves no room in max_llm_seq_len={self.max_llm_seq_len}"
            )

        outputs = self.llm_model.generate(
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
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in output_text]
