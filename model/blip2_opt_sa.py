"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import disabled_train
from model.blip2_sa import Blip2Base
from transformers import AutoTokenizer
from transformers import OPTForCausalLM
from transformers import BitsAndBytesConfig
from model.help_funcs import hf_enable_gradient_checkpointing
from model.hf_local import hf_from_pretrained_kwargs
from model.plm_encode import encode_prot_hidden

opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

def get_gpu_memory(device=0):
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total

def mask_by_len(input, lens, fill_value=0):
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


class Blip2OPT(Blip2Base):
    def __init__(
        self,
        bert_name,
        num_query_token=32,
        cross_attention_freq=2,
        plm_model="facebook/esm2_t30_150M_UR50D",
        plm_tune='freeze',
        llm_name="facebook/galactica-1.3b",
        llm_tune='freeze',
        peft_dir='',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.enbale_gradient_checkpointing = args.enbale_gradient_checkpointing

        self.plm_tokenizer, self.plm, self.ln_layer = self.init_protein_encoder(plm_model)
        self.plm_tune = plm_tune
        if plm_tune == 'freeze':
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
            self.plm = self.plm.eval()
            self.plm.train = disabled_train
            logging.info("freeze plm encoder")
        elif plm_tune == 'lora':
            from opendelta import LoraModel
            from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
            lora_config = DeltaLoraConfig(args.lora_r, 
                                          args.lora_alpha, 
                                          args.lora_dropout,
                                          modified_modules=["query", "value"])
            self.delta = LoraModel.from_config(lora_config, self.plm)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError()
        
        self.num_query_token = num_query_token
        _, self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.plm.num_features, cross_attention_freq)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.llm_model, self.llm_tokenizer = self.load_llm(llm_name, load_4bit=False, enable_gradient_checkpointing=self.enbale_gradient_checkpointing)

        if llm_tune == 'freeze':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True
        elif llm_tune == 'lora':
            from opendelta import LoraModel
            from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
            lora_config = DeltaLoraConfig(args.lora_r, 
                                          args.lora_alpha, 
                                          args.lora_dropout,)
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        elif llm_tune == 'mid_lora':
            from opendelta import LoraModel
            from opendelta.delta_models.lora import LoraConfig as DeltaLoraConfig
            lora_config = DeltaLoraConfig(args.lora_r, args.lora_alpha, args.lora_dropout, modified_modules=["q_proj", "v_proj", 'k_proj', "out_proj", "fc1", "fc2"])
            self.delta = LoraModel.from_config(lora_config, self.llm_model)
            self.delta.freeze_module(set_state_dict=False)
            self.delta.log()
        else:
            raise NotImplementedError()

        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

    def _encode_protein(self, prot_batch):
        prot_embeds, prot_mask = encode_prot_hidden(
            self.plm, self.ln_layer, prot_batch, self.plm_tune, is_saprot=True
        )
        return prot_embeds, prot_mask

    def load_llm(self, llm_model, load_4bit=False, enable_gradient_checkpointing=True):
        hf_kw = hf_from_pretrained_kwargs(llm_model)
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right', **hf_kw)
        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            outputs = get_gpu_memory(0)
            used_memory = outputs[1]
            if used_memory > 1:
                device_map = {"": 1}
            else:
                device_map = {"": 0}
            llm_model = OPTForCausalLM.from_pretrained(
                llm_model, 
                quantization_config=quant_config,
                load_in_4bit=True,
                load_in_8bit=False,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                **hf_kw,
            )
            llm_model.resize_token_embeddings(len(llm_tokenizer))
        else:
            llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16, **hf_kw)
            llm_model.resize_token_embeddings(len(llm_tokenizer))
            if enable_gradient_checkpointing:
                llm_model = hf_enable_gradient_checkpointing(llm_model)
        return llm_model, llm_tokenizer

    def forward(self, batch):
        prot_batch, text_batch = batch
        prot_embeds, prot_mask = self._encode_protein(prot_batch)
        device = prot_embeds.device
        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=text_batch.attention_mask.dtype, device=device)
        prot_empty_targets = torch.ones(prot_tokens.shape[:2], dtype=torch.long, device=device).fill_(-100)
        
        targets = text_batch.input_ids.masked_fill(text_batch.input_ids == self.llm_tokenizer.pad_token_id, -100)
        targets = targets.masked_fill(text_batch.token_type_ids == 0, -100)
        targets = torch.cat([prot_empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, inputs_embeds), dim=1)
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
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        prot_batch = samples['prot_batch']
        prompt_batch = samples['prompt_batch']
        
        prot_embeds, prot_mask = self._encode_protein(prot_batch)

        query_tokens = self.query_tokens.expand(prot_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=prot_embeds,
            encoder_attention_mask=prot_mask,
            return_dict=True,
        )
        prot_tokens = self.opt_proj(query_output.last_hidden_state)
        
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_batch.input_ids)
        inputs_embeds = torch.cat((prot_tokens, prompt_embeds), dim=1)
        prot_mask = torch.ones(prot_tokens.shape[:2], dtype=prompt_batch.attention_mask.dtype, device=prompt_embeds.device)
        attention_mask = torch.cat([prot_mask, prompt_batch.attention_mask], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            use_cache=True,
        )
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [text.strip() for text in output_text]
