"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn

from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer, BitsAndBytesConfig
from transformers import EsmTokenizer, EsmModel
from transformers import T5Tokenizer, T5EncoderModel
from model.esm_flash_attention import replace_esm_attn_with_flash_attn
from model.hf_local import hf_from_pretrained_kwargs
import sys
sys.path.insert(0, '/home/zhuoyang')
from SaProt.model.saprot.base import SaprotBaseModel    

def get_gpu_memory(device=0):
    # t = torch.cuda.get_device_properties(device).total_memory
    # r = torch.cuda.memory_reserved(device)
    # a = torch.cuda.memory_allocated(device)
    # f = r-a  # free inside reserved
    free, total = torch.cuda.mem_get_info(device)
    free = free / (1024 ** 3)
    total = total / (1024 ** 3)
    return free, total-free, total


class Blip2Base(BaseModel):
    # @classmethod
    # def init_tokenizer(cls):
    #     tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
    #     tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    #     return tokenizer

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, plm_width, cross_attention_freq=2):
        pass
        #assert model_name == 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
        #print("bert load microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        
        encoder_config = BertConfig.from_pretrained(model_name, **hf_from_pretrained_kwargs(model_name))
        encoder_config.encoder_width = plm_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(model_name, config=encoder_config, **hf_from_pretrained_kwargs(model_name))
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        tokenizer = BertTokenizer.from_pretrained(model_name, **hf_from_pretrained_kwargs(model_name))
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer, Qformer, query_tokens
    

    def init_protein_encoder(self, plm_name, load_4bit=False):
        #assert plm_name.startswith('facebook')
        hf_kw = hf_from_pretrained_kwargs(plm_name)
        if 'esm' in plm_name:
            plm_tokenizer = EsmTokenizer.from_pretrained(plm_name, trust_remote_code=True, **hf_kw)
        elif 't5' in plm_name:
            plm_tokenizer = T5Tokenizer.from_pretrained(plm_name, do_lower_case=False, trust_remote_code=True, **hf_kw)
        else:
            plm_tokenizer = EsmTokenizer.from_pretrained(plm_name, trust_remote_code=True, **hf_kw)
        if not load_4bit:
            if 'esm' in plm_name:
                plm = EsmModel.from_pretrained(plm_name, add_pooling_layer=False, torch_dtype=torch.bfloat16, trust_remote_code=True, **hf_kw)
                replace_esm_attn_with_flash_attn()
            elif 't5' in plm_name:
                plm = T5EncoderModel.from_pretrained(plm_name, torch_dtype=torch.bfloat16, trust_remote_code=True, **hf_kw)
            else:
                plm = SaprotBaseModel(task="base",config_path=plm_name,load_pretrained=True)
                replace_esm_attn_with_flash_attn()
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            ## give a device map that assign all layers to device 0
            outputs = get_gpu_memory(0)
            used_memory = outputs[1]
            if used_memory > 1:
                device_map = {"": 1}
            else:
                device_map = {"": 0}
            plm = EsmModel.from_pretrained(
                plm_name, 
                add_pooling_layer=False,
                quantization_config=quant_config,
                load_in_4bit=True,
                load_in_8bit=False,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                **hf_kw,
            )
            replace_esm_attn_with_flash_attn()

        if '650M' in plm_name:
            plm.num_features = 1280 #480 for 35M, 1280 for 650M
        elif '35M' in plm_name:
            plm.num_features = 480
        else:
            plm.num_features = plm.config.hidden_size
        
        ln_layer = nn.LayerNorm(plm.num_features)
        return plm_tokenizer, plm, ln_layer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

