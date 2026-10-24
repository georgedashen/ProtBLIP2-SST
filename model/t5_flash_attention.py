import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.t5.modeling_t5 import T5Attention

try:
    from flash_attn.flash_attn_triton import flash_attn_func as triton_flash_attn_func
    _TRITON_FLASH_AVAILABLE = True
except Exception:
    triton_flash_attn_func = None
    _TRITON_FLASH_AVAILABLE = False


_forward_original = T5Attention.forward
_USE_TRITON_FLASH = False


def _compute_position_bias(self, query_states, key_states, mask, position_bias, query_length, cache_position):
    batch_size, seq_length = query_states.shape[0], query_states.shape[2]
    if position_bias is not None:
        return position_bias

    key_length = key_states.shape[-2]
    if cache_position is not None:
        real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
    else:
        real_seq_length = query_length if query_length is not None else seq_length

    if not self.has_relative_attention_bias:
        position_bias = torch.zeros(
            (1, self.n_heads, seq_length, key_length),
            device=query_states.device,
            dtype=query_states.dtype,
        )
        if self.gradient_checkpointing and self.training:
            position_bias.requires_grad = True
    else:
        position_bias = self.compute_bias(
            real_seq_length,
            key_length,
            device=query_states.device,
            cache_position=cache_position,
        )
        position_bias = position_bias[:, :, -seq_length:, :]

    if mask is not None:
        causal_mask = mask[:, :, :, : key_states.shape[-2]]
        position_bias = position_bias + causal_mask
    return position_bias


def _apply_head_mask(query_states, key_states, value_states, position_bias, pruned_heads):
    if not pruned_heads:
        return query_states, key_states, value_states, position_bias

    head_mask = torch.ones(position_bias.shape[1], device=position_bias.device)
    head_mask[list(pruned_heads)] = 0
    return (
        query_states[:, head_mask.bool()],
        key_states[:, head_mask.bool()],
        value_states[:, head_mask.bool()],
        position_bias[:, head_mask.bool()],
    )


def _attn_sdpa(query_states, key_states, value_states, position_bias_masked, dropout_p):
    return F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=position_bias_masked.to(query_states.dtype),
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0,
    )


def _attn_triton(query_states, key_states, value_states, position_bias_masked):
    # flash_attn_triton expects [batch, seqlen, heads, dim] and does not support dropout.
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    bias = position_bias_masked.to(q.dtype)
    out = triton_flash_attn_func(q, k, v, bias, False, 1.0)
    return out.transpose(1, 2).contiguous()


def _run_attention(self, query_states, key_states, value_states, position_bias_masked, seq_length):
    dropout_p = self.dropout.p if self.training else 0.0
    use_triton = _USE_TRITON_FLASH and dropout_p == 0.0

    if use_triton:
        try:
            attn_output = _attn_triton(query_states, key_states, value_states, position_bias_masked)
        except Exception as exc:
            logging.warning("Triton flash attention failed (%s); falling back to SDPA.", exc)
            attn_output = _attn_sdpa(query_states, key_states, value_states, position_bias_masked, dropout_p)
    else:
        attn_output = _attn_sdpa(query_states, key_states, value_states, position_bias_masked, dropout_p)

    if self.pruned_heads:
        head_mask = torch.ones(self.n_heads, device=attn_output.device, dtype=torch.bool)
        head_mask[list(self.pruned_heads)] = False
        attn_output_full = torch.zeros(
            (query_states.shape[0], self.n_heads, seq_length, self.key_value_proj_dim),
            device=attn_output.device,
            dtype=attn_output.dtype,
        )
        attn_output_full[:, head_mask] = attn_output
        attn_output = attn_output_full

    return attn_output


def forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """Encoder self-attention for T5 with Triton flash (bias) or SDPA fallback."""
    is_cross_attention = key_value_states is not None
    if (
        is_cross_attention
        or past_key_value is not None
        or output_attentions
        or layer_head_mask is not None
        or self.is_decoder
        or use_cache
    ):
        return _forward_original(
            self,
            hidden_states,
            mask=mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            layer_head_mask=layer_head_mask,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

    batch_size, seq_length = hidden_states.shape[:2]

    query_states = self.q(hidden_states)
    query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    key_states = self.k(hidden_states)
    value_states = self.v(hidden_states)
    key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    position_bias = _compute_position_bias(
        self, query_states, key_states, mask, position_bias, query_length, cache_position
    )
    query_states, key_states, value_states, position_bias_masked = _apply_head_mask(
        query_states, key_states, value_states, position_bias, self.pruned_heads
    )

    attn_output = _run_attention(
        self, query_states, key_states, value_states, position_bias_masked, seq_length
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, -1, self.inner_dim)
    attn_output = self.o(attn_output)

    return (attn_output, past_key_value, position_bias)


def replace_t5_attn_with_flash_attn(use_triton=False):
    """Patch T5 encoder attention. SDPA is default (exact); Triton is faster but ~0.1/layer drift."""
    global _USE_TRITON_FLASH
    cuda_major, _ = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    _USE_TRITON_FLASH = use_triton and _TRITON_FLASH_AVAILABLE
    if use_triton and not _TRITON_FLASH_AVAILABLE:
        logging.warning("flash_attn_triton is unavailable; using SDPA fallback for T5 attention.")
    elif _USE_TRITON_FLASH:
        logging.warning(
            "Triton flash+bias enabled for T5: faster but not bit-exact vs HuggingFace T5 attention. "
            "Use only for experiments, not production training."
        )
    else:
        logging.info("Using SDPA for T5 encoder attention (exact match with original).")

    T5Attention.forward = forward


def replace_flash_attn_with_t5_attn():
    global _USE_TRITON_FLASH
    _USE_TRITON_FLASH = False
    T5Attention.forward = _forward_original


if __name__ == '__main__':
    import time

    from transformers import T5EncoderModel, T5Tokenizer

    def bench(model, inp, label, n=20):
        for _ in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model(**inp)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(**inp)
        torch.cuda.synchronize()
        print(f'{label}: {(time.time() - t0) / n:.6f} sec')
        return out

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    tok = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    text = ['<AA2fold> M K T A V L I'] * 4
    inp = tok(text, return_tensors='pt', padding='max_length', max_length=1024, return_attention_mask=True)
    inp = {k: v.cuda() for k, v in inp.items()}

    model_orig = T5EncoderModel.from_pretrained('Rostlab/ProstT5', torch_dtype=torch.bfloat16).cuda().eval()
    out_orig = bench(model_orig, inp, 'Original T5 attention')

    replace_t5_attn_with_flash_attn(use_triton=False)
    model_sdpa = T5EncoderModel.from_pretrained('Rostlab/ProstT5', torch_dtype=torch.bfloat16).cuda().eval()
    out_sdpa = bench(model_sdpa, inp, 'SDPA fallback')

    replace_t5_attn_with_flash_attn(use_triton=True)
    model_triton = T5EncoderModel.from_pretrained('Rostlab/ProstT5', torch_dtype=torch.bfloat16).cuda().eval()
    out_triton = bench(model_triton, inp, 'Triton flash + bias')

    print(f'SDPA diff vs original: {(out_orig.last_hidden_state.float() - out_sdpa.last_hidden_state.float()).abs().mean().item():.6f}')
    print(f'Triton diff vs original: {(out_orig.last_hidden_state.float() - out_triton.last_hidden_state.float()).abs().mean().item():.6f}')
