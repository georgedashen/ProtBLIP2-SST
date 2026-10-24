"""Shared PLM encoding helpers (live forward or cached hidden states)."""
from data_provider.plm_embed_cache import CachedProtBatch, is_cached_prot_batch


def is_saprot_plm(plm_name: str) -> bool:
    name = (plm_name or "").lower()
    return "saprot" in name or name.endswith("_af2")


def batch_size_of(prot_batch) -> int:
    if is_cached_prot_batch(prot_batch):
        return prot_batch.batch_size
    return prot_batch["input_ids"].shape[0]


def encode_prot_hidden(plm, ln_layer, prot_batch, plm_tune: str, is_saprot: bool = False):
    """Return ln_layer(PLM hidden states) for token batch or CachedProtBatch."""
    if is_cached_prot_batch(prot_batch):
        prot_embeds = prot_batch.hidden_states
        target = ln_layer.weight.device
        if prot_embeds.device != target:
            prot_embeds = prot_embeds.to(target, non_blocking=True)
        if prot_embeds.dtype != ln_layer.weight.dtype:
            prot_embeds = prot_embeds.to(dtype=ln_layer.weight.dtype)
        if plm_tune == "freeze":
            prot_embeds = prot_embeds.detach()
        return ln_layer(prot_embeds), prot_batch.attention_mask.to(target, non_blocking=True)

    if is_saprot:
        prot_embeds = plm.get_embed(prot_batch)
    else:
        prot_embeds = plm(**prot_batch, return_dict=True).last_hidden_state
    if plm_tune == "freeze":
        prot_embeds = prot_embeds.detach()
    prot_embeds = ln_layer(prot_embeds)
    return prot_embeds, prot_batch["attention_mask"]
