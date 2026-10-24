"""Safe per-sample caption generation with optional time limits."""
import logging
from typing import Any, List, Sequence

from data_provider.plm_embed_cache import CachedProtBatch, is_cached_prot_batch
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


def slice_prot_batch(prot_batch: Any, index: int):
    if is_cached_prot_batch(prot_batch):
        return CachedProtBatch(
            prot_batch.hidden_states[index : index + 1],
            prot_batch.attention_mask[index : index + 1],
        )
    return {key: value[index : index + 1] for key, value in prot_batch.items()}


def slice_prompt_batch(prompt_batch: Any, index: int):
    sliced = prompt_batch[index : index + 1]
    if isinstance(sliced, BatchEncoding):
        return sliced
    if isinstance(sliced, dict):
        return BatchEncoding(sliced, tensor_type="pt")
    return sliced


def generate_captions_safe(
    blip2,
    prot_batch,
    prompt_batch,
    targets: Sequence[str],
    indices: Sequence[int],
    *,
    do_sample: bool,
    num_beams: int,
    max_new_tokens: int,
    min_length: int,
    gen_max_time_sec: float = 0.0,
    rank: int = 0,
) -> List[str]:
    """Generate one caption per sample; on timeout/error use partial output or empty string."""
    predictions: List[str] = []
    batch_size = len(targets)
    for i in range(batch_size):
        sample_idx = indices[i] if i < len(indices) else i
        try:
            samples = {
                "prot_batch": slice_prot_batch(prot_batch, i),
                "prompt_batch": slice_prompt_batch(prompt_batch, i),
            }
            gen_kwargs = {
                "do_sample": do_sample,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
            }
            if gen_max_time_sec > 0:
                gen_kwargs["max_time"] = float(gen_max_time_sec)
            out = blip2.generate(samples, **gen_kwargs)
            pred = out[0] if out else ""
        except Exception as exc:
            logger.warning(
                "Caption generation failed (rank=%s dataset_idx=%s): %s",
                rank,
                sample_idx,
                exc,
            )
            pred = ""
        if not pred or not str(pred).strip():
            logger.warning(
                "Caption generation empty (rank=%s dataset_idx=%s, max_time=%ss)",
                rank,
                sample_idx,
                gen_max_time_sec,
            )
            pred = ""
        predictions.append(pred)
    return predictions
