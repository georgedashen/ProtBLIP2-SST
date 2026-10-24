"""Disk cache for frozen PLM hidden states (before ln_layer) to skip repeated encoder forwards."""
from __future__ import annotations

import logging
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_VERSION = 1
STALE_BUILD_LOCK_SECS = 3600.0
DEFAULT_PLM_EMBED_CACHE_DIR = "/data2/zhuoyang/ProtT3/data/plm_embed_cache"
STORAGE_MMAP = "mmap"
STORAGE_INLINE = "inline"
MMAP_WRITE_CHUNK = 256


@dataclass
class CachedProtBatch:
    """Precomputed PLM hidden states; ln_layer is applied in the model forward path."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor

    def to(self, device):
        self.hidden_states = self.hidden_states.to(device, non_blocking=True)
        self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        return self

    @property
    def batch_size(self) -> int:
        return int(self.hidden_states.shape[0])


def is_cached_prot_batch(batch) -> bool:
    return isinstance(batch, CachedProtBatch)


def add_plm_cache_args(parser):
    parser.set_defaults(use_cached_plm_embeds=True)
    parser.add_argument(
        "--disable_cached_plm_embeds",
        dest="use_cached_plm_embeds",
        action="store_false",
        help="Run PLM forward each step instead of loading precomputed embeddings.",
    )
    parser.add_argument(
        "--plm_embed_cache_dir",
        type=str,
        default=DEFAULT_PLM_EMBED_CACHE_DIR,
        help="Root directory for per-PLM / per-split embedding caches.",
    )
    parser.add_argument(
        "--plm_embed_cache_batch_size",
        type=int,
        default=16,
        help="Batch size per cache-build worker (lower when using multiple workers per GPU).",
    )
    parser.add_argument(
        "--plm_embed_cache_gpus",
        type=str,
        default="auto",
        help="Comma-separated GPU ids for parallel cache build (default: auto = all visible CUDA devices). "
        "Use a single id, e.g. '0', to force single-GPU build.",
    )
    parser.add_argument(
        "--plm_embed_cache_workers_per_gpu",
        type=int,
        default=1,
        help="Number of parallel cache-build processes per GPU (total workers = len(gpus) * N). "
        "Increase when GPU memory allows multiple PLM replicas on one card.",
    )
    return parser


def _slugify(name: str) -> str:
    name = name.replace("\\", "/").strip("/")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:180] or "plm"


def _cache_path(
    cache_dir: Path,
    plm_name: str,
    prot_max_len: int,
    suffix: str,
    dataset_path: Union[str, Path],
) -> Path:
    path_str = str(dataset_path)
    match_tag = ""
    if path_str.endswith(":match"):
        path_str = path_str[: -len(":match")]
        match_tag = "_match"
    ds_name = Path(path_str).name
    stem = Path(ds_name).stem + match_tag
    sub = _slugify(plm_name)
    suffix_part = _slugify(suffix) if suffix else "nosuffix"
    return cache_dir / sub / f"len{prot_max_len}" / suffix_part / f"{stem}.pt"


def preprocess_prot_seqs(prot_seqs: Sequence[str], plm_name: str) -> List[str]:
    """Match collater preprocessing for T5 / ProstT5 protein inputs."""
    plm_name = plm_name or ""
    plm_lower = plm_name.lower()
    if "t5" in plm_lower:
        prot_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in prot_seqs]
    if "prost" in plm_lower:
        prot_seqs = [
            "<AA2fold> " + seq if seq.isupper() else "<fold2AA> " + seq
            for seq in prot_seqs
        ]
    return list(prot_seqs)


def _tokenize_prot_seqs(
    prot_seqs: Sequence[str],
    prot_tokenizer,
    prot_max_len: int,
    plm_name: str,
) -> dict:
    prot_seqs = preprocess_prot_seqs(prot_seqs, plm_name)
    return prot_tokenizer(
        list(prot_seqs),
        truncation=True,
        padding="max_length",
        max_length=prot_max_len,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )


@torch.no_grad()
def _run_plm(plm, prot_batch: dict, is_saprot: bool) -> torch.Tensor:
    if is_saprot:
        return plm.get_embed(prot_batch)
    outputs = plm(**prot_batch, return_dict=True)
    return outputs.last_hidden_state


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _distributed_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _resolve_cache_gpus(args) -> List[int]:
    spec = getattr(args, "plm_embed_cache_gpus", "auto")
    if spec is None or str(spec).strip().lower() in {"", "auto", "all"}:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    ids = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids


def _resolve_workers_per_gpu(args) -> int:
    n = int(getattr(args, "plm_embed_cache_workers_per_gpu", 1) or 1)
    return max(1, n)


def _expand_worker_gpus(gpu_ids: List[int], workers_per_gpu: int) -> List[int]:
    """Flatten GPU list to one entry per worker, e.g. [0,1] x2 -> [0,0,1,1]."""
    worker_gpus: List[int] = []
    for gpu_id in gpu_ids:
        worker_gpus.extend([gpu_id] * workers_per_gpu)
    return worker_gpus


def _extract_prot_seq(item) -> str:
    if isinstance(item, (list, tuple)):
        return item[0]
    raise TypeError(f"unexpected dataset item type: {type(item)}")


def _collect_prot_seqs(dataset: Dataset, n: int) -> List[str]:
    indices = range(n)
    if n > 5000:
        indices = tqdm(indices, desc="collect prot seqs", leave=False)
    return [_extract_prot_seq(dataset[i]) for i in indices]


def _create_plm_and_tokenizer(plm_name: str, is_saprot: bool, device: torch.device):
    """Load a fresh PLM + tokenizer on the given device (for cache-build workers)."""
    from model.hf_local import hf_from_pretrained_kwargs

    plm_lower = (plm_name or "").lower()
    hf_kw = hf_from_pretrained_kwargs(plm_name)
    if is_saprot:
        from transformers import EsmTokenizer
        from SaProt.model.saprot.base import SaprotBaseModel
        from model.esm_flash_attention import replace_esm_attn_with_flash_attn

        replace_esm_attn_with_flash_attn()
        tok = EsmTokenizer.from_pretrained(plm_name, trust_remote_code=True, **hf_kw)
        plm = SaprotBaseModel(task="base", config_path=plm_name, load_pretrained=True)
        plm = plm.to(torch.bfloat16)
    elif "esm" in plm_lower:
        from transformers import EsmModel, EsmTokenizer
        from model.esm_flash_attention import replace_esm_attn_with_flash_attn

        tok = EsmTokenizer.from_pretrained(plm_name, trust_remote_code=True, **hf_kw)
        plm = EsmModel.from_pretrained(
            plm_name,
            add_pooling_layer=False,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **hf_kw,
        )
        replace_esm_attn_with_flash_attn()
    elif "t5" in plm_lower or plm_name.startswith("Rostlab"):
        from transformers import T5EncoderModel, T5Tokenizer

        if "prost" in plm_lower:
            from model.t5_flash_attention import replace_t5_attn_with_flash_attn

            replace_t5_attn_with_flash_attn(use_triton=False)
        tok = T5Tokenizer.from_pretrained(plm_name, do_lower_case=False, **hf_kw)
        plm = T5EncoderModel.from_pretrained(plm_name, torch_dtype=torch.bfloat16, **hf_kw)
        if "prost" not in plm_lower:
            plm = plm.half()
    else:
        raise NotImplementedError(f"unsupported PLM for cache build: {plm_name}")

    plm = plm.to(device).eval()
    return tok, plm


def _mmap_stem_paths(stem_path: Path) -> Tuple[Path, Path]:
    return stem_path.with_suffix(stem_path.suffix + ".hidden.mmap"), stem_path.with_suffix(
        stem_path.suffix + ".mask.mmap"
    )


def _cleanup_mmap_files(stem_path: Path):
    for path in _mmap_stem_paths(stem_path):
        path.unlink(missing_ok=True)


def _is_mmap_manifest(payload: dict) -> bool:
    return payload.get("storage") == STORAGE_MMAP or (
        bool(payload.get("hidden_mmap")) and bool(payload.get("mask_mmap"))
    )


def _mmap_byte_sizes(n: int, seq_len: int, hidden_dim: int) -> Tuple[int, int]:
    return n * seq_len * hidden_dim * 2, n * seq_len * 8


def _mmap_files_ready(h_path: Path, m_path: Path, n: int, seq_len: int, hidden_dim: int) -> bool:
    hidden_bytes, mask_bytes = _mmap_byte_sizes(n, seq_len, hidden_dim)
    return (
        h_path.is_file()
        and m_path.is_file()
        and h_path.stat().st_size == hidden_bytes
        and m_path.stat().st_size == mask_bytes
    )


def _write_tensors_to_mmap(
    h_path: Path,
    m_path: Path,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_samples: int = MMAP_WRITE_CHUNK,
):
    n = int(hidden_states.shape[0])
    seq_len = int(hidden_states.shape[1])
    hidden_dim = int(hidden_states.shape[2])
    hidden_mm = np.memmap(h_path, dtype=np.float16, mode="w+", shape=(n, seq_len, hidden_dim))
    mask_mm = np.memmap(m_path, dtype=np.int64, mode="w+", shape=(n, seq_len))
    for start in range(0, n, chunk_samples):
        end = min(n, start + chunk_samples)
        hidden_mm[start:end] = hidden_states[start:end].cpu().numpy()
        mask_mm[start:end] = attention_mask[start:end].cpu().numpy()
    hidden_mm.flush()
    mask_mm.flush()
    del hidden_mm, mask_mm


def _open_mmap_arrays(
    h_path: Path,
    m_path: Path,
    n: int,
    seq_len: int,
    hidden_dim: int,
) -> Tuple[np.memmap, np.memmap]:
    hidden_mm = np.memmap(
        h_path, dtype=np.float16, mode="r", shape=(n, seq_len, hidden_dim)
    )
    mask_mm = np.memmap(m_path, dtype=np.int64, mode="r", shape=(n, seq_len))
    return hidden_mm, mask_mm


def _run_cache_batches(
    prot_seqs: Sequence[str],
    plm,
    prot_tokenizer,
    plm_name: str,
    prot_max_len: int,
    is_saprot: bool,
    batch_size: int,
    device: torch.device,
    desc: str = "cache",
    spill_stem: Optional[Path] = None,
) -> tuple:
    """Encode protein sequences and return CPU tensors for cache storage.

    When ``spill_stem`` is set, embeddings are written through numpy memmap files
    so peak host RAM stays near one encoder batch instead of the full shard
    (critical for ProstT5 @ prot_max_len=1024, where a full shard is ~200GB).
    """
    n = len(prot_seqs)
    hidden_buf: List[torch.Tensor] = []
    mask_buf: List[torch.Tensor] = []
    hidden_mm = None
    mask_mm = None

    try:
        for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
            batch_seqs = prot_seqs[start : start + batch_size]
            prot_batch = _tokenize_prot_seqs(batch_seqs, prot_tokenizer, prot_max_len, plm_name)
            prot_batch = {k: v.to(device) for k, v in prot_batch.items()}
            hidden = _run_plm(plm, prot_batch, is_saprot).detach().cpu().half()
            mask = prot_batch["attention_mask"].detach().cpu()
            end = start + hidden.shape[0]

            if spill_stem is not None:
                if hidden_mm is None:
                    seq_len = int(hidden.shape[1])
                    hidden_dim = int(hidden.shape[2])
                    h_path, m_path = _mmap_stem_paths(spill_stem)
                    hidden_mm = np.memmap(
                        h_path, dtype=np.float16, mode="w+", shape=(n, seq_len, hidden_dim)
                    )
                    mask_mm = np.memmap(m_path, dtype=np.int64, mode="w+", shape=(n, seq_len))
                hidden_mm[start:end] = hidden.numpy()
                mask_mm[start:end] = mask.numpy()
            else:
                hidden_buf.append(hidden)
                mask_buf.append(mask)

        if hidden_mm is not None:
            hidden_mm.flush()
            mask_mm.flush()
            # Zero-copy views; do not np.asarray() the memmap or it copies the full shard into RAM.
            return torch.from_numpy(hidden_mm), torch.from_numpy(mask_mm)
        return torch.cat(hidden_buf, dim=0), torch.cat(mask_buf, dim=0)
    finally:
        pass


def _parallel_cache_worker(
    gpu_id: int,
    shard_rank: int,
    prot_seqs: List[str],
    plm_name: str,
    prot_max_len: int,
    is_saprot: bool,
    batch_size: int,
    shard_path: str,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    tok, plm = _create_plm_and_tokenizer(plm_name, is_saprot, device)
    shard = Path(shard_path)
    hidden_states, attention_mask = _run_cache_batches(
        prot_seqs,
        plm,
        tok,
        plm_name,
        prot_max_len,
        is_saprot,
        batch_size,
        device,
        desc=f"cache shard{shard_rank} gpu{gpu_id}",
        spill_stem=shard,
    )
    h_path, m_path = _mmap_stem_paths(shard)
    torch.save(
        {
            "shard_rank": shard_rank,
            "format": "mmap",
            "hidden_mmap": str(h_path),
            "mask_mmap": str(m_path),
            "num_samples": int(hidden_states.shape[0]),
            "seq_len": int(hidden_states.shape[1]),
            "hidden_dim": int(hidden_states.shape[2]),
        },
        shard_path,
    )


def _load_shard_arrays(payload: dict) -> Tuple[np.memmap, np.memmap, int]:
    if payload.get("format") == "mmap":
        count = int(payload["num_samples"])
        hs = np.memmap(
            payload["hidden_mmap"],
            dtype=np.float16,
            mode="r",
            shape=(count, int(payload["seq_len"]), int(payload["hidden_dim"])),
        )
        ms = np.memmap(
            payload["mask_mmap"],
            dtype=np.int64,
            mode="r",
            shape=(count, int(payload["seq_len"])),
        )
        return hs, ms, count
    hs = payload["hidden_states"].numpy()
    ms = payload["attention_mask"].numpy()
    return hs, ms, int(hs.shape[0])


def _merge_cache_shards(shard_paths: List[Path], n: int, spill_stem: Optional[Path] = None) -> tuple:
    ordered = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu")
        ordered.append((int(payload["shard_rank"]), payload))
    ordered.sort(key=lambda x: x[0])

    first = ordered[0][1]
    if first.get("format") == "mmap":
        seq_len = int(first["seq_len"])
        hidden_dim = int(first["hidden_dim"])
    else:
        seq_len = int(first["hidden_states"].shape[1])
        hidden_dim = int(first["hidden_states"].shape[2])

    if spill_stem is not None:
        h_path, m_path = _mmap_stem_paths(spill_stem)
        hidden_mm = np.memmap(
            h_path, dtype=np.float16, mode="w+", shape=(n, seq_len, hidden_dim)
        )
        mask_mm = np.memmap(m_path, dtype=np.int64, mode="w+", shape=(n, seq_len))
        offset = 0
        for _, payload in ordered:
            hs, ms, count = _load_shard_arrays(payload)
            hidden_mm[offset : offset + count] = hs
            mask_mm[offset : offset + count] = ms
            offset += count
            del payload, hs, ms
        assert offset == n, (offset, n)
        hidden_mm.flush()
        mask_mm.flush()
        return torch.from_numpy(hidden_mm), torch.from_numpy(mask_mm)

    hidden_states = torch.cat([p["hidden_states"] for _, p in ordered], dim=0)
    attention_mask = torch.cat([p["attention_mask"] for _, p in ordered], dim=0)
    assert hidden_states.shape[0] == n, (hidden_states.shape[0], n)
    return hidden_states, attention_mask


def _plm_hidden_dim(plm) -> Optional[int]:
    if plm is None:
        return None
    config = getattr(plm, "config", None)
    if config is not None:
        for attr in ("hidden_size", "d_model"):
            if hasattr(config, attr):
                return int(getattr(config, attr))
    esm = getattr(plm, "esm", None)
    if esm is not None and hasattr(esm, "embeddings"):
        return int(esm.embeddings.embedding_dim)
    return None


def _validate_mmap_manifest(
    payload: dict,
    n: int,
    expected_hidden_dim: Optional[int] = None,
    prot_max_len: Optional[int] = None,
) -> Optional[str]:
    if not isinstance(payload, dict):
        return f"payload is {type(payload).__name__}, expected dict"
    if payload.get("version") != CACHE_VERSION:
        return f"version {payload.get('version')} != {CACHE_VERSION}"

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return "missing or invalid meta"
    if meta.get("num_samples") != n:
        return f"num_samples {meta.get('num_samples')} != {n}"

    seq_len = int(payload.get("seq_len") or meta.get("seq_len") or 0)
    hidden_dim = int(payload.get("hidden_dim") or meta.get("hidden_dim") or 0)
    if seq_len <= 0 or hidden_dim <= 0:
        return "missing seq_len/hidden_dim in mmap manifest"
    if prot_max_len is not None and seq_len > int(prot_max_len):
        return f"sequence length {seq_len} > prot_max_len {prot_max_len}"
    if expected_hidden_dim is not None and hidden_dim != int(expected_hidden_dim):
        return f"hidden_dim {hidden_dim} != expected {expected_hidden_dim}"

    h_path = Path(str(payload["hidden_mmap"]))
    m_path = Path(str(payload["mask_mmap"]))
    if not _mmap_files_ready(h_path, m_path, n, seq_len, hidden_dim):
        return (
            f"mmap sidecar missing or wrong size for {h_path.name} / {m_path.name} "
            f"(n={n}, seq_len={seq_len}, hidden_dim={hidden_dim})"
        )
    return None


def _validate_cache_payload(
    payload,
    n: int,
    expected_hidden_dim: Optional[int] = None,
    prot_max_len: Optional[int] = None,
) -> Optional[str]:
    """Return an error string when invalid; None when the cache payload is usable."""
    if not isinstance(payload, dict):
        return f"payload is {type(payload).__name__}, expected dict"
    if payload.get("version") != CACHE_VERSION:
        return f"version {payload.get('version')} != {CACHE_VERSION}"

    if _is_mmap_manifest(payload):
        return _validate_mmap_manifest(payload, n, expected_hidden_dim, prot_max_len)

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return "missing or invalid meta"
    if meta.get("num_samples") != n:
        return f"num_samples {meta.get('num_samples')} != {n}"

    hidden_states = payload.get("hidden_states")
    attention_mask = payload.get("attention_mask")
    if not torch.is_tensor(hidden_states):
        return "hidden_states is not a tensor"
    if not torch.is_tensor(attention_mask):
        return "attention_mask is not a tensor"
    if hidden_states.ndim != 3:
        return f"hidden_states rank {hidden_states.ndim} != 3"
    if attention_mask.ndim != 2:
        return f"attention_mask rank {attention_mask.ndim} != 2"
    if hidden_states.shape[0] != n or attention_mask.shape[0] != n:
        return (
            f"batch size mismatch hidden={hidden_states.shape[0]} "
            f"mask={attention_mask.shape[0]} expected={n}"
        )
    if hidden_states.shape[:2] != attention_mask.shape:
        return (
            f"hidden/mask shape mismatch hidden={tuple(hidden_states.shape[:2])} "
            f"mask={tuple(attention_mask.shape)}"
        )

    hidden_dim = int(hidden_states.shape[-1])
    if hidden_dim <= 0:
        return f"invalid hidden_dim {hidden_dim}"
    meta_hidden_dim = meta.get("hidden_dim")
    if meta_hidden_dim is not None and int(meta_hidden_dim) != hidden_dim:
        return f"meta hidden_dim {meta_hidden_dim} != tensor {hidden_dim}"
    if expected_hidden_dim is not None and hidden_dim != int(expected_hidden_dim):
        return f"hidden_dim {hidden_dim} != expected {expected_hidden_dim}"

    if prot_max_len is not None and hidden_states.shape[1] > int(prot_max_len):
        return f"sequence length {hidden_states.shape[1]} > prot_max_len {prot_max_len}"
    return None


def _safe_load_cache_payload(out_path: Path):
    try:
        return torch.load(out_path, map_location="cpu"), None
    except Exception as exc:
        return None, str(exc)


def _cache_lock_path(out_path: Path) -> Path:
    return out_path.with_suffix(".building")


def _is_stale_build_lock(lock_path: Path, max_age_s: float = STALE_BUILD_LOCK_SECS) -> bool:
    if not lock_path.exists():
        return False
    return (time.time() - lock_path.stat().st_mtime) > max_age_s


def _remove_invalid_cache_file(out_path: Path):
    if out_path.exists() or any(p.exists() for p in _mmap_stem_paths(out_path)):
        logger.warning("removing invalid or incomplete PLM embed cache: %s", out_path)
    out_path.unlink(missing_ok=True)
    _cleanup_mmap_files(out_path)


def _remove_stale_build_lock(lock_path: Path):
    if lock_path.exists():
        logger.warning("removing stale PLM embed cache build lock: %s", lock_path)
        lock_path.unlink(missing_ok=True)


def _cleanup_cache_shards(out_path: Path):
    for shard in out_path.parent.glob(f"{out_path.stem}.shard*.pt"):
        shard.unlink(missing_ok=True)


class PlmEmbedCache:
    """Per-split cache keyed by dataset sample index."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        mmap_hidden: Optional[np.memmap] = None,
        mmap_mask: Optional[np.memmap] = None,
    ):
        self.hidden_states = hidden_states
        self.attention_mask = attention_mask
        self._mmap_hidden = mmap_hidden
        self._mmap_mask = mmap_mask

    def get(self, indices: Sequence[int], device: Optional[torch.device] = None) -> CachedProtBatch:
        idx_list = list(indices)
        if self._mmap_hidden is not None and self._mmap_mask is not None:
            idx = np.asarray(idx_list, dtype=np.int64)
            hidden = torch.from_numpy(
                np.array(self._mmap_hidden[idx], dtype=np.float16, copy=True)
            )
            mask = torch.from_numpy(np.array(self._mmap_mask[idx], dtype=np.int64, copy=True))
        else:
            idx = torch.as_tensor(idx_list, dtype=torch.long)
            hidden = self.hidden_states.index_select(0, idx)
            mask = self.attention_mask.index_select(0, idx)
        if device is not None:
            hidden = hidden.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
        return CachedProtBatch(hidden_states=hidden, attention_mask=mask)

    @classmethod
    def _from_mmap_manifest(cls, payload: dict, out_path: Path) -> "PlmEmbedCache":
        meta = payload["meta"]
        n = int(meta["num_samples"])
        seq_len = int(payload.get("seq_len") or meta.get("seq_len"))
        hidden_dim = int(payload.get("hidden_dim") or meta["hidden_dim"])
        h_path = Path(str(payload["hidden_mmap"]))
        m_path = Path(str(payload["mask_mmap"]))
        hidden_mm, mask_mm = _open_mmap_arrays(h_path, m_path, n, seq_len, hidden_dim)
        hidden_t = torch.from_numpy(hidden_mm)
        mask_t = torch.from_numpy(mask_mm)
        logger.info(
            "load PLM embed cache (mmap): %s (%d samples, lazy batch reads)", out_path, n
        )
        return cls(hidden_t, mask_t, mmap_hidden=hidden_mm, mmap_mask=mask_mm)

    @classmethod
    def _try_load(
        cls,
        out_path: Path,
        n: int,
        expected_hidden_dim: Optional[int] = None,
        prot_max_len: Optional[int] = None,
    ) -> Optional["PlmEmbedCache"]:
        if not out_path.exists():
            return None
        payload, load_err = _safe_load_cache_payload(out_path)
        if load_err is not None:
            logger.warning("unreadable PLM embed cache %s: %s", out_path, load_err)
            return None
        err = _validate_cache_payload(payload, n, expected_hidden_dim, prot_max_len)
        if err is not None:
            logger.warning("invalid PLM embed cache %s: %s", out_path, err)
            return None
        if _is_mmap_manifest(payload):
            return cls._from_mmap_manifest(payload, out_path)
        logger.info("load PLM embed cache (inline): %s (%d samples, full RAM)", out_path, n)
        return cls(payload["hidden_states"], payload["attention_mask"])

    @classmethod
    def _wait_for_cache_file(
        cls,
        out_path: Path,
        n: int,
        expected_hidden_dim: Optional[int] = None,
        prot_max_len: Optional[int] = None,
        poll_s: float = 5.0,
        timeout_s: float = 86400.0,
    ) -> "PlmEmbedCache":
        deadline = time.time() + timeout_s
        lock_path = _cache_lock_path(out_path)
        last_log = 0.0

        while time.time() < deadline:
            cached = cls._try_load(
                out_path, n, expected_hidden_dim=expected_hidden_dim, prot_max_len=prot_max_len
            )
            if cached is not None:
                return cached

            if _is_stale_build_lock(lock_path):
                _remove_stale_build_lock(lock_path)

            building = lock_path.exists()
            if out_path.exists() and not building:
                _remove_invalid_cache_file(out_path)
                _cleanup_cache_shards(out_path)
            elif out_path.exists() and building:
                if time.time() - last_log > 60.0:
                    logger.info(
                        "waiting for valid PLM embed cache build (existing file not valid yet): %s",
                        out_path,
                    )
                    last_log = time.time()
            elif building:
                if time.time() - last_log > 60.0:
                    logger.info("waiting for PLM embed cache build in progress: %s", out_path)
                    last_log = time.time()
            else:
                if time.time() - last_log > 60.0:
                    logger.info("waiting for PLM embed cache to appear: %s", out_path)
                    last_log = time.time()

            time.sleep(poll_s)

        raise TimeoutError(
            f"timed out waiting for a valid PLM embed cache ({n} samples, "
            f"hidden_dim={expected_hidden_dim}): {out_path}"
        )

    @classmethod
    def _save_payload(
        cls,
        out_path: Path,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        plm_name: str,
        prot_max_len: int,
        suffix: str,
        dataset_path: Union[str, Path],
    ) -> "PlmEmbedCache":
        n = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])
        hidden_dim = int(hidden_states.shape[-1])
        h_path, m_path = _mmap_stem_paths(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not _mmap_files_ready(h_path, m_path, n, seq_len, hidden_dim):
            _write_tensors_to_mmap(h_path, m_path, hidden_states, attention_mask)

        payload = {
            "version": CACHE_VERSION,
            "storage": STORAGE_MMAP,
            "meta": {
                "plm_name": plm_name,
                "prot_max_len": prot_max_len,
                "suffix": suffix,
                "dataset_path": str(dataset_path),
                "num_samples": n,
                "hidden_dim": hidden_dim,
                "seq_len": seq_len,
            },
            "hidden_mmap": str(h_path),
            "mask_mmap": str(m_path),
            "num_samples": n,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
        }
        torch.save(payload, out_path)
        logger.info("saved PLM embed cache (mmap manifest): %s", out_path)
        return cls._from_mmap_manifest(payload, out_path)

    @classmethod
    def _build_single_gpu(
        cls,
        prot_seqs: List[str],
        plm,
        prot_tokenizer,
        plm_name: str,
        prot_max_len: int,
        is_saprot: bool,
        batch_size: int,
        device: torch.device,
        out_path: Path,
        suffix: str,
        dataset_path: Union[str, Path],
    ) -> "PlmEmbedCache":
        plm.eval()
        hidden_states, attention_mask = _run_cache_batches(
            prot_seqs,
            plm,
            prot_tokenizer,
            plm_name,
            prot_max_len,
            is_saprot,
            batch_size,
            device,
            desc=f"cache {out_path.name}",
            spill_stem=out_path,
        )
        return cls._save_payload(
            out_path, hidden_states, attention_mask, plm_name, prot_max_len, suffix, dataset_path
        )

    @classmethod
    def _build_parallel(
        cls,
        prot_seqs: List[str],
        plm_name: str,
        prot_max_len: int,
        is_saprot: bool,
        batch_size: int,
        worker_gpus: List[int],
        out_path: Path,
        suffix: str,
        dataset_path: Union[str, Path],
    ) -> "PlmEmbedCache":
        n = len(prot_seqs)
        num_workers = len(worker_gpus)
        shard_size = (n + num_workers - 1) // num_workers
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = mp.get_context("spawn")
        processes = []
        shard_paths: List[Path] = []

        unique_gpus = sorted(set(worker_gpus))
        per_gpu = {g: worker_gpus.count(g) for g in unique_gpus}
        logger.info(
            "building PLM embed cache in parallel -> %s (%d samples, %d workers on GPUs %s, layout=%s)",
            out_path,
            n,
            num_workers,
            ",".join(str(g) for g in unique_gpus),
            per_gpu,
        )

        for shard_rank, gpu_id in enumerate(worker_gpus):
            start = shard_rank * shard_size
            end = min(n, start + shard_size)
            if start >= end:
                continue
            shard_path = out_path.with_suffix(f".shard{shard_rank}.pt")
            shard_paths.append(shard_path)
            proc = ctx.Process(
                target=_parallel_cache_worker,
                args=(
                    gpu_id,
                    shard_rank,
                    prot_seqs[start:end],
                    plm_name,
                    prot_max_len,
                    is_saprot,
                    batch_size,
                    str(shard_path),
                ),
            )
            proc.start()
            processes.append(proc)

        failed = []
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failed.append(proc.exitcode)

        if failed:
            for path in shard_paths:
                path.unlink(missing_ok=True)
            raise RuntimeError(f"parallel PLM cache build failed, worker exit codes: {failed}")

        try:
            hidden_states, attention_mask = _merge_cache_shards(shard_paths, n, spill_stem=out_path)
            return cls._save_payload(
                out_path, hidden_states, attention_mask, plm_name, prot_max_len, suffix, dataset_path
            )
        except Exception:
            _cleanup_mmap_files(out_path)
            raise
        finally:
            for path in shard_paths:
                payload = None
                try:
                    if path.exists():
                        payload = torch.load(path, map_location="cpu")
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("format") == "mmap":
                    for key in ("hidden_mmap", "mask_mmap"):
                        mmap_path = payload.get(key)
                        if mmap_path:
                            Path(mmap_path).unlink(missing_ok=True)
                path.unlink(missing_ok=True)

    @classmethod
    def build_or_load(
        cls,
        dataset: Dataset,
        dataset_path: Union[str, Path],
        plm,
        prot_tokenizer,
        plm_name: str,
        prot_max_len: int,
        cache_dir: Union[str, Path],
        suffix: str = "",
        is_saprot: bool = False,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
        cache_gpus: Optional[List[int]] = None,
        workers_per_gpu: int = 1,
        rank: int = 0,
    ) -> "PlmEmbedCache":
        cache_dir = Path(cache_dir)
        out_path = _cache_path(cache_dir, plm_name, prot_max_len, suffix, dataset_path)
        n = len(dataset)
        expected_hidden_dim = _plm_hidden_dim(plm)
        lock_path = _cache_lock_path(out_path)

        cached = cls._try_load(
            out_path,
            n,
            expected_hidden_dim=expected_hidden_dim,
            prot_max_len=prot_max_len,
        )
        if cached is not None:
            return cached

        if rank != 0:
            cached = cls._wait_for_cache_file(
                out_path,
                n,
                expected_hidden_dim=expected_hidden_dim,
                prot_max_len=prot_max_len,
            )
            _distributed_barrier()
            return cached

        if out_path.exists():
            _remove_invalid_cache_file(out_path)
            _cleanup_cache_shards(out_path)

        if _is_stale_build_lock(lock_path):
            _remove_stale_build_lock(lock_path)

        if lock_path.exists():
            logger.info("another process is building %s; waiting", out_path)
            cached = cls._wait_for_cache_file(
                out_path,
                n,
                expected_hidden_dim=expected_hidden_dim,
                prot_max_len=prot_max_len,
            )
            _distributed_barrier()
            return cached

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()
        try:
            logger.info("collecting protein sequences for cache build (%d samples)", n)
            prot_seqs = _collect_prot_seqs(dataset, n)

            gpu_ids = [g for g in (cache_gpus or []) if torch.cuda.is_available()]
            worker_gpus = _expand_worker_gpus(gpu_ids, workers_per_gpu) if gpu_ids else []
            use_parallel = len(worker_gpus) > 1

            if use_parallel:
                result = cls._build_parallel(
                    prot_seqs,
                    plm_name,
                    prot_max_len,
                    is_saprot,
                    batch_size,
                    worker_gpus,
                    out_path,
                    suffix,
                    dataset_path,
                )
            else:
                if device is None:
                    if torch.cuda.is_available():
                        gpu = gpu_ids[0] if gpu_ids else 0
                        device = torch.device(f"cuda:{gpu}")
                    else:
                        device = torch.device("cpu")
                if plm is None or prot_tokenizer is None:
                    prot_tokenizer, plm = _create_plm_and_tokenizer(plm_name, is_saprot, device)
                else:
                    plm = plm.to(device).eval()
                logger.info("building PLM embed cache -> %s (%d samples, 1 GPU)", out_path, n)
                result = cls._build_single_gpu(
                    prot_seqs,
                    plm,
                    prot_tokenizer,
                    plm_name,
                    prot_max_len,
                    is_saprot,
                    batch_size,
                    device,
                    out_path,
                    suffix,
                    dataset_path,
                )
            _distributed_barrier()
            return result
        finally:
            lock_path.unlink(missing_ok=True)


class PlmEmbedCacheStore:
    """Maps dataset object id -> PlmEmbedCache."""

    def __init__(self):
        self._by_dataset: Dict[int, PlmEmbedCache] = {}

    def register(self, dataset: Dataset, cache: PlmEmbedCache):
        self._by_dataset[id(dataset)] = cache

    def for_dataset(self, dataset: Dataset) -> Optional[PlmEmbedCache]:
        return self._by_dataset.get(id(dataset))

    def for_batch_dataset(self, batch: Sequence, dataset: Dataset) -> Optional[PlmEmbedCache]:
        return self.for_dataset(dataset)


def prepare_plm_embed_caches(
    store: PlmEmbedCacheStore,
    datasets: Iterable[tuple],
    plm,
    prot_tokenizer,
    args,
    is_saprot: bool = False,
):
    """Build/load caches for (dataset, dataset_path) pairs when enabled."""
    if not getattr(args, "use_cached_plm_embeds", True):
        logger.info("PLM embed cache disabled")
        return store

    plm_tune = getattr(args, "plm_tune", "freeze")
    if plm_tune != "freeze":
        logger.warning("PLM embed cache requires plm_tune=freeze; disabling cache (got %s)", plm_tune)
        args.use_cached_plm_embeds = False
        return store

    if getattr(args, "prot_aug", "None") == "rand_crop":
        logger.warning("rand_crop is incompatible with PLM embed cache; disabling prot_aug")
        args.prot_aug = "None"

    rank = _distributed_rank()
    cache_gpus = _resolve_cache_gpus(args)
    workers_per_gpu = _resolve_workers_per_gpu(args)
    suffix = getattr(args, "suffix", "")
    cache_dir = getattr(args, "plm_embed_cache_dir", DEFAULT_PLM_EMBED_CACHE_DIR)
    batch_size = getattr(args, "plm_embed_cache_batch_size", 16)
    plm_name = getattr(args, "plm_model", None) or getattr(args, "plm_name", "plm")

    worker_gpus = _expand_worker_gpus(cache_gpus, workers_per_gpu) if cache_gpus else []
    device = None
    if rank == 0 and torch.cuda.is_available():
        if len(worker_gpus) <= 1 and len(cache_gpus) == 1:
            plm = plm.cuda(cache_gpus[0])
            device = torch.device(f"cuda:{cache_gpus[0]}")
        elif len(cache_gpus) == 0:
            plm = plm.cuda()
            device = next(plm.parameters()).device

    for dataset, dataset_path in datasets:
        cache = PlmEmbedCache.build_or_load(
            dataset=dataset,
            dataset_path=dataset_path,
            plm=plm if rank == 0 else None,
            prot_tokenizer=prot_tokenizer if rank == 0 else None,
            plm_name=plm_name,
            prot_max_len=args.prot_max_len,
            cache_dir=cache_dir,
            suffix=suffix,
            is_saprot=is_saprot,
            batch_size=batch_size,
            device=device,
            cache_gpus=cache_gpus,
            workers_per_gpu=workers_per_gpu,
            rank=rank,
        )
        store.register(dataset, cache)
    return store
