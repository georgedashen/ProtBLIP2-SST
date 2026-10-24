#!/usr/bin/env python3
"""Quick smoke test for multi-GPU PLM embed cache build (small subset only)."""
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.plm_embed_cache import PlmEmbedCacheStore, prepare_plm_embed_caches, _create_plm_and_tokenizer
from data_provider.stage1_dm import SwissProtDataset


def main():
    cache_dir = ROOT / "data/plm_embed_cache_test_parallel"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    ds = SwissProtDataset(str(ROOT / "data/SwissProtV3/train_set_afdbfilt.json"), test=64)
    plm_name = "fredzzp/esm2_t12_35M_UR50D"
    tok, plm = _create_plm_and_tokenizer(plm_name, False, torch.device("cuda:0"))
    args = SimpleNamespace(
        use_cached_plm_embeds=True,
        plm_tune="freeze",
        prot_aug="None",
        suffix="_afdbfilt",
        plm_embed_cache_dir=str(cache_dir),
        plm_embed_cache_batch_size=16,
        plm_embed_cache_gpus="0,1",
        plm_embed_cache_workers_per_gpu=2,
        prot_max_len=1024,
        plm_name=plm_name,
    )
    store = PlmEmbedCacheStore()
    prepare_plm_embed_caches(store, [(ds, ds.data_path)], plm, tok, args, is_saprot=False)
    cache = store.for_dataset(ds)
    assert cache.hidden_states.shape[0] == 64
    print("parallel cache build ok", tuple(cache.hidden_states.shape))
    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
