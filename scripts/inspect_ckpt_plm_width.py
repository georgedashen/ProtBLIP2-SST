#!/usr/bin/env python3
"""Quick check: infer PLM width from ln_layer in a stage2 fp32 or DeepSpeed checkpoint."""
import argparse
import torch
from pathlib import Path


def ln_dim_from_state_dict(sd):
    for k, v in sd.items():
        if "ln_layer.weight" in k:
            return int(v.shape[0]), k
    return None, None


def load_sd(path: Path):
    path = Path(path)
    if path.is_file():
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("state_dict", ckpt)
    mp = path / "checkpoint" / "mp_rank_00_model_states.pt"
    st = torch.load(mp, map_location="cpu")
    module = st.get("module", st)
    return {k.replace("_forward_module.", ""): v for k, v in module.items() if hasattr(v, "shape")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="fp32 .ckpt file or DeepSpeed *..ckpt directory")
    args = p.parse_args()
    sd = load_sd(Path(args.path))
    dim, key = ln_dim_from_state_dict(sd)
    label = {480: "35M (ESM2-t12)", 1280: "650M (ESM2-t33)", 640: "150M (ESM2-t30)"}.get(dim, f"unknown({dim})")
    print(f"path: {args.path}")
    print(f"ln_layer dim: {dim} ({label}) via {key}")


if __name__ == "__main__":
    main()
