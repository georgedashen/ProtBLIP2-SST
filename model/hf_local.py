"""Prefer local Hugging Face cache for from_pretrained (skip hub when weights are cached)."""
import os
from typing import Any, Dict


def _offline_forced() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"


def _has_local_hf_file(model_name: str, filename: str = "config.json") -> bool:
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, filename)):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model_name, filename)
        return path is not None and os.path.isfile(path)
    except Exception:
        return False


def hf_from_pretrained_kwargs(model_name: str, config_file: str = "config.json") -> Dict[str, Any]:
    """Return local_files_only=True when offline is forced or weights are already cached."""
    if _offline_forced() or _has_local_hf_file(model_name, config_file):
        return {"local_files_only": True}
    return {}


def configure_hf_offline_if_cached(*model_names: str, config_file: str = "config.json") -> None:
    """Set HF offline env vars when all listed models are available locally."""
    if _offline_forced():
        return
    if model_names and all(_has_local_hf_file(name, config_file) for name in model_names if name):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
