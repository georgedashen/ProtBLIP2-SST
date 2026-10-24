# Extras (legacy ProtT3 utilities)

Optional scripts inherited from ProtT3 that are **not** part of the ProtBLIP2-SST main paper experiments:

- `llm_tuning.py` — LLM captioning without the SST Q-Former path
- `proteinchat_tuning.py` / `train_protclap.py` — ProteinChat / ProtCLAP
- `esmc_demo.py` — ESM-C demo

Modules live under `extras/model/` and `extras/data_provider/`. Top-level `model/` / `data_provider/` keep compatibility shims.

Prefer the root Stage-1 / Stage-2 entry points for ProtBLIP2-SST.
