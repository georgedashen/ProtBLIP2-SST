# Ablations

Code for paper **ablation / baseline** experiments (not the main SST pipeline).

Run all commands from the **repository root** (the shell scripts already `cd` there):

```bash
bash ablations/stage2aaraw_run.sh
bash ablations/stage1sa_ablation_35m.sh
python ablations/stage2_pLM.py --help
```

| Experiment | Entry | What it tests |
|------------|-------|----------------|
| Stage-1 w/o PTM / w/o LM | `stage1sa_ablation_35m.sh` | Loss ablation on SaProt-35M |
| Raw AA → Galactica (+ LoRA) | `stage2_AAraw.py`, `stage2aaraw_*.sh` | No PLM / no Q-Former |
| Marker + AA prefix | `stage2_AAraw_presuffix.py` | Prompt-format ablation |
| Joint marker summary | `stage2_AAraw_joint.py` | Zero-shot / marker variants |
| Unaligned PLM hidden states | `stage2_pLM.py`, `stage2_pLM8.py`, `stage2plm_*.sh` | Skip Q-Former alignment |

Implementation modules live under `ablations/model/` and `ablations/data_provider/`. Thin shims in top-level `model/` / `data_provider/` keep old import paths working.

Reported numbers: [`../results/ablation/`](../results/ablation/) and the Stage-2 ablation table in the root README.
