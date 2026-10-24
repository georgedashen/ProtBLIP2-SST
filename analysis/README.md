# ProtT3 Analysis

Qualitative and quantitative analyses supporting stage-2 checkpoint comparison and SaProt vs ProtT3 interpretation.

## Contents

| File | Description |
|------|-------------|
| [saprot_vs_prott3_stage2_analysis.md](./saprot_vs_prott3_stage2_analysis.md) | Architecture comparison, win-case patterns, and practical guidance from aligned 9,239-sample eval |
| [saprot_improvement_proposals.md](./saprot_improvement_proposals.md) | Data, CoT/SSF, architecture, and training proposals based on the analysis |

## Data sources

- Checkpoints and predictions: `all_checkpoints/stage2_main/`
- Case lists: `sa_win_cross_size_cases.tsv`, `prott3_win_cross_size_cases.tsv` (same directory)

## Regenerate case lists

```bash
python scripts/export_sa_win_cross_size_cases.py
```
