# Experiment results (ProtBLIP2-SST)

Numbers below are re-scored from local checkpoints on the **aligned 9,239-sample** Swiss-Prot test split (`*_saprot.json` / `*_afdbfilt.json`, identical captions and order).

Machine-readable copies: `stage2_main_summary.json`, `ablation/stage2_35m_summary.json`.  
Qualitative win cases: `stage2_main/sa_win_cross_size_cases.tsv` (69), `stage2_main/prott3_win_cross_size_cases.tsv` (45).  
Long-form analysis: [`../analysis/saprot_vs_prott3_stage2_analysis.md`](../analysis/saprot_vs_prott3_stage2_analysis.md).

## Stage 2 — main captioning

| Model | EM | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|-------|---:|-------:|-------:|--------:|-------:|
| ProtT3 / ESM-2 35M | 30.33 | 57.86 | 54.74 | 64.29 | 65.99 |
| ProtBLIP2-SST / SaProt 35M | 30.59 | 58.33 | 55.37 | 65.26 | 66.62 |
| ProtT3 / ESM-2 650M | 32.83 | 61.01 | 58.12 | 67.77 | 69.10 |
| **ProtBLIP2-SST / SaProt 650M** | **33.07** | **61.41** | **58.53** | **68.23** | **69.54** |

## Stage 2 — ablation (35M scale)

| Variant | EM | BLEU-4 | ROUGE-L | METEOR |
|---------|---:|-------:|--------:|-------:|
| No finetune (marker + AA, Galactica zero-shot) | 0.00 | 0.58 | 6.20 | 5.11 |
| LoRA finetune (raw AA) | 24.43 | 37.15 | 53.05 | 57.40 |
| LoRA finetune (marker + AA) | 24.22 | 36.97 | 53.01 | 57.16 |
| ESM-2 hidden states → Galactica (no Q-Former alignment) | 30.92 | 43.66 | 62.91 | 64.51 |
| ProtT3 / ESM-2 + 8 query tokens | 30.33 | 54.74 | 64.29 | 65.99 |
| **ProtBLIP2-SST / SaProt + 8 queries** | **30.59** | **55.37** | **65.26** | **66.62** |

**Takeaway:** Q-Former protein–text alignment is the main driver of n-gram overlap (BLEU/ROUGE). Structure-aware SaProt adds a smaller but consistent gain over ESM-2 under the same BLIP-2 skeleton. Raw-AA LoRA and zero-shot Galactica are far weaker.

## Stage 1 — retrieval (from manuscript)

Average of Acc / Rec@20 (in-batch + test, P→T and T→P) after PTM re-ranking (128 candidates):

| Encoder | Average |
|---------|--------:|
| ESM-2 35M | 88.44 |
| SaProt 35M | 89.53 |
| ESM-2 650M | 89.92 |
| ProstT5 1.5B | 90.03 |
| **SaProt 650M** | **90.83** |

Stage-1 PTM/LM loss ablations (`stage1sa_ablation_35m.sh`) were trained locally; full retrieval re-eval numbers are still pending for the w/o-PTM and w/o-LM variants.
