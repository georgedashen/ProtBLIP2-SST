# SaProt Stage-2 Improvement Proposals

Based on aligned eval (9,239 samples) and qualitative analysis in [saprot_vs_prott3_stage2_analysis.md](./saprot_vs_prott3_stage2_analysis.md).

**Current strengths to preserve:** fold-specific enzyme disambiguation, multi-field Swiss-Prot formatting, cross-scale stability (35M follows 650M when structure read is correct).

**Current gaps to address:** ~24% `FUNCTION:` hallucination on short `SIMILARITY` targets (both models); 45 ProtT3 cross-size wins where SaProt over-generates; modest aggregate BLEU gap (+0.41 @ 650M) leaves room on easy family/localization lines.

---

## 1. Data collection and curation

### 1.1 Field-balanced and length-stratified sampling

Swiss-Prot captions are dominated by long `FUNCTION:` entries (~87% of test set). Training likely mirrors this, which encourages verbose FUNCTION priors from Galactica.

**Proposal:**

- Stratify batches by **caption lead field** (`FUNCTION` / `SUBCELLULAR` / `SIMILARITY`) and **token length** (short ≤15 words vs long).
- Target ratios e.g. 60% FUNCTION / 20% SUBCELLULAR / 20% SIMILARITY-short in each epoch (vs natural ~80/10/10).
- Oversample **minimal captions** where ProtT3 currently wins (HSP70, asp23, IS21 families — see `prott3_win_cross_size_cases.tsv`).

**Expected effect:** Reduce Sa-specific over-hallucination on one-line family labels without losing enzyme disambiguation gains.

### 1.2 Hard negatives and confusable pairs

47/69 Sa wins are “same field (`FUNCTION:`), wrong pathway.” These are **confusable superfamilies** — ideal supervision if mined explicitly.

**Proposal:**

- Build a **confusion graph** from Swiss-Prot: proteins sharing InterPro/Pfam family but different EC/reaction keywords.
- Add **paired training examples**: (protein A, correct caption A) vs (protein A, wrong caption B) with contrastive or ranking loss in stage 1 or auxiliary stage-2 head.
- Import hard negatives from `prott3_win_cross_size_cases.tsv` (Sa wrong, Pt right) and `sa_win_cross_size_cases.tsv` (Pt wrong, Sa right) as a **curated 114-sample replay buffer** each epoch.

### 1.3 Explicit sequence–structure–function annotations

SaProt already consumes AA+3Di, but stage-2 trains **protein → caption** end-to-end with no intermediate supervision.

**Proposal — curated SSF layers:**

| Layer | Source | Example |
|-------|--------|---------|
| Sequence motif | PROSITE / Pfam | “Walker A”, “HSP70 signature” |
| Structure | AF2 + 3Di cluster | “TIM barrel”, “Rossmann fold” |
| Function | Swiss-Prot / Rhea / EC | “dienoyl-CoA isomerase” |

- Extend JSONL with optional fields: `fold_label`, `motif_span`, `ec_number`, `reaction_text`.
- Train **multi-task**: predict compact SSF JSON first, then full Swiss-Prot caption (see §2.2 CoT).

### 1.4 Data quality and coverage

- **Structure coverage:** Proteins with low pLDDT regions — mask or down-weight noisy 3Di tokens; add pLDDT as extra input channel to SaProt encoder.
- **AF vs experimental structures:** Mix AF2 with PDB-AlphaFold hybrid entries where available to reduce fold mislabel risk.
- **Beyond Swiss-Prot:** TrEMBL high-confidence subset with 3Di for scale; keep Swiss-Prot for eval only.

---

## 2. Sequence–structure–function chain-of-thought (CoT)

### 2.1 Structured intermediate captions (training only)

Instead of jumping to full Swiss-Prot prose, use a **two-stage target** during training:

```
[FOLD] enoyl-CoA hydratase/isomerase superfamily
[SITE] C-terminal catalytic domain
[FUNCTION] Isomerization of 3-trans,5-cis-dienoyl-CoA ...
[FORMAT] FUNCTION + SUBCELLULAR + SIMILARITY  (field order)
```

At inference: generate CoT with `max_length` budget, then **compress** to eval format (or train compressor with second pass). Alternatively use **hidden CoT** — predict SSF tokens as auxiliary labels without decoding them at test time.

**Expected effect:** Forces Q-Former to encode fold before Galactica narrates; directly attacks pathway-swap errors (idx 4215, 5255, 1610).

### 2.2 Field-first decoding policy

Analysis shows failures when the model picks the wrong **leading field** (SIMILARITY target → FUNCTION output).

**Proposal:**

1. Lightweight **field classifier** on Q-Former pooled output: `{SIMILARITY-only, SUBCELLULAR-led, FUNCTION-led}`.
2. Condition Galactica prompt on prediction:  
   - `Write only a SIMILARITY line.`  
   - `Start with SUBCELLULAR LOCATION, then SIMILARITY.`
3. Train classifier with CE loss; captions remain main LM loss.

**Low-cost variant:** Rule from classifier at inference only (no CoT text).

### 2.3 Structure-grounded rationale distillation

Use GPT-4 / manual curation on a **small SSF-annotated set** (~2–5k) to produce rationales:

> “3Di cluster X and active-site geometry match TRM44 tRNA methyltransferases, not mRNA cap MTases.”

Distill into SaProt via **KL on intermediate tokens** or **DPO** (chosen: correct caption vs rejected: mRNA-cap caption from same protein).

---

## 3. Model architecture

### 3.1 Protein encoder: unfreeze and fuse

Current setup: **SaProt PLM frozen** (`plm_tune: freeze`), only Q-Former + LoRA train.

| Change | Rationale | Cost |
|--------|-----------|------|
| SaProt **LoRA** (layers last 25%) | Adapt 3Di readout to caption task | +~5–10% train memory |
| **Dual encoder fusion** (SaProt + ESM2 seq) | Keep Sa disambiguation + ESM2 family-motif strength (ProtT3 wins on HSP70) | 2× encoder forward; linear fusion before Q-Former |
| **pLDDT-weighted pooling** | Down-weight unreliable structure tokens | Minimal |

Dual-path sketch:

```
AA+3Di → SaProt ──┐
                  ├→ concat/proj → Q-Former → LLM
AA only → ESM2  ──┘   (ESM2 frozen or small LoRA)
```

### 3.2 Q-Former and protein–LLM bridge

- **Increase `num_query_token`** from 8 → 16 or 32 for long FUNCTION captions (avg ~26 words in Sa-win set; 8 queries may bottleneck).
- **Structure-aware cross-attention:** extra cross-attn layer where queries attend only to residues with high pLDDT or catalytic site masks (if annotations exist).
- **Per-field query banks:** 4 queries for function, 2 for localization, 2 for similarity — soft partition before LLM projection.

### 3.3 Language model and decoding

- **Instruction-tuned LM** (e.g. Galactica-1.3b is dated): migrate to a modern bio/instruct model with same LoRA recipe; keep Q-Former interface.
- **Decoding constraints:**  
  - `min_inference_len` / `max_inference_len` **per field class** (short for SIMILARITY-only).  
  - **No-repeat n-gram** for boilerplate (“Component of the ribosome”).  
  - **Length normalization** in beam search favoring shorter hypotheses when classifier says SIMILARITY-only.

### 3.4 Auxiliary heads (multi-task stage 2)

Train alongside caption LM loss:

| Head | Target | Weight |
|------|--------|--------|
| EC number (multi-label) | Swiss-Prot EC | 0.1 |
| Fold class (3Di cluster) | derived from 3Di | 0.1 |
| Field-type classifier | lead field | 0.05 |
| Caption LM | full text | 1.0 |

Auxiliary losses tie structure representation to function taxonomy without requiring CoT at inference.

---

## 4. Training efficiency and schedule

### 4.1 Evaluation and early stopping

- `caption_eval_epoch: 10` → caption metrics **once per training run**; val loss alone is misleading (broken metrics logging was fixed separately).
- **Proposal:** `caption_eval_epoch: 1` on valid subset (1k samples) every epoch; full 9,239 eval only at end.
- Use **correct** `flatten_caption_field` metrics for checkpoint selection (already in code).

### 4.2 Curriculum and replay

| Phase | Epochs | Data mix |
|-------|--------|----------|
| A | 1–3 | Short SIMILARITY + SUBCELLULAR (field discipline) |
| B | 4–7 | Full Swiss-Prot + hard pairs from TSV replay |
| C | 8–10 | Full mix + optional CoT targets |

**Hard-example replay:** 114 curated indices each epoch (69 Sa-win + 45 Pt-win) with 2–5× sampling weight.

### 4.3 Parameter-efficient training

- **Stage 2 only:** SaProt LoRA + Q-Former + mid LoRA (current) — baseline.
- **Stage 1.5:** Brief continued contrastive on **confusable pairs** only (cheap, no LLM).
- **650M → 35M distillation:** KL on LLM logits for structure-disambiguation subset to improve 35M without full 650M inference at train time.

### 4.4 Compute optimizations

Already using DeepSpeed, bf16, flash attention. Further:

- **Gradient checkpointing** on LLM (`enbale_gradient_checkpointing: true`) to allow larger batch or more query tokens.
- **Cached SaProt embeddings** for stage-2 sweeps (precompute PLM outputs, train only Q-Former + LoRA) — faster hyperparameter search.
- **Mixed caption eval** in TB: track per-field exact match, not just BLEU-4.

---

## 5. Proposed experiment roadmap (priority order)

| Priority | Experiment | Addresses | Effort |
|----------|------------|-----------|--------|
| P0 | Field-balanced sampling + field classifier | SIMILARITY hallucination, ProtT3-win gap | Low |
| P0 | Fix/evaluate with per-field metrics each epoch | Training visibility | Low |
| P1 | Hard-pair replay buffer (114 TSV cases) | Pathway swap, symmetric errors | Low |
| P1 | SaProt LoRA (unfreeze light) | Encoder–task alignment | Medium |
| P2 | SSF CoT targets (compact JSON intermediate) | 47/69 wrong-pathway wins | Medium |
| P2 | Dual encoder SaProt + ESM2 fusion | Best of both win modes | High |
| P3 | Auxiliary EC/fold heads | Structure–function grounding | Medium |
| P3 | LM upgrade + constrained decoding | Generic boilerplate | High |

---

## 6. Success metrics (beyond aggregate BLEU)

Track separately on valid/test:

1. **Per-field exact match** (FUNCTION / SUBCELLULAR / SIMILARITY-led).
2. **Short-caption exact** (≤15 words).
3. **Confusable-pair accuracy** on mined same-family pairs.
4. **Cross-size consistency:** fraction where 650M direction matches 35M.
5. **Hallucination rate:** `FUNCTION:` lead when target is SIMILARITY-only.

Target: improve SIMILARITY-only exact toward ProtT3 parity while **maintaining or widening** enzyme disambiguation gap on hard FUNCTION pairs.

---

## 7. Risks and ablations

| Risk | Mitigation |
|------|------------|
| Field balancing hurts long FUNCTION BLEU | Mix ratio ablation (70/15/15 vs 60/20/20) |
| CoT adds length → trunc at 128 tokens | CoT only at train; single-line distill |
| Dual encoder overfits | Freeze ESM2; ablate fusion on AA-only eval |
| LoRA on SaProt degrades structure pretraining | Small rank (r=4), early stop on fold auxiliary |

**Critical ablation:** SaProt with **3Di masked vs full** on same backbone — quantifies how much gain is structure vs SaProt sequence representation.

---

## Related artifacts

- [saprot_vs_prott3_stage2_analysis.md](./saprot_vs_prott3_stage2_analysis.md)
- `all_checkpoints/stage2_main/sa_win_cross_size_cases.tsv`
- `all_checkpoints/stage2_main/prott3_win_cross_size_cases.tsv`
- `scripts/export_sa_win_cross_size_cases.py`
