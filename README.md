# ProtBLIP2-SST

**Protein BLIP-2 training with Sequence, Structure, and Text (SST)**

ProtBLIP2-SST is a modular BLIP-2-style framework for open-ended protein function captioning. We keep a frozen protein encoder, an 8-query Q-Former, and Galactica-1.3B (mid-layer LoRA) fixed, and swap only the protein language model (PLM):

| Variant | PLM | Input |
|---------|-----|--------|
| SST-ESM | ESM-2 (35M / 650M) | amino-acid sequence |
| SST-SaProt | SaProt AF2 (35M / 650M) | interleaved AA + Foldseek 3Di |
| SST-ProstT5 | ProstT5 (~1.5B) | bilingual sequence–structure T5 |

**Stage 1** aligns protein and text (PTC / PTM / PCap). **Stage 2** generates Swiss-Prot-style captions (`FUNCTION:`, `SUBCELLULAR LOCATION:`, `SIMILARITY:`).

On an **aligned 9,239-sample** test split, **SaProt-650M** reaches **33.07% exact match** and **58.53 BLEU-4**, modestly above ESM-2-650M (32.83% / 58.12) with complementary errors: structure helps fold-specific enzyme disambiguation; sequence-only encoding stays stronger on short family labels.

> Built on the [ProtT3](https://github.com/georgedashen/ProtT3) / BLIP-2 codebase ([Liu et al., ACL 2024](https://openreview.net/forum?id=ZmIjOPil2b)). This repository is an independent SST extension, not a fork PR.

Paper draft: [`paper/`](paper/) · conference write-up: [`manuscripts/conference.tex`](manuscripts/conference.tex) · analysis: [`analysis/`](analysis/) · tabulated results: [`results/`](results/)

---

## Repository layout

```
# Main paper pipeline (repo root)
stage1.py / stage1sa.py / stage1pt5.py    # Stage-1: ESM / SaProt / ProstT5
stage2.py / stage2sa.py                    # Stage-2 captioning
sa_run.sh / stage1_run.sh / stage2_sa_run.sh / pt5_*.sh / fa_run.sh
convert.py / read_results.py
model/                                     # Main Q-Former + encoder/decoder modules
data_provider/                             # Main dataloaders + PLM embed cache

# Supporting
results/  analysis/  paper/  manuscripts/  scripts/

# Moved out of the root
ablations/   # AA-raw / unaligned-PLM / Stage-1 loss ablations (see ablations/README.md)
extras/      # Legacy ProtT3 utilities (ProteinChat, ProtCLAP, …)
```

Checkpoints and datasets stay local (`all_checkpoints/`, `data/`) and are gitignored.

---

## Dependencies

Python 3.8+, CUDA GPU recommended (experiments used 2× A100 80GB).

Typical stack (see ProtT3 notes for RTX 3090 / A100 tips):

```bash
# PyTorch + CUDA matching your driver
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install git+https://github.com/thunlp/OpenDelta.git
pip install rouge_score nltk spacy==3.7.2 salesforce-lavis
pip install -U transformers pytorch-lightning==2.1 huggingface_hub==0.25.2
pip install git+https://github.com/microsoft/DeepSpeed.git

python -c "import nltk; nltk.download('wordnet')"
```

External weights:

- **ESM-2**: Hugging Face `fredzzp/esm2_t12_35M_UR50D`, `fredzzp/esm2_t33_650M_UR50D` (or Facebook ESM equivalents)
- **SaProt**: local `SaProt_*_AF2` checkpoints (see [SaProt](https://github.com/westlake-repl/SaProt))
- **ProstT5**: `Rostlab/ProstT5`
- **LLM**: `facebook/galactica-1.3b`
- **Text encoders**: PubMedBERT / BiomedBERT as used in each variant

---

## Dataset

Base corpus: **Swiss-Prot V3** from ProtT3 ([OSF download](https://osf.io/23azs/?view_only=185575515e714f4798499bf06513a730)). Unzip under `./data`.

ProtBLIP2-SST keeps proteins with AlphaFold DB structures and ships two parallel suffixes:

| Suffix | Used by | Tokenization |
|--------|---------|--------------|
| `*_afdbfilt.json` | ESM-2, ProstT5, AA-raw ablations | amino acids only |
| `*_saprot.json` | SaProt | interleaved AA + lowercase 3Di |

Approximate split sizes: **410,925** train / **9,236** val / **9,239** test. Stage-2 comparisons use the **aligned** test subset (same captions and order across suffixes).

Stage-2 prompt: `Swiss-Prot description:` (or `Describe the function of this protein.` depending on script flags).

---

## Stage 1 — protein–text alignment

Train Q-Former with frozen PLM (`plm_tune=freeze`), DeepSpeed ZeRO-2, bf16, 8 query tokens, 50 epochs.

### SaProt (structure-aware)

```bash
# see also sa_run.sh
python stage1sa.py \
  --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_sa --num_query_token 8 --num_workers 8 \
  --plm_name "../SaProt/weights/PLMs/SaProt_35M_AF2" \
  --plm_tune freeze --suffix _saprot \
  --save_every_n_epochs 10 --batch_size 32 --match_batch_size 64 \
  --precision bf16-mixed --max_epochs 50 --use_tb_logger --seed 42
```

### ESM-2 (sequence-only)

```bash
# see also stage1_run.sh
python stage1.py \
  --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_afdbfilt --num_query_token 8 --num_workers 8 \
  --plm_name "fredzzp/esm2_t12_35M_UR50D" \
  --suffix _afdbfilt \
  --save_every_n_epochs 10 --batch_size 32 --match_batch_size 64 \
  --precision bf16-mixed --max_epochs 50 --use_tb_logger --seed 42
```

### ProstT5

```bash
python stage1pt5.py \
  --strategy deepspeed --devices '0,1' --mode train \
  --filename stage1_pt5 --num_query_token 8 --num_workers 8 \
  --plm_name "Rostlab/ProstT5" --suffix _afdbfilt \
  --save_every_n_epochs 10 --batch_size 32 --match_batch_size 64 \
  --precision bf16-mixed --max_epochs 50 --use_tb_logger
```

Convert DeepSpeed shards to a single fp32 checkpoint before Stage 2:

```bash
python convert.py --input all_checkpoints/stage1_sa/epoch=49.ckpt \
                  --output all_checkpoints/stage1_sa/SaProt_35M_fp32.ckpt
```

### Stage-1 ablation (PTM / LM losses)

```bash
bash ablations/stage1sa_ablation_35m.sh   # trains SaProt-35M w/o PTM and w/o LM
```

### Reported Stage-1 retrieval (average Acc / Rec@20)

| Encoder | Average |
|---------|--------:|
| ESM-2 35M | 88.44 |
| SaProt 35M | 89.53 |
| ESM-2 650M | 89.92 |
| ProstT5 1.5B | 90.03 |
| **SaProt 650M** | **90.83** |

---

## Stage 2 — caption generation

Load Stage-1 Q-Former, attach Galactica-1.3B with `mid_lora` (`r=8`, `α=16`), train 10 epochs, evaluate with beam search.

### SaProt

```bash
# see stage2_sa_run.sh
python stage2sa.py \
  --strategy deepspeed --devices '0,1' \
  --stage1_path all_checkpoints/stage1_sa/SaProt_35M_fp32.ckpt \
  --filename stage2_sa_35m_protein_captioning \
  --plm_model "../SaProt/weights/PLMs/SaProt_35M_AF2" \
  --llm_name "facebook/galactica-1.3b" \
  --llm_tune mid_lora --enable_flash --use_tb_logger \
  --root data/SwissProtV3 --suffix _saprot \
  --num_query_token 8 --batch_size 32 --max_epochs 10 \
  --precision bf16-mixed --num_workers 8 --save_every_n_epochs 10
```

### ESM-2 (ProtT3-style baseline under the same skeleton)

```bash
python stage2.py \
  --strategy deepspeed --devices '0,1' \
  --stage1_path /path/to/stage1_esm_fp32.ckpt \
  --filename stage2_esm_35m \
  --llm_tune mid_lora --enable_flash \
  --root ./data/SwissProtV3 --suffix _afdbfilt \
  --num_query_token 8 --batch_size 32 --max_epochs 10 \
  --precision bf16-mixed --num_workers 8
```

Predictions are written under `all_checkpoints/<filename>/.../dataset0_predictions.txt`. Score with:

```bash
python read_results.py --path /path/to/dataset0_predictions.txt
```

### Reported Stage-2 captioning (aligned 9,239)

| Model | EM | BLEU-4 | ROUGE-L | METEOR |
|-------|---:|-------:|--------:|-------:|
| ProtT3 / ESM-2 35M | 30.33 | 54.74 | 64.29 | 65.99 |
| ProtBLIP2-SST / SaProt 35M | 30.59 | 55.37 | 65.26 | 66.62 |
| ProtT3 / ESM-2 650M | 32.83 | 58.12 | 67.77 | 69.10 |
| **ProtBLIP2-SST / SaProt 650M** | **33.07** | **58.53** | **68.23** | **69.54** |

JSON + win-case TSVs: [`results/`](results/).

---

## Stage-2 ablations (what we tested)

Scripts live under [`ablations/`](ablations/). They isolate **alignment** and **input format** at 35M scale (same Galactica + LoRA where applicable):

| Variant | Script | EM | BLEU-4 | ROUGE-L | METEOR |
|---------|--------|---:|-------:|--------:|-------:|
| No finetune (marker + AA, zero-shot) | `ablations/stage2aaraw_joint_summary_eval_afdbfilt.sh` | 0.00 | 0.58 | 6.20 | 5.11 |
| LoRA on raw AA letters | `ablations/stage2aaraw_run.sh` | 24.43 | 37.15 | 53.05 | 57.40 |
| LoRA on marker + AA | `ablations/stage2aaraw_presuffix_train.sh` | 24.22 | 36.97 | 53.01 | 57.16 |
| Unaligned ESM-2 hidden states → LLM | `ablations/stage2plm_run.sh` | 30.92 | 43.66 | 62.91 | 64.51 |
| ProtT3 / ESM-2 + 8 Q-Former queries | `stage2.py` | 30.33 | 54.74 | 64.29 | 65.99 |
| **ProtBLIP2-SST / SaProt + 8 queries** | `stage2sa.py` | **30.59** | **55.37** | **65.26** | **66.62** |

**Interpretation**

1. **Q-Former alignment matters most for BLEU/ROUGE** — unaligned PLM tokens reach similar exact match but much weaker n-gram overlap; raw-AA LoRA lags further.
2. **Structure (SaProt) helps beyond ESM-2** under identical Q-Former + Galactica — modest metric gains, clearer qualitative wins on enzyme/pathway confusions ([`analysis/saprot_vs_prott3_stage2_analysis.md`](analysis/saprot_vs_prott3_stage2_analysis.md)).
3. **Zero-shot Galactica fails** on Swiss-Prot captions even with AA prefixes — Stage-2 finetuning is required.

Field-level exclusive exact-match wins (650M): SaProt leads on `FUNCTION:` / `SIMILARITY:`; ESM-2 leads on `SUBCELLULAR LOCATION:`.

---

## Cached PLM embeddings

When the PLM is frozen, hidden states can be precomputed (before Q-Former LayerNorm) via `data_provider/plm_embed_cache.py`. Observed Stage-1 speedups on A100: ~2× (ESM-35M), ~6–7× (ESM-650M), ~12× (ProstT5).

---

## Citation

If you use this code, please cite ProtT3 and the relevant PLMs (ESM-2, SaProt, ProstT5, Galactica, BLIP-2). ProtBLIP2-SST manuscript citation will be added upon publication.

```bibtex
@inproceedings{liu2024prott,
  title={ProtT3: Protein-to-Text Generation for Text-based Protein Understanding},
  author={Liu, Zhiyuan and Zhang, An and Fei, Hao and Zhang, Enzhi and Wang, Xiang and Kawaguchi, Kenji and Chua, Tat-Seng},
  booktitle={ACL},
  year={2024}
}
```

---

## License / acknowledgments

Code derives from ProtT3. Dataset and pretrained weights follow their original licenses. Thanks to the ProtT3, SaProt, ProstT5, ESM, Galactica, and BLIP-2 authors.
