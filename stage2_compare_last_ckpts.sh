#!/bin/bash
# Compare last.ckpt vs last-v1.ckpt via eval-only runs (no retraining).
# Writes to an isolated subfolder; does not modify training checkpoints or version_0 logs.
#
# Usage:
#   bash stage2_compare_last_ckpts.sh
#   bash stage2_compare_last_ckpts.sh --skip-convert   # use existing fp32 in eval_compare/converted/
#   bash stage2_compare_last_ckpts.sh --only-summary  # parse metrics only (after eval finished)
#
# Outputs:
#   all_checkpoints/prott3_650m_protein_captioning_swiss_dataset/eval_compare/
#     converted/last.ckpt.fp32, last-v1.ckpt.fp32
#     last_ckpt/lightning_logs/version_*/{metrics.csv,dataset0_predictions.txt}
#     last-v1_ckpt/lightning_logs/version_*/...
#     comparison_summary.csv

set -euo pipefail
cd "$(dirname "$0")"

SKIP_CONVERT=false
ONLY_SUMMARY=false
OVERWRITE_CONVERT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-convert) SKIP_CONVERT=true; shift ;;
    --only-summary) ONLY_SUMMARY=true; shift ;;
    --overwrite-convert) OVERWRITE_CONVERT=true; shift ;;
    -h|--help)
      echo "Usage: bash stage2_compare_last_ckpts.sh [--skip-convert] [--only-summary] [--overwrite-convert]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

CKPT_DIR="${CKPT_DIR:-all_checkpoints/prott3_650m_protein_captioning_swiss_dataset}"
EVAL_ROOT="${CKPT_DIR}/eval_compare"
CONVERT_DIR="${EVAL_ROOT}/converted"
CKPT_NAMES=( "last.ckpt" "last-v1.ckpt" )
LOG_TAGS=( "last_ckpt" "last-v1_ckpt" )

# Same settings as stage2_reproduce.sh (650M block)
PLM_MODEL="${PLM_MODEL:-fredzzp/esm2_t33_650M_UR50D}"
DEVICES="${DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${CONVERT_DIR}"

convert_one() {
  local ds_name="$1"
  local fp32_out="${CONVERT_DIR}/${ds_name}.fp32"
  if [[ "${SKIP_CONVERT}" == "true" && -f "${fp32_out}" ]]; then
    echo "[convert] skip (exists): ${fp32_out}"
    return 0
  fi
  if [[ -f "${fp32_out}" && "${OVERWRITE_CONVERT}" != "true" ]]; then
    echo "[convert] skip (exists): ${fp32_out}"
    return 0
  fi
  if [[ ! -d "${CKPT_DIR}/${ds_name}" ]]; then
    echo "Error: missing ${CKPT_DIR}/${ds_name}" >&2
    exit 1
  fi
  echo "[convert] ${CKPT_DIR}/${ds_name} -> ${fp32_out}"
  python convert.py --input "${CKPT_DIR}/${ds_name}" --output "${fp32_out}"
}

eval_one() {
  local ds_name="$1"
  local log_tag="$2"
  local fp32_ckpt="${CONVERT_DIR}/${ds_name}.fp32"
  local run_name="$(basename "${CKPT_DIR}")/eval_compare/${log_tag}"

  if [[ ! -f "${fp32_ckpt}" ]]; then
    echo "Error: fp32 not found: ${fp32_ckpt} (run convert first)" >&2
    exit 1
  fi

  echo ""
  echo "========== eval: ${ds_name} -> all_checkpoints/${run_name} =========="
  python stage2.py \
    --mode eval \
    --devices "${DEVICES}" \
    --strategy deepspeed \
    --filename "${run_name}" \
    --init_checkpoint "${fp32_ckpt}" \
    --plm_model "${PLM_MODEL}" \
    --num_query_token 8 \
    --caption_eval_epoch 1 \
    --batch_size 32 \
    --inference_batch_size 4 \
    --precision bf16-mixed \
    --num_workers 8 \
    --llm_tune mid_lora \
    --enable_flash \
    --root data/SwissProtV3
}

if [[ "${ONLY_SUMMARY}" != "true" ]]; then
  for i in "${!CKPT_NAMES[@]}"; do
    convert_one "${CKPT_NAMES[$i]}"
  done
  for i in "${!CKPT_NAMES[@]}"; do
    eval_one "${CKPT_NAMES[$i]}" "${LOG_TAGS[$i]}"
  done
fi

echo ""
echo "========== comparison summary =========="
CKPT_DIR="${CKPT_DIR}" EVAL_ROOT="${EVAL_ROOT}" python - <<'PY'
import csv
import os
from pathlib import Path

ckpt_dir = Path(os.environ["CKPT_DIR"])
eval_root = Path(os.environ["EVAL_ROOT"])
tags = ["last_ckpt", "last-v1_ckpt"]
ds_names = ["last.ckpt", "last-v1.ckpt"]

cols = [
    "dataloader0/val loss",
    "dataset0/acc",
    "dataset0/bleu2",
    "dataset0/bleu4",
    "dataset0/meteor_score",
    "dataset0/rouge_1",
    "dataset0/rouge_2",
    "dataset0/rouge_l",
]

def last_metric_row(metrics_path: Path):
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    # Prefer last row with caption metrics (bleu4 present)
    for row in reversed(rows):
        if row.get("dataset0/bleu4", "").strip():
            return row
    return rows[-1]

def find_metrics(tag: str):
    log_dir = eval_root / tag / "lightning_logs"
    if not log_dir.exists():
        return None, None
    versions = sorted(log_dir.glob("version_*"), key=lambda p: p.stat().st_mtime)
    if not versions:
        return None, None
    m = versions[-1] / "metrics.csv"
    return m if m.exists() else None, versions[-1]

rows_out = []
for tag, ds in zip(tags, ds_names):
    mpath, vdir = find_metrics(tag)
    entry = {"checkpoint": ds, "log_tag": tag, "metrics_file": str(mpath) if mpath else ""}
    if mpath:
        row = last_metric_row(mpath)
        for c in cols:
            entry[c] = row.get(c, "") if row else ""
    rows_out.append(entry)

out_csv = eval_root / "comparison_summary.csv"
fieldnames = ["checkpoint", "log_tag", "metrics_file"] + cols
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)

print(f"Wrote {out_csv}\n")
header = f"{'checkpoint':<14} {'bleu4':>8} {'acc':>8} {'meteor':>8} {'val_loss':>10}  likely"
print(header)
print("-" * len(header))

def score(r):
    try:
        return float(r.get("dataset0/bleu4") or -1)
    except ValueError:
        return -1

ranked = sorted(rows_out, key=score, reverse=True)
best = ranked[0]["checkpoint"] if ranked and score(ranked[0]) >= 0 else None

for r in rows_out:
    bleu4 = r.get("dataset0/bleu4", "n/a")
    acc = r.get("dataset0/acc", "n/a")
    meteor = r.get("dataset0/meteor_score", "n/a")
    vloss = r.get("dataloader0/val loss", "n/a")
    hint = ""
    if best and r["checkpoint"] == best:
        hint = "  <- higher caption metrics (likely 650M stage1 init)"
    elif best and len(ranked) == 2:
        hint = "  <- lower caption metrics (suspect 35M stage1 init)"
    print(f"{r['checkpoint']:<14} {str(bleu4):>8} {str(acc):>8} {str(meteor):>8} {str(vloss):>10}{hint}")

# Reference: original training run version_0 (if present)
ref = ckpt_dir / "lightning_logs/version_0/metrics.csv"
if ref.exists():
    row = last_metric_row(ref)
    if row:
        print(f"\nReference (training version_0 / last completed run):")
        print(f"  bleu4={row.get('dataset0/bleu4')} acc={row.get('dataset0/acc')} "
              f"meteor={row.get('dataset0/meteor_score')} val_loss={row.get('dataloader0/val loss')}")
PY

echo ""
echo "Eval outputs under: ${EVAL_ROOT}/"
echo "  (does not modify ${CKPT_DIR}/lightning_logs/version_0 or DeepSpeed checkpoints)"
