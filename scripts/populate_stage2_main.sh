#!/usr/bin/env bash
# Copy best-scoring stage2 checkpoints into all_checkpoints/stage2_main/
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="all_checkpoints/stage2_main"

# name | source_run_dir | checkpoint_folder | log_dir (predictions+metrics)
declare -A RUN_DIR CKPT_DIR LOG_DIR

RUN_DIR[prott3_35m]="/home/zhuoyang/ProtT3/all_checkpoints/prott3_35m_protein_captioning_swiss_dataset"
CKPT_DIR[prott3_35m]="epoch=09.ckpt"
LOG_DIR[prott3_35m]="tb_logs/lightning_logs/version_0"

RUN_DIR[prott3_650m]="/home/zhuoyang/ProtT3/all_checkpoints/prott3_650m_protein_captioning_swiss_dataset"
CKPT_DIR[prott3_650m]="epoch=09-v1.ckpt"
LOG_DIR[prott3_650m]="eval_compare/last-v1_ckpt/lightning_logs/version_0"

RUN_DIR[sa_35m]="/data2/zhuoyang/ProtT3/all_checkpoints/stage2_sa_35m_protein_captioning"
CKPT_DIR[sa_35m]="epoch=09-v2.ckpt"
LOG_DIR[sa_35m]="tb_logs/lightning_logs/version_2"

RUN_DIR[sa_650m]="/data2/zhuoyang/ProtT3/all_checkpoints/stage2_sa_650m_protein_captioning"
CKPT_DIR[sa_650m]="epoch=09.ckpt"
LOG_DIR[sa_650m]="lightning_logs/version_0"

mkdir -p "$ROOT"

for name in prott3_35m prott3_650m sa_35m sa_650m; do
  dest="$ROOT/$name"
  src_run="${RUN_DIR[$name]}"
  ckpt="${CKPT_DIR[$name]}"
  logs="${LOG_DIR[$name]}"

  echo "==> $name"
  mkdir -p "$dest"

  if [[ ! -d "$src_run/$ckpt" ]]; then
    echo "ERROR: missing checkpoint $src_run/$ckpt" >&2
    exit 1
  fi

  if [[ -e "$dest/$ckpt" ]]; then
    echo "  skip checkpoint (exists): $dest/$ckpt"
  else
    echo "  copying checkpoint $ckpt ..."
    cp -a "$src_run/$ckpt" "$dest/$ckpt"
  fi

  # also copy matching last*.ckpt alias when present
  last_alias=""
  if [[ "$ckpt" == *-v* ]]; then
    ver="${ckpt#epoch=09-}"
    ver="${ver%.ckpt}"
    last_alias="last-${ver}.ckpt"
  else
    last_alias="last.ckpt"
  fi
  if [[ -d "$src_run/$last_alias" && ! -e "$dest/$last_alias" ]]; then
    echo "  copying alias $last_alias ..."
    cp -a "$src_run/$last_alias" "$dest/$last_alias"
  fi

  mkdir -p "$dest/lightning_logs/best"
  for f in metrics.csv dataset0_predictions.txt hparams.yaml; do
    if [[ -f "$src_run/$logs/$f" ]]; then
      cp -a "$src_run/$logs/$f" "$dest/lightning_logs/best/$f"
    fi
  done
  # backfill metrics from TB if missing
  if [[ ! -f "$dest/lightning_logs/best/metrics.csv" ]] && [[ -f "$src_run/$logs/events.out.tfevents."* ]]; then
    python3 scripts/tb_events_to_metrics_csv.py "$src_run/$logs" || true
    [[ -f "$src_run/$logs/metrics.csv" ]] && cp -a "$src_run/$logs/metrics.csv" "$dest/lightning_logs/best/metrics.csv"
  fi

  echo "  -> $dest ($ckpt)"
done

echo ""
echo "Re-scoring caption metrics from predictions (renames broken metrics.csv -> metrics_error.csv) ..."
python3 scripts/rescore_stage2_main_metrics.py

echo ""
echo "Done. Layout: $ROOT/{prott3_35m,prott3_650m,sa_35m,sa_650m}/"
