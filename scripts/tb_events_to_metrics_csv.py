#!/usr/bin/env python3
"""Rebuild Lightning-style metrics.csv from TensorBoard event files."""
import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Match stage2 CSVLogger column order (captioning runs).
COLUMNS = [
    "dataloader0/val loss/dataloader_idx_0",
    "dataset0/acc",
    "dataset0/bleu2",
    "dataset0/bleu4",
    "dataset0/meteor_score",
    "dataset0/rouge_1",
    "dataset0/rouge_2",
    "dataset0/rouge_l",
    "epoch",
    "loss",
    "lr",
    "step",
]

SKIP_TAGS = {"hp_metric"}


def load_rows(log_dir: Path):
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    tags = [t for t in ea.Tags().get("scalars", []) if t not in SKIP_TAGS]
    by_step = {}
    for tag in tags:
        for ev in ea.Scalars(tag):
            step = int(ev.step)
            row = by_step.setdefault(step, {})
            row[tag] = float(ev.value)
            if tag == "epoch":
                row["epoch"] = float(ev.value)
    return by_step


def write_metrics_csv(log_dir: Path, overwrite: bool = False) -> bool:
    out = log_dir / "metrics.csv"
    if out.exists() and not overwrite:
        print(f"skip (exists): {out}")
        return False
    if not list(log_dir.glob("events.out.tfevents.*")):
        print(f"skip (no tfevents): {log_dir}")
        return False

    by_step = load_rows(log_dir)
    if not by_step:
        print(f"skip (no scalars): {log_dir}")
        return False

    rows = []
    for step in sorted(by_step):
        src = by_step[step]
        row = {col: "" for col in COLUMNS}
        row["step"] = step
        for col in COLUMNS:
            if col == "step":
                continue
            if col in src:
                row[col] = src[col]
        rows.append(row)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return True


def discover_tb_log_dirs(roots):
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pred in root.rglob("dataset0_predictions.txt"):
            log_dir = pred.parent
            if list(log_dir.glob("events.out.tfevents.*")):
                yield log_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="lightning_logs version dirs or run roots to scan",
    )
    parser.add_argument(
        "--scan-checkpoints",
        action="store_true",
        help="scan all_checkpoints under repo and /data2/.../all_checkpoints",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    targets = []
    if args.scan_checkpoints:
        repo = Path(__file__).resolve().parents[1]
        for base in [repo / "all_checkpoints", Path("/data2/zhuoyang/ProtT3/all_checkpoints")]:
            targets.extend(discover_tb_log_dirs([base]))
    for p in args.paths:
        path = Path(p)
        if path.is_dir() and list(path.glob("events.out.tfevents.*")):
            targets.append(path)
        else:
            targets.extend(discover_tb_log_dirs([path]))

    seen = set()
    n = 0
    for log_dir in sorted(set(targets)):
        key = str(log_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        metrics = log_dir / "metrics.csv"
        if metrics.exists() and not args.overwrite:
            continue
        if write_metrics_csv(log_dir, overwrite=args.overwrite):
            n += 1
    print(f"done: created/updated {n} metrics.csv file(s)")


if __name__ == "__main__":
    main()
