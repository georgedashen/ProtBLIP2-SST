#!/usr/bin/env python3
"""Filter ProtT3 10k-test predictions to 9239 afdbfilt-aligned samples.

Maps each afdbfilt test index i to the matching entry in test_set.json (by caption),
then rewrites predictions so index i matches SaProt _saprot indices/targets.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_caption(text: str) -> str:
    return text.strip() + "\n"


def build_afdbfilt_mapping(data_root: Path):
    default = load_jsonl(data_root / "test_set.json")
    afdbfilt = load_jsonl(data_root / "test_set_afdbfilt.json")

    cap_to_default_idx = defaultdict(list)
    for idx, (_prot, caption) in enumerate(default):
        cap_to_default_idx[norm_caption(caption)].append(idx)

    mapping = []
    for af_idx, (_prot, caption) in enumerate(afdbfilt):
        cap = norm_caption(caption)
        hits = cap_to_default_idx[cap]
        if len(hits) != 1:
            raise RuntimeError(
                f"afdbfilt index {af_idx}: expected 1 caption match, got {len(hits)}"
            )
        mapping.append((af_idx, hits[0], cap))

    return mapping, len(afdbfilt)


def load_predictions_by_index(pred_path: Path):
    by_idx = {}
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_idx[row["indices"]] = row
    return by_idx


def filter_predictions(pred_path: Path, mapping, dry_run: bool = False):
    backup_name = "dataset0_predictions_10000sample.txt"
    backup_path = pred_path.parent / backup_name

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    if backup_path.exists():
        print(f"  backup already exists: {backup_path}")
    elif not dry_run:
        shutil.move(pred_path, backup_path)
        print(f"  renamed -> {backup_path.name}")
    else:
        print(f"  would rename -> {backup_name}")

    src_path = backup_path if backup_path.exists() else pred_path
    preds_by_idx = load_predictions_by_index(src_path)

    out_rows = []
    missing = 0
    for af_idx, default_idx, target in mapping:
        src = preds_by_idx.get(default_idx)
        if src is None:
            missing += 1
            continue
        out_rows.append(
            {
                "targets": target,
                "indices": af_idx,
                "predictions": src["predictions"],
            }
        )

    if missing:
        raise RuntimeError(f"missing {missing} source predictions in {src_path}")

    if not dry_run:
        with open(pred_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {len(out_rows)} lines -> {pred_path}")

    return len(out_rows)


def verify_against_saprot(filtered_path: Path, saprot_path: Path, n_check: int = 5):
    filtered = load_predictions_by_index(filtered_path) if filtered_path.exists() else {}
    saprot = load_predictions_by_index(saprot_path)
    common = sorted(set(filtered) & set(saprot))
    match = sum(1 for i in common if filtered[i]["targets"] == saprot[i]["targets"])
    print(f"  target match vs saprot: {match}/{len(common)}")
    for i in common[:n_check]:
        ok = filtered[i]["targets"] == saprot[i]["targets"]
        print(f"    idx {i}: {'OK' if ok else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Path to dataset0_predictions.txt (repeatable)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/SwissProtV3"),
    )
    parser.add_argument(
        "--saprot-pred",
        type=Path,
        default=None,
        help="Optional SaProt predictions for alignment verification",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping, n_af = build_afdbfilt_mapping(args.data_root)
    print(f"afdbfilt mapping ready: {n_af} samples")

    for pred in args.pred:
        pred_path = Path(pred)
        print(f"\n{pred_path}")
        n = filter_predictions(pred_path, mapping, dry_run=args.dry_run)
        if args.saprot_pred and not args.dry_run:
            verify_against_saprot(pred_path, args.saprot_pred)

    print("\nDone.")


if __name__ == "__main__":
    main()
