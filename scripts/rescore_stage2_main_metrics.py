#!/usr/bin/env python3
"""Rename broken training metrics.csv and write corrected caption scores from predictions."""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from model.blip2_stage2 import evaluate_exact_match
from model.help_funcs import caption_evaluate, flatten_caption_field

ROOT = Path(__file__).resolve().parents[1] / "all_checkpoints" / "stage2_main"
MODELS = ("prott3_35m", "prott3_650m", "sa_35m", "sa_650m")
TEXT_TRUNC_LENGTH = 128

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


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        "facebook/galactica-1.3b", use_fast=False, padding_side="right"
    )
    tok.add_special_tokens({"pad_token": "<pad>"})
    tok.add_special_tokens({"bos_token": "[DEC]"})
    return tok


def score_predictions(pred_path: Path, tokenizer):
    rows = []
    with pred_path.open(encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    preds = flatten_caption_field(rows, "predictions")
    tgts = flatten_caption_field(rows, "targets")
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor = caption_evaluate(
        preds, tgts, tokenizer, TEXT_TRUNC_LENGTH
    )
    acc = evaluate_exact_match(preds, tgts)
    return {
        "n_samples": len(preds),
        "dataset0/acc": acc,
        "dataset0/bleu2": round(bleu2, 2),
        "dataset0/bleu4": round(bleu4, 2),
        "dataset0/meteor_score": round(meteor, 2),
        "dataset0/rouge_1": round(rouge_1, 2),
        "dataset0/rouge_2": round(rouge_2, 2),
        "dataset0/rouge_l": round(rouge_l, 2),
    }


def has_caption_metrics(row: dict) -> bool:
    val = row.get("dataset0/bleu4", "")
    return str(val).strip() != ""


def patch_metrics_csv(src: Path, dst: Path, scores: dict) -> int:
    """Copy training history and replace broken caption rows with correct scores."""
    with src.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or COLUMNS
        rows = list(reader)

    patched = 0
    for row in rows:
        if has_caption_metrics(row):
            for key in scores:
                if key.startswith("dataset0/"):
                    row[key] = scores[key]
            patched += 1

    if patched == 0:
        row = {col: "" for col in fieldnames}
        for key, value in scores.items():
            if key in row:
                row[key] = value
        row["epoch"] = rows[-1]["epoch"] if rows else "0"
        row["step"] = rows[-1]["step"] if rows else "0"
        rows.append(row)
        patched = 1

    with dst.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return patched


def update_best_score(model_dir: Path, scores: dict):
    best_path = model_dir / "BEST_SCORE.json"
    data = {}
    if best_path.exists():
        data = json.loads(best_path.read_text(encoding="utf8"))
    data.update(
        {
            "acc": scores["dataset0/acc"],
            "bleu4": scores["dataset0/bleu4"],
            "bleu2": scores["dataset0/bleu2"],
            "meteor": scores["dataset0/meteor_score"],
            "rouge_1": scores["dataset0/rouge_1"],
            "rouge_2": scores["dataset0/rouge_2"],
            "rouge_l": scores["dataset0/rouge_l"],
            "n_samples": scores["n_samples"],
            "note": (
                "Caption metrics re-scored from dataset0_predictions.txt "
                f"(text_trunc_length={TEXT_TRUNC_LENGTH}). "
                "Original training-time metrics.csv saved as metrics_error.csv."
            ),
        }
    )
    best_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")


def process_model(name: str, tokenizer):
    log_dir = ROOT / name / "lightning_logs" / "best"
    metrics_path = log_dir / "metrics.csv"
    error_path = log_dir / "metrics_error.csv"
    pred_path = log_dir / "dataset0_predictions.txt"

    if not pred_path.exists():
        raise FileNotFoundError(f"missing predictions: {pred_path}")

    print(f"==> {name}")
    scores = score_predictions(pred_path, tokenizer)
    print(
        f"    n={scores['n_samples']} acc={scores['dataset0/acc']}% "
        f"bleu4={scores['dataset0/bleu4']}"
    )

    if metrics_path.exists() and not error_path.exists():
        shutil.move(str(metrics_path), str(error_path))
        print(f"    renamed metrics.csv -> metrics_error.csv")
    elif metrics_path.exists() and error_path.exists():
        metrics_path.unlink()
        print(f"    removed stale metrics.csv (metrics_error.csv already present)")

    if error_path.exists():
        patched = patch_metrics_csv(error_path, metrics_path, scores)
        print(f"    wrote metrics.csv ({patched} caption row(s) corrected)")
    else:
        with metrics_path.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            row = {col: "" for col in COLUMNS}
            row.update(scores)
            row["epoch"] = "0"
            row["step"] = "0"
            writer.writerow(row)
        print(f"    wrote metrics.csv (new eval-only row)")

    update_best_score(ROOT / name, scores)


def main():
    tokenizer = load_tokenizer()
    for name in MODELS:
        process_model(name, tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()
