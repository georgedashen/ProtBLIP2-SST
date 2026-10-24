#!/usr/bin/env python3
"""Export cross-size win cases (SaProt or ProtT3) from aligned stage2_main predictions."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1] / "all_checkpoints" / "stage2_main"
MODELS = ("prott3_35m", "prott3_650m", "sa_35m", "sa_650m")

COLUMNS = [
    "index",
    "target_lead_field",
    "sa_650m_exact",
    "prott3_650m_exact",
    "sa_35m_exact",
    "prott3_35m_exact",
    "sa_650m_token_f1",
    "prott3_650m_token_f1",
    "sa_35m_token_f1",
    "prott3_35m_token_f1",
    "gap650_f1",
    "gap35_f1",
    "sa_650m_lead_field",
    "prott3_650m_lead_field",
    "sa_35m_lead_field",
    "prott3_35m_lead_field",
    "target",
    "sa_650m_prediction",
    "prott3_650m_prediction",
    "sa_35m_prediction",
    "prott3_35m_prediction",
]


def load_preds(name):
    by_idx = {}
    path = ROOT / name / "lightning_logs/best" / "dataset0_predictions.txt"
    with path.open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_idx[r["indices"]] = r
    return by_idx


def exact(pred, tgt):
    return pred.strip() == tgt.strip()


def token_f1(pred, tgt):
    pt = set(pred.lower().split())
    tt = set(tgt.lower().split())
    if not pt or not tt:
        return 0.0
    inter = len(pt & tt)
    prec, rec = inter / len(pt), inter / len(tt)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def lead_field(text):
    s = text.strip()
    return s.split(":", 1)[0].split()[0] if s else ""


def build_rows():
    data = {m: load_preds(m) for m in MODELS}
    common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))

    rows = []
    for idx in common:
        tgt = data["prott3_650m"][idx]["targets"]
        r = {"index": idx, "target": tgt, "target_lead_field": lead_field(tgt)}
        for m in MODELS:
            pred = data[m][idx]["predictions"]
            r[f"{m}_prediction"] = pred
            r[f"{m}_exact"] = exact(pred, tgt)
            r[f"{m}_token_f1"] = round(token_f1(pred, tgt), 4)
            r[f"{m}_lead_field"] = lead_field(pred)
        rows.append(r)
    return rows


def filter_sa_win(rows):
    out = [
        r
        for r in rows
        if r["sa_650m_exact"]
        and not r["prott3_650m_exact"]
        and r["sa_35m_token_f1"] >= 0.8
        and r["prott3_35m_token_f1"] <= r["sa_35m_token_f1"] - 0.15
    ]
    for r in out:
        r["gap650_f1"] = round(r["sa_650m_token_f1"] - r["prott3_650m_token_f1"], 4)
        r["gap35_f1"] = round(r["sa_35m_token_f1"] - r["prott3_35m_token_f1"], 4)
    out.sort(key=lambda r: (r["gap650_f1"], r["gap35_f1"]), reverse=True)
    return out


def filter_prott3_win(rows):
    out = [
        r
        for r in rows
        if r["prott3_650m_exact"]
        and not r["sa_650m_exact"]
        and r["prott3_35m_token_f1"] >= 0.8
        and r["sa_35m_token_f1"] <= r["prott3_35m_token_f1"] - 0.15
    ]
    for r in out:
        r["gap650_f1"] = round(r["prott3_650m_token_f1"] - r["sa_650m_token_f1"], 4)
        r["gap35_f1"] = round(r["prott3_35m_token_f1"] - r["sa_35m_token_f1"], 4)
    out.sort(key=lambda r: (r["gap650_f1"], r["gap35_f1"]), reverse=True)
    return out


def write_tsv(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in COLUMNS})


def main():
    rows = build_rows()

    sa_out = ROOT / "sa_win_cross_size_cases.tsv"
    sa_rows = filter_sa_win(rows)
    write_tsv(sa_out, sa_rows)
    print(f"wrote {len(sa_rows)} rows -> {sa_out}")

    pt_out = ROOT / "prott3_win_cross_size_cases.tsv"
    pt_rows = filter_prott3_win(rows)
    write_tsv(pt_out, pt_rows)
    print(f"wrote {len(pt_rows)} rows -> {pt_out}")


if __name__ == "__main__":
    main()
