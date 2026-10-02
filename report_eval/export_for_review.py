"""
Export free-text VLM outputs with the parser's reading, for hand validation.

Each row shows what the model wrote and the side / zone / extent the rule-based
parser extracted. A reviewer fills the three `ok_*` columns (y / n) and an
optional note; summarise_review.py then reports the parser's accuracy, which
must accompany any S3/S4 number in the paper.

Usage (repository root):
    python -m report_eval.export_for_review --results runs/report_eval/vlm.jsonl \
        --n 40 --out runs/report_eval/parser_review.csv
"""
import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FREE_TEXT_GENERATORS = {"s3", "s4"}
COLUMNS = ["key", "generator", "condition", "text", "parsed_side", "parsed_zone",
           "parsed_extent", "ok_side", "ok_zone", "ok_extent", "note"]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rows_for_review(records, n, seed=0):
    pool = [r for r in records if r["generator"] in FREE_TEXT_GENERATORS and not r.get("error")]
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    out = []
    for r in sorted(picked, key=lambda r: (r["generator"], r["condition"], r["key"])):
        claim = r.get("claim") or {}
        out.append({"key": r["key"], "generator": r["generator"], "condition": r["condition"],
                    "text": " ".join(r.get("text", "").split()),
                    "parsed_side": claim.get("side") or "", "parsed_zone": claim.get("zone") or "",
                    "parsed_extent": claim.get("extent") or "",
                    "ok_side": "", "ok_zone": "", "ok_extent": "", "note": ""})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    rows = rows_for_review(load(args.results), args.n, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
