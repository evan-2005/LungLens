"""
Collect eval JSONs from runs/lungattn into one Markdown comparison table.

Usage (repository root):
    python -m lung_attention.summarise_runs --split val --out runs/lungattn/summary_val.md
"""
import argparse
import glob
import json
import os

COLUMNS = (
    ("Run", None),
    ("Acc %", "accuracy"),
    ("Macro-F1 %", "macro_f1"),
    ("Pneu. recall %", "recall.Pneumonia"),
    ("GT-lung: region mostly outside %", "heatmap_gt_lung.region_majority_outside_frac"),
    ("GT-lung: region in lung (median)", "heatmap_gt_lung.region_in_lung_median"),
    ("GT-lung: no region %", "heatmap_gt_lung.no_region_frac"),
    ("Pred-lung: region mostly outside %", "heatmap_pred_lung.region_majority_outside_frac"),
    ("Worst source acc %", "worst_source"),
)
PERCENT_FRACTIONS = {"heatmap_gt_lung.region_majority_outside_frac", "heatmap_gt_lung.no_region_frac",
                     "heatmap_pred_lung.region_majority_outside_frac"}


def lookup(result, key):
    if key == "worst_source":
        accs = [v["acc"] for v in result.get("per_source", {}).values()]
        return min(accs) if accs else None
    node = result
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def fmt(value, key):
    if value is None:
        return "n/a"
    if key in PERCENT_FRACTIONS:
        value = 100.0 * value
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def table(named_results):
    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [head, rule]
    for name, res in named_results:
        cells = [name] + [fmt(lookup(res, k), k) for _, k in COLUMNS[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", default=os.path.join("runs", "lungattn"))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    found = sorted(glob.glob(os.path.join(args.runs, "**", f"eval_{args.split}.json"), recursive=True))
    if not found:
        raise SystemExit(f"No eval_{args.split}.json files under {args.runs}")
    named = []
    for path in found:
        with open(path, encoding="utf-8") as fh:
            named.append((os.path.basename(os.path.dirname(path)), json.load(fh)))
    md = (f"# E1 results ({args.split} split)\n\n"
          "GT-lung columns use the Radiography Database's own lung masks (independent of training); "
          "Pred-lung columns use the lung U-Net that also supervises E1, so they flatter E1.\n\n"
          + table(named) + "\n")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
