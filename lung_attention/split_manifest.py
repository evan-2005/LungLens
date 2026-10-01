"""
Machine-independent train/val/test split manifest.

app.collect_dataset sorts absolute paths. Windows and Linux separators ('\\'
vs '/') sort differently against '_', '-' and '.', and stratified_subsample
depends on that order, so re-deriving the split on a Linux cloud machine would
pick a different 32,000-image subsample than the one behind the paper's
numbers. This module freezes a split as (split, source, dataset-relative
POSIX path, label) rows on the machine that produced the paper, and rebuilds
the exact same split anywhere by resolving those rows against the local
dataset roots.

Usage (repository root):
    # once, on the laptop whose split matches the paper
    python -m lung_attention.split_manifest export --out lung_attention/splits/split_s32000_seed42.csv
    # on any machine, after the datasets are downloaded
    python -m lung_attention.split_manifest verify --manifest lung_attention/splits/split_s32000_seed42.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault("LUNGLENS_SKIP_STARTUP", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIELDS = ("split", "source", "relpath", "label")
SPLITS = ("train", "val", "test")


def relative_posix(path, root):
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    if rel.startswith(".."):
        raise ValueError(f"{path} is not under dataset root {root}")
    return rel.replace(os.sep, "/")


def resolve(relpath, root):
    return os.path.join(root, *relpath.split("/"))


def dataset_roots():
    """Source tag -> local dataset root, matching the tags app.collect_dataset uses."""
    import app
    (tb, pn, cov, radio, shenzhen, montgomery, tbx11k, legrande) = app.get_dataset_paths()
    return {"tb_ds": tb, "pneu_ds": pn, "covid_ds": cov, "radiography_db": radio,
            "shenzhen_tb": shenzhen, "montgomery_tb": montgomery, "tbx11k": tbx11k,
            "legrande_tb": legrande,
            "custom": os.path.join(os.path.dirname(os.path.abspath(app.__file__)), "custom_dataset")}


def rows_from_split(split, roots):
    """split: {name: (paths, labels, sources)} -> list of manifest rows."""
    rows = []
    for name in SPLITS:
        paths, labels, sources = split[name]
        for p, lab, src in zip(paths, labels, sources):
            rows.append({"split": name, "source": src,
                         "relpath": relative_posix(p, roots[src]), "label": int(lab)})
    return rows


def fingerprint(rows):
    """Order-independent SHA-256 of split membership and labels."""
    keys = sorted(f"{r['split']}|{r['source']}|{r['relpath']}|{int(r['label'])}" for r in rows)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    meta = {"fingerprint": fingerprint(rows), "counts": counts}
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r, label=int(r["label"])) for r in csv.DictReader(fh)]


def load_split(path, roots=None):
    """
    Rebuild {name: (abs_paths, labels, sources)} from a manifest.

    Fails loudly if any file is missing, since a partial split would silently
    change every reported number.
    """
    rows = read_manifest(path)
    roots = roots or dataset_roots()
    out = {s: ([], [], []) for s in SPLITS}
    missing = []
    for r in rows:
        p = resolve(r["relpath"], roots[r["source"]])
        if not os.path.exists(p):
            missing.append(p)
            continue
        out[r["split"]][0].append(p)
        out[r["split"]][1].append(r["label"])
        out[r["split"]][2].append(r["source"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest images are missing locally (first: {missing[0]}). "
            "A dataset version probably differs from the one the manifest was built on.")
    return out


def _is_lung_opacity(path):
    return "Lung_Opacity" in os.path.normpath(path).split(os.sep)


def apply_split_options(split, drop_lung_opacity=False, holdout_source=None):
    """
    Variants of a frozen split, returned as new dicts (the input is not changed).

    drop_lung_opacity (#7): remove the Radiography Database's Lung_Opacity
        folder from every partition. 82 of the 102 pneumonia-as-Normal test
        errors come from it, and "lung opacity" is a broader finding than
        pneumonia. Compare models only on the same filtered test set.
    holdout_source (#9): leave-one-source-out. Train and validate on every
        other source; test on ALL images of the held-out source, from all
        three partitions, so the model never saw that acquisition pipeline.
    """
    def keep(name, pred):
        paths, labels, sources = split[name]
        rows = [(p, l, s) for p, l, s in zip(paths, labels, sources) if pred(p, s)]
        return tuple(list(col) for col in zip(*rows)) if rows else ([], [], [])

    out = dict(split)
    if drop_lung_opacity:
        out = {name: keep(name, lambda p, s: not _is_lung_opacity(p)) for name in SPLITS}
        split = out
    if holdout_source:
        all_sources = {s for name in SPLITS for s in split[name][2]}
        if holdout_source not in all_sources:
            raise ValueError(f"unknown source {holdout_source!r}; have {sorted(all_sources)}")
        held = [keep(name, lambda p, s: s == holdout_source) for name in SPLITS]
        out = {"train": keep("train", lambda p, s: s != holdout_source),
               "val": keep("val", lambda p, s: s != holdout_source),
               "test": tuple(sum((list(h[i]) for h in held), []) for i in range(3))}
    return out


def export(out, samples, seed=42):
    import app
    paths, labels, groups, sources = app.collect_dataset()
    paths, labels, groups, sources = app.stratified_subsample(
        paths, labels, groups, sources, samples, seed=seed)
    tr, va, te = app.patient_grouped_split(paths, labels, groups, sources, seed=seed)
    return write_manifest(rows_from_split({"train": tr, "val": va, "test": te}, dataset_roots()), out)


def main(argv=None):
    p = argparse.ArgumentParser(description="Export or verify a split manifest.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--samples", type=int, default=32000)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    args = p.parse_args(argv)

    if args.cmd == "export":
        print(json.dumps(export(args.out, args.samples), indent=2))
        return
    split = load_split(args.manifest)
    rows = read_manifest(args.manifest)
    with open(os.path.splitext(args.manifest)[0] + ".json", encoding="utf-8") as fh:
        expected = json.load(fh)["fingerprint"]
    ok = fingerprint(rows) == expected
    print(json.dumps({"counts": {s: len(split[s][0]) for s in SPLITS},
                      "fingerprint_matches_export": ok}, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
