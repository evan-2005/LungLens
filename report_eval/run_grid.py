"""
Run report generators across heatmap conditions on the 245 scored films.

Conditions
  actual     the film's own served heatmap
  none       no heatmap (H0)
  permuted   another film's heatmap under a fixed derangement (H5)
Generators
  s0 deployed template, s1 descriptor template   (no model needed)
  s2 constrained text LLM                         (needs --backend)
  s3 VLM, radiograph only                         (runs under "none" only)
  s4 VLM, radiograph + overlay                    (runs under "actual" and "permuted")
  s4i as s4, but sides asked in image coordinates and converted by the code

Every output is appended to a JSONL file as soon as it exists, and finished
(generator, condition, film) triples are skipped on re-run, so an interrupted
run resumes where it stopped. Model failures are recorded per film, not dropped.

Examples (repository root):
    python -m report_eval.run_grid --generators s0,s1 --out runs/report_eval/templates.jsonl
    python -m report_eval.run_grid --generators s2 --backend ollama --model qwen2.5:7b-instruct \
        --sample 20 --out runs/report_eval/s2_pilot.jsonl
    python -m report_eval.run_grid --generators s2,s3,s4 --backend claude \
        --out runs/report_eval/claude.jsonl
"""
import argparse
import dataclasses
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from report_eval.claims import claim_from_descriptors, parse_claim, response_kind  # noqa: E402
from report_eval.films import derangement, load_films  # noqa: E402
from report_eval.llm_backends import BackendError, ClaudeBackend, OllamaBackend  # noqa: E402
from report_eval.llm_generators import image_side_claim  # noqa: E402
from report_eval.score import score_output, summarise  # noqa: E402
from report_eval.templates import s0_template, s1_descriptor  # noqa: E402

CONDITIONS = ("actual", "none", "permuted")
GENERATOR_CONDITIONS = {"s0": CONDITIONS, "s1": CONDITIONS, "s2": CONDITIONS,
                        "s3": ("none",), "s4": ("actual", "permuted"),
                        "s4i": ("actual", "permuted")}
MODEL_GENERATORS = {"s2", "s3", "s4", "s4i"}
H5_SEEDS = 1000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--generators", default="s0,s1")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--backend", choices=["ollama", "claude"], default=None)
    p.add_argument("--model", default=None, help="model name for the backend")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--no-fallbacks", action="store_true",
                   help="Claude: disable server-side refusal fallbacks")
    p.add_argument("--effort", default=None, help="Claude: output_config.effort")
    p.add_argument("--sample", type=int, default=0, help="evenly spaced subset of films; 0 = all")
    p.add_argument("--seed", type=int, default=42, help="derangement seed for 'permuted'")
    p.add_argument("--out", required=True, help="JSONL file (appended to, resumable)")
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args(argv)
    args.generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = (set(args.generators) - set(GENERATOR_CONDITIONS)) | (set(args.conditions) - set(CONDITIONS))
    if unknown:
        p.error(f"unknown generator/condition: {sorted(unknown)}")
    if set(args.generators) & MODEL_GENERATORS and not args.backend and not args.summarise_only:
        p.error("s2/s3/s4 need --backend (ollama or claude)")
    if args.backend == "ollama" and not args.model:
        p.error("--backend ollama needs --model")
    return args


def make_backend(args):
    if args.backend == "ollama":
        return OllamaBackend(args.model, host=args.ollama_host)
    if args.backend == "claude":
        kw = {"fallbacks": not args.no_fallbacks, "effort": args.effort}
        if args.model:
            kw["model"] = args.model
        return ClaudeBackend(**kw)
    return None


def sample_films(films, n):
    if not n or n >= len(films):
        return films
    step = len(films) / float(n)
    return [films[int(i * step)] for i in range(n)]


def heatmap_source(film, condition, by_key, perm):
    """The film whose heatmap is shown, or None when no heatmap is given."""
    if condition == "none":
        return None
    return film if condition == "actual" else by_key[perm[film.key]]


def generate(gen, film, source, backend):
    """Returns (text, claim, extras) for one generator on one film."""
    pred = film.prediction
    d = source.descriptors if source is not None else None
    if gen == "s0":
        text = s0_template(pred)
        return text, parse_claim(text), {}
    if gen == "s1":
        return s1_descriptor(pred, d), claim_from_descriptors(d), {}
    from report_eval.llm_generators import s2_llm, s3_vlm, s4_vlm, s4i_vlm
    if gen == "s2":
        out = s2_llm(pred, d, backend)
    else:
        from report_eval.images import overlay_png, radiograph_png
        radio = radiograph_png(film)
        if gen == "s3":
            out = s3_vlm(pred, radio, backend)
        else:
            vlm = s4i_vlm if gen == "s4i" else s4_vlm
            out = vlm(pred, radio, overlay_png(film, source), backend)
    extras = {"model": out.model, "latency_s": round(out.latency_s, 3), "raw": out.raw,
              "rejected": out.rejected, "reject_reason": out.reject_reason}
    return out.text, out.claim, extras


def load_done(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(args, films, all_films, backend):
    """`films` is the (possibly sampled) set to run; heatmaps may come from any of `all_films`."""
    all_by_key = {f.key: f for f in all_films}
    perm = derangement(list(all_by_key), seed=args.seed)
    done = {(r["generator"], r["condition"], r["key"]) for r in load_done(args.out)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for gen in args.generators:
            for cond in (c for c in args.conditions if c in GENERATOR_CONDITIONS[gen]):
                todo = [f for f in films if (gen, cond, f.key) not in done]
                print(f"[run_grid] {gen} x {cond}: {len(todo)} to do", flush=True)
                for i, film in enumerate(todo, 1):
                    source = heatmap_source(film, cond, all_by_key, perm)
                    rec = {"generator": gen, "condition": cond, "key": film.key,
                           "heatmap_from": source.key if source else None, "error": None}
                    try:
                        text, claim, extras = generate(gen, film, source, backend)
                        rec.update(extras, text=text,
                                   claim=dataclasses.asdict(claim) if claim else None)
                        rec.update(score_output(claim, film.descriptors, text))
                    except BackendError as e:
                        rec["error"] = str(e)
                        print(f"  ERROR {film.key}: {e}", flush=True)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    if i % 25 == 0:
                        print(f"  {i}/{len(todo)}", flush=True)


def h5_distribution(films, seeds=H5_SEEDS):
    """S1 grounding precision under many derangements (templates only: free)."""
    by_key = {f.key: f for f in films}
    values = []
    for seed in range(seeds):
        perm = derangement(list(by_key), seed=seed)
        rows = [score_output(claim_from_descriptors(by_key[perm[f.key]].descriptors), f.descriptors)
                for f in films]
        values.append(summarise(rows)["grounding_precision"])
    values.sort()
    return {"seeds": seeds, "mean": statistics.mean(values),
            "p2_5": values[int(0.025 * seeds)], "p97_5": values[int(0.975 * seeds) - 1]}


def _fmt(v, pct=False):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{100 * v:.1f}%" if pct else f"{v:.2f}"
    return str(v)


def summary_table(records):
    groups = {}
    for r in records:
        groups.setdefault((r["generator"], r["condition"]), []).append(r)
    head = ("| Generator | Heatmap | Films | Errors | Claims | Claim rate | Grounding precision "
            "| Phantom claims | Outline phantoms | LLM rejected | Mean latency (s) |")
    lines = [head, "|" + " --- |" * 11]
    for (gen, cond), rows in sorted(groups.items()):
        ok = [r for r in rows if not r.get("error")]
        s = summarise(ok) if ok else {}
        lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
        rej = sum(1 for r in ok if r.get("rejected"))
        lines.append("| " + " | ".join([
            gen, cond, str(len(rows)), str(len(rows) - len(ok)), _fmt(s.get("claims")),
            _fmt(s.get("claim_rate"), True), _fmt(s.get("grounding_precision"), True),
            _fmt(s.get("phantom_claims")), _fmt(s.get("outline_phantoms")),
            f"{rej}" if gen == "s2" else "n/a",
            _fmt(statistics.mean(lat)) if lat else "n/a"]) + " |")
    return "\n".join(lines)


FREE_TEXT = {"s3", "s4", "s4i"}
MIRROR = {"right lung": "left lung", "left lung": "right lung"}


def rescore_free_text(records, all_films):
    """Re-parse S3/S4 text with the current parser, so parser fixes need no model re-run."""
    truth = {f.key: f.descriptors for f in all_films}
    out = []
    for r in records:
        if r["generator"] not in FREE_TEXT or r.get("error"):
            out.append(r)
            continue
        claim = (image_side_claim if r["generator"] == "s4i" else parse_claim)(r["text"])
        new = {**r, "claim": dataclasses.asdict(claim) if claim else None,
               "kind": response_kind(r["text"])}
        new.update(score_output(claim, truth[r["key"]], r["text"]))
        side, true_side = (claim.side if claim else None), truth[r["key"]].side
        new["single_side"] = side in MIRROR and true_side in MIRROR
        new["mirrored"] = new["single_side"] and MIRROR[side] == true_side
        out.append(new)
    return out


def vlm_table(records):
    rows = [r for r in records if r["generator"] in FREE_TEXT and not r.get("error")]
    if not rows:
        return ""
    groups = {}
    for r in rows:
        groups.setdefault((r["generator"], r["condition"]), []).append(r)
    lines = ["| Generator | Heatmap | Specific claim | Non-specific (\"throughout the lungs\") "
             "| Sees no evidence | Other | One-side claims: correct / mirrored |",
             "|" + " --- |" * 7]
    for (gen, cond), rs in sorted(groups.items()):
        n = len(rs)
        kinds = {k: sum(r["kind"] == k for r in rs)
                 for k in ("specific", "non_specific", "no_evidence", "other")}
        single = [r for r in rs if r["single_side"]]
        correct = sum(r["side_correct"] for r in single)
        mirrored = sum(r["mirrored"] for r in single)
        lines.append("| " + " | ".join(
            [gen, cond] + [f"{kinds[k]} ({100 * kinds[k] / n:.0f}%)" for k in kinds]
            + [f"{correct} / {mirrored} of {len(single)}"]) + " |")
    return "\n".join(lines)


def write_summary(args, all_films):
    records = rescore_free_text(load_done(args.out), all_films)
    if not records:
        raise SystemExit(f"No records in {args.out}")
    md = ["# Report-generator grid", "",
          "Truth = each film's served heatmap region. Claims are scored against the heatmap, "
          "not against pathology.", "", summary_table(records), ""]
    vlm = vlm_table(records)
    if vlm:
        md += ["## Free-text answers (S3/S4)", "",
               "S3/S4 claims are re-parsed from the saved text with the current parser. "
               "'Mirrored' = the model named the opposite single side to the heatmap's, the "
               "signature of describing image-left/right instead of the patient's.", "", vlm, ""]
    if "s1" in {r["generator"] for r in records}:
        h5 = h5_distribution(all_films)
        md += [f"S1 grounding precision under the permuted control, over {h5['seeds']} "
               f"derangements: mean {100 * h5['mean']:.1f}%, 95% of derangements between "
               f"{100 * h5['p2_5']:.1f}% and {100 * h5['p97_5']:.1f}%.", ""]
    text = "\n".join(md)
    with open(os.path.splitext(args.out)[0] + "_summary.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main(argv=None):
    args = parse_args(argv)
    all_films = load_films()
    if not args.summarise_only:
        run(args, sample_films(all_films, args.sample), all_films, make_backend(args))
    write_summary(args, all_films)


if __name__ == "__main__":
    main()
