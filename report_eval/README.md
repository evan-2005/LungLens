# report_eval: does the summary read the heatmap?

Evaluation harness for the paper's central question. It runs several report generators on the 245 scored test films, gives each one different heatmaps, and measures whether the location it states matches the heatmap. Design and paper placement are in `paper/PAPER_PLAN.md`, Sections 4, 5 and 6.

## Generators

| ID | What it is | Location comes from | Checked |
| --- | --- | --- | --- |
| S0 | The deployed template, rebuilt verbatim from `app.py` | fixed text per class | n/a |
| S1 | Descriptor template: side, zone, extent, focality | heatmap descriptors, gated | by construction |
| S2 | Text LLM, **constrained**: JSON schema, then a checker verifies every field, every location in its sentence, and that it never calls the heatmap region a finding. A failed reply falls back to S1 and is counted as rejected. | heatmap descriptors | yes |
| S3 | VLM, radiograph only | the model's own reading | parsed from free text |
| S4 | VLM, radiograph plus heatmap overlay | the model's reading of image and overlay | parsed from free text |

## Heatmap conditions

`actual` (the served heatmap), `none` (no heatmap), `permuted` (another film's heatmap under a fixed derangement; for S4 it is drawn onto this film's radiograph).

## Backends

- `--backend ollama --model <name>`: local, default choice, nothing leaves the machine.
- `--backend claude`: Anthropic SDK, default model `claude-opus-5` with server-side refusal fallbacks on (`--no-fallbacks` to disable). Needs `pip install anthropic` and credentials in the environment (`ANTHROPIC_API_KEY` or `ant auth login`).

## Run

From the repository root:

```bash
python -m unittest discover -s report_eval/tests -t .
```

```bash
python -m report_eval.run_grid --generators s0,s1 --out runs/report_eval/templates.jsonl
```

```bash
python -m report_eval.run_grid --generators s2 --backend ollama --model qwen2.5-coder:7b --sample 20 --out runs/report_eval/s2_pilot.jsonl
```

Every output is appended to the JSONL as it is produced (prompt reply, parsed claim, scores, latency, model); re-running skips finished work. A `_summary.md` table is written next to it.

## Scoring caveat

"Truth" is the film's **served heatmap region**, not pathology. These numbers measure whether the text agrees with the evidence shown beside it. The claim parser for free text (S3/S4) is rule-based and must be validated by hand on about 40 outputs before S3/S4 numbers are reported.

## Findings so far (2026-09-25)

| Result | Value |
| --- | --- |
| S1 claims, served heatmap | 92 / 245 (37.6%), reproduces the paper exactly |
| S0 phantom-outline claims | 2 / 245, reproduces the paper exactly |
| S1 grounding under the permuted control | mean **12.0%** over 1,000 derangements (95% range 6.5% to 18.5%). The paper's single-draw 15.2% is inside this range; report the distribution instead. |
| S0 location claims | **10 / 245**: every confident Covid-19 prediction prints the fixed word "bilateral", whatever the heatmap shows (6 of 10 happen to match). The paper's "S0 makes no spatial claim" must become "S0's only spatial claim is hardcoded per class". |
| S2 pilot, 20 films, local `qwen2.5-coder:7b` | 100% grounded with the served heatmap, silent with none, 0% grounded when permuted; 0 rejections; about 4.4 s per summary on the laptop GPU |
| S2 prompt wording | With the first prompt, 9 of 14 location sentences (64%) called the heatmap region a finding ("abnormality", "opacity"), including on a film predicted Normal. After telling the model the heatmap is not a finding, 0 of 14 did. The checker now rejects such sentences. |

Not run yet: S3/S4 (no vision model installed locally and no Claude credentials on this machine), the full 245-film S2 run, and the hand validation of the claim parser.
