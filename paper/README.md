# Paper: *The Summary Never Reads the Mask*

LaTeX source for the LungLens localisation-grounded report-faithfulness audit.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Manuscript source. |
| `main.bib` | Bibliography (30 entries, one per cited key). |
| `neurips_2026.sty` | **Compatibility placeholder**, not the official venue style. See the header of that file. Replace with the official `neurips_2026.sty` before submission and recompile. |
| `figures/` | Figure PDFs. Two are referenced: `fig3_revised.pdf` (system diagram) and `paired_summary.pdf` (S0-vs-S1 comparison). If a file is absent, `main.tex` renders an inline placeholder via `\IfFileExists`, so the document still compiles. |
| `main.pdf` | Built output, committed for convenience. |

## Build

```bash
cd paper
latexmk -pdf main.tex        # preferred
# or, without latexmk:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

MiKTeX installs any missing packages (`natbib`, `titlesec`, `tikz`, `microtype`, ...) on first run.

## Status of the `\todo{}` markers

All `\todo{}` markers from the drafting passes are resolved:

- **Seg epochs = 20** and **Stage-2 loss = unweighted Dice+BCE alone**: confirmed against
  the public repo (`chest_classifier_metrics.json` byte-identical between commit `a25f54f`
  and HEAD; `app.py` `train_segmentation_head` uses `DiceBCELoss()` with no cross-entropy
  term). Stated as fact in Section 3.3.
- **`MultiTaskUNet` parameter count = 7,705,224**: from the served checkpoint's state dict.
- **RUN-04 / RUN-05** (per-film H1-vs-H3 pairing; per-stage latency split): not run;
  now written as scoped future-work sentences with no invented numbers.
- **Checklist**: expanded to a full answered list.
- **Dataset licenses**: the eight Kaggle IDs are named; per-dataset terms are left to be
  transcribed individually for the camera-ready (they are not uniform).

## Known repo-hygiene items (not paper issues)

- `README.md` in the repo root documents the Stage-2 loss as
  `CrossEntropyLoss + 2.0 * DiceBCELoss`, which does not match any code path that runs.
- `README.md`'s confusion-matrix table does not match `chest_classifier_metrics.json`;
  the paper uses the JSON matrix.
