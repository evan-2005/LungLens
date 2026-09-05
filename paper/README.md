# Paper: *The Summary Never Reads the Mask*

LaTeX source for the LungLens localisation-grounded report-faithfulness audit.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Manuscript. Body is **7 pages** (limit is 9; references, checklist, and appendices do not count). |
| `main.bib` | Bibliography, 30 entries, one per cited key. |
| `checklist.tex` | NeurIPS paper checklist, `\input` at the end of `main.tex`. Placeholder for the official `checklist.tex`; answers transfer directly. |
| `neurips_2026.sty` | **Compatibility placeholder**, built to match the documented layout (17 pt title between a 4 pt and a 1 pt rule, 5.5 x 9 in text block, 1.5 in left margin, 10/11 pt Times, line numbers in submission mode, sentence-case bold headings, hidden `\begin{ack}`). Replace with the official file from <https://neurips.cc> before submission. |
| `figures/` | System diagram, 3 data charts, and the qualitative-overlay figure; see `figures/README.md`. `paired_summary.pdf` (S0 vs S1) isn't committed yet -- `main.tex` renders an inline placeholder via `\IfFileExists`, so it still compiles. |
| `main.pdf` | Built output. 15 pages total; body is still 7 pages (references/checklist/appendices don't count). |
| `OVERLEAF_STEPS.md` | How to open this project in Overleaf and what to edit where. |

A ready-to-upload `LungLens_paper_overleaf.zip` (this directory, minus the
build log) is generated for Overleaf; see `OVERLEAF_STEPS.md`.

## Track

`main.tex` currently uses `\usepackage{neurips_2026}` (the shell default = Main
Track, double-blind, line numbers on). To target a workshop, uncomment
`\usepackage[dblblindworkshop]{neurips_2026}` and add `\workshoptitle{...}`.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

MiKTeX installs missing packages (`natbib`, `titlesec`, `lineno`, `comment`,
`tikz`, `microtype`, ...) on first run.

## Fact-check log

Every repository-dependent number was checked against the public code, the
served `chest_classifier_metrics.json`, and the analysis outputs in
`figures/RESULTS_FOR_PAPER.md` / `fig7_work/`. Corrections applied in this pass:

- **Per-class recall range** was "92.2% (pneumonia) to 98.5% (tuberculosis)".
  98.5% is Normal's recall; tuberculosis recall is 95.1% (from the confusion
  matrix). Now reads "to 98.5% (Normal)".
- **Latency** was a table captioned "N=11 with the first excluded". The per-path
  figures (293 / 560 / 318 ms) come from a 40-request run split by overlay path;
  the "N=11" runs were a separate check. The table is now one prose sentence
  with the per-path means and the 91% / 9% test-set serving split.

Verified unchanged: 96.18% accuracy, 95.77 macro-F1, the confusion matrix,
per-class precision/recall/F1/specificity (all consistent with the matrix),
7,705,224 U-Net parameters, median Dice 0.22 (27% zero overlap, 15% > 0.5),
in-lung gate 92/245, H5 grounding 14/92, the 5.9 / 8.9 / 4.0 grey-level probe,
per-source 83.3%–99.5%, and the 22,419/4,785/4,796 -> 22,420/4,759/4,821 /
96.62% -> 98.11% reproducibility incident (commit `7a2cd81`).

## Figures and tables added this pass

Section 3.1 gained a Dice histogram; Appendix B gained a per-source accuracy
chart; Appendix C gained the six-panel qualitative overlay figure; Appendix D
gained a region-descriptor (zone/focality) bar chart; Appendix E is new, a
per-path latency table. All four charts are generated from verified files
already in the repo (see `figures/README.md` for exactly which file backs
each one); the zone/focality counts were independently recomputed from the
raw mask arrays this session and matched the previously-logged numbers
exactly, which is now a second confirmation of those two descriptors.

Figure 1 (the system diagram) had two genuine layout overlaps -- the
Grad-CAM-layer arrow label touching the cached-target box, and the
label+confidence arrow cutting through the "S0--S4" tag -- both fixed by
widening the affected gap and re-routing the arrow to a different anchor;
neither was a placeholder-vs-final issue, just tight spacing.

## Known repo-hygiene items (not paper issues)

- Repo `README.md` documents the Stage-2 loss as
  `CrossEntropyLoss + 2.0 * DiceBCELoss`; no code path runs that (it is
  `DiceBCELoss()` alone).
- Repo `README.md`'s confusion matrix does not match
  `chest_classifier_metrics.json`; the paper uses the JSON matrix.
