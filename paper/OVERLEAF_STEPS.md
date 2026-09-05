# Opening this paper in Overleaf

The whole project is packaged as **`LungLens_paper_overleaf.zip`**. It compiles
as-is (verified: extracted to a clean folder and rebuilt from scratch with
`pdflatex` + `bibtex`, 14 pages, no errors) and already uses the **official
NeurIPS 2026 style file** (not a placeholder — see step 2). You do not need to
install anything locally to edit it — Overleaf compiles in the browser.

## 1. Upload it

1. Go to <https://www.overleaf.com> and log in (or create a free account).
2. Click **New Project -> Upload Project** (on the dashboard, top left).
3. Choose `LungLens_paper_overleaf.zip` and upload it.
4. Overleaf unzips it into a new project and should detect `main.tex` as the
   main file automatically (it's the only `.tex` file at the project root).
   If it opens a different file as the "main" document, click the menu icon
   (top left, next to the project name) -> **Settings** -> set **Main document**
   to `main.tex`.
5. Click **Recompile** (green button, top of the PDF preview pane). You should
   get the same 14-page PDF, with a NeurIPS-style layout and line numbers down
   the left margin.

## 2. Style file (already the official one — nothing to swap)

`neurips_2026.sty` in this project **is the official NeurIPS 2026 file**
(2026-01-29 revision, downloaded verbatim from the venue's own template
source and diffed against it byte-for-byte — not a hand-written stand-in).
Earlier drafts of this project used a placeholder `.sty` I'd built by hand;
that has been replaced, so there is no swap step left to do here.

`main.tex` is currently set to target the **Med-Reasoner workshop** at
NeurIPS 2026:
```latex
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{Medical Reasoning with Vision Language Foundation Models (Med-Reasoner)}
```
If you instead need the main track, change that line back to
`\usepackage{neurips_2026}` and remove the `\workshoptitle{...}` line.

Note: in submission mode the PDF's footer always reads "Submitted to the
40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do
not distribute." — it does **not** print the workshop name until camera-ready
(`\usepackage[final]{neurips_2026}`). That's the official file's own
behaviour, not a bug in this project.

## 3. What's in the project

| File | What it is |
|---|---|
| `main.tex` | The manuscript. Edit this for text changes. |
| `main.bib` | The 30 references, in BibTeX format. Add new ones here; cite with `\citep{key}` / `\citet{key}`. |
| `checklist.tex` | The NeurIPS paper checklist, `\input` at the very end of `main.tex`. Edit answers here, not in `main.tex`. |
| `neurips_2026.sty` | The official venue style file (see step 2 above). |
| `figures/fig3.tex` + `fig3.pdf` | The system-overview diagram (Figure 1), hand-drawn in TikZ. `main.tex` includes the `.pdf`; if you edit `fig3.tex` you need to recompile it separately (see below) and re-upload `fig3.pdf`, since Overleaf does not compile standalone sub-documents automatically. |
| `figures/fig_dice_hist.pdf` | Dice histogram (Section 3.1). |
| `figures/fig_descriptors.pdf` | Region-descriptor bar charts (Appendix D). |
| `figures/fig_persource.pdf` | Per-source accuracy bar chart (Appendix B). |
| `figures/fig_qualitative.png` | The six-panel good/bad overlay figure (Appendix C). |

`main.tex` also references `figures/paired_summary.pdf`, which does not exist
yet — it renders an inline text placeholder via `\IfFileExists` until you drop
a real file in with that exact name.

## 4. Editing `figures/fig3.tex` (the system diagram)

This one file is not plain text you can tweak line-by-line and see instantly —
it is a **separate LaTeX document** that Overleaf does not recompile on its
own. Two ways to change it:

- **Small tweaks (recommended in Overleaf):** create a second project (or a
  scratch project) with just `fig3.tex`, set it as the main document, add
  `\usepackage{tikz}` etc. are already in the file, hit Recompile, then
  download the resulting PDF and re-upload it over `figures/fig3.pdf` in the
  main project.
- **Local edits:** if you have a LaTeX install, run `pdflatex fig3.tex` in the
  `figures/` folder and re-upload the new `fig3.pdf`.

Either way, `main.tex` itself never changes — it just does
`\includegraphics[width=\linewidth]{figures/fig3.pdf}`.

## 5. Checking the page count

The body (Introduction through Conclusion) must stay at or under **9 pages**;
references, the checklist, and the appendices do not count. After any edit,
recompile and check which page "References" starts on (Overleaf's PDF viewer
shows page numbers at the top). It currently starts on page 7 (Conclusion
and References both fit on the same page), so there is headroom.

## 6. Getting the PDF back out

Click the **Download PDF** button (top right of the PDF preview) any time, or
**Menu -> Download -> Source** to get a zip of the whole project back (useful
if you want to hand it to someone else, or move it back to a local git repo).
