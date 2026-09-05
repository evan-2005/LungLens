# figures/

| File | Content | Referenced as |
| --- | --- | --- |
| `fig3.tex` / `fig3.pdf` | System diagram: Stage 1 (classifier training), Stage 2 (explanation amortisation, drawn as four distinct things: frozen teacher, stop-gradient, cached CAM target, trainable U-Net student), and Serving (both inference paths converging on the report generator with labelled incoming arrows). Standalone TikZ, `\includegraphics[width=\linewidth]{figures/fig3.pdf}` from `main.tex`. Rebuild with `pdflatex fig3.tex` inside this directory. |
| `paired_summary.pdf` | Paired S0-vs-S1 summary comparison for one film. Not committed yet; `main.tex` renders an inline placeholder via `\IfFileExists` (the S0/S1 text reconstructed from the verbatim template in Appendix C plus the film's measured descriptors) until a real PDF is dropped in here. |

## `fig3.tex` design notes

- Okabe-Ito colour-blind-safe palette, one colour per role (data/image = sky
  blue, trainable = bluish green, frozen = neutral grey, derived artefact =
  orange, generator/text = reddish purple, loss/gate/stop-grad = vermillion),
  stated in an on-figure key.
- Redundant encoding so it survives greyscale printing: frozen is dashed,
  trainable has a heavier border, losses are ellipses, gates are diamonds,
  everything else is a rounded rectangle; forward flow is solid, supervision
  is dashed, the Grad-CAM fallback branch is dotted.
- The small concentric blue/yellow/red glyph beside "mask + overlay" is a
  drawn (vector) heatmap icon, not a raster image.
- Two dashed vermillion tags mark the paper's experimental switch points:
  H0--H5 enters at the mask, S0--S4 enters at the report generator.
- Measured at 5.51 x 3.87 in (target was 5.5 x 3.7 in; width is on target,
  height is about 4.7% over after several compression passes -- three bands
  of this much detail don't fit 3.7in without either dropping content or
  going below the type-size floor, so height was kept over the target
  instead). Base font is 6.3pt; legend and the smallest annotation notes run
  4.6-5pt, below the nominal 7pt minimum for the same reason. Verified in
  greyscale (shape/border redundancy holds) and against a rough deuteranopia
  approximation (artefact-orange and loss-vermillion become close, but
  shape -- rounded rectangle vs ellipse/diamond -- keeps them apart); the
  approximation is a simple RGB matrix, not a validated CVD simulator.

The repo root also has `figures/` (PNG report figures + `Fig7_localisation_table.csv`,
`RESULTS_FOR_PAPER.md`) from the analysis scripts; that directory is separate
from this one.
