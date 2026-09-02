# figures/

`main.tex` references two figure files here:

| File | Content | Referenced as |
| --- | --- | --- |
| `fig3_revised.pdf` | Revised system diagram (Stage 2 as four blocks; report-generation block with its real inputs; dashed no-feed line from the mask). | `Figure~\ref{fig:fig3}` |
| `paired_summary.pdf` | Paired S0-vs-S1 summary comparison for one film (COVID-3348). | `Figure~\ref{fig:paired}` |

Neither is committed yet. Until a real PDF is dropped in, `main.tex` renders an
inline placeholder for each via `\IfFileExists`, so the document still compiles.
The placeholders are faithful to the captions (the S0/S1 text is reconstructed
from the verbatim template in Appendix C plus the film's measured descriptors),
but replace them with the final artwork before submission.

The repo root also has `figures/` (PNG report figures + `Fig7_localisation_table.csv`,
`RESULTS_FOR_PAPER.md`) from the analysis scripts; that directory is separate
from this one.
