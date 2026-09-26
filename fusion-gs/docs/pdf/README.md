# Building the teaching-report PDF

Source: `fusion_gs_report.tex` in this directory. Figures are read from `../figures/` and from `figures/` (the extracted elongation frames).

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode fusion_gs_report.tex
cp fusion_gs_report.pdf ../REPORT.pdf
```

The reader-facing copy is `docs/REPORT.pdf`. Auxiliary files (`*.aux`, `*.log`, `*.toc`, and the rest listed in `.gitignore`) are build products.
