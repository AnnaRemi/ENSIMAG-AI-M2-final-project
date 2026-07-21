# EDBT 2027 Paper Project

This directory is based on the official EDBT 2027 A4 LaTeX template linked
from the research-track call for papers. The project is configured as a regular
research paper because it proposes a new execution strategy.

## Build

Use either:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Upload the entire directory to Overleaf if compiling online. Do not replace
`acmart.cls`, `edbt-macros.tex`, or `ACM-Reference-Format.bst` with a different
version, and do not change the class options or geometry.

## EDBT 2027 submission checklist

- [ ] Use A4 and the supplied EDBT template without formatting changes.
- [ ] Keep all non-reference content, including appendices, within 12 pages.
- [ ] Keep author names and affiliations: reviewing is single-anonymous.
- [ ] Make the PDF author list identical to the author list in CMT.
- [ ] Register every author in CMT and provide each author's ORCID.
- [ ] Keep the title in title case.
- [ ] Submit one printable PDF through CMT.
- [ ] Keep the section named `Artifacts` immediately before the references.
- [ ] Submit code, data, and experimental artifacts as a ZIP or an
      access-unmonitored repository link.
- [ ] Confirm that the paper is not simultaneously under review elsewhere.
- [ ] Check the 12-month EDBT resubmission restriction if applicable.
- [ ] Complete the TODO items in `main.tex`.
- [ ] Inspect the final PDF for overfull boxes, missing references, embedded
      fonts, page count, and accidental local paths.

The third-cycle paper deadline is 7 October 2026 at 5 p.m. PST. The call lists
10 February 2027 as the corresponding camera-ready deadline.

## Current evidence used in the draft

The results table transcribes the same-run aggregate reported in the workspace
root README for the ten-query `gemma4:e2b` to `gemma4:e4b` comparison. It shows
a strong latency and call-count improvement, but lower recall and F1 than SUQL.
The draft therefore does not claim that overall answer quality is preserved.
Re-check every value against the final frozen experiment before submission.
