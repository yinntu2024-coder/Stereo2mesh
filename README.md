# PA-DSDB / Meta-Diff LaTeX Manuscript

This repository contains an IEEE TIP-style LaTeX draft for:

**Degradation-State Diffusion Bridges for Metalens Photography**

## Files

- `main.tex` — main IEEE-style manuscript draft, including abstract, method, theoretical analysis, TikZ idea figures, training objective, inference algorithm, experiment protocol, tables, and appendices.
- `refs.bib` — starter BibTeX database for diffusion and image-quality references.
- `figures/fig1_pipeline.svg` — polished standalone vector version of the PA-DSDB pipeline overview, matching the Fig. 1 image2 prompt.
- `figures/image2_prompts.md` — optional image2/gpt-image-2 prompt pack for generating polished bitmap alternatives to the built-in TikZ figures.

## Local Compilation

Recommended command:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current execution environment does not include a TeX engine, so PDF compilation must be performed on a local machine or an environment with TeX Live/MiKTeX installed.

## Writing Notes

- Do not fill quantitative tables with estimated numbers. Replace table dashes only after real experiments are completed.
- Keep the method story centered on: clean-domain anchor, fixed forward optical uncertainty prior, dynamic reverse degradation state, explicit measurement consistency, and theory-backed bridge derivations.
- The manuscript includes multiple schematic TikZ figures that compile without external image assets: pipeline overview, concept illustration, bridge geometry, network architecture, and experiment-evidence panels. A polished standalone SVG for Fig. 1 is also provided under `figures/`. Replace schematic experiment curves and `measured` table entries with real experimental outputs before submission.
- For an IEEE TIP submission, update author affiliation, acknowledgments, real citations, dataset details, and all experiment values before submission.
