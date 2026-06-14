# DownstreamDOE — Monograph

A detailed, book-length treatment of the science inside the `DownstreamDOE`
package, in two chapters:

1. **Mechanistic Models of Preparative Chromatography** — the governing
   conservation laws, the transport-dispersive engine and its numerics, the
   unified adsorption-isotherm core, a mode-by-mode treatment (CEX/AEX, HIC,
   RP-HPLC) with, for each: the full mathematical formulation, every parameter
   and its units, the real-world separation it represents, its limits of
   validity, and the experiments/data used to estimate its parameters; and a
   **general rate model** (finite-volume / PyFVTool) that resolves the film and
   intraparticle-diffusion mass transfer and is mass-conservative where the
   lumped engine is not.
2. **The Methods of Experimental Design** — full factorial designs with
   response-surface/ANOVA analysis, Latin hypercube sampling, PCA/PLS
   latent-variable analysis, and Gaussian-process Bayesian optimisation, each
   with its theory, mathematical formulation, and a frank account of strengths
   and weaknesses.

Every chromatogram and design figure is generated **directly from the package
source** by `scripts/make_figures.py` — the plots are real numerical solutions
of the models in the text, not schematics.

## Layout

```
doc/
├── main.tex                 # master document (preamble + front/back matter)
├── chapters/
│   ├── preface.tex
│   ├── ch1_models.tex       # Chapter 1 — chromatography models
│   ├── ch2_design.tex       # Chapter 2 — experimental design methods
│   └── bibliography.tex
├── figures/                 # generated PNGs (git-ignored)
├── scripts/
│   ├── make_figures.py      # builds every figure from the package
│   ├── make_grm_figures.py  # GRM-vs-MoL comparison figures (called by make_figures)
│   ├── build_pdf.sh         # -> build/monograph.pdf   (pdflatex/latexmk)
│   ├── build_html.sh        # -> build/html/monograph.html (pandoc + MathJax)
│   └── monograph.css        # styling for the HTML build
├── Makefile
└── build/                   # all output (git-ignored)
```

## Building

From the `doc/` directory:

```bash
make            # figures + PDF + HTML
make pdf        # build/monograph.pdf
make html       # build/html/monograph.html
make figures    # just (re)generate the figures
```

Or call the scripts directly (they regenerate figures first; set `REGEN=0` to
skip that, and `PYTHON=...` to choose an interpreter):

```bash
bash scripts/build_pdf.sh
bash scripts/build_html.sh
```

## Requirements

- **PDF:** a TeX Live installation with `pdflatex` and `latexmk` (the document
  uses `amsmath`, `mathtools`, `booktabs`, `tcolorbox`, `hyperref`, `titlesec`,
  `fancyhdr`, `cleveref`, …).
- **HTML:** `pandoc` (≥ 3). Math is rendered client-side with MathJax.
- **Figures:** the project's Python environment (the package itself plus
  `matplotlib`, and `pyfvtool` for the general-rate-model figures); the scripts
  auto-detect `../.venv/bin/python`.

## Notes on the dual PDF/HTML source

The LaTeX source is written so a single set of files compiles richly with
`pdflatex` *and* converts cleanly with Pandoc:

- Custom commands are limited to **math macros** and **titled `tcolorbox`
  environments**; both survive the LaTeX→HTML conversion. Pandoc emits each
  call-out box as a `<div>` whose class matches the environment name and whose
  title text is preserved, so `monograph.css` can style them.
- Figures are plain `\includegraphics` (no conditional wrappers), so Pandoc
  embeds them.
- Cross-references use `\ref`/`\eqref`, which Pandoc resolves to numbered links.
