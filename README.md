# 384‑Well Potency Assay Web App

Interactive Dash application for designing and analyzing 384‑well cytotoxicity / viability potency assays in a single workflow.

The app is organized into four main tabs:

1. **Dilution Block** – serial dilution design and concentration series
2. **Reagent Prep** – antibody upload (up to 24), 384‑well plate layout, stock plate prep
3. **Cell Calc** – target and effector cell seeding math
4. **Data Analysis** – raw plate data handling, normalization, 4PL fitting, EC50 summaries

> Plate format: 384‑well, 2×2 quadruplicates, up to 12 antibodies per plate × 2 plates (24 total).

---

## Features

- Serial dilution designer with support for up to 6 proteins per dilution block and log‑spaced concentration series. [file:1]
- Automated plate mapping for 384‑well plates with 2×2 quadruplicate layouts and color‑coded antibodies. [file:1]
- Reagent preparation calculators (mg ⇄ nM conversion, stock/working concentration math). [file:1]
- Cell seeding calculator for cancer/effector cells with step‑by‑step volume and dilution instructions plus sanity‑check warnings. [file:1]
- Data ingestion from plate reader output (RLU/OD/etc.) and aggregation into Mean, SD, CV per concentration. [file:1]
- Normalization to controls, 4‑parameter logistic (4PL) model fitting, EC50 extraction, and R² quality metrics. [file:1]
- Multi‑panel dose–response plots (log‑scaled x‑axis) with markers, fitted curves, and EC50 highlight markers per antibody. [file:1]
- Built‑in simulation modes for:
  - Plate‑level raw signal (RLU‑like) data. [file:1]
  - Antibody‑level viability % data for positive/negative control profiles. [file:1]
- Modern Dash UI using `dash-ag-grid` tables and Plotly figures with compact layouts optimized for lab laptop screens. [file:1]

---

## Installation

Create and activate a Python environment (conda, venv, etc.), then install dependencies:

```bash
pip install dash dash-ag-grid plotly pandas openpyxl scipy numpy
