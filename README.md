# REF Gender Analysis
![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-GPLv3-green) ![Status](https://img.shields.io/badge/status-active-red) ![Plots](https://img.shields.io/badge/plots-matplotlib-orange) ![Data](https://img.shields.io/badge/data-REF%202021-9cf)

Analytical code and notebooks for studying gender representation in the UK REF (Research Excellence Framework) data, blending large-scale scientometric data from Dimensions with LLM-derived indicators. The repo produces cleaned datasets, descriptive figures, and regression outputs. If you like plots, tables, and a bit of stats, you’re in the right place.

## Repository Layout
- `src/make_figure_one.ipynb` — builds the four-panel gender comparison figure from cleaned data.
- `src/make_figure_two.ipynb` — plots % female authors across `llm_*` industries (overall + by REF panel).
- `src/make_table_one.ipynb` — prints a readable regression table and writes the LaTeX version.
- `src/build_regression_models.py` — fits the OLS/GLM models and serializes regression artifacts.
- `src/build_statistics.ipynb` — printable descriptive and inference statistics for ICS vs Outputs.
- `src/statistics_summary.py` — helpers used by the statistics notebook.
- `src/figure_two_llm.py` — helpers for the new Figure 2 showing llm_* female shares.
- `src/figure_one_data.py`, `src/figure_one_helpers.py`, `src/figure_one_plots.py` — helpers for data prep and plotting Figure 1.
- `src/figure_two_regression.py` — helpers for regression tables/LaTeX (used by Table 1).
- `src/make_enhanced_data.py`, `src/make_ref_staff.py` — data acquisition and enrichment utilities.
- `data/` — expected location for input CSV/XLSX files (not tracked here).
- `outputs/` — generated figures (`outputs/figures`) and tables (`outputs/tables`); regression artifacts in `outputs/models`.
- `requirements.txt` — Python dependencies.

## Prerequisites
- Python 3.9+.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Input data should be placed under `data/` (see paths referenced in the code, e.g., `data/final/enhanced_ref_data.csv`, `data/dimensions_outputs/outputs_concat_with_positive_authors.csv`).

## Data Preparation
- Run `src/make_enhanced_data.py` as needed to download and clean REF datasets (requires network access).
- Figure 1 notebooks/scripts expect cleaned/wrangled CSVs under `data/`:
  - `data/final/enhanced_ref_data.csv`
  - `data/dimensions_outputs/outputs_concat_with_positive_authors.csv`
  - Optional lookup: `data/manual/university_category/ref_unique_institutions.csv`

## Figure 1 (Gender Distributions)
1) Open and run `src/make_figure_one.ipynb`, or execute the helper pipeline manually:
   - `prepare_figure_one_data` to build `df_ics`, `df_uoa_m`, `df_uni_m`, `df_uniuoa_m`.
   - `save_wrangled` writes `data/wrangled/{uoa_gender,uni_gender,uniunoa_gender}.csv`.
   - `plot_figure_one` + `save_figure` write `outputs/figures/gender_output_ics_four_panel.{pdf,svg,png}`.

## Figure 2 (LLM Female Shares)
1) Run `src/make_figure_two.ipynb` to load llm_* indicators from `data/final/enhanced_ref_data.csv`.
2) The notebook writes `outputs/figures/regressions.{pdf,svg,png}` with:
   - Panel a: overall % female across llm_* industries.
   - Panel b: % female across llm_* industries split by REF panel.

## Regression Models and Table 1
1) Fit models and serialize artifacts:
   ```bash
   python -m src.build_regression_models
   ```
   - Produces `outputs/models/regression_results.pkl` containing:
     - `coef_df`, `var_order`, `latex_str`, `metrics_df`.
2) Table 1: run `src/make_table_one.ipynb` to print readable coefficients/metrics and write `outputs/tables/regression_results.tex`.

## Descriptive & Inference Statistics
- Run `src/build_statistics.ipynb` to print high-level summaries and one-sided tests comparing female shares in ICS vs outputs (uses `statistics_summary.py` helpers).

## Additional Utilities
- `src/make_ref_staff.py` — parses REF PDFs and enriches staff records (requires OpenAI key in `keys/OPENAI_API_KEY`).
- `src/make_enhanced_data.py` — end-to-end data download/clean steps for REF results, ICS, and outputs.

## Notes
- Outputs are written to `outputs/`; ensure the directory exists or is created by the scripts.
- Some scripts require network access (downloading REF data) and API keys (OpenAI) for text processing.
- If running notebooks in VS Code/Jupyter, reload the file after code changes to pick up the latest cells.
- Want the figures fast? Run `python -m src.build_regression_models` once, then open `src/make_figure_two.ipynb` to plot. For Figure 1, `src/make_figure_one.ipynb` handles both data prep and plotting.

## License
GNU GPLv3. See `LICENSE` for details.
