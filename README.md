# REF Gender Analysis (Journal Submission Repository)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)
![Status](https://img.shields.io/badge/status-active-red)
![Plots](https://img.shields.io/badge/plots-matplotlib-orange)
![Data](https://img.shields.io/badge/data-REF%202021-9cf)

This repository contains the full computational pipeline for analysing gender representation in UK REF 2021 Impact Case Studies (ICS) and research outputs, including:
- data construction from REF source files,
- staff-name extraction from ICS PDFs,
- output-author enrichment from Dimensions,
- thematic indicator construction (regex and LLM-based),
- regression modelling,
- publication-ready figures/tables,
- additional word-level text association analysis.

The project is script-first and intended for reproducible reruns from the command line.

## 1. Study Scope
The central empirical objective is to evaluate gender composition differences between:
- REF Impact Case Studies,
- REF-linked research outputs,

and to examine how institutional, panel, and thematic features relate to women’s representation.

The repository operationalises this with:
- case-level and aggregated gender counts,
- panel/UoA splits,
- thematic indicators (regex + LLM model variants),
- OLS and GLM model families,
- validation and agreement diagnostics.

## 2. Repository Structure

### 2.1 Core scripts (analysis pipeline)
- `src/step01_make_enhanced_data.py`
  - Builds canonical enhanced ICS dataset in `data/gold/enhanced_ref_data.{parquet,csv}`.
  - Can run with or without LLM thematic classification.
  - Supports model/prompt cache backfilling in `data/openai/categories.csv`.
- `src/step02_make_ref_staff.py`
  - Extracts staff blocks from ICS PDFs and derives case-level staff gender counts.
  - Writes outputs under `data/ics_staff_rows/`.
- `src/step03_get_dimensions_research_outputs.py`
  - Builds output-level author-gender tables using Dimensions API (or existing local chunks/outputs in offline mode).
  - Writes `outputs_concat_with_any_number_authors` and `outputs_concat_with_positive_authors` to `data/gold/`.
- `src/step04_make_figure_one.py`
  - Rebuilds Figure 1.
- `src/step05_build_regression_models.py`
  - Fits weighted OLS + GLM specifications and writes `outputs/models/regression_results.pkl`.
- `src/step06_make_figure_two.py`
  - Rebuilds Figure 2 and `supplementary_figure_2`.
- `src/step07_make_table_one.py`
  - Rebuilds Table 1 LaTeX (`outputs/tables/regression_results.tex`).
- `src/step08_build_statistics.py`
  - Rebuilds descriptive/inference report + summary LaTeX tables.
- `src/step09_evaluate_thematic_indicators.py`
  - Cross-method thematic comparison across Regex / GPT-5-mini / GPT-5.1 / GPT-5.4.
  - Produces `supplementary_figure_1`, `supplementary_figure_4`, and agreement/coverage tables.
- `src/step10_analyze_ics_text_gender.py`
  - Word-level association analysis for ICS text vs case-level female share outcome.

### 2.2 Shared modules
- `src/figure_one_*.py`, `src/figure_two_*.py`, `src/statistics_helpers.py`
- `src/pipeline_*.py` (config, paths, I/O, schema, drift checks)

### 2.3 Data and outputs
- `data/bronze/` raw/downloaded inputs
- `data/silver/` intermediate cleaned tables
- `data/gold/` canonical analysis-ready tables
- `outputs/figures/` publication figures
- `outputs/tables/` publication and diagnostic tables
- `outputs/models/` fitted model artifacts

## 3. Data Provenance

### 3.1 REF inputs
The pipeline consumes REF 2021 public workbooks (environment/results/ICS/ICS tags/outputs) via step01 download routines, with local persistence under `data/bronze/` (mirrored to legacy paths for compatibility).

### 3.2 ICS PDF text
Step02 downloads ICS PDFs from the REF website and extracts text/staff blocks.

### 3.3 Dimensions metadata
Step03 queries Dimensions publication metadata (unless `--skip-api`), and enriches output records with author-level gender counts.

### 3.4 LLM thematic indicators
Step01 can generate thematic indicator flags from ICS text and cache results in:
- `data/openai/categories.csv`

Cache keys are deterministic hashes of:
- prompt version,
- model,
- normalized ICS text.

## 4. Environment Setup

### 4.1 Python environment
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

If parquet read/write fails, ensure `pyarrow` is installed (already listed in requirements).

### 4.2 Optional API credentials
- OpenAI:
  - env var: `OPENAI_API_KEY`
  - file fallback: `keys/OPENAI_API_KEY`
- Dimensions:
  - env var: `DIMENSIONS_API_KEY`
  - file fallback: `keys/dimensions_apikey.txt`

### 4.3 Configuration
Runtime configuration is controlled by `pipeline.yaml`.
Key defaults include:
- OpenAI model: `gpt-5.4`
- OpenAI prompt version: `v2`
- OpenAI service tier: `flex`
- thematic batch size: `12`

## 5. Step-by-Step Pipeline Map

| Step | Script | External calls by default | Primary outputs |
|---|---|---|---|
| 01 | `step01_make_enhanced_data.py` | REF workbook downloads; OpenAI (if `--with-llm`) | `data/gold/enhanced_ref_data.*`, `data/openai/categories.csv` |
| 02 | `step02_make_ref_staff.py` | ICS PDF downloads; OpenAI (if `--with-llm`) | `data/ics_staff_rows/*.csv` |
| 03 | `step03_get_dimensions_research_outputs.py` | Dimensions API (unless `--skip-api`) | `data/gold/outputs_concat_*.{parquet,csv}` |
| 04 | `step04_make_figure_one.py` | none | `outputs/figures/figure_one.{pdf,svg,png}` |
| 05 | `step05_build_regression_models.py` | none | `outputs/models/regression_results.pkl` |
| 06 | `step06_make_figure_two.py` | none | `outputs/figures/figure_two.*`, `outputs/figures/supplementary_figure_2.*` |
| 07 | `step07_make_table_one.py` | none | `outputs/tables/regression_results.tex` |
| 08 | `step08_build_statistics.py` | none | `outputs/tables/statistics_report.txt`, summary `.tex` tables |
| 09 | `step09_evaluate_thematic_indicators.py` | none | `outputs/figures/supplementary_figure_1.*`, `outputs/figures/supplementary_figure_4.*`, thematic diagnostics `.csv` |
| 10 | `step10_analyze_ics_text_gender.py` | none | `outputs/figures/supplementary_figure_5.*`, step10 tables |

## 6. Reproducibility Workflows

### 6.1 Full refresh (uses external services)
Use when you explicitly want to refresh data/classifications from external sources:
```bash
python -m src.step01_make_enhanced_data --without-llm --force
python -m src.step02_make_ref_staff --with-llm
python -m src.step01_make_enhanced_data --with-llm --force
python -m src.step03_get_dimensions_research_outputs --force
python -m src.step04_make_figure_one
python -m src.step05_build_regression_models
python -m src.step06_make_figure_two
python -m src.step07_make_table_one
python -m src.step08_build_statistics
python -m src.step09_evaluate_thematic_indicators --model-mini gpt-5-nano --prompt-mini v2 --model-51 gpt-5.1 --prompt-51 v2 --model-54 gpt-5.4 --prompt-54 v2
python -m src.step10_analyze_ics_text_gender
```

### 6.2 Full rerun with no new API calls (local snapshot only)
Use when journal reproducibility requires regeneration from frozen local data/cache:
```bash
cd ~/Dropbox/ics_work/ref_gender && REF_SKIP_MANIFEST=1 PYTHONUNBUFFERED=1 bash -lc 'set -euo pipefail; ENH_PATH="$( [ -f data/gold/enhanced_ref_data.parquet ] && echo data/gold/enhanced_ref_data.parquet || { [ -f data/gold/enhanced_ref_data.csv ] && echo data/gold/enhanced_ref_data.csv || true; } )"; [ -n "$ENH_PATH" ] || { echo "Missing existing enhanced_ref_data in data/gold"; exit 1; }; printf "REF impact case study identifier\n" > /tmp/ref_gender_empty_ids.csv; python -m src.step01_make_enhanced_data --with-llm --output "$ENH_PATH"; python -m src.step02_make_ref_staff --without-llm --input /tmp/ref_gender_empty_ids.csv --out-dir /tmp/ref_gender_step02_noapi; python -m src.step03_get_dimensions_research_outputs --skip-api; python -m src.step04_make_figure_one; python -m src.step05_build_regression_models; python -m src.step06_make_figure_two; python -m src.step07_make_table_one; python -m src.step08_build_statistics; python -m src.step09_evaluate_thematic_indicators --model-mini gpt-5-nano --prompt-mini v2 --model-51 gpt-5.1 --prompt-51 v2 --model-54 gpt-5.4 --prompt-54 v2; python -m src.step10_analyze_ics_text_gender'
```

Notes:
- This mode regenerates outputs from current local artifacts.
- It does not refresh external datasets/classifications.
- In `step03 --skip-api`, existing-output contract mismatches are downgraded to warnings to allow offline continuation.

### 6.3 Downstream-only rerun (fast)
If `data/gold/` is already current:
```bash
python -m src.step04_make_figure_one
python -m src.step05_build_regression_models
python -m src.step06_make_figure_two
python -m src.step07_make_table_one
python -m src.step08_build_statistics
python -m src.step09_evaluate_thematic_indicators --model-mini gpt-5-nano --prompt-mini v2 --model-51 gpt-5.1 --prompt-51 v2 --model-54 gpt-5.4 --prompt-54 v2
python -m src.step10_analyze_ics_text_gender
```

## 7. Thematic Indicator Model Slices
Step09 expects four method families:
- Regex (`regex_*`)
- GPT-5-mini defaults (configured as `gpt-5-nano` + prompt `v2`)
- GPT-5.1 (`v2`)
- GPT-5.4 (`v2`)

If cache slices are incomplete, backfill per model:
```bash
python -m src.step01_make_enhanced_data --backfill-model gpt-5-nano --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 12 --backfill-prompt-cache-key thematic_indicators_v2
python -m src.step01_make_enhanced_data --backfill-model gpt-5.1   --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 12 --backfill-prompt-cache-key thematic_indicators_v2
python -m src.step01_make_enhanced_data --backfill-model gpt-5.4   --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 12 --backfill-prompt-cache-key thematic_indicators_v2
```
Then rebuild enhanced data:
```bash
python -m src.step01_make_enhanced_data --with-llm --force
```

## 8. Key Manuscript Artifacts

### 8.1 Figures
- `outputs/figures/figure_one.{pdf,svg,png}`
- `outputs/figures/figure_two.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_1.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_2.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_4.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_5.{pdf,svg,png}`

### 8.2 Tables
- `outputs/tables/regression_results.tex`
- `outputs/tables/panel_summary.tex`
- `outputs/tables/uoa_summary.tex`
- `outputs/tables/llm_summary.tex`
- `outputs/tables/llm_panel_summary.tex`
- `outputs/tables/thematic_model_health_checks.csv`
- `outputs/tables/thematic_pairwise_agreement_by_topic.csv`
- `outputs/tables/thematic_pairwise_agreement_summary.csv`
- `outputs/tables/thematic_topic_positive_rates.csv`
- `outputs/tables/statistics_report.txt`
- `outputs/tables/supplementary_figure_5_all.csv`
- `outputs/tables/supplementary_figure_5_top_positive.csv`
- `outputs/tables/supplementary_figure_5_top_negative.csv`

### 8.3 Model artifacts
- `outputs/models/regression_results.pkl`

## 9. Validation and QA Checks
- Schema validation for enhanced/output tables (`src/pipeline_schema.py`).
- Drift checks with configurable thresholds (`pipeline.yaml` -> `drift_checks`).
- Step06 enforces GPT-5.4 thematic indicator availability for Figure 2.
- Step09 performs strict method-coverage checks and fails fast on missing/disabled/error/parse_error statuses.

## 10. Determinism and Caching
- LLM thematic cache keys are deterministic (`prompt_version + model + normalized text`).
- Step09 adjudication sample uses fixed default random seed (`42`).
- Manifest logging has been intentionally disabled (compatibility shim in `src/pipeline_manifest.py`).

## 11. License
GNU GPLv3. See `LICENSE`.
