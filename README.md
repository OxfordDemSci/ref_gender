# Gendered Impact in Academic Research

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)
![Status](https://img.shields.io/badge/status-active-red)
![Plots](https://img.shields.io/badge/plots-matplotlib-orange)
![Data](https://img.shields.io/badge/data-REF%202021-9cf)

This repository contains the reproducible pipeline for analysing gender representation in UK REF 2021 Impact Case Studies (ICS) and REF-linked research outputs.

The pipeline builds:

- case-level ICS gender counts from extracted staff names,
- output-level author gender counts from Dimensions metadata,
- regex and OpenAI thematic indicators from REF case-study text,
- model-comparison diagnostics across thematic classifiers,
- regression models, manuscript figures, tables, and supplementary text-analysis outputs.

All commands below are intended to be run from the repository root.

## Repository Layout

Core pipeline scripts:

- `src/step01_make_enhanced_data.py` builds `data/analysis/enhanced_ref_data.{parquet,csv}` and manages OpenAI thematic-indicator cache rows in `data/openai/categories.csv`.
- `src/step02_make_ref_staff.py` downloads/caches ICS PDFs, extracts staff blocks, uses LLM extraction where enabled, and writes `data/ics_staff_rows/`.
- `src/step03_get_dimensions_research_outputs.py` queries or reuses Dimensions chunks and writes `data/analysis/outputs_concat_with_*_authors.{parquet,csv}`.
- `src/step04_make_figure_one.py` rebuilds Figure 1.
- `src/step05_build_regression_models.py` fits regression models and writes `outputs/models/regression_results.pkl`.
- `src/step06_make_figure_two.py` rebuilds Figure 2 plus supplementary regression figures.
- `src/step07_make_table_one.py` writes `outputs/tables/regression_results.tex`.
- `src/step08_build_statistics.py` writes descriptive statistics and summary tables.
- `src/step09_evaluate_thematic_indicators.py` compares regex, GPT-5-nano, GPT-5.1, GPT-5.4, and GPT-5.5 thematic indicators.
- `src/step10_analyze_ics_text_gender.py` runs word-level ICS text/gender association analysis.

Important directories:

- `data/source/`: downloaded source workbooks and cached Dimensions chunks.
- `data/cache/ref_pdfs/`: cached ICS PDFs.
- `data/ics_staff_rows/`: staff extraction rows, case-level counts, and audit files.
- `data/openai/`: OpenAI thematic cache.
- `data/analysis/`: canonical analysis-ready datasets.
- `data/final/`: final enhanced dataset exports.
- `data/dimensions_outputs/`: legacy mirror of Dimensions output-author tables.
- `outputs/figures/`: figures in PDF/SVG/PNG.
- `outputs/tables/`: manuscript and diagnostic tables.
- `outputs/models/`: fitted model artifacts.

## Current Pipeline Defaults

Configuration is in `pipeline.yaml`.

Current OpenAI defaults:

- primary model: `gpt-5.5`
- prompt version: `v2`
- service tier: `flex`
- processing mode: `sync`
- OpenAI Batch API: not used by default
- thematic request batch size: `1`
- staff extraction request batch size: `1`
- transient OpenAI retry budget for staff extraction: `5`
- prompt cache key: `thematic_indicators_v2`

Thematic classification uses all five REF ICS text fields:

- `1. Summary of the impact`
- `2. Underpinning research`
- `3. References to the research`
- `4. Details of the impact`
- `5. Sources to corroborate the impact`

Staff extraction is robust by design:

- PDF text extraction is cached under `data/cache/ref_pdfs/`.
- LLM extraction retries transient OpenAI failures.
- Cases that remain genuinely unresolved are kept as unresolved/blank counts, not silently converted to zero people.
- `step02.require_people: false` allows the pipeline to continue while preserving unresolved status in audit outputs.

Gender inference for both ICS staff names and Dimensions author forenames uses:

1. `gender_guesser` as the primary deterministic classifier.
2. `gender_detector` as a fallback only when `gender_guesser` returns `unknown`.

The stored output labels remain exactly:

- `male`
- `female`
- `unknown`

## Credentials

OpenAI credentials can be provided by either:

- environment variable: `OPENAI_API_KEY`
- file fallback: `keys/OPENAI_API_KEY`

Dimensions credentials can be provided by either:

- environment variable: `DIMENSIONS_API_KEY`
- file fallback: `keys/dimensions_apikey.txt`

`keys/` is intentionally preserved by rebuild commands.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If parquet read/write fails, check that `pyarrow` is installed.

## One-Command Workflows

### Offline Rerun

Use this when local data/cache artifacts already exist and you do not want new API calls:

```bash
./rerun_pipeline.sh --mode offline
```

This reuses existing enhanced data, staff rows, OpenAI cache rows, and Dimensions chunks/outputs. It then rebuilds downstream figures, tables, models, and diagnostics.

### Downstream-Only Rerun

Use this when `data/analysis/` is already current and only outputs need rebuilding:

```bash
./rerun_pipeline.sh --mode downstream
```

This runs steps 04-10 only.

### Full Refresh

Use this when you want to refresh source data and external-service outputs while preserving cached/generated directories:

```bash
./rerun_pipeline.sh --mode refresh
```

This may call REF downloads, OpenAI, and Dimensions.

### Destructive Rebuild

Use only when you want to delete generated data and rebuild from scratch:

```bash
./rerun_pipeline.sh --mode rebuild --i-understand-this-deletes-data
```

This deletes generated data/output directories, preserves `data/manual/` and `keys/`, then runs the refresh pipeline.

### Resume OpenAI Comparison Backfills

Use when the primary enhanced dataset and staff rows already exist, but comparison-model thematic cache rows still need filling:

```bash
./rerun_pipeline.sh --mode resume-flex
```

This backfills:

- `gpt-5-nano`
- `gpt-5.1`
- `gpt-5.4`

using flex-tier synchronous calls with one item per request, then continues through Dimensions and downstream steps.

## Recovery Workflows

### Recover After a Dimensions Write/Serialization Failure

If Dimensions API chunking completed but `step03` failed while assembling or writing outputs, do not rerun expensive API calls. Rebuild from saved raw chunks:

```bash
python -m src.step03_get_dimensions_research_outputs \
  --config pipeline.yaml \
  --project-root . \
  --skip-api \
  --force
```

Then continue:

```bash
./rerun_pipeline.sh --mode downstream
```

One-line version:

```bash
mkdir -p logs && { python -m src.step03_get_dimensions_research_outputs --config pipeline.yaml --project-root . --skip-api --force && ./rerun_pipeline.sh --mode downstream; } 2>&1 | tee logs/resume_from_step03_$(date +%Y%m%d_%H%M%S).log
```

The current `step03` implementation serializes complex Dimensions fields such as `authors` and `category_for_2020` as JSON strings before writing parquet. This avoids PyArrow failures caused by mixed nested types in Dimensions metadata.

### Check Whether Expensive Data Already Exists

Dimensions raw chunks:

```bash
find data/source/dimensions_api/raw -name 'df_*.csv' -printf '%h\n' | sort | uniq -c
```

OpenAI thematic cache:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/openai/categories.csv", usecols=["model", "prompt_version", "llm_status", "cache_key"])
print(df.groupby(["model", "prompt_version", "llm_status"]).size())
print("bad rows:", len(df[~df["llm_status"].isin(["ok", "ok_prompt_cache_fallback"])]))
print(df.groupby(["model", "prompt_version"])["cache_key"].nunique())
PY
```

## Step Map

| Step | Script | External calls by default | Primary outputs |
|---|---|---|---|
| 01 | `step01_make_enhanced_data.py` | REF downloads; OpenAI when `--with-llm` needs uncached rows | `data/analysis/enhanced_ref_data.*`, `data/openai/categories.csv` |
| 02 | `step02_make_ref_staff.py` | ICS PDF downloads; OpenAI when `--with-llm` needs extraction | `data/ics_staff_rows/*.csv` |
| 03 | `step03_get_dimensions_research_outputs.py` | Dimensions unless `--skip-api` | `data/analysis/outputs_concat_*.{parquet,csv}` |
| 04 | `step04_make_figure_one.py` | none | `outputs/figures/figure_one.{pdf,svg,png}` |
| 05 | `step05_build_regression_models.py` | none | `outputs/models/regression_results.pkl` |
| 06 | `step06_make_figure_two.py` | none | `outputs/figures/figure_two.*`, supplementary regression figures |
| 07 | `step07_make_table_one.py` | none | `outputs/tables/regression_results.tex` |
| 08 | `step08_build_statistics.py` | none | `outputs/tables/statistics_report.txt`, summary `.tex` tables |
| 09 | `step09_evaluate_thematic_indicators.py` | none | thematic diagnostics and comparison figures |
| 10 | `step10_analyze_ics_text_gender.py` | none | `supplementary_figure_5.*`, word-association tables |

## Thematic Indicator Methods

Step09 compares five method families:

- Regex rules.
- GPT-5-nano, displayed in some outputs as `GPT-5-mini`.
- GPT-5.1.
- GPT-5.4.
- GPT-5.5, the primary model used by the main enhanced dataset and Figure 2.

If comparison cache slices need filling manually:

```bash
python -m src.step01_make_enhanced_data --config pipeline.yaml --project-root . --backfill-model gpt-5-nano --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 1 --backfill-prompt-cache-key thematic_indicators_v2
python -m src.step01_make_enhanced_data --config pipeline.yaml --project-root . --backfill-model gpt-5.1   --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 1 --backfill-prompt-cache-key thematic_indicators_v2
python -m src.step01_make_enhanced_data --config pipeline.yaml --project-root . --backfill-model gpt-5.4   --backfill-prompt-version v2 --backfill-service-tier flex --backfill-batch-size 1 --backfill-prompt-cache-key thematic_indicators_v2
```

Then rebuild the enhanced dataset if needed:

```bash
python -m src.step01_make_enhanced_data --config pipeline.yaml --project-root . --with-llm --force
```

## Key Outputs

Figures:

- `outputs/figures/figure_one.{pdf,svg,png}`
- `outputs/figures/figure_two.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_1.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_2.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_3.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_4.{pdf,svg,png}`
- `outputs/figures/supplementary_figure_5.{pdf,svg,png}`

Tables:

- `outputs/tables/regression_results.tex`
- `outputs/tables/panel_summary.tex`
- `outputs/tables/uoa_summary.tex`
- `outputs/tables/llm_summary.tex`
- `outputs/tables/llm_panel_summary.tex`
- `outputs/tables/statistics_report.txt`
- `outputs/tables/thematic_model_health_checks.csv`
- `outputs/tables/thematic_pairwise_agreement_by_topic.csv`
- `outputs/tables/thematic_pairwise_agreement_summary.csv`
- `outputs/tables/thematic_topic_positive_rates.csv`
- `outputs/tables/thematic_adjudication_sample.csv`
- `outputs/tables/supplementary_figure_5_all.csv`
- `outputs/tables/supplementary_figure_5_top_positive.csv`
- `outputs/tables/supplementary_figure_5_top_negative.csv`

Models:

- `outputs/models/regression_results.pkl`

## Validation Checks

Run targeted tests for the recent pipeline fixes:

```bash
python -m unittest tests.test_step03_gender_inference tests.test_pipeline_schema
```

Check the latest thematic method coverage:

```bash
cat outputs/tables/thematic_model_health_checks.csv
```

Check Dimensions output row contracts:

```bash
python - <<'PY'
import pandas as pd
any_df = pd.read_parquet("data/analysis/outputs_concat_with_any_number_authors.parquet")
pos_df = pd.read_parquet("data/analysis/outputs_concat_with_positive_authors.parquet")
print("any rows:", len(any_df))
print("positive rows:", len(pos_df))
print("positive ids subset of any ids:", set(pos_df["REF2ID"].astype(str)).issubset(set(any_df["REF2ID"].astype(str))))
print("zero-author rows in any:", int((pd.to_numeric(any_df["number_people"], errors="coerce").fillna(-1) == 0).sum()))
print("non-positive rows in positive:", int((pd.to_numeric(pos_df["number_people"], errors="coerce").fillna(0) <= 0).sum()))
PY
```

## Determinism and Caching

- OpenAI thematic cache keys are deterministic hashes of model, prompt version, and normalized five-field ICS text.
- Successful prompt-cache fallback rows are stored with `llm_status=ok_prompt_cache_fallback` and are treated as valid completions.
- `step03 --skip-api` reuses saved Dimensions chunks and does not call Dimensions.
- Step09 strict health checks fail on missing, disabled, parse-error, or otherwise bad thematic rows.
- `rerun_pipeline.sh` sets `REF_SKIP_MANIFEST=1` by default, so manifest rows may be absent unless this is overridden.

## License

GNU GPLv3. See `LICENSE`.
