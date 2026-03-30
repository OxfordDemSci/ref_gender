#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CONFIG_PATH="$PROJECT_ROOT/pipeline.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="offline"
CLEAN_OUTPUTS=0
DRY_RUN=0
FORCE_STEP01=0

usage() {
  cat <<'EOF'
Usage:
  ./rerun_pipeline.sh [options]

Modes:
  offline     Rebuild all outputs with no new API calls (default).
  refresh     Full refresh using external services/APIs.
  downstream  Rebuild only downstream artifacts (steps 04-10).

Options:
  --mode <offline|refresh|downstream>  Pipeline mode (default: offline)
  --clean-outputs                      Delete files under outputs/ before running
  --config <path>                      Path to pipeline.yaml
  --project-root <path>                Project root (default: script directory)
  --python <python_bin>                Python executable (default: python)
  --force-step01                       Force step01 rewrite in offline mode
  --dry-run                            Print commands without executing
  -h, --help                           Show this help text

Notes:
  - --clean-outputs only clears outputs/ files.
  - Cached data and precomputed API/LLM artifacts in data/ are preserved.
EOF
}

run_step() {
  echo
  echo ">>> $*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

run_downstream_steps() {
  run_step "$PYTHON_BIN" -m src.step04_make_figure_one --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
  run_step "$PYTHON_BIN" -m src.step05_build_regression_models --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
  run_step "$PYTHON_BIN" -m src.step06_make_figure_two --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
  run_step "$PYTHON_BIN" -m src.step07_make_table_one --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
  run_step "$PYTHON_BIN" -m src.step08_build_statistics --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
  run_step "$PYTHON_BIN" -m src.step09_evaluate_thematic_indicators \
    --config "$CONFIG_PATH" \
    --project-root "$PROJECT_ROOT" \
    --model-mini gpt-5-nano --prompt-mini v2 \
    --model-51 gpt-5.1 --prompt-51 v2 \
    --model-54 gpt-5.4 --prompt-54 v2
  run_step "$PYTHON_BIN" -m src.step10_analyze_ics_text_gender --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --clean-outputs)
      CLEAN_OUTPUTS=1
      shift
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --project-root)
      PROJECT_ROOT="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --force-step01)
      FORCE_STEP01=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$MODE" in
  offline|refresh|downstream) ;;
  *)
    echo "Invalid --mode '$MODE' (use: offline, refresh, downstream)." >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
DATA_DIR="$PROJECT_ROOT/data"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config file: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/src" ]]; then
  echo "Missing src/ under project root: $PROJECT_ROOT" >&2
  exit 1
fi

export REF_SKIP_MANIFEST="${REF_SKIP_MANIFEST:-1}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/ref_gender_mplconfig}"
mkdir -p "$MPLCONFIGDIR"

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Config:       $CONFIG_PATH"
echo "Python:       $PYTHON_BIN"
echo "Mode:         $MODE"
echo "Clean outputs:$CLEAN_OUTPUTS"
echo "Force step01: $FORCE_STEP01"
echo "Dry run:      $DRY_RUN"

if [[ "$CLEAN_OUTPUTS" -eq 1 ]]; then
  if [[ -d "$OUTPUTS_DIR" ]]; then
    echo
    echo "Cleaning output files under $OUTPUTS_DIR (preserving cached data in $DATA_DIR)"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      find "$OUTPUTS_DIR" -type f
    else
      find "$OUTPUTS_DIR" -type f -print -delete
    fi
  else
    echo
    echo "Outputs directory not found: $OUTPUTS_DIR"
  fi
fi

if [[ "$MODE" == "refresh" ]]; then
  run_step "$PYTHON_BIN" -m src.step01_make_enhanced_data --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --without-llm --force
  run_step "$PYTHON_BIN" -m src.step02_make_ref_staff --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --with-llm
  run_step "$PYTHON_BIN" -m src.step01_make_enhanced_data --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --with-llm --force
  run_step "$PYTHON_BIN" -m src.step03_get_dimensions_research_outputs --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --force
  run_downstream_steps
  exit 0
fi

if [[ "$MODE" == "offline" ]]; then
  ENH_PATH=""
  if [[ -f "$DATA_DIR/gold/enhanced_ref_data.parquet" ]]; then
    ENH_PATH="$DATA_DIR/gold/enhanced_ref_data.parquet"
  elif [[ -f "$DATA_DIR/gold/enhanced_ref_data.csv" ]]; then
    ENH_PATH="$DATA_DIR/gold/enhanced_ref_data.csv"
  else
    echo "Missing existing enhanced_ref_data in $DATA_DIR/gold (need parquet or csv)." >&2
    exit 1
  fi

  EMPTY_IDS_FILE="$(mktemp "${TMPDIR:-/tmp}/ref_gender_empty_ids.XXXXXX.csv")"
  STEP02_NOAPI_DIR="${TMPDIR:-/tmp}/ref_gender_step02_noapi"
  cleanup() {
    rm -f "$EMPTY_IDS_FILE"
  }
  trap cleanup EXIT

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "REF impact case study identifier"
  else
    printf "REF impact case study identifier\n" > "$EMPTY_IDS_FILE"
    mkdir -p "$STEP02_NOAPI_DIR"
  fi

  step01_cmd=(
    "$PYTHON_BIN" -m src.step01_make_enhanced_data
    --config "$CONFIG_PATH"
    --project-root "$PROJECT_ROOT"
    --with-llm
    --output "$ENH_PATH"
  )
  if [[ "$FORCE_STEP01" -eq 1 ]]; then
    step01_cmd+=(--force)
  fi
  run_step "${step01_cmd[@]}"
  run_step "$PYTHON_BIN" -m src.step02_make_ref_staff --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --without-llm --input "$EMPTY_IDS_FILE" --out-dir "$STEP02_NOAPI_DIR"
  run_step "$PYTHON_BIN" -m src.step03_get_dimensions_research_outputs --config "$CONFIG_PATH" --project-root "$PROJECT_ROOT" --skip-api
  run_downstream_steps
  exit 0
fi

run_downstream_steps
