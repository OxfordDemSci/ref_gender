import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable

try:  # pragma: no cover
    from .pipeline_config import load_config_and_paths
    from .pipeline_drift import apply_enhanced_drift_checks
    from .pipeline_io import atomic_write_csv, atomic_write_parquet, build_retry_session, download_file, read_secret
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import PipelinePaths, default_project_root, ensure_core_dirs, format_relative_path
    from .pipeline_schema import validate_enhanced_ref_data
    from .openai_batch import OpenAIBatchPending, create_or_retrieve_batch, read_jsonl
except ImportError:  # pragma: no cover
    from pipeline_config import load_config_and_paths
    from pipeline_drift import apply_enhanced_drift_checks
    from pipeline_io import atomic_write_csv, atomic_write_parquet, build_retry_session, download_file, read_secret
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import PipelinePaths, default_project_root, ensure_core_dirs, format_relative_path
    from pipeline_schema import validate_enhanced_ref_data
    from openai_batch import OpenAIBatchPending, create_or_retrieve_batch, read_jsonl


# ===============================
# Utilities and pipeline helpers
# ===============================

def log_row_count(func):
    """Decorator to log the number of rows in a DataFrame after applying a function."""
    def wrapper(df, *args, **kwargs):
        result = func(df, *args, **kwargs)
        print(f"Number of rows after {func.__name__}: {len(result)}")
        return result
    return wrapper


def _response_output_text(resp: Any) -> str:
    """
    Extract visible text from an OpenAI Responses API object without falling
    back to the whole response repr, which is not model output.
    """
    if isinstance(resp, str):
        return resp

    direct = getattr(resp, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    def _to_plain(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump(mode="python")
            except Exception:  # noqa: BLE001
                return obj
        return obj

    def _collect(obj: Any, *, depth: int = 0) -> list[str]:
        if obj is None or depth > 8:
            return []
        obj = _to_plain(obj)
        if isinstance(obj, str):
            return [obj] if obj.strip() else []
        if isinstance(obj, list | tuple):
            out: list[str] = []
            for item in obj:
                out.extend(_collect(item, depth=depth + 1))
            return out
        if not isinstance(obj, dict):
            return []

        out = []
        if isinstance(obj.get("output_text"), str):
            out.append(obj["output_text"])
        if obj.get("type") == "output_text" and isinstance(obj.get("text"), str):
            out.append(obj["text"])
        parsed = obj.get("parsed")
        if parsed:
            out.append(json.dumps(parsed, ensure_ascii=False))
        for key in ("output", "content", "message", "messages"):
            if key in obj:
                out.extend(_collect(obj[key], depth=depth + 1))
        return out

    return "".join(_collect(resp)).strip()


def _response_diagnostic(resp: Any) -> str:
    parts: list[str] = []
    for attr in ("status", "incomplete_details", "error"):
        value = getattr(resp, attr, None)
        if value:
            parts.append(f"{attr}={value}")
    return "; ".join(parts)


def _loads_json_payload(raw_text: str) -> Any:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        raise ValueError("empty_response")

    candidates = [raw_text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for idx, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
    raise ValueError("json_parse_failed")


def _parse_thematic_single_flags(raw_text: str, ordered_groups: list[str]) -> tuple[dict[str, int], str, str]:
    try:
        parsed = _loads_json_payload(raw_text)
    except ValueError as exc:
        return {g: 0 for g in ordered_groups}, "parse_error", str(exc)

    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list) and parsed["results"]:
        first = parsed["results"][0]
        if isinstance(first, dict):
            parsed = first

    if not isinstance(parsed, dict):
        return {g: 0 for g in ordered_groups}, "parse_error", "json_object_missing"

    missing = [g for g in ordered_groups if g not in parsed]
    if missing:
        return {g: 0 for g in ordered_groups}, "parse_error", f"indicator_fields_missing:{','.join(missing)}"

    return {g: int(bool(parsed.get(g, False))) for g in ordered_groups}, "ok", ""


def _parse_thematic_batch_flags(
    raw_text: str,
    ordered_groups: list[str],
    expected_ids: list[str] | None = None,
) -> tuple[dict[str, dict[str, int]], str, str]:
    try:
        parsed = _loads_json_payload(raw_text)
    except ValueError as exc:
        return {}, "parse_error", str(exc)

    expected_ids = expected_ids or []
    results: Any
    if isinstance(parsed, dict) and isinstance(parsed.get("results"), list):
        results = parsed["results"]
    elif isinstance(parsed, list):
        results = parsed
    elif isinstance(parsed, dict) and ("id" in parsed or len(expected_ids) == 1):
        results = [parsed]
    else:
        return {}, "parse_error", "batch_results_missing"

    out: dict[str, dict[str, int]] = {}
    malformed_items = 0
    for item in results:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        item_id = str(item.get("id", "")).strip()
        if not item_id and len(expected_ids) == 1:
            item_id = expected_ids[0]
        if not item_id:
            malformed_items += 1
            continue
        if any(g not in item for g in ordered_groups):
            malformed_items += 1
            continue
        out[item_id] = {g: int(bool(item.get(g, False))) for g in ordered_groups}

    if not out:
        return {}, "parse_error", "batch_results_missing"
    if malformed_items and len(out) < len(results):
        return out, "parse_error", "malformed_batch_items"
    return out, "ok", ""


def mirror_legacy_raw_to_source(raw_path: Path, legacy_raw_path: Path) -> None:
    """
    Copy legacy raw files into the source-data directory when source is empty.
    """
    raw_path = Path(raw_path)
    legacy_raw_path = Path(legacy_raw_path)
    raw_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "raw_ref_environment_data.xlsx",
        "raw_ref_results_data.xlsx",
        "raw_ref_ics_data.xlsx",
        "raw_ref_ics_tags_data.xlsx",
        "raw_ref_outputs_data.xlsx",
    ):
        source_file = raw_path / name
        legacy_file = legacy_raw_path / name
        if not source_file.exists() and legacy_file.exists():
            shutil.copy2(legacy_file, source_file)


def get_impact_data(raw_path: Path, session, timeout_seconds: int) -> None:
    """Download ICS data + tags to raw_path (unrelated to results sheet)."""
    print("Getting ICS Data!")
    download_file(
        "https://results2021.ref.ac.uk/impact/export-all",
        raw_path / "raw_ref_ics_data.xlsx",
        session=session,
        timeout_seconds=timeout_seconds,
    )
    download_file(
        "https://results2021.ref.ac.uk/impact/export-tags-all",
        raw_path / "raw_ref_ics_tags_data.xlsx",
        session=session,
        timeout_seconds=timeout_seconds,
    )


def get_environmental_data(raw_path: Path, session, timeout_seconds: int) -> None:
    """Download Environmental data to raw_path."""
    print("Getting Environmental Data!")
    download_file(
        "https://results2021.ref.ac.uk/environment/export-all",
        raw_path / "raw_ref_environment_data.xlsx",
        session=session,
        timeout_seconds=timeout_seconds,
    )


def get_all_results(raw_path: Path, session, timeout_seconds: int) -> None:
    """
    Ensure the REF results workbook exists (XLSX). If missing, download it.
    """
    xlsx_path = raw_path / "raw_ref_results_data.xlsx"
    if xlsx_path.exists():
        print("Results XLSX present locally; not downloading.")
        return
    print("Getting Results Data (XLSX)…")
    download_file(
        "https://results2021.ref.ac.uk/profiles/export-all",
        xlsx_path,
        session=session,
        timeout_seconds=timeout_seconds,
    )


def get_output_data(raw_path: Path, session, timeout_seconds: int) -> None:
    """Download Outputs data to raw_path."""
    print("Getting Outputs Data!")
    download_file(
        "https://results2021.ref.ac.uk/outputs/export-all",
        raw_path / "raw_ref_outputs_data.xlsx",
        session=session,
        timeout_seconds=timeout_seconds,
    )


def format_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise institution + UoA identifiers and build 'uoa_id'.
    Expects either 'Institution UKPRN code' or 'Institution code (UKPRN)'.
    """
    if "Institution UKPRN code" in df.columns:
        df = df.rename(columns={"Institution UKPRN code": "inst_id"})
    if "Institution code (UKPRN)" in df.columns:
        df = df.rename(columns={"Institution code (UKPRN)": "inst_id"})
    df = df[df["inst_id"] != " "]
    df = df.astype({"inst_id": "int"})
    df["uoa_id"] = (
        df["Unit of assessment number"].astype(int).astype(str)
        + df["Multiple submission letter"].fillna("").astype(str)
    )
    return df


def merge_ins_uoa(df1: pd.DataFrame, df2: pd.DataFrame, id1: str = "inst_id", id2: str = "uoa_id") -> pd.DataFrame:
    """Left-merge df2 into df1 on inst_id and uoa_id with key assertions."""
    assert all(df1[id1].isin(df2[id1]))
    assert all(df1[id2].isin(df2[id2]))
    return df1.merge(df2, how="left", on=[id1, id2])


@log_row_count
def clean_ics_level(raw_path: Path, edit_path: Path) -> pd.DataFrame:
    """Clean ICS-level data and persist a cleaned Excel."""
    print("Cleaning ICS Level Data!")
    raw_ics = pd.read_excel(raw_path / "raw_ref_ics_data.xlsx", engine="openpyxl")
    raw_ics["Title"] = raw_ics["Title"].apply(
        lambda val: unicodedata.normalize("NFKD", str(val)).encode("ascii", "ignore").decode()
    )
    raw_ics = format_ids(raw_ics)
    raw_ics.to_excel(edit_path / "clean_ref_ics_data.xlsx", index=False)
    return raw_ics


# ---------- Results reader (Excel) ----------

def read_results_table(raw_path: Path, sheet: int = 0) -> pd.DataFrame:
    """
    Load the REF results workbook by sheet using pandas.read_excel.
    Assumptions:
      - The header row is at Excel row 7 (0-index header=6).
      - We keep everything as object dtype to avoid coercion.
    """
    xlsx_path = raw_path / "raw_ref_results_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing {xlsx_path}. Download step should have created it.")
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet,
        header=6,  # header row starts at line 7 in the workbook
        engine="openpyxl",
        dtype=object,
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_dep_level(raw_path: Path, edit_path: Path) -> None:
    """
    Build a wide department-level scorecard and merge with environmental metrics.
    Results table is loaded from the original Excel workbook.
    """
    print("Cleaning Department Level Data!")

    raw_results = read_results_table(raw_path, sheet=0)
    raw_results = raw_results.rename(columns={"Institution UKPRN code": "inst_id", "Institution code (UKPRN)": "inst_id"})
    raw_results = format_ids(raw_results)
    raw_results = raw_results.rename(columns={"FTE of submitted staff": "fte", "% of eligible staff submitted": "fte_pc"})

    score_types = ["4*", "3*", "2*", "1*", "Unclassified"]
    wide_score_card = pd.pivot_table(
        raw_results[["inst_id", "uoa_id", "Profile"] + score_types],
        index=["inst_id", "uoa_id"],
        columns="Profile",
        values=score_types,
        aggfunc="first",
    )
    wide_score_card.columns = wide_score_card.columns.map("_".join)
    wide_score_card = wide_score_card.reset_index()

    raw_env_path = raw_path / "raw_ref_environment_data.xlsx"

    raw_env_doctoral = pd.read_excel(
        raw_env_path,
        sheet_name="ResearchDoctoralDegreesAwarded",
        skiprows=4,
        engine="openpyxl",
    )
    raw_env_doctoral = format_ids(raw_env_doctoral)
    number_cols = [c for c in raw_env_doctoral.columns if "Number of doctoral" in c]
    raw_env_doctoral["num_doc_degrees_total"] = raw_env_doctoral[number_cols].sum(axis=1)

    raw_env_income = pd.read_excel(raw_env_path, sheet_name="ResearchIncome", skiprows=4, engine="openpyxl")
    raw_env_income = format_ids(raw_env_income)
    raw_env_income = raw_env_income.rename(
        columns={
            "Average income for academic years 2013-14 to 2019-20": "av_income",
            "Total income for academic years 2013-14 to 2019-20": "tot_income",
        }
    )
    tot_inc = raw_env_income[raw_env_income["Income source"] == "Total income"]

    raw_env_income_inkind = pd.read_excel(
        raw_env_path,
        sheet_name="ResearchIncomeInKind",
        skiprows=4,
        engine="openpyxl",
    )
    raw_env_income_inkind = format_ids(raw_env_income_inkind)
    raw_env_income_inkind = raw_env_income_inkind.rename(
        columns={"Total income for academic years 2013-14 to 2019-20": "tot_inc_kind"}
    )
    tot_inc_kind = raw_env_income_inkind.loc[raw_env_income_inkind["Income source"] == "Total income-in-kind"]

    raw_dep = merge_ins_uoa(raw_results[["inst_id", "uoa_id", "fte", "fte_pc"]].drop_duplicates(), wide_score_card)
    raw_dep = merge_ins_uoa(raw_dep, raw_env_doctoral[["inst_id", "uoa_id", "num_doc_degrees_total"]])
    raw_dep = merge_ins_uoa(raw_dep, tot_inc[["inst_id", "uoa_id", "av_income", "tot_income"]])
    raw_dep = merge_ins_uoa(raw_dep, tot_inc_kind[["inst_id", "uoa_id", "tot_inc_kind"]])
    raw_dep.to_excel(edit_path / "clean_ref_dep_data.xlsx", index=False)


def get_paths(paths: PipelinePaths):
    """
    Build project data paths while preserving legacy compatibility directories.
    """
    raw_path = paths.source_dir
    edit_path = paths.working_dir
    sup_path = paths.data_dir / "supplementary"
    manual_path = paths.manual_dir
    final_path = paths.analysis_dir
    topic_path = paths.data_dir / "reassignments"
    dim_path = paths.data_dir / "dimensions_returns"
    openalex_path = paths.data_dir / "openalex_returns"
    ics_staff_rows_path = paths.data_dir / "ics_staff_rows"
    ics_grants_path = paths.data_dir / "ics_grants"
    return (
        raw_path,
        edit_path,
        sup_path,
        manual_path,
        final_path,
        topic_path,
        dim_path,
        openalex_path,
        ics_staff_rows_path,
        ics_grants_path,
    )


def _base_stata_column_name(name: str) -> str:
    """
    Convert an arbitrary column label into a Stata-safe base variable name.
    """
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "var"
    if re.match(r"^[0-9]", value):
        value = f"v_{value}"
    return value


def _make_unique_stata_column_names(columns: list[str], max_length: int = 32) -> tuple[list[str], pd.DataFrame]:
    """
    Build unique, <=32-char Stata variable names and return a mapping table.
    """
    used: set[str] = set()
    new_cols: list[str] = []
    mapping_rows: list[dict[str, str]] = []

    for col in columns:
        base = _base_stata_column_name(col)
        candidate = base[:max_length]
        suffix_idx = 1
        while candidate in used:
            suffix = f"_{suffix_idx}"
            candidate = f"{base[:max_length - len(suffix)]}{suffix}"
            suffix_idx += 1
        used.add(candidate)
        new_cols.append(candidate)
        mapping_rows.append({"original_column": str(col), "stata_column": candidate})

    return new_cols, pd.DataFrame(mapping_rows)


def _atomic_write_stata(df: pd.DataFrame, out_path: Path, *, version: int = 118, convert_strl: list[str] | None = None) -> None:
    """
    Write a Stata .dta file atomically to avoid partial files on interruption.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_stata(
        tmp_path,
        write_index=False,
        version=version,
        convert_strl=convert_strl or [],
    )
    tmp_path.replace(out_path)


def write_final_clean_exports(df: pd.DataFrame, final_dir: Path) -> dict[str, Path]:
    """
    Write cleaned final exports for downstream users:
      - Stata-safe cleaned CSV
      - Stata .dta
      - original->Stata column mapping CSV
    """
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = final_dir / "enhanced_ref_data_clean_final.csv"
    cleaned_dta_path = final_dir / "enhanced_ref_data_clean_final.dta"
    column_map_path = final_dir / "enhanced_ref_data_stata_column_map.csv"

    cleaned_df = df.copy()
    stata_columns, column_map_df = _make_unique_stata_column_names([str(c) for c in cleaned_df.columns])
    cleaned_df.columns = stata_columns

    object_cols = [c for c in cleaned_df.columns if cleaned_df[c].dtype == "object"]
    for col in object_cols:
        cleaned_df[col] = cleaned_df[col].where(cleaned_df[col].notna(), None)

    atomic_write_csv(cleaned_df, cleaned_csv_path)
    _atomic_write_stata(cleaned_df, cleaned_dta_path, version=118, convert_strl=object_cols)
    atomic_write_csv(column_map_df, column_map_path)

    return {
        "cleaned_csv_path": cleaned_csv_path,
        "cleaned_dta_path": cleaned_dta_path,
        "column_map_path": column_map_path,
    }


@log_row_count
def load_dept_vars(df: pd.DataFrame, edit_path: Path) -> pd.DataFrame:
    """Load department vars, compute GPAs, and merge."""
    print("Loading department variables")
    dept_vars = pd.read_excel(edit_path / "clean_ref_dep_data.xlsx", engine="openpyxl")

    dept_vars["ICS_GPA"] = (
        pd.to_numeric(dept_vars["4*_Impact"], errors="coerce") * 4
        + pd.to_numeric(dept_vars["3*_Impact"], errors="coerce") * 3
        + pd.to_numeric(dept_vars["2*_Impact"], errors="coerce") * 2
        + pd.to_numeric(dept_vars["1*_Impact"], errors="coerce")
    ) / 100
    dept_vars["Environment_GPA"] = (
        pd.to_numeric(dept_vars["4*_Environment"], errors="coerce") * 4
        + pd.to_numeric(dept_vars["3*_Environment"], errors="coerce") * 3
        + pd.to_numeric(dept_vars["2*_Environment"], errors="coerce") * 2
        + pd.to_numeric(dept_vars["1*_Environment"], errors="coerce")
    ) / 100
    dept_vars["Output_GPA"] = (
        pd.to_numeric(dept_vars["4*_Outputs"], errors="coerce") * 4
        + pd.to_numeric(dept_vars["3*_Outputs"], errors="coerce") * 3
        + pd.to_numeric(dept_vars["2*_Outputs"], errors="coerce") * 2
        + pd.to_numeric(dept_vars["1*_Outputs"], errors="coerce")
    ) / 100
    dept_vars["Overall_GPA"] = (
        pd.to_numeric(dept_vars["4*_Overall"], errors="coerce") * 4
        + pd.to_numeric(dept_vars["3*_Overall"], errors="coerce") * 3
        + pd.to_numeric(dept_vars["2*_Overall"], errors="coerce") * 2
        + pd.to_numeric(dept_vars["1*_Overall"], errors="coerce")
    ) / 100

    cols = [
        "inst_id",
        "uoa_id",
        "fte",
        "num_doc_degrees_total",
        "av_income",
        "tot_income",
        "tot_inc_kind",
        "ICS_GPA",
        "Environment_GPA",
        "Output_GPA",
        "Overall_GPA",
    ]

    return pd.merge(df, dept_vars[cols], how="left", left_on=["inst_id", "uoa_id"], right_on=["inst_id", "uoa_id"]).drop(
        "uoa_id", axis=1
    )


STAFF_COUNT_COLUMNS = ["number_male", "number_female", "number_unknown", "number_people"]
STAFF_METADATA_DEFAULTS = {
    "staff_block": pd.NA,
    "extraction_status": "not_run",
    "staff_extraction_status": "unknown",
    "staff_extraction_error": "",
    "names": "[]",
    "given_names": "[]",
    "roles": "[]",
    "genders": "[]",
}


def get_ics_staff_rows(df: pd.DataFrame, ics_staff_rows_path: Path) -> pd.DataFrame:
    staff_path = Path(ics_staff_rows_path) / "ref_case_level.csv"
    if not staff_path.exists():
        raise FileNotFoundError(
            f"Missing staff enrichment file: {staff_path}. "
            "Run step01 with --prepare-source-only, then run step02_make_ref_staff --with-llm, "
            "then rerun step01 to build enhanced_ref_data."
        )
    staff_rows = pd.read_csv(staff_path)
    print(f"Staff rows loaded: {len(staff_rows)}")
    required_staff_cols = ["REF impact case study identifier", *STAFF_COUNT_COLUMNS]
    missing_staff_cols = [c for c in required_staff_cols if c not in staff_rows.columns]
    if missing_staff_cols:
        raise ValueError(f"{staff_path} missing required staff columns: {missing_staff_cols}")

    staff_enrichment_cols = [
        c for c in [*STAFF_METADATA_DEFAULTS.keys(), *STAFF_COUNT_COLUMNS] if c in df.columns
    ]
    base = df.drop(columns=staff_enrichment_cols) if staff_enrichment_cols else df
    merged = pd.merge(
        base,
        staff_rows,
        how="left",
        on="REF impact case study identifier",
        validate="one_to_one",
        indicator=True,
    )

    missing_rows = merged["_merge"].eq("left_only")
    if bool(missing_rows.any()):
        missing_ids = (
            merged.loc[missing_rows, "REF impact case study identifier"].astype(str).head(20).tolist()
        )
        raise ValueError(
            f"Staff enrichment file does not cover {int(missing_rows.sum())} case(s). "
            f"First missing case IDs: {', '.join(missing_ids)}"
        )
    merged = merged.drop(columns=["_merge"])

    for col in STAFF_COUNT_COLUMNS:
        numeric = pd.to_numeric(merged[col], errors="coerce")
        invalid = numeric.isna() & merged[col].notna()
        if bool(invalid.any()):
            raise ValueError(f"Staff enrichment column {col} contains non-numeric values.")
        merged[col] = numeric
    for col, default in STAFF_METADATA_DEFAULTS.items():
        if col not in merged.columns:
            merged[col] = default
        elif col in {"extraction_status", "staff_extraction_status", "staff_extraction_error"}:
            merged[col] = merged[col].fillna(default)
    return merged


def get_ics_grants(df: pd.DataFrame, ics_grants_path: Path) -> pd.DataFrame:
    grants_path = Path(ics_grants_path) / "ICS_grants_aggregated.csv"
    if not grants_path.exists():
        print(f"No ICS grants data at {format_relative_path(grants_path)}; continuing without grants enrichment.")
        return df
    grants_rows = pd.read_csv(grants_path)
    print(f"ICS grants rows loaded: {len(grants_rows)}")
    return pd.merge(df, grants_rows, how="left", on="REF impact case study identifier")


def get_university_class(df: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    class_path = Path(manual_path) / "university_category" / "ref_unique_institutions.csv"
    if not class_path.exists():
        print(f"No university classifications file at {format_relative_path(class_path)}; continuing without this lookup.")
        return df
    university_class = pd.read_csv(class_path)
    print("Merged in university classifications data.")
    return pd.merge(df, university_class, how="left", on="Institution name")


def get_panel_and_UoA_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = [
        {"Unit of assessment number": 1, "Unit of assessment": "Clinical Medicine", "Main Panel": "A"},
        {
            "Unit of assessment number": 2,
            "Unit of assessment": "Public Health, Health Services and Primary Care",
            "Main Panel": "A",
        },
        {
            "Unit of assessment number": 3,
            "Unit of assessment": "Allied Health Professions, Dentistry, Nursing and Pharmacy",
            "Main Panel": "A",
        },
        {"Unit of assessment number": 4, "Unit of assessment": "Psychology, Psychiatry and Neuroscience", "Main Panel": "A"},
        {"Unit of assessment number": 5, "Unit of assessment": "Biological Sciences", "Main Panel": "A"},
        {"Unit of assessment number": 6, "Unit of assessment": "Agriculture, Food and Veterinary Sciences", "Main Panel": "A"},
        {"Unit of assessment number": 7, "Unit of assessment": "Earth Systems and Environmental Sciences", "Main Panel": "B"},
        {"Unit of assessment number": 8, "Unit of assessment": "Chemistry", "Main Panel": "B"},
        {"Unit of assessment number": 9, "Unit of assessment": "Physics", "Main Panel": "B"},
        {"Unit of assessment number": 10, "Unit of assessment": "Mathematical Sciences", "Main Panel": "B"},
        {"Unit of assessment number": 11, "Unit of assessment": "Computer Science and Informatics", "Main Panel": "B"},
        {"Unit of assessment number": 12, "Unit of assessment": "Engineering", "Main Panel": "B"},
        {"Unit of assessment number": 13, "Unit of assessment": "Architecture, Built Environment and Planning", "Main Panel": "C"},
        {"Unit of assessment number": 14, "Unit of assessment": "Geography and Environmental Studies", "Main Panel": "C"},
        {"Unit of assessment number": 15, "Unit of assessment": "Archaeology", "Main Panel": "C"},
        {"Unit of assessment number": 16, "Unit of assessment": "Economics and Econometrics", "Main Panel": "C"},
        {"Unit of assessment number": 17, "Unit of assessment": "Business and Management Studies", "Main Panel": "C"},
        {"Unit of assessment number": 18, "Unit of assessment": "Law", "Main Panel": "C"},
        {"Unit of assessment number": 19, "Unit of assessment": "Politics and International Studies", "Main Panel": "C"},
        {"Unit of assessment number": 20, "Unit of assessment": "Social Work and Social Policy", "Main Panel": "C"},
        {"Unit of assessment number": 21, "Unit of assessment": "Sociology", "Main Panel": "C"},
        {"Unit of assessment number": 22, "Unit of assessment": "Anthropology and Development Studies", "Main Panel": "C"},
        {"Unit of assessment number": 23, "Unit of assessment": "Education", "Main Panel": "C"},
        {"Unit of assessment number": 24, "Unit of assessment": "Sport and Exercise Sciences, Leisure and Tourism", "Main Panel": "C"},
        {"Unit of assessment number": 25, "Unit of assessment": "Area Studies", "Main Panel": "D"},
        {"Unit of assessment number": 26, "Unit of assessment": "Modern Languages and Linguistics", "Main Panel": "D"},
        {"Unit of assessment number": 27, "Unit of assessment": "English Language and Literature", "Main Panel": "D"},
        {"Unit of assessment number": 28, "Unit of assessment": "History", "Main Panel": "D"},
        {"Unit of assessment number": 29, "Unit of assessment": "Classics", "Main Panel": "D"},
        {"Unit of assessment number": 30, "Unit of assessment": "Philosophy", "Main Panel": "D"},
        {"Unit of assessment number": 31, "Unit of assessment": "Theology and Religious Studies", "Main Panel": "D"},
        {"Unit of assessment number": 32, "Unit of assessment": "Art and Design: History, Practice and Theory", "Main Panel": "D"},
        {
            "Unit of assessment number": 33,
            "Unit of assessment": "Music, Drama, Dance, Performing Arts, Film and Screen Studies",
            "Main Panel": "D",
        },
        {
            "Unit of assessment number": 34,
            "Unit of assessment": "Communication, Cultural and Media Studies, Library and Information Management",
            "Main Panel": "D",
        },
    ]
    mapping_df = pd.DataFrame(mapping)
    print("Merged on UoA and Panel Names")
    return df.merge(mapping_df, on="Unit of assessment number", how="left", validate="many_to_one")


def get_thematic_indicators(
    df: pd.DataFrame,
    *,
    llm_enabled: bool = True,
    model: str = "gpt-5.5",
    service_tier: str = "flex",
    prompt_version: str = "v2",
    llm_batch_size: int = 12,
    prompt_cache_key: str | None = "thematic_indicators_v2",
    prompt_cache_retention: str | None = "24h",
    key_env_var: str = "OPENAI_API_KEY",
    key_path: str | Path | None = None,
    cache_path: str | Path = "./data/openai/categories.csv",
    openai_processing_mode: str = "sync",
    batch_wait: bool = False,
    batch_poll_interval_seconds: float = 60.0,
    batch_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add thematic indicator columns using regex and optionally an online LLM.

    For each semantic group g in:
        ['charity', 'startup', 'patent', 'museum', 'nhs', 'drug_trial',
         'school', 'legislation', 'heritage', 'manufacturing', 'software']

    this function adds two integer columns:

        - 'regex_g' : regex-based indicator (0/1)
        - 'llm_g'   : LLM-based indicator (0/1, cached in CSV)
    """
    thematic_text_fields = [
        "1. Summary of the impact",
        "2. Underpinning research",
        "3. References to the research",
        "4. Details of the impact",
        "5. Sources to corroborate the impact",
    ]

    def _normalise_text(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str)
        s = s.str.replace("[\u2012\u2013\u2014\u2015]", "-", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        return s

    df = df.copy()
    missing_text_fields = [c for c in thematic_text_fields if c not in df.columns]
    if missing_text_fields:
        raise ValueError(f"Missing thematic text fields: {missing_text_fields}")
    text_parts = []
    for col in thematic_text_fields:
        normalised_col = _normalise_text(df[col])
        text_parts.append(normalised_col.map(lambda value, col=col: f"{col}: {value}" if value else ""))
    norm = pd.concat(text_parts, axis=1).agg(" ".join, axis=1)
    norm = norm.str.replace(r"\s+", " ", regex=True).str.strip()
    df["_impact_text_norm"] = norm

    HX = r"(?:\s|[-–—])?"

    def wb(p: str) -> str:
        return rf"(?<![A-Za-z]){p}(?![A-Za-z])"

    patterns = {
        "charity": [
            wb(r"charit(?:y|ies)"),
            wb(rf"non{HX}?profit(?:{HX}organi[sz]ation(?:s)?)?"),
            wb(r"NGOs?"),
            wb(rf"non{HX}?governmental{HX}organi[sz]ation(?:s)?"),
            wb(rf"voluntary{HX}organi[sz]ation(?:s)?"),
            wb(rf"philanthropic{HX}organi[sz]ation(?:s)?"),
            wb(rf"third{HX}sector{HX}organi[sz]ation(?:s)?"),
            wb(rf"charitable{HX}trusts?"),
            wb(rf"charitable{HX}foundations?"),
            wb(rf"social{HX}enterprises?"),
        ],
        "startup": [
            wb(rf"start{HX}?ups?"),
            wb(rf"spin{HX}?outs?"),
            wb(rf"spin{HX}?offs?"),
        ],
        "patent": [wb(r"patent(?:s|ed|able|ing)?")],
        "museum": [
            wb(r"museums?"),
            wb(rf"exhibition{HX}?s?"),
            wb(r"galler(?:y|ies)"),
        ],
        "nhs": [
            wb(r"NHS"),
            wb(rf"National{HX}Health{HX}Service"),
        ],
        "drug_trial": [
            wb(rf"(?:drug|pharmaceutical|clinical|medical){HX}trial(?:s)?"),
            wb(rf"(?:drug|pharmaceutical|therapeutic|medicine){HX}(?:development|discovery)"),
            wb(rf"new{HX}drug"),
        ],
        "school": [wb(r"schools?")],
        "legislation": [
            wb(r"legislations?"),
            wb(rf"legislative{HX}reform"),
            wb(rf"law{HX}(?:reform|change)"),
            wb(rf"legal{HX}reform"),
        ],
        "heritage": [
            wb(rf"National{HX}Trust"),
            wb(rf"English{HX}Heritage"),
            wb(rf"Historic{HX}England"),
            wb(r"UNESCO"),
        ],
        "manufacturing": [wb(r"manufacturing")],
        "software": [wb(r"software")],
    }

    groups = list(patterns.keys())
    compiled = {g: re.compile("|".join(patts), flags=re.IGNORECASE) for g, patts in patterns.items()}

    for g, rx in compiled.items():
        df[f"regex_{g}"] = df["_impact_text_norm"].str.contains(rx, na=False)
    for g in groups:
        df[f"regex_{g}"] = df[f"regex_{g}"].astype("int8")

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_base_instructions(version: str, ordered_groups: list[str]) -> str:
        if str(version).lower() != "v2":
            return (
                f"[PROMPT_VERSION={version}]\n"
                "You are classifying academic impact case studies into thematic indicators.\n"
                "Given the text, decide for EACH indicator whether it applies.\n\n"
                "The indicators are:\n"
                "  - charity\n"
                "  - startup\n"
                "  - patent\n"
                "  - museum\n"
                "  - nhs\n"
                "  - drug_trial\n"
                "  - school\n"
                "  - legislation\n"
                "  - heritage\n"
                "  - manufacturing\n"
                "  - software\n\n"
                "Interpret 'applies' as: the impact materially involves that type of activity\n"
                "(e.g. working with a charity, starting a company, influencing legislation),\n"
                "not just a passing mention. If you are uncertain, use false.\n\n"
                "Return ONLY valid json, a single JSON object of the form:\n"
                "{\n"
                '  \"charity\": true/false,\n'
                '  \"startup\": true/false,\n'
                '  \"patent\": true/false,\n'
                '  \"museum\": true/false,\n'
                '  \"nhs\": true/false,\n'
                '  \"drug_trial\": true/false,\n'
                '  \"school\": true/false,\n'
                '  \"legislation\": true/false,\n'
                '  \"heritage\": true/false,\n'
                '  \"manufacturing\": true/false,\n'
                '  \"software\": true/false\n'
                "}\n"
            )

        indicator_definitions = {
            "charity": "True only when charities/NGOs/third-sector actors are a material impact route, partner, or beneficiary.",
            "startup": "True only when startup/spinout creation, growth, or deployment is part of the impact pathway.",
            "patent": "True only when patents/patenting/licensing are materially involved in the impact.",
            "museum": "True only when museums/galleries/exhibitions are direct impact venues, partners, or beneficiaries.",
            "nhs": "True only when NHS bodies, services, pathways, or policy/practice are materially affected.",
            "drug_trial": "True only when drug/therapeutic development or clinical trial activity is materially involved.",
            "school": "True only when school-level policy/practice/curriculum/outcomes are materially affected.",
            "legislation": "True only when law/regulation/statutory guidance is created, changed, or implemented as impact.",
            "heritage": "True only when heritage institutions, assets, conservation policy, or practice are materially affected.",
            "manufacturing": "True only when industrial manufacturing processes, plants, or production outcomes are materially affected.",
            "software": "True only when software tools/platforms/systems are central to the delivered impact.",
        }
        defs_block = "\n".join([f"  - {g}: {indicator_definitions[g]}" for g in ordered_groups])
        return (
            f"[PROMPT_VERSION={version}]\n"
            "You are a conservative multi-label classifier for REF impact case studies.\n"
            "Given ITEMS (a JSON array with fields `id` and `text`), classify each item.\n\n"
            "Decision rules:\n"
            "1) Mark true only if that theme is materially involved in the impact claim.\n"
            "2) Passing mention, background context, or weak association => false.\n"
            "3) Multiple true labels are allowed.\n"
            "4) If uncertain, choose false.\n\n"
            "Indicator definitions:\n"
            f"{defs_block}\n\n"
            "Return only valid JSON in this structure:\n"
            "{\n"
            '  "results": [\n'
            "    {\n"
            '      "id": "<id-from-input>",\n'
            + ",\n".join([f'      "{g}": true/false' for g in ordered_groups])
            + "\n    }\n"
            "  ]\n"
            "}\n"
            "Include exactly one result object per input item id.\n"
        )

    def _json_schema_for_batch(ordered_groups: list[str]) -> dict[str, Any]:
        item_properties: dict[str, Any] = {"id": {"type": "string"}}
        for g in ordered_groups:
            item_properties[g] = {"type": "boolean"}
        item_required = ["id"] + ordered_groups
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": item_required,
                        "properties": item_properties,
                    },
                }
            },
        }

    def _extract_json_flags(raw_text: str, ordered_groups: list[str]) -> tuple[dict[str, int], str, str]:
        return _parse_thematic_single_flags(raw_text, ordered_groups)

    def _extract_batch_flags(
        raw_text: str,
        ordered_groups: list[str],
        expected_ids: list[str] | None = None,
    ) -> tuple[dict[str, dict[str, int]], str, str]:
        return _parse_thematic_batch_flags(raw_text, ordered_groups, expected_ids=expected_ids)

    def _chunked(items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _build_batch_prompt(instructions: str, batch_items: list[dict[str, str]]) -> str:
        return instructions + "\n\nITEMS:\n" + json.dumps(batch_items, ensure_ascii=False)

    def _responses_create_with_fallback(
        api_client: OpenAI,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str, str]:
        def _is_transient(exc: Exception) -> bool:
            msg = str(exc).lower()
            transient_markers = (
                "timeout",
                "temporar",
                "connection",
                "connect",
                "disconnect",
                "reset",
                "upstream",
                "no healthy upstream",
                "rate limit",
                "429",
                "500",
                "502",
                "503",
                "504",
                "service unavailable",
            )
            return any(marker in msg for marker in transient_markers)

        candidates: list[dict[str, Any]] = [request_kwargs]
        has_cache_fields = any(k in request_kwargs for k in ("prompt_cache_key", "prompt_cache_retention"))
        has_schema = "text" in request_kwargs
        if has_cache_fields:
            candidates.append({k: v for k, v in request_kwargs.items() if k not in {"prompt_cache_key", "prompt_cache_retention"}})
        if has_schema:
            candidates.append({k: v for k, v in request_kwargs.items() if k != "text"})
        if has_schema and has_cache_fields:
            candidates.append(
                {
                    k: v
                    for k, v in request_kwargs.items()
                    if k not in {"text", "prompt_cache_key", "prompt_cache_retention"}
                }
            )
        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = tuple(sorted(candidate.keys()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(candidate)

        errors: list[str] = []
        final_kwargs: dict[str, Any] | None = None
        resp = None
        for candidate in deduped:
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = api_client.responses.create(**candidate)
                    final_kwargs = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    if _is_transient(exc) and attempt < max_attempts:
                        sleep_for = min(60.0, 1.5 ** (attempt - 1))
                        print(
                            "Transient OpenAI thematic classification error; "
                            f"retrying in {sleep_for:.1f}s ({attempt}/{max_attempts}): {exc}",
                            file=sys.stderr,
                        )
                        time.sleep(sleep_for)
                        continue
                    errors.append(str(exc))
                    break
            if resp is not None:
                break
        if resp is None or final_kwargs is None:
            raise RuntimeError(errors[-1] if errors else "responses_create_failed")

        fallback_tags: list[str] = []
        if has_schema and "text" not in final_kwargs:
            fallback_tags.append("schema")
        if has_cache_fields and not any(k in final_kwargs for k in ("prompt_cache_key", "prompt_cache_retention")):
            fallback_tags.append("prompt_cache")

        if not fallback_tags:
            return resp, "ok", ""
        return resp, f"ok_{'_'.join(fallback_tags)}_fallback", "; ".join(errors[:2])

    def _log_llm_issue(prefix: str, cache_key: str, status: str, error: str) -> None:
        msg = str(error or "").strip()
        print(
            f"[LLM {prefix}] model={model} prompt={prompt_version} cache_key={cache_key} "
            f"status={status} error={msg}",
            file=sys.stderr,
        )

    def _build_request_kwargs(prompt_input: str, item_count: int, structured: bool) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt_input,
            "max_output_tokens": max(2048, 384 * int(item_count)) if structured else 512,
            "service_tier": service_tier,
        }
        model_l = str(model).lower()
        # Use one consistent reasoning setting across GPT-5 models for comparability.
        if model_l.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "low"}
            request_kwargs["max_output_tokens"] = max(int(request_kwargs["max_output_tokens"]), 8192 if structured else 2048)
        else:
            request_kwargs["temperature"] = 0.0
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = str(prompt_cache_key)
        if structured:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "thematic_indicators_batch",
                    "strict": True,
                    "schema": strict_json_schema_batch,
                }
            }
        return request_kwargs

    use_structured_outputs = str(prompt_version).lower() == "v2"
    base_instructions = _build_base_instructions(prompt_version, groups)
    strict_json_schema_batch = _json_schema_for_batch(groups)
    effective_batch_size = max(1, int(llm_batch_size)) if use_structured_outputs else 1

    def _cache_key(text: str) -> str:
        basis = f"{prompt_version}\n{model}\n{text}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["cache_key", "text", "model", "prompt_version", "llm_status", "llm_error"] + groups)

    for col in ["cache_key", "text", "model", "prompt_version", "llm_status", "llm_error"]:
        if col not in cache_df.columns:
            cache_df[col] = ""
    for g in groups:
        if g not in cache_df.columns:
            cache_df[g] = 0
        cache_df[g] = cache_df[g].fillna(0).astype("int8")

    cache_df["cache_key"] = cache_df["cache_key"].astype(str)
    cache_df = cache_df.drop_duplicates(subset=["cache_key"], keep="last")

    cache_map = {
        row.cache_key: {
            **{g: int(getattr(row, g)) for g in groups},
            "llm_status": getattr(row, "llm_status", "cached") or "cached",
            "llm_error": getattr(row, "llm_error", "") or "",
        }
        for row in cache_df.itertuples(index=False)
        if getattr(row, "cache_key", "")
    }

    df["_cache_key"] = df["_impact_text_norm"].apply(lambda t: _cache_key(t) if t and t.strip() else "")
    df["llm_status"] = "not_run"
    df["llm_error"] = ""
    for g in groups:
        df[f"llm_{g}"] = 0

    unique_texts = (
        df.loc[df["_cache_key"] != "", ["_cache_key", "_impact_text_norm"]]
        .drop_duplicates(subset=["_cache_key"])
        .set_index("_cache_key")["_impact_text_norm"]
        .to_dict()
    )
    retryable_statuses = {"disabled", "error", "parse_error"}
    keys_to_query: list[str] = []
    for k in unique_texts:
        cached = cache_map.get(k)
        if cached is None:
            keys_to_query.append(k)
            continue
        cached_status = str(cached.get("llm_status", "")).strip().lower()
        if llm_enabled and cached_status in retryable_statuses:
            keys_to_query.append(k)

    client = None
    llm_query_allowed = llm_enabled and bool(keys_to_query)
    llm_disable_reason = ""
    if llm_query_allowed:
        try:
            secret = read_secret(key_env_var, key_path, required=True)
            client = OpenAI(api_key=secret)
        except Exception as exc:  # noqa: BLE001
            llm_query_allowed = False
            llm_disable_reason = f"openai_key_unavailable: {exc}"
            warnings.warn(llm_disable_reason)

    new_cache_rows: list[dict[str, Any]] = []
    if keys_to_query and not llm_query_allowed:
        for ck in keys_to_query:
            new_cache_rows.append(
                {
                    "cache_key": ck,
                    "text": unique_texts[ck],
                    "model": model,
                    "prompt_version": prompt_version,
                    "llm_status": "disabled",
                    "llm_error": llm_disable_reason or "llm_disabled",
                    **{g: 0 for g in groups},
                }
            )

    def _batch_groups(keys: list[str]) -> list[list[str]]:
        grouped: list[list[str]] = []
        current: list[str] = []
        for ck in keys:
            cached_status = str(cache_map.get(ck, {}).get("llm_status", "")).strip().lower()
            force_single = cached_status in {"error", "parse_error"}
            if force_single:
                if current:
                    grouped.append(current)
                    current = []
                grouped.append([ck])
                continue
            current.append(ck)
            if len(current) >= effective_batch_size:
                grouped.append(current)
                current = []
        if current:
            grouped.append(current)
        return grouped

    def _run_openai_batch_thematic(batch_keys: list[str]) -> list[dict[str, Any]]:
        out_rows: list[dict[str, Any]] = []
        if not batch_keys:
            return out_rows
        root = Path(batch_dir) if batch_dir is not None else cache_path.parent / "batches"
        root.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model))
        safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", str(prompt_version))
        keys_digest = hashlib.sha256("\n".join(batch_keys).encode("utf-8")).hexdigest()[:16]
        stem = f"thematic_{safe_model}_{safe_prompt}_{keys_digest}"
        manifest_path = root / f"{stem}.manifest.json"
        jsonl_path = root / f"{stem}.input.jsonl"
        output_path = root / f"{stem}.output.jsonl"
        error_path = root / f"{stem}.errors.jsonl"
        index_path = root / f"{stem}.index.json"

        custom_index: dict[str, list[dict[str, str]]] = {}
        requests: list[dict[str, Any]] = []
        for request_number, ck_group in enumerate(_batch_groups(batch_keys), start=1):
            batch_items = [{"id": ck, "text": unique_texts[ck]} for ck in ck_group]
            if use_structured_outputs:
                prompt_input = _build_batch_prompt(base_instructions, batch_items)
            else:
                prompt_input = base_instructions + "\n\nTEXT:\n\"\"\"\n" + batch_items[0]["text"] + "\n\"\"\"\n"
            body = _build_request_kwargs(prompt_input, len(batch_items), use_structured_outputs)
            body.pop("service_tier", None)
            custom_id = f"thematic-{request_number:06d}-{hashlib.sha256('|'.join(ck_group).encode('utf-8')).hexdigest()[:12]}"
            custom_index[custom_id] = batch_items
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
            )

        if not index_path.exists():
            index_path.write_text(json.dumps(custom_index, indent=2, sort_keys=True), encoding="utf-8")
        else:
            custom_index = json.loads(index_path.read_text(encoding="utf-8"))

        state, manifest = create_or_retrieve_batch(
            client,
            project_root=default_project_root(),
            manifest_path=manifest_path,
            jsonl_path=jsonl_path,
            output_path=output_path,
            error_path=error_path,
            endpoint="/v1/responses",
            requests=requests,
            metadata={
                "project": "ref_gender",
                "task": "thematic_classification",
                "model": str(model),
                "prompt_version": str(prompt_version),
            },
            wait=bool(batch_wait),
            poll_interval_seconds=float(batch_poll_interval_seconds),
        )
        if state != "completed":
            raise OpenAIBatchPending(
                "OpenAI thematic batch is pending. "
                f"batch_id={manifest.get('batch_id')} status={manifest.get('status')} "
                f"manifest={format_relative_path(manifest_path)}. Re-run the same pipeline command later to collect it."
            )

        output_lines = read_jsonl(output_path)
        seen_custom_ids: set[str] = set()
        for line in output_lines:
            custom_id = str(line.get("custom_id", ""))
            seen_custom_ids.add(custom_id)
            batch_items = custom_index.get(custom_id, [])
            response = line.get("response") or {}
            status_code = int(response.get("status_code") or 0)
            request_error = line.get("error")
            if request_error or status_code >= 400:
                error_text = json.dumps(request_error or response.get("body") or response, ensure_ascii=False)
                for item in batch_items:
                    out_rows.append(
                        {
                            "cache_key": item["id"],
                            "text": item["text"],
                            "model": model,
                            "prompt_version": prompt_version,
                            "llm_status": "error",
                            "llm_error": error_text,
                            **{g: 0 for g in groups},
                        }
                    )
                continue

            raw = _response_output_text(response.get("body") or {})
            if use_structured_outputs:
                expected_ids = [item["id"] for item in batch_items]
                parsed_batch, parse_status, parse_error = _extract_batch_flags(raw, groups, expected_ids=expected_ids)
            else:
                flags, parse_status, parse_error = _extract_json_flags(raw, groups)
                parsed_batch = {batch_items[0]["id"]: flags} if batch_items else {}
            for item in batch_items:
                flags = parsed_batch.get(item["id"], {g: 0 for g in groups})
                status = "ok_batch" if parse_status == "ok" and item["id"] in parsed_batch else "parse_error"
                error = "" if status == "ok_batch" else parse_error or "missing_id_in_batch_response"
                out_rows.append(
                    {
                        "cache_key": item["id"],
                        "text": item["text"],
                        "model": model,
                        "prompt_version": prompt_version,
                        "llm_status": status,
                        "llm_error": error,
                        **flags,
                    }
                )

        missing_custom_ids = set(custom_index) - seen_custom_ids
        for custom_id in sorted(missing_custom_ids):
            for item in custom_index.get(custom_id, []):
                out_rows.append(
                    {
                        "cache_key": item["id"],
                        "text": item["text"],
                        "model": model,
                        "prompt_version": prompt_version,
                        "llm_status": "error",
                        "llm_error": "missing_batch_output_line",
                        **{g: 0 for g in groups},
                    }
                )
        return out_rows

    if llm_query_allowed and client is not None and str(openai_processing_mode).strip().lower() == "batch":
        new_cache_rows.extend(_run_openai_batch_thematic(keys_to_query))

    if llm_query_allowed and client is not None and str(openai_processing_mode).strip().lower() != "batch":
        force_single_item = False
        consecutive_repaired_batch_parse_errors = 0
        cursor = 0
        pbar = tqdm(
            total=len(keys_to_query),
            desc=f"LLM thematic classification ({model}, {service_tier})",
            unit="item",
        )
        while cursor < len(keys_to_query):
            current_batch_size = 1 if force_single_item else effective_batch_size
            ck_batch = keys_to_query[cursor : cursor + current_batch_size]
            cursor += current_batch_size
            batch_items: list[dict[str, str]] = []
            for ck in ck_batch:
                text = unique_texts[ck]
                if not text or not text.strip():
                    new_cache_rows.append(
                        {
                            "cache_key": ck,
                            "text": text,
                            "model": model,
                            "prompt_version": prompt_version,
                            "llm_status": "empty_text",
                            "llm_error": "",
                            **{g: 0 for g in groups},
                        }
                    )
                    continue
                batch_items.append({"id": ck, "text": text})
            if not batch_items:
                pbar.update(len(ck_batch))
                continue

            batch_status = "ok"
            batch_error = ""
            batch_parse_error = False
            batch_results: dict[str, dict[str, int]] = {}
            item_status_overrides: dict[str, tuple[str, str]] = {}
            try:
                if use_structured_outputs:
                    prompt_input = _build_batch_prompt(base_instructions, batch_items)
                else:
                    # v1 compatibility path: preserve one-text prompt shape.
                    prompt_input = base_instructions + "\n\nTEXT:\n\"\"\"\n" + batch_items[0]["text"] + "\n\"\"\"\n"
                request_kwargs = _build_request_kwargs(prompt_input, len(batch_items), use_structured_outputs)
                resp, status_fallback, error_fallback = _responses_create_with_fallback(client, request_kwargs)
                batch_status = status_fallback
                batch_error = error_fallback
                raw = _response_output_text(resp)
                response_detail = _response_diagnostic(resp)
                if use_structured_outputs:
                    expected_ids = [item["id"] for item in batch_items]
                    parsed_batch, parse_status, parse_error = _extract_batch_flags(raw, groups, expected_ids=expected_ids)
                    batch_results = parsed_batch
                else:
                    # v1 compatibility path (single-item only due effective_batch_size=1)
                    parsed_single, parse_status, parse_error = _extract_json_flags(raw, groups)
                    batch_results = {batch_items[0]["id"]: parsed_single}
                if parse_status != "ok":
                    if response_detail:
                        parse_error = "; ".join([p for p in (parse_error, response_detail) if p])
                    batch_parse_error = True
                    batch_status = parse_status
                    batch_error = parse_error
                    if len(batch_items) == 1:
                        _log_llm_issue("BATCH", ck_batch[0], batch_status, batch_error)

                # Repair partial/missing batch outputs by retrying only missing ids one-by-one.
                if use_structured_outputs:
                    missing_ids = [item["id"] for item in batch_items if item["id"] not in batch_results]
                    for miss_id in missing_ids:
                        miss_text = unique_texts.get(miss_id, "")
                        retry_flags = {g: 0 for g in groups}
                        retry_status = "parse_error"
                        retry_error = "missing_id_in_batch_response"
                        try:
                            retry_prompt = _build_batch_prompt(base_instructions, [{"id": miss_id, "text": miss_text}])
                            retry_kwargs = _build_request_kwargs(retry_prompt, 1, True)
                            retry_resp, retry_fb_status, retry_fb_error = _responses_create_with_fallback(client, retry_kwargs)
                            retry_raw = _response_output_text(retry_resp)
                            retry_detail = _response_diagnostic(retry_resp)
                            retry_parsed, retry_parse_status, retry_parse_error = _extract_batch_flags(
                                retry_raw,
                                groups,
                                expected_ids=[miss_id],
                            )
                            if retry_parse_status != "ok" and retry_detail:
                                retry_parse_error = "; ".join(
                                    [p for p in (retry_parse_error, retry_detail) if p]
                                )
                            parsed_item = retry_parsed.get(miss_id)
                            # If single-item retry returns one item with missing/mismatched id,
                            # still accept it because this retry is for exactly one cache key.
                            if parsed_item is None and len(retry_parsed) == 1:
                                parsed_item = next(iter(retry_parsed.values()))
                            # Final fallback: some models may emit a flat JSON object instead of
                            # {"results":[...]} on one-item retries.
                            if parsed_item is None:
                                flat_flags, flat_status, flat_error = _extract_json_flags(retry_raw, groups)
                                if flat_status == "ok":
                                    parsed_item = flat_flags
                                else:
                                    retry_parse_error = flat_error or retry_parse_error
                            if retry_parse_status == "ok" and parsed_item is not None:
                                retry_flags = parsed_item
                                retry_status = retry_fb_status
                                retry_error = retry_fb_error
                                batch_results[miss_id] = retry_flags
                            else:
                                retry_status = "parse_error"
                                retry_error = retry_parse_error or "missing_id_in_retry_response"
                            if not str(retry_status).startswith("ok"):
                                _log_llm_issue("RETRY", miss_id, retry_status, retry_error)
                        except Exception as retry_exc:  # noqa: BLE001
                            retry_status = "error"
                            retry_error = str(retry_exc)
                            warnings.warn(f"OpenAI retry failed for item {miss_id[:8]}...: {retry_exc}")
                            _log_llm_issue("RETRY", miss_id, retry_status, retry_error)
                        item_status_overrides[miss_id] = (retry_status, retry_error)

                    unrepaired_ids = [
                        item["id"]
                        for item in batch_items
                        if item["id"] not in batch_results
                        or not str(item_status_overrides.get(item["id"], ("ok", ""))[0]).startswith("ok")
                    ]
                    if batch_parse_error and len(batch_items) > 1 and not unrepaired_ids:
                        batch_status = status_fallback
                        batch_error = error_fallback
                        consecutive_repaired_batch_parse_errors += 1
                        if consecutive_repaired_batch_parse_errors >= 2 and not force_single_item:
                            force_single_item = True
                            print(
                                "[LLM BATCH] Structured batch responses are not parseable, "
                                "but single-item repair is working; switching remaining "
                                f"{len(keys_to_query) - cursor} items to single-item mode.",
                                file=sys.stderr,
                            )
                    elif batch_parse_error:
                        consecutive_repaired_batch_parse_errors = 0
            except Exception as exc:  # noqa: BLE001
                batch_status = "error"
                batch_error = str(exc)
                warnings.warn(f"OpenAI call failed for batch starting {ck_batch[0][:8]}...: {exc}")
                _log_llm_issue("BATCH", ck_batch[0], batch_status, batch_error)

            for item in batch_items:
                ck = item["id"]
                text = item["text"]
                flags = batch_results.get(ck, {g: 0 for g in groups})
                if ck in item_status_overrides:
                    item_status, item_error = item_status_overrides[ck]
                else:
                    item_status = batch_status
                    item_error = batch_error
                    if ck not in batch_results and batch_status.startswith("ok"):
                        item_status = "parse_error"
                        item_error = "missing_id_in_batch_response"
                if not str(item_status).startswith("ok") and str(item_status) not in {"cached", "empty_text"}:
                    _log_llm_issue("ITEM", ck, str(item_status), str(item_error))
                new_cache_rows.append(
                    {
                        "cache_key": ck,
                        "text": text,
                        "model": model,
                        "prompt_version": prompt_version,
                        "llm_status": item_status,
                        "llm_error": item_error,
                        **flags,
                    }
                )
            pbar.update(len(ck_batch))
        pbar.close()

    if new_cache_rows:
        new_df = pd.DataFrame(new_cache_rows)
        for g in groups:
            new_df[g] = new_df[g].fillna(0).astype("int8")
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["cache_key"], keep="last")
        atomic_write_csv(cache_df, cache_path)
        cache_map = {
            row.cache_key: {
                **{g: int(getattr(row, g)) for g in groups},
                "llm_status": getattr(row, "llm_status", "cached") or "cached",
                "llm_error": getattr(row, "llm_error", "") or "",
            }
            for row in cache_df.itertuples(index=False)
            if getattr(row, "cache_key", "")
        }
        if llm_query_allowed:
            status_lower = new_df["llm_status"].fillna("").astype(str).str.strip().str.lower()
            bad_mask = status_lower.isin({"error", "parse_error"})
            if bad_mask.any():
                bad_counts = status_lower[bad_mask].value_counts(dropna=False).to_dict()
                sample = new_df.loc[bad_mask, ["cache_key", "llm_status", "llm_error"]].head(5)
                raise RuntimeError(
                    "OpenAI thematic classification did not complete cleanly; "
                    f"bad statuses in new cache rows: {bad_counts}. "
                    f"Cache was written to {cache_path} for audit/retry. "
                    "Sample failures:\n"
                    f"{sample.to_string(index=False)}"
                )

    for idx, row in df.iterrows():
        ck = row["_cache_key"]
        if not ck:
            df.at[idx, "llm_status"] = "empty_text"
            continue
        cached = cache_map.get(ck)
        if not cached:
            # keep deterministic fallback so downstream can still run without LLM
            df.at[idx, "llm_status"] = "missing_cache_regex_fallback"
            for g in groups:
                df.at[idx, f"llm_{g}"] = int(df.at[idx, f"regex_{g}"])
            continue
        df.at[idx, "llm_status"] = cached.get("llm_status", "cached")
        df.at[idx, "llm_error"] = cached.get("llm_error", "")
        use_regex_proxy = cached.get("llm_status") in {"disabled", "error", "parse_error"}
        for g in groups:
            if use_regex_proxy:
                df.at[idx, f"llm_{g}"] = int(df.at[idx, f"regex_{g}"])
            else:
                df.at[idx, f"llm_{g}"] = int(cached.get(g, 0))

    for g in groups:
        df[f"llm_{g}"] = df[f"llm_{g}"].astype("int8")
    df = df.drop(columns=["_impact_text_norm", "_cache_key"])
    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the enhanced REF dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--output", type=str, default=None, help="Override output CSV path.")
    parser.add_argument("--force", action="store_true", help="Overwrite output CSV if it exists.")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip REF download steps.")
    parser.add_argument(
        "--prepare-source-only",
        action="store_true",
        help="Download/prepare REF source workbooks only; do not build enhanced_ref_data.",
    )
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--with-llm", action="store_true", help="Enable LLM thematic indicators.")
    llm_group.add_argument("--without-llm", action="store_true", help="Disable LLM thematic indicators.")
    parser.add_argument(
        "--backfill-model",
        type=str,
        default=None,
        help=(
            "Backfill thematic cache rows for a specific model/prompt from existing enhanced data "
            "(updates data/openai/categories.csv only; does not rewrite enhanced_ref_data outputs)."
        ),
    )
    parser.add_argument(
        "--backfill-prompt-version",
        type=str,
        default=None,
        help="Prompt version to use with --backfill-model (defaults to v1 for gpt-5.1, else config/default).",
    )
    parser.add_argument(
        "--backfill-service-tier",
        type=str,
        default="flex",
        help="Service tier for --backfill-model (default: flex).",
    )
    parser.add_argument(
        "--backfill-batch-size",
        type=int,
        default=None,
        help="LLM batch size for --backfill-model (defaults to config thematic_batch_size).",
    )
    parser.add_argument(
        "--backfill-prompt-cache-key",
        type=str,
        default=None,
        help="Optional prompt cache key override for --backfill-model (defaults to thematic_indicators_<prompt_version>).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    project_root = Path(args.project_root).resolve() if args.project_root else default_project_root()
    config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    (
        raw_path,
        edit_path,
        _sup_path,
        manual_path,
        final_path,
        _topic_path,
        _dim_path,
        _openalex_path,
        ics_staff_rows_path,
        ics_grants_path,
    ) = get_paths(paths)
    for p in (raw_path, edit_path, final_path, ics_staff_rows_path, ics_grants_path, manual_path):
        Path(p).mkdir(parents=True, exist_ok=True)
    mirror_legacy_raw_to_source(raw_path, paths.legacy_raw_dir)

    output_path = Path(args.output).resolve() if args.output else final_path / "enhanced_ref_data.parquet"
    analysis_parquet_path = final_path / "enhanced_ref_data.parquet"
    analysis_csv_path = final_path / "enhanced_ref_data.csv"
    legacy_csv_path = paths.legacy_final_dir / "enhanced_ref_data.csv"
    legacy_zip_path = paths.legacy_final_dir / "enhanced_ref_data.zip"

    backfill_mode = bool(args.backfill_model)
    prepare_source_only = bool(args.prepare_source_only)
    if output_path.exists() and not args.force and not backfill_mode and not prepare_source_only:
        print(f"Output already exists: {format_relative_path(output_path, project_root)}. Use --force to overwrite.")
        return 0

    started_at = datetime.now(timezone.utc)
    manifest_status = "success"
    manifest_notes = ""
    row_counts: dict[str, Any] = {}
    exit_code = 0
    params = {
        "llm_mode": "with_llm" if args.with_llm else ("without_llm" if args.without_llm else "config_default"),
        "skip_downloads": args.skip_downloads,
        "prepare_source_only": args.prepare_source_only,
        "output": output_path,
        "backfill_model": args.backfill_model,
        "backfill_prompt_version": args.backfill_prompt_version,
    }

    input_paths = {
        "raw_environment": raw_path / "raw_ref_environment_data.xlsx",
        "raw_results": raw_path / "raw_ref_results_data.xlsx",
        "raw_ics": raw_path / "raw_ref_ics_data.xlsx",
        "raw_ics_tags": raw_path / "raw_ref_ics_tags_data.xlsx",
        "raw_outputs": raw_path / "raw_ref_outputs_data.xlsx",
        "staff_case_level": Path(ics_staff_rows_path) / "ref_case_level.csv",
        "university_class": Path(manual_path) / "university_category" / "ref_unique_institutions.csv",
    }
    output_paths = {
        "enhanced_analysis_parquet": analysis_parquet_path,
        "enhanced_analysis_csv": analysis_csv_path,
        "enhanced_legacy_csv": legacy_csv_path,
        "enhanced_legacy_zip": legacy_zip_path,
        "enhanced_clean_final_csv": paths.legacy_final_dir / "enhanced_ref_data_clean_final.csv",
        "enhanced_clean_final_dta": paths.legacy_final_dir / "enhanced_ref_data_clean_final.dta",
        "enhanced_stata_column_map": paths.legacy_final_dir / "enhanced_ref_data_stata_column_map.csv",
        "clean_ref_dep_data": Path(edit_path) / "clean_ref_dep_data.xlsx",
        "clean_ref_ics_data": Path(edit_path) / "clean_ref_ics_data.xlsx",
        "llm_categories_cache": paths.data_dir / "openai" / "categories.csv",
    }

    try:
        openai_cfg = config.get("openai", {})
        openai_processing_mode = str(openai_cfg.get("processing_mode", "sync")).strip().lower()
        batch_wait = bool(openai_cfg.get("batch_wait", False))
        batch_poll_interval_seconds = float(openai_cfg.get("batch_poll_interval_seconds", 60))

        if backfill_mode:
            print(f"Backfilling thematic cache for model={args.backfill_model} ...")
            candidates = [
                final_path / "enhanced_ref_data.parquet",
                final_path / "enhanced_ref_data.csv",
                paths.legacy_final_dir / "enhanced_ref_data.csv",
            ]
            src_path = next((p for p in candidates if p.exists()), None)
            if src_path is None:
                raise FileNotFoundError(
                    "Backfill mode requires an existing enhanced dataset. "
                    f"Looked in: {', '.join(str(p) for p in candidates)}"
                )
            if src_path.suffix.lower() == ".parquet":
                df_backfill = pd.read_parquet(src_path)
            else:
                df_backfill = pd.read_csv(src_path)
            print(f"Loaded {len(df_backfill)} rows from {format_relative_path(src_path, paths.project_root)}")

            backfill_model = str(args.backfill_model)
            backfill_prompt_version = str(
                args.backfill_prompt_version
                if args.backfill_prompt_version is not None
                else ("v1" if backfill_model == "gpt-5.1" else openai_cfg.get("prompt_version", "v2"))
            )
            backfill_batch_size = int(
                args.backfill_batch_size
                if args.backfill_batch_size is not None
                else openai_cfg.get("thematic_batch_size", 12)
            )
            if backfill_batch_size < 1:
                backfill_batch_size = 1
            backfill_cache_key = (
                args.backfill_prompt_cache_key
                if args.backfill_prompt_cache_key is not None
                else f"thematic_indicators_{backfill_prompt_version}"
            )

            _ = get_thematic_indicators(
                df_backfill,
                llm_enabled=True,
                model=backfill_model,
                service_tier=str(args.backfill_service_tier or "flex"),
                prompt_version=backfill_prompt_version,
                llm_batch_size=backfill_batch_size,
                prompt_cache_key=backfill_cache_key,
                prompt_cache_retention=openai_cfg.get("prompt_cache_retention", "24h"),
                key_env_var=str(openai_cfg.get("key_env_var", "OPENAI_API_KEY")),
                key_path=paths.project_root / str(openai_cfg.get("key_file", "keys/OPENAI_API_KEY")),
                cache_path=paths.data_dir / "openai" / "categories.csv",
                openai_processing_mode=openai_processing_mode,
                batch_wait=batch_wait,
                batch_poll_interval_seconds=batch_poll_interval_seconds,
                batch_dir=paths.data_dir / "openai" / "batches",
            )
            row_counts = {"backfill_source_rows": int(len(df_backfill))}
            params.update(
                {
                    "mode": "backfill_cache_only",
                    "backfill_model": backfill_model,
                    "backfill_prompt_version": backfill_prompt_version,
                    "backfill_service_tier": str(args.backfill_service_tier or "flex"),
                    "backfill_batch_size": backfill_batch_size,
                }
            )
            print(
                "Backfill complete. Updated cache at "
                f"{format_relative_path(paths.data_dir / 'openai' / 'categories.csv', paths.project_root)} "
                f"for model={backfill_model}, prompt_version={backfill_prompt_version}."
            )
            return 0

        http_cfg = config.get("http", {})
        session = build_retry_session(
            max_retries=int(http_cfg.get("max_retries", 5)),
            backoff_factor=float(http_cfg.get("backoff_factor", 1.5)),
        )
        timeout_seconds = int(http_cfg.get("timeout_seconds", 60))

        if not args.skip_downloads:
            if not input_paths["raw_environment"].exists():
                get_environmental_data(raw_path, session=session, timeout_seconds=timeout_seconds)
            get_all_results(raw_path, session=session, timeout_seconds=timeout_seconds)
            if not (input_paths["raw_ics"].exists() and input_paths["raw_ics_tags"].exists()):
                get_impact_data(raw_path, session=session, timeout_seconds=timeout_seconds)
            if not input_paths["raw_outputs"].exists():
                get_output_data(raw_path, session=session, timeout_seconds=timeout_seconds)

        if prepare_source_only:
            raw_files = [
                input_paths["raw_environment"],
                input_paths["raw_results"],
                input_paths["raw_ics"],
                input_paths["raw_ics_tags"],
                input_paths["raw_outputs"],
            ]
            missing_raw = [str(p) for p in raw_files if not Path(p).exists()]
            if missing_raw:
                raise FileNotFoundError(
                    "REF source preparation did not produce all required workbooks: "
                    + ", ".join(missing_raw)
                )
            row_counts = {"prepared_source_files": len(raw_files)}
            params["mode"] = "prepare_source_only"
            print(f"Prepared REF source workbooks in: {format_relative_path(raw_path, paths.project_root)}")
            return 0

        clean_dep_level(raw_path, edit_path)
        df = clean_ics_level(raw_path, edit_path)
        df = load_dept_vars(df, edit_path)
        df = get_ics_staff_rows(df, ics_staff_rows_path)
        df = get_ics_grants(df, ics_grants_path)
        df = get_university_class(df, manual_path)
        df = get_panel_and_UoA_names(df)

        llm_enabled = bool(openai_cfg.get("enabled", True))
        if args.with_llm:
            llm_enabled = True
        if args.without_llm:
            llm_enabled = False

        df = get_thematic_indicators(
            df,
            llm_enabled=llm_enabled,
            model=str(openai_cfg.get("model", "gpt-5.5")),
            service_tier=str(openai_cfg.get("service_tier", "flex")),
            prompt_version=str(openai_cfg.get("prompt_version", "v2")),
            llm_batch_size=int(openai_cfg.get("thematic_batch_size", 12)),
            prompt_cache_key=openai_cfg.get("prompt_cache_key", "thematic_indicators_v2"),
            prompt_cache_retention=openai_cfg.get("prompt_cache_retention", "24h"),
            key_env_var=str(openai_cfg.get("key_env_var", "OPENAI_API_KEY")),
            key_path=paths.project_root / str(openai_cfg.get("key_file", "keys/OPENAI_API_KEY")),
            cache_path=paths.data_dir / "openai" / "categories.csv",
            openai_processing_mode=openai_processing_mode,
            batch_wait=batch_wait,
            batch_poll_interval_seconds=batch_poll_interval_seconds,
            batch_dir=paths.data_dir / "openai" / "batches",
        )

        df = validate_enhanced_ref_data(df)
        apply_enhanced_drift_checks(df, config.get("drift_checks", {}))
        row_counts = {"enhanced_ref_data_rows": int(len(df))}

        atomic_write_parquet(df, analysis_parquet_path)
        atomic_write_csv(df, analysis_csv_path)
        atomic_write_csv(df, legacy_csv_path)
        df.to_csv(legacy_zip_path, index=False, compression=dict(method="zip", archive_name="enhanced_ref_data.csv"))
        final_exports = write_final_clean_exports(df, paths.legacy_final_dir)
        if output_path.suffix.lower() == ".csv":
            atomic_write_csv(df, output_path)
        elif output_path.suffix.lower() == ".parquet":
            atomic_write_parquet(df, output_path)
        elif output_path != analysis_parquet_path:
            # default to CSV for unknown extension requests
            atomic_write_csv(df, output_path)
        print(f"Saved enhanced dataset to: {format_relative_path(output_path, paths.project_root)}")
        print(f"Saved cleaned final CSV to: {format_relative_path(final_exports['cleaned_csv_path'], paths.project_root)}")
        print(f"Saved Stata DTA to: {format_relative_path(final_exports['cleaned_dta_path'], paths.project_root)}")
        print(f"Saved Stata column map to: {format_relative_path(final_exports['column_map_path'], paths.project_root)}")
    except OpenAIBatchPending as exc:
        manifest_status = "pending"
        manifest_notes = str(exc)
        print(str(exc))
        exit_code = 75
    except Exception as exc:  # noqa: BLE001
        manifest_status = "failed"
        manifest_notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            paths.manifest_csv,
            project_root=paths.project_root,
            step="step01_make_enhanced_data",
            status=manifest_status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters=params,
            input_paths=input_paths,
            output_paths=output_paths,
            row_counts=row_counts,
            notes=manifest_notes,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
