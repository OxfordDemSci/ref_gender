import argparse
import hashlib
import json
import re
import shutil
import sys
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
    from .pipeline_paths import PipelinePaths, ensure_core_dirs
    from .pipeline_schema import validate_enhanced_ref_data
except ImportError:  # pragma: no cover
    from pipeline_config import load_config_and_paths
    from pipeline_drift import apply_enhanced_drift_checks
    from pipeline_io import atomic_write_csv, atomic_write_parquet, build_retry_session, download_file, read_secret
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import PipelinePaths, ensure_core_dirs
    from pipeline_schema import validate_enhanced_ref_data


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


def mirror_legacy_raw_to_bronze(raw_path: Path, legacy_raw_path: Path) -> None:
    """
    Copy legacy raw files into the bronze layer when bronze is empty.
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
        bronze_file = raw_path / name
        legacy_file = legacy_raw_path / name
        if not bronze_file.exists() and legacy_file.exists():
            shutil.copy2(legacy_file, bronze_file)


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
    Build canonical paths while preserving legacy compatibility directories.
    """
    raw_path = paths.bronze_dir
    edit_path = paths.silver_dir
    sup_path = paths.data_dir / "supplementary"
    manual_path = paths.manual_dir
    final_path = paths.gold_dir
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


def get_ics_staff_rows(df: pd.DataFrame, ics_staff_rows_path: Path) -> pd.DataFrame:
    staff_path = Path(ics_staff_rows_path) / "ref_case_level.csv"
    if not staff_path.exists():
        print(f"No staff rows data at {staff_path}; continuing without staff enrichment.")
        return df
    staff_rows = pd.read_csv(staff_path)
    print(f"Staff rows loaded: {len(staff_rows)}")
    return pd.merge(df, staff_rows, how="left", on="REF impact case study identifier")


def get_ics_grants(df: pd.DataFrame, ics_grants_path: Path) -> pd.DataFrame:
    grants_path = Path(ics_grants_path) / "ICS_grants_aggregated.csv"
    if not grants_path.exists():
        print(f"No ICS grants data at {grants_path}; continuing without grants enrichment.")
        return df
    grants_rows = pd.read_csv(grants_path)
    print(f"ICS grants rows loaded: {len(grants_rows)}")
    return pd.merge(df, grants_rows, how="left", on="REF impact case study identifier")


def get_university_class(df: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    class_path = Path(manual_path) / "university_category" / "ref_unique_institutions.csv"
    if not class_path.exists():
        print(f"No university classifications file at {class_path}; continuing without this lookup.")
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
    model: str = "gpt-5.4",
    service_tier: str = "flex",
    prompt_version: str = "v2",
    llm_batch_size: int = 12,
    prompt_cache_key: str | None = "thematic_indicators_v2",
    prompt_cache_retention: str | None = "24h",
    key_env_var: str = "OPENAI_API_KEY",
    key_path: str | Path | None = None,
    cache_path: str | Path = "./data/openai/categories.csv",
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
    COLS = ["1. Summary of the impact", "4. Details of the impact"]

    def _normalise_text(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str)
        s = s.str.replace("[\u2012\u2013\u2014\u2015]", "-", regex=True)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        return s

    df = df.copy()
    norm = _normalise_text(df[COLS[0]]) + " " + _normalise_text(df[COLS[1]])
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
        llm_status_local = "ok"
        llm_error_local = ""
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw_text[start : end + 1])
                except Exception:  # noqa: BLE001
                    parsed = {}
                    llm_status_local = "parse_error"
                    llm_error_local = "json_parse_failed"
            else:
                parsed = {}
                llm_status_local = "parse_error"
                llm_error_local = "json_parse_failed"
        return {g: int(bool(parsed.get(g, False))) for g in ordered_groups}, llm_status_local, llm_error_local

    def _extract_batch_flags(raw_text: str, ordered_groups: list[str]) -> tuple[dict[str, dict[str, int]], str, str]:
        llm_status_local = "ok"
        llm_error_local = ""
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw_text[start : end + 1])
                except Exception:  # noqa: BLE001
                    parsed = {}
                    llm_status_local = "parse_error"
                    llm_error_local = "json_parse_failed"
            else:
                parsed = {}
                llm_status_local = "parse_error"
                llm_error_local = "json_parse_failed"
        out: dict[str, dict[str, int]] = {}
        results = parsed.get("results", []) if isinstance(parsed, dict) else []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                if not item_id:
                    continue
                out[item_id] = {g: int(bool(item.get(g, False))) for g in ordered_groups}
        if llm_status_local == "ok" and not out:
            llm_status_local = "parse_error"
            llm_error_local = "batch_results_missing"
        return out, llm_status_local, llm_error_local

    def _chunked(items: list[str], size: int) -> list[list[str]]:
        return [items[i : i + size] for i in range(0, len(items), size)]

    def _build_batch_prompt(instructions: str, batch_items: list[dict[str, str]]) -> str:
        return instructions + "\n\nITEMS:\n" + json.dumps(batch_items, ensure_ascii=False)

    def _responses_create_with_fallback(
        api_client: OpenAI,
        request_kwargs: dict[str, Any],
    ) -> tuple[Any, str, str]:
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
            try:
                resp = api_client.responses.create(**candidate)
                final_kwargs = candidate
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
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
            "max_output_tokens": max(512, 128 * int(item_count)) if structured else 256,
            "service_tier": service_tier,
        }
        model_l = str(model).lower()
        # Use one consistent reasoning setting across GPT-5 models for comparability.
        if model_l.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "low"}
            request_kwargs["max_output_tokens"] = max(int(request_kwargs["max_output_tokens"]), 1536 if structured else 1024)
        else:
            request_kwargs["temperature"] = 0.0
        if prompt_cache_key:
            request_kwargs["prompt_cache_key"] = str(prompt_cache_key)
            if prompt_cache_retention:
                request_kwargs["prompt_cache_retention"] = str(prompt_cache_retention)
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

    if llm_query_allowed and client is not None:
        key_batches = _chunked(keys_to_query, size=effective_batch_size)
        for ck_batch in tqdm(key_batches, desc=f"LLM thematic classification ({model}, {service_tier})", unit="batch"):
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
                continue

            batch_status = "ok"
            batch_error = ""
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
                raw = getattr(resp, "output_text", str(resp)) or ""
                if use_structured_outputs:
                    parsed_batch, parse_status, parse_error = _extract_batch_flags(raw, groups)
                    batch_results = parsed_batch
                else:
                    # v1 compatibility path (single-item only due effective_batch_size=1)
                    parsed_single, parse_status, parse_error = _extract_json_flags(raw, groups)
                    batch_results = {batch_items[0]["id"]: parsed_single}
                if parse_status != "ok":
                    batch_status = parse_status
                    batch_error = parse_error
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
                            retry_raw = getattr(retry_resp, "output_text", str(retry_resp)) or ""
                            retry_parsed, retry_parse_status, retry_parse_error = _extract_batch_flags(retry_raw, groups)
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

    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
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
    mirror_legacy_raw_to_bronze(raw_path, paths.legacy_raw_dir)

    output_path = Path(args.output).resolve() if args.output else final_path / "enhanced_ref_data.parquet"
    gold_parquet_path = final_path / "enhanced_ref_data.parquet"
    gold_csv_path = final_path / "enhanced_ref_data.csv"
    legacy_csv_path = paths.legacy_final_dir / "enhanced_ref_data.csv"
    legacy_zip_path = paths.legacy_final_dir / "enhanced_ref_data.zip"

    backfill_mode = bool(args.backfill_model)
    if output_path.exists() and not args.force and not backfill_mode:
        print(f"Output already exists: {output_path}. Use --force to overwrite.")
        return 0

    started_at = datetime.now(timezone.utc)
    manifest_status = "success"
    manifest_notes = ""
    row_counts: dict[str, Any] = {}
    params = {
        "llm_mode": "with_llm" if args.with_llm else ("without_llm" if args.without_llm else "config_default"),
        "skip_downloads": args.skip_downloads,
        "output": str(output_path),
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
        "enhanced_gold_parquet": gold_parquet_path,
        "enhanced_gold_csv": gold_csv_path,
        "enhanced_legacy_csv": legacy_csv_path,
        "enhanced_legacy_zip": legacy_zip_path,
        "clean_ref_dep_data": Path(edit_path) / "clean_ref_dep_data.xlsx",
        "clean_ref_ics_data": Path(edit_path) / "clean_ref_ics_data.xlsx",
        "llm_categories_cache": paths.data_dir / "openai" / "categories.csv",
    }

    try:
        openai_cfg = config.get("openai", {})

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
            print(f"Loaded {len(df_backfill)} rows from {src_path}")

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
                f"{paths.data_dir / 'openai' / 'categories.csv'} "
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
            model=str(openai_cfg.get("model", "gpt-5.4")),
            service_tier=str(openai_cfg.get("service_tier", "flex")),
            prompt_version=str(openai_cfg.get("prompt_version", "v2")),
            llm_batch_size=int(openai_cfg.get("thematic_batch_size", 12)),
            prompt_cache_key=openai_cfg.get("prompt_cache_key", "thematic_indicators_v2"),
            prompt_cache_retention=openai_cfg.get("prompt_cache_retention", "24h"),
            key_env_var=str(openai_cfg.get("key_env_var", "OPENAI_API_KEY")),
            key_path=paths.project_root / str(openai_cfg.get("key_file", "keys/OPENAI_API_KEY")),
            cache_path=paths.data_dir / "openai" / "categories.csv",
        )

        df = validate_enhanced_ref_data(df)
        apply_enhanced_drift_checks(df, config.get("drift_checks", {}))
        row_counts = {"enhanced_ref_data_rows": int(len(df))}

        atomic_write_parquet(df, gold_parquet_path)
        atomic_write_csv(df, gold_csv_path)
        atomic_write_csv(df, legacy_csv_path)
        df.to_csv(legacy_zip_path, index=False, compression=dict(method="zip", archive_name="enhanced_ref_data.csv"))
        if output_path.suffix.lower() == ".csv":
            atomic_write_csv(df, output_path)
        elif output_path.suffix.lower() == ".parquet":
            atomic_write_parquet(df, output_path)
        elif output_path != gold_parquet_path:
            # default to CSV for unknown extension requests
            atomic_write_csv(df, output_path)
        print(f"Saved enhanced dataset to: {output_path}")
    except Exception as exc:  # noqa: BLE001
        manifest_status = "failed"
        manifest_notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            paths.manifest_csv,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
