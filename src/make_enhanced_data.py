# %%
import json
import os
import re
import sys
import unicodedata
import warnings
import hashlib
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


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


def get_impact_data(raw_path: Path) -> None:
    """Download ICS data + tags to raw_path (unrelated to results sheet)."""
    print("Getting ICS Data!")
    url = "https://results2021.ref.ac.uk/impact/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / "raw_ref_ics_data.xlsx", "wb").write(r.content)

    url = "https://results2021.ref.ac.uk/impact/export-tags-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / "raw_ref_ics_tags_data.xlsx", "wb").write(r.content)


def get_environmental_data(raw_path: Path) -> None:
    """Download Environmental data to raw_path."""
    print("Getting Environmental Data!")
    url = "https://results2021.ref.ac.uk/environment/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / "raw_ref_environment_data.xlsx", "wb").write(r.content)


def get_all_results(raw_path: Path) -> None:
    """
    Ensure the REF results workbook exists (XLSX). If missing, download it.
    """
    xlsx_path = raw_path / "raw_ref_results_data.xlsx"
    if xlsx_path.exists():
        print("Results XLSX present locally; not downloading.")
        return
    print("Getting Results Data (XLSX)…")
    url = "https://results2021.ref.ac.uk/profiles/export-all"
    r = requests.get(url, allow_redirects=True)
    open(xlsx_path, "wb").write(r.content)


def get_output_data(raw_path: Path) -> None:
    """Download Outputs data to raw_path."""
    print("Getting Outputs Data!")
    url = "https://results2021.ref.ac.uk/outputs/export-all"
    r = requests.get(url, allow_redirects=True)
    open(raw_path / "raw_ref_outputs_data.xlsx", "wb").write(r.content)


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


def get_paths(data_path: Path):
    """Build standard folder paths."""
    data_path = Path(data_path)
    raw_path = data_path / "raw"
    edit_path = data_path / "edit"
    sup_path = data_path / "supplementary"
    manual_path = data_path / "manual"
    final_path = data_path / "final"
    topic_path = data_path / "reassignments"
    dim_path = data_path / "dimensions_returns"
    openalex_path = data_path / "openalex_returns"
    ics_staff_rows_path = data_path / "ics_staff_rows"
    ics_grants_path = data_path / "ics_grants"
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
    if os.path.join(ics_staff_rows_path, "ref_case_level.csv") is False:
        print("No staff rows data! Run ref_staff.py")
        return df
    staff_rows = pd.read_csv(os.path.join(ics_staff_rows_path, "ref_case_level.csv"))
    print(f"Were missing {len(df)-len(staff_rows)} staff rows! investigate!")
    return pd.merge(df, staff_rows, how="left", on="REF impact case study identifier")


def get_ics_grants(df: pd.DataFrame, ics_grants_path: Path) -> pd.DataFrame:
    if os.path.join(ics_grants_path, "ICS_grants_aggregated.csv") is False:
        print("No ICS grants data! Run get_ics_grants.py")
        return df
    grants_rows = pd.read_csv(os.path.join(ics_grants_path, "ICS_grants_aggregated.csv"))
    print(f"Were missing {len(df)-len(grants_rows)} grants rows! investigate!")
    return pd.merge(df, grants_rows, how="left", on="REF impact case study identifier")


def get_university_class(df: pd.DataFrame, manual_path: Path) -> pd.DataFrame:
    if os.path.join(manual_path, "university_category", "ref_unique_institutions.csv") is False:
        print("No university classifications file! Go make one?!")
        return df
    university_class = pd.read_csv(os.path.join(manual_path, "university_category", "ref_unique_institutions.csv"))
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
    model: str = "gpt-5.1",
    key_path: str = "./keys/OPENAI_API_KEY",
    cache_path: str = "./data/openai/categories.csv",
) -> pd.DataFrame:
    """
    Add thematic indicator columns using both regex and an online LLM (GPT-5).

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

    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["text_hash", "text"] + groups)

    if "text_hash" not in cache_df.columns:
        cache_df["text_hash"] = ""
    if "text" not in cache_df.columns:
        cache_df["text"] = ""
    for g in groups:
        if g not in cache_df.columns:
            cache_df[g] = 0

    cache_df["text_hash"] = cache_df["text_hash"].astype(str)
    cache_df = cache_df.drop_duplicates(subset=["text_hash"], keep="last")
    for g in groups:
        cache_df[g] = cache_df[g].fillna(0).astype("int8")

    cache_map = {
        row.text_hash: {g: int(getattr(row, g)) for g in groups}
        for row in cache_df.itertuples(index=False)
        if getattr(row, "text_hash", "")
    }

    def _hash_text(t: str) -> str:
        if not t or not t.strip():
            return ""
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    df["_impact_hash"] = df["_impact_text_norm"].apply(_hash_text)

    for g in groups:
        df[f"llm_{g}"] = 0

    unique_map = (
        df.loc[df["_impact_hash"] != "", ["_impact_hash", "_impact_text_norm"]]
        .drop_duplicates(subset=["_impact_hash"])
        .set_index("_impact_hash")["_impact_text_norm"]
        .to_dict()
    )

    unique_hashes = list(unique_map.keys())
    hashes_to_query = [h for h in unique_hashes if h not in cache_map]

    if hashes_to_query:
        base_instructions = (
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

        new_cache_rows = []

        for h in tqdm(hashes_to_query, desc=f"LLM thematic classification ({model}, flex tier)", unit="text"):
            text = unique_map[h]

            if not text or not text.strip():
                result_flags = {g: 0 for g in groups}
                cache_map[h] = result_flags
                new_cache_rows.append({"text_hash": h, "text": text, **result_flags})
                continue

            prompt = base_instructions + "\n\nTEXT:\n\"\"\"\n" + text + "\n\"\"\"\n"

            try:
                resp = client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=256,
                    temperature=0.0,
                    service_tier="flex",
                )

                try:
                    raw = resp.output_text  # type: ignore[attr-defined]
                except AttributeError:
                    raw = str(resp)

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        try:
                            parsed = json.loads(raw[start : end + 1])
                        except json.JSONDecodeError:
                            warnings.warn(f"Failed to parse JSON for hash {h[:8]}...; raw response was:\n{raw}")
                            parsed = {}
                    else:
                        warnings.warn(f"Failed to parse JSON for hash {h[:8]}...; raw response was:\n{raw}")
                        parsed = {}

                result_flags = {g: int(bool(parsed.get(g, False))) for g in groups}

            except Exception as e:  # noqa: BLE001
                warnings.warn(f"OpenAI call failed for hash {h[:8]}...: {e}")
                result_flags = {g: 0 for g in groups}

            cache_map[h] = result_flags
            new_cache_rows.append({"text_hash": h, "text": text, **result_flags})

        if new_cache_rows:
            new_df = pd.DataFrame(new_cache_rows)
            for g in groups:
                new_df[g] = new_df[g].astype("int8")

            cache_df = pd.concat([cache_df, new_df], ignore_index=True)
            cache_df = cache_df.drop_duplicates(subset=["text_hash"], keep="last")
            cache_df.to_csv(cache_path, index=False)

    for idx, row in df.iterrows():
        h = row["_impact_hash"]
        if not h:
            continue
        flags = cache_map.get(h)
        if not flags:
            continue
        for g in groups:
            df.at[idx, f"llm_{g}"] = int(flags.get(g, 0))

    for g in groups:
        df[f"llm_{g}"] = df[f"llm_{g}"].astype("int8")

    df = df.drop(columns=["_impact_text_norm", "_impact_hash"])
    return df


if __name__ == "__main__":
    project_root = Path.cwd()
    data_path = project_root / "data"

    (
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
    ) = get_paths(data_path)

    csv_out = [arg for arg in sys.argv if ".csv" in arg]
    if csv_out:
        output_path = Path(csv_out[0])
        print(f"Will write final dataset to provided path: {output_path}")
        write = True
    else:
        output_path = final_path / "enhanced_ref_data.zip"
        if not output_path.exists():
            print(f"Will write final dataset to default path: {output_path}")
            write = True
        elif "-f" in sys.argv:
            print(f"WARNING: Will force overwrite dataset at default path: {output_path}")
            write = True
        else:
            print(
                "WARNING: Not overwriting final dataset as file already exists.\n"
                "Use -f to force overwrite or provide a custom path.\n"
                "Only intermittent files will be generated and saved."
            )
            write = False

    if not os.path.exists(raw_path):
        os.makedirs(raw_path, exist_ok=True)
    if not (raw_path / "raw_ref_environment_data.xlsx").exists():
        get_environmental_data(raw_path)
    get_all_results(raw_path)
    if not ((raw_path / "raw_ref_ics_tags_data.xlsx").exists() and (raw_path / "raw_ref_ics_data.xlsx").exists()):
        get_impact_data(raw_path)
    if not (raw_path / "raw_ref_outputs_data.xlsx").exists():
        get_output_data(raw_path)

    clean_dep_level(raw_path, edit_path)
    df = clean_ics_level(raw_path, edit_path)
    df = load_dept_vars(df, edit_path)
    df = get_ics_staff_rows(df, ics_staff_rows_path)
    df = get_university_class(df, manual_path)
    df = get_panel_and_UoA_names(df)
    df = get_thematic_indicators(df)

    if write:
        print(f'Writing final enhanced dataset to CSV (zipped): ', {output_path})
        df.to_csv(output_path, index=False, compression=dict(method="zip", archive_name="enhanced_ref_data.csv"))
