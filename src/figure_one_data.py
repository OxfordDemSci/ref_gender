from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from figure_one_helpers import DEFAULT_DATA_ROOT, DEFAULT_UNICLASS_PATH, uoa_to_panel

# REF 2021 Units of Assessment
UOA_MAP = {
    1: "Clinical Medicine",
    2: "Public Health, Health Services and Primary Care",
    3: "Allied Health Professions, Dentistry, Nursing and Pharmacy",
    4: "Psychology, Psychiatry and Neuroscience",
    5: "Biological Sciences",
    6: "Agriculture, Food and Veterinary Sciences",
    7: "Earth Systems and Environmental Sciences",
    8: "Chemistry",
    9: "Physics",
    10: "Mathematical Sciences",
    11: "Computer Science and Informatics",
    12: "Engineering",
    13: "Architecture, Built Environment and Planning",
    14: "Geography and Environmental Studies",
    15: "Archaeology",
    16: "Economics and Econometrics",
    17: "Business and Management Studies",
    18: "Law",
    19: "Politics and International Studies",
    20: "Social Work and Social Policy",
    21: "Sociology",
    22: "Anthropology and Development Studies",
    23: "Education",
    24: "Sport and Exercise Sciences, Leisure and Tourism",
    25: "Area Studies",
    26: "Modern Languages and Linguistics",
    27: "English Language and Literature",
    28: "History",
    29: "Classics",
    30: "Philosophy",
    31: "Theology and Religious Studies",
    32: "Art and Design: History, Practice and Theory",
    33: "Music, Drama, Dance, Performing Arts, Film and Screen Studies",
    34: "Communication, Cultural and Media Studies, Library and Information Management",
}

STEM_UOAS = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]


def _load_inputs(
    data_root: Path = DEFAULT_DATA_ROOT,
    uniclass_path: Path = DEFAULT_UNICLASS_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw ICS/output tables and the university classification lookup."""
    data_root = Path(data_root)
    uniclass_path = Path(uniclass_path)
    df_output = pd.read_csv(data_root / "dimensions_outputs/outputs_concat_with_positive_authors.csv")
    df_ics = pd.read_csv(data_root / "final/enhanced_ref_data.csv")
    df_uniclass = pd.read_csv(uniclass_path)
    return df_output, df_ics, df_uniclass


def _pct_female(df: pd.DataFrame, female_col: str, male_col: str, target: str) -> pd.DataFrame:
    df[target] = df[female_col] / (df[male_col] + df[female_col])
    return df


def _group_person_counts(df: pd.DataFrame, group_cols) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)[["number_people", "number_male", "number_female", "number_unknown"]]
        .sum()
    )


def prepare_figure_one_data(
    data_root: Path = DEFAULT_DATA_ROOT,
    uniclass_path: Path = DEFAULT_UNICLASS_PATH,
):
    """
    Build all data frames needed for Figure 1.

    Returns df_ics (with Panel), df_uoa_m, df_uni_m, df_uniuoa_m.
    """
    df_output, df_ics, df_uniclass = _load_inputs(data_root, uniclass_path)

    # --- Institution level ---
    df_group_uni_ics = _group_person_counts(df_ics, "Institution name")
    _pct_female(df_group_uni_ics, "number_female", "number_male", "pct_female_ics")

    df_group_uni_output = _group_person_counts(df_output, "Institution name")
    _pct_female(df_group_uni_output, "number_female", "number_male", "pct_female_output")

    df_uni_m = pd.merge(
        df_group_uni_output,
        df_group_uni_ics,
        how="left",
        on="Institution name",
    )

    # --- UoA level ---
    df_group_uoa_ics = _group_person_counts(df_ics, "Unit of assessment number")
    _pct_female(df_group_uoa_ics, "number_female", "number_male", "pct_female_ics")

    df_group_uoa_output = _group_person_counts(df_output, "Unit of assessment number")
    _pct_female(df_group_uoa_output, "number_female", "number_male", "pct_female_output")

    df_uoa_m = pd.merge(
        df_group_uoa_output,
        df_group_uoa_ics,
        how="left",
        on="Unit of assessment number",
    )

    df_uoa_m["Unit of assessment name"] = df_uoa_m["Unit of assessment number"].map(UOA_MAP)
    df_uoa_m["Discipline_group"] = np.where(
        df_uoa_m["Unit of assessment number"].isin(STEM_UOAS),
        "STEM",
        "SHAPE",
    )
    df_uoa_m["Panel"] = df_uoa_m["Unit of assessment number"].apply(uoa_to_panel)

    # --- Institution + UoA level ---
    df_group_uniuoa_ics = _group_person_counts(df_ics, ["inst_id", "Unit of assessment number"])
    _pct_female(df_group_uniuoa_ics, "number_female", "number_male", "pct_female_ics")

    df_group_uniuoa_output = _group_person_counts(df_output, ["Institution UKPRN code", "Unit of assessment number"])
    _pct_female(df_group_uniuoa_output, "number_female", "number_male", "pct_female_output")

    df_uniuoa_m = pd.merge(
        df_group_uniuoa_output,
        df_group_uniuoa_ics,
        how="left",
        left_on=["Institution UKPRN code", "Unit of assessment number"],
        right_on=["inst_id", "Unit of assessment number"],
        suffixes=("_output", "_ics"),
    )

    # --- Lookups ---
    df_uni_m = pd.merge(df_uni_m, df_uniclass, how="left", on="Institution name")
    df_ics["Panel"] = df_ics["Unit of assessment number"].apply(uoa_to_panel)

    return df_ics, df_uoa_m, df_uni_m, df_uniuoa_m


def save_wrangled(out_dir: Path, df_uoa_m: pd.DataFrame, df_uni_m: pd.DataFrame, df_uniuoa_m: pd.DataFrame):
    """Persist cleaned tables used by the plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_uoa_m.to_csv(out_dir / "uoa_gender.csv", index=False)
    df_uni_m.to_csv(out_dir / "uni_gender.csv", index=False)
    df_uniuoa_m.to_csv(out_dir / "uniunoa_gender.csv", index=False)
