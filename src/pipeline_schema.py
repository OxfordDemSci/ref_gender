from __future__ import annotations

from typing import Iterable

import pandas as pd

try:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema

    HAS_PANDERA = True
except ModuleNotFoundError:  # pragma: no cover
    pa = None
    Check = None
    Column = None
    DataFrameSchema = None
    HAS_PANDERA = False


ENHANCED_REQUIRED = [
    "REF impact case study identifier",
    "Institution name",
    "Unit of assessment number",
    "number_male",
    "number_female",
    "number_unknown",
    "number_people",
]

OUTPUTS_REQUIRED = [
    "Institution UKPRN code",
    "Institution name",
    "Main panel",
    "Unit of assessment number",
    "number_male",
    "number_female",
    "number_unknown",
    "number_people",
]


def _assert_required_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{dataset_name} missing required columns: {joined}")


def _assert_non_negative(df: pd.DataFrame, cols: Iterable[str], dataset_name: str) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        bad = pd.to_numeric(df[col], errors="coerce").fillna(0) < 0
        if bool(bad.any()):
            raise ValueError(f"{dataset_name} contains negative values in column '{col}'.")


def validate_enhanced_ref_data(df: pd.DataFrame, use_pandera: bool = True) -> pd.DataFrame:
    _assert_required_columns(df, ENHANCED_REQUIRED, "enhanced_ref_data")
    _assert_non_negative(df, ("number_male", "number_female", "number_unknown", "number_people"), "enhanced_ref_data")

    if use_pandera and HAS_PANDERA:
        schema = DataFrameSchema(
            {
                "REF impact case study identifier": Column(str, nullable=False),
                "Institution name": Column(str, nullable=False),
                "Unit of assessment number": Column(int, nullable=False),
                "number_male": Column(float, checks=Check.ge(0), nullable=True),
                "number_female": Column(float, checks=Check.ge(0), nullable=True),
                "number_unknown": Column(float, checks=Check.ge(0), nullable=True),
                "number_people": Column(float, checks=Check.ge(0), nullable=True),
            },
            strict=False,
            coerce=False,
        )
        return schema.validate(df, lazy=True)
    return df


def validate_outputs_concat(df: pd.DataFrame, use_pandera: bool = True) -> pd.DataFrame:
    _assert_required_columns(df, OUTPUTS_REQUIRED, "outputs_concat_with_positive_authors")
    _assert_non_negative(
        df,
        ("number_male", "number_female", "number_unknown", "number_people"),
        "outputs_concat_with_positive_authors",
    )

    if use_pandera and HAS_PANDERA:
        schema = DataFrameSchema(
            {
                "Institution UKPRN code": Column(float, nullable=True),
                "Institution name": Column(str, nullable=False),
                "Main panel": Column(str, nullable=True),
                "Unit of assessment number": Column(float, nullable=True),
                "number_male": Column(float, checks=Check.ge(0), nullable=True),
                "number_female": Column(float, checks=Check.ge(0), nullable=True),
                "number_unknown": Column(float, checks=Check.ge(0), nullable=True),
                "number_people": Column(float, checks=Check.ge(0), nullable=True),
            },
            strict=False,
            coerce=False,
        )
        return schema.validate(df, lazy=True)
    return df

