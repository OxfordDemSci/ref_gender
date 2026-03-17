from __future__ import annotations

from typing import Any

import pandas as pd


def _check_range(name: str, value: float, min_value: float | None, max_value: float | None) -> None:
    if min_value is not None and value < min_value:
        raise ValueError(f"Drift check failed for {name}: {value} < min {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"Drift check failed for {name}: {value} > max {max_value}")


def _check_null_rate(df: pd.DataFrame, column: str, max_rate: float) -> None:
    if column not in df.columns:
        raise ValueError(f"Drift check missing expected column: {column}")
    rate = float(df[column].isna().mean())
    if rate > max_rate:
        raise ValueError(f"Drift check failed for null rate {column}: {rate:.4f} > {max_rate:.4f}")


def apply_enhanced_drift_checks(df: pd.DataFrame, drift_cfg: dict[str, Any]) -> None:
    cfg = drift_cfg.get("enhanced_ref_data", {}) if isinstance(drift_cfg, dict) else {}
    _check_range(
        "enhanced_ref_data_rows",
        float(len(df)),
        cfg.get("row_count_min"),
        cfg.get("row_count_max"),
    )
    for col, max_rate in (cfg.get("max_null_rate", {}) or {}).items():
        _check_null_rate(df, col, float(max_rate))


def apply_outputs_drift_checks(any_df: pd.DataFrame, positive_df: pd.DataFrame, drift_cfg: dict[str, Any]) -> None:
    cfg = drift_cfg.get("outputs_concat", {}) if isinstance(drift_cfg, dict) else {}
    _check_range(
        "outputs_any_rows",
        float(len(any_df)),
        cfg.get("row_count_any_min"),
        cfg.get("row_count_any_max"),
    )
    _check_range(
        "outputs_positive_rows",
        float(len(positive_df)),
        cfg.get("row_count_positive_min"),
        cfg.get("row_count_positive_max"),
    )
    if len(any_df) <= len(positive_df):
        raise ValueError(
            f"Outputs contract violated: any rows ({len(any_df)}) must be greater than positive rows ({len(positive_df)})."
        )
    if "number_people" in any_df.columns:
        _check_range(
            "outputs_zero_author_rows",
            float((pd.to_numeric(any_df["number_people"], errors="coerce").fillna(0) == 0).sum()),
            cfg.get("zero_author_rows_min", 1),
            None,
        )
    for col, max_rate in (cfg.get("max_null_rate", {}) or {}).items():
        _check_null_rate(any_df, col, float(max_rate))

