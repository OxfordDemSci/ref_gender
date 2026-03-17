"""
Fit OLS and GLM regressions used in Figure 2/Table 1 and persist artifacts.

Creates outputs/models/regression_results.pkl containing a dictionary with:
    - coef_df: tidy coefficient frame for plotting
    - var_order: variable ordering
    - latex_str: LaTeX regression table string
    - metrics_df: model metrics table
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import statsmodels.api as sm

try:  # pragma: no cover
    from .figure_one_helpers import DEFAULT_DATA_ROOT
    from .figure_two_regression import build_coef_df, build_regression_latex
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import read_table
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs, resolve_enhanced_ref_data_path
except ImportError:  # pragma: no cover
    from figure_one_helpers import DEFAULT_DATA_ROOT
    from figure_two_regression import build_coef_df, build_regression_latex
    from pipeline_config import load_config_and_paths
    from pipeline_io import read_table
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs, resolve_enhanced_ref_data_path


def _load_data(data_csv_path: Path) -> pd.DataFrame:
    df = read_table(data_csv_path)
    df = df.copy()
    df = df[df["number_male"] + df["number_female"] > 0]
    df["pct_female"] = df["number_female"] / (df["number_male"] + df["number_female"])
    df["MainPanel"] = df["Main Panel"]
    return df


def _fit_models(df: pd.DataFrame) -> Tuple[List[Tuple[str, object]], List[Tuple[str, object]]]:
    predictors_base = "C(MainPanel)"
    inst_cols = ["OxBridge", "RussellGroup", "Redbrick", "Ancient"]
    llm_cols = [
        "llm_museum",
        "llm_nhs",
        "llm_drug_trial",
        "llm_school",
        "llm_legislation",
        "llm_heritage",
        "llm_manufacturing",
        "llm_software",
        "llm_patent",
        "llm_startup",
        "llm_charity",
    ]

    for col in inst_cols + llm_cols:
        if col not in df:
            df[col] = 0
        df[col] = df[col].fillna(0)

    predictors_m1 = predictors_base
    predictors_m2 = predictors_base + " + " + " + ".join(inst_cols)
    predictors_m3 = predictors_m2 + " + " + " + ".join(llm_cols)

    weight_col = df["number_male"] + df["number_female"]

    ols_results = []
    for name, formula in [
        ("OLS (1)", f"pct_female ~ {predictors_m1}"),
        ("OLS (2)", f"pct_female ~ {predictors_m2}"),
        ("OLS (3)", f"pct_female ~ {predictors_m3}"),
    ]:
        model = sm.WLS.from_formula(formula, data=df, weights=weight_col)
        res = model.fit()
        ols_results.append((name, res))

    glm_results = []
    for name, formula in [
        ("GLM (1)", f"pct_female ~ {predictors_m1}"),
        ("GLM (2)", f"pct_female ~ {predictors_m2}"),
        ("GLM (3)", f"pct_female ~ {predictors_m3}"),
    ]:
        model = sm.GLM.from_formula(
            formula,
            data=df,
            family=sm.families.Binomial(),
            freq_weights=weight_col,
        )
        res = model.fit()
        glm_results.append((name, res))

    return ols_results, glm_results


def build_and_save_models(
    data_csv_path: Path,
    out_path: Path = Path("outputs/models/regression_results.pkl"),
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_data(data_csv_path)
    ols_results, glm_results = _fit_models(df)

    coef_df, var_order = build_coef_df(ols_results, glm_results)
    replacements = {
        "C(MainPanel)[T.B]": "Panel B",
        "C(MainPanel)[T.C]": "Panel C",
        "C(MainPanel)[T.D]": "Panel D",
        "RussellGroup": "Russell Group",
        "Redbrick": "Red Brick",
    }
    latex_str = build_regression_latex(
        ols_results=ols_results,
        glm_results=glm_results,
        var_order=var_order,
        replacements=replacements,
    )

    metrics_rows = []
    for name, res in ols_results + glm_results:
        metrics_rows.append(
            {
                "model": name,
                "N": int(res.nobs),
                "R-squared": getattr(res, "rsquared", None),
                "Adj. R2": getattr(res, "rsquared_adj", None),
                "Pseudo R2": getattr(res, "prsquared", None),
                "LogLik": getattr(res, "llf", None),
                "AIC": getattr(res, "aic", None),
                "BIC": getattr(res, "bic", None),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).set_index("model")

    payload = {
        "coef_df": coef_df,
        "var_order": var_order,
        "latex_str": latex_str,
        "metrics_df": metrics_df,
    }

    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved regression results to {out_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and persist REF regression models.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--data-csv", type=str, default=None, help="Override enhanced_ref_data CSV path.")
    parser.add_argument("--out", type=str, default=None, help="Output pickle path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    _config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    if args.data_csv:
        data_csv_path = Path(args.data_csv).resolve()
    else:
        try:
            data_csv_path = resolve_enhanced_ref_data_path(paths, must_exist=True)
        except FileNotFoundError:
            data_csv_path = DEFAULT_DATA_ROOT / "final" / "enhanced_ref_data.csv"

    out_path = Path(args.out).resolve() if args.out else (paths.outputs_dir / "models" / "regression_results.pkl")

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    try:
        build_and_save_models(data_csv_path=data_csv_path, out_path=out_path)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step05_build_regression_models",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={"data_csv": str(data_csv_path), "out_path": str(out_path)},
            input_paths={"enhanced_ref_data": data_csv_path},
            output_paths={"regression_results": out_path},
            row_counts={},
            notes=notes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
