"""
Fit OLS and GLM regressions used in Figure 2/Table 1 and persist artifacts.

Creates outputs/models/regression_results.pkl containing a dictionary with:
    - coef_df: tidy coefficient frame for plotting
    - var_order: variable ordering
    - latex_str: LaTeX regression table string
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import statsmodels.api as sm

from figure_one_helpers import DEFAULT_DATA_ROOT
from figure_two_regression import build_coef_df, build_regression_latex


def _load_data(data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(data_root / "final/enhanced_ref_data.csv")
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
    data_root: Path = DEFAULT_DATA_ROOT,
    out_path: Path = Path("outputs/models/regression_results.pkl"),
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_data(data_root)
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

    # Collect model-level metrics in a table for display
    metrics_rows = []
    for name, res in ols_results + glm_results:
        r2 = getattr(res, "rsquared", None)
        r2_adj = getattr(res, "rsquared_adj", None)
        pseudo = getattr(res, "prsquared", None)
        loglik = getattr(res, "llf", None)
        aic = getattr(res, "aic", None)
        bic = getattr(res, "bic", None)
        metrics_rows.append(
            {
                "model": name,
                "N": int(res.nobs),
                "R-squared": r2,
                "Adj. R2": r2_adj,
                "Pseudo R2": pseudo,
                "LogLik": loglik,
                "AIC": aic,
                "BIC": bic,
            }
        )
    metrics_df = pd.DataFrame(metrics_rows).set_index("model")

    payload = {
        "coef_df": coef_df,
        "var_order": var_order,
        "latex_str": latex_str,
        "metrics_df": metrics_df,
    }

    import pickle

    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved regression results to {out_path}")


if __name__ == "__main__":
    build_and_save_models()
