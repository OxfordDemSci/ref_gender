from pathlib import Path
import pickle
from typing import Callable, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col


ColorList = Sequence[str]
ModelResult = Tuple[str, object]  # (model_name, statsmodels result)


def pretty_label(var: str) -> str:
    """Human-readable labels for common dummy/indicator variables."""
    mapping = {
        "C(MainPanel)[T.B]": "Panel B",
        "C(MainPanel)[T.C]": "Panel C",
        "C(MainPanel)[T.D]": "Panel D",
        "MainPanel_B": "Panel B",
        "MainPanel_C": "Panel C",
        "MainPanel_D": "Panel D",
        "RussellGroup": "Russell Group",
        "Redbrick": "Red Brick",
    }
    if var in mapping:
        return mapping[var]
    if var.startswith("llm_"):
        tail = var[4:]
        return "NHS" if tail.lower() == "nhs" else tail.replace("_", " ").title()
    return var


def coef_table(result, model_name: str, estimator_name: str) -> pd.DataFrame:
    """Construct a tidy coefficient table from a statsmodels result."""
    params = result.params
    conf = result.conf_int()
    return pd.DataFrame(
        {
            "variable": params.index,
            "coef": params.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
            "model": model_name,
            "estimator": estimator_name,
        }
    )


def build_coef_df(
    ols_results: List[ModelResult],
    glm_results: List[ModelResult],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combine OLS/GLM results into a tidy coefficient frame and variable order.

    Assumes the richest OLS specification is the last in the ols_results list.
    """
    frames = []
    for name, res in ols_results:
        frames.append(coef_table(res, name, "OLS"))
    for name, res in glm_results:
        frames.append(coef_table(res, name, "GLM"))

    coef_df = pd.concat(frames, ignore_index=True)
    coef_df = coef_df[~coef_df["variable"].isin(["const", "Intercept"])]

    var_order = [v for v in ols_results[-1][1].params.index if v not in ("const", "Intercept")]
    coef_df["variable"] = pd.Categorical(coef_df["variable"], categories=var_order, ordered=True)
    coef_df["y"] = coef_df["variable"].cat.codes.astype(float)
    return coef_df, var_order


def plot_regression_coefficients(
    coef_df: pd.DataFrame,
    var_order: List[str],
    colors: ColorList = ("#B2182B", "#0072B2", "#E76F00"),
    label_fn: Callable[[str], str] = pretty_label,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Render side-by-side OLS/GLM coefficient plots."""
    variables = var_order
    n_vars = len(variables)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    display_labels = [label_fn(v) for v in variables]

    ax_left = axes[0]
    models_ols = sorted(coef_df.loc[coef_df["estimator"] == "OLS", "model"].unique())
    offsets_ols = np.linspace(-0.2, 0.2, len(models_ols))
    for offset, m, color in zip(offsets_ols, models_ols, colors):
        dfm = coef_df[(coef_df["estimator"] == "OLS") & (coef_df["model"] == m)]
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        ax_left.errorbar(
            dfm["coef"],
            dfm["y"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor="k",
            capsize=5,
            label=m,
            color=color,
        )
    ax_left.axvline(0, linestyle="--", linewidth=2, color="k")
    ax_left.set_yticks(range(n_vars))
    ax_left.set_yticklabels(display_labels)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Coefficient (on proportion scale)", fontsize=15)
    ax_left.legend(frameon=True, edgecolor="k", title="OLS", fontsize=12, facecolor=(1, 1, 1, 1), framealpha=1.0)

    ax_right = axes[1]
    models_glm = sorted(coef_df.loc[coef_df["estimator"] == "GLM", "model"].unique())
    offsets_glm = np.linspace(-0.2, 0.2, len(models_glm))
    for offset, m, color in zip(offsets_glm, models_glm, colors):
        dfm = coef_df[(coef_df["estimator"] == "GLM") & (coef_df["model"] == m)]
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        ax_right.errorbar(
            dfm["coef"],
            dfm["y"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor="k",
            color=color,
            capsize=5,
            label=m,
        )
    ax_right.axvline(0, linestyle="--", linewidth=2, color="k")
    ax_right.set_xlabel("Coefficient (log-odds)", fontsize=15)
    ax_right.legend(
        frameon=True,
        title="Binomial GLM (logit)",
        edgecolor="k",
        fontsize=12,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )
    ax_right.set_yticks(range(n_vars))
    ax_right.set_ylim(ax_left.get_ylim())
    ax_right.set_yticklabels([])

    ax_left.set_title("a.", loc="left", fontweight="bold", fontsize=17)
    ax_right.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax_left.grid(linestyle="--", color="grey", alpha=0.15)
    ax_right.grid(linestyle="--", color="grey", alpha=0.15)

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig, (ax_left, ax_right)


def _pseudo_r2(res) -> str:
    if hasattr(res, "prsquared"):
        return f"{res.prsquared:.3f}"
    return ""


def _r2(res) -> str:
    if hasattr(res, "rsquared"):
        return f"{res.rsquared:.3f}"
    return ""


def _r2_adj(res) -> str:
    if hasattr(res, "rsquared_adj"):
        return f"{res.rsquared_adj:.3f}"
    return ""


def build_regression_latex(
    ols_results: List[ModelResult],
    glm_results: List[ModelResult],
    var_order: List[str],
    replacements: dict | None = None,
) -> str:
    """Create LaTeX regression table string for all models."""
    info_dict = {
        "N": lambda x: f"{int(x.nobs)}",
        "R-squared": _r2,
        "Adj. R2": _r2_adj,
        "Pseudo R2": _pseudo_r2,
        "LogLik": lambda x: f"{x.llf:.1f}",
        "AIC": lambda x: f"{x.aic:.1f}",
        "BIC": lambda x: f"{x.bic:.1f}",
    }
    regressor_order = ["const"] + var_order
    results = [res for _, res in ols_results + glm_results]
    model_names = [f"{name}" for name, _ in ols_results + glm_results]

    summary = summary_col(
        results=results,
        stars=True,
        float_format="%.3f",
        model_names=model_names,
        info_dict=info_dict,
        regressor_order=regressor_order,
    )
    latex_str = summary.as_latex()
    if replacements:
        for old, new in replacements.items():
            latex_str = latex_str.replace(old, new)
    return latex_str


def save_figure(fig: plt.Figure, out_dir: Path, basename: str = "regressions"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.svg")
    fig.savefig(out_dir / f"{basename}.png", dpi=800)


def save_latex(latex_str: str, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_str, encoding="utf-8")


def load_prefit_results(path: Path) -> Tuple[List[ModelResult], List[ModelResult]]:
    """
    Load pre-fit OLS/GLM results from a pickle file.

    Expected format: (ols_results, glm_results) where each is a list of (name, result).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Regression results pickle not found at {path}. Provide (ols_results, glm_results) or artifacts dict."
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, tuple) and len(payload) == 2:
        ols_results, glm_results = payload
        return ols_results, glm_results
    if isinstance(payload, dict) and "coef_df" in payload:
        raise ValueError(
            "Regression pickle now stores artifacts (coef_df/var_order/latex_str). "
            "Use load_regression_artifacts instead of load_prefit_results."
        )
    raise ValueError("Unexpected regression results format. Expected (ols_results, glm_results) or artifacts dict.")


def load_regression_artifacts(path: Path) -> dict:
    """
    Load precomputed regression artifacts:
        - coef_df: tidy coefficients DataFrame
        - var_order: list of variable order
        - latex_str: LaTeX table string
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Regression artifacts not found at {path}.")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and {"coef_df", "var_order", "latex_str"} <= set(payload.keys()):
        return payload
    raise ValueError("Regression artifacts file missing expected keys (coef_df, var_order, latex_str).")
