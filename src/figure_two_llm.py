from pathlib import Path
import re
from typing import Iterable, Tuple, Sequence, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import blended_transform_factory
import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm

try:  # pragma: no cover
    from .figure_one_helpers import (
        DEFAULT_DATA_ROOT,
        PANEL_COLORS,
        apply_mpl_defaults,
        resolve_enhanced_ref_data_path,
    )
    from .figure_one_plots import _format_uoa_label, _load_uoa_label_lookup
    from .figure_two_regression import build_coef_df, load_regression_artifacts, pretty_label, save_figure
    from .statistics_helpers import _ensure_panel, llm_female_share_tables
    from .pipeline_io import read_table
except ImportError:  # pragma: no cover
    from figure_one_helpers import (
        DEFAULT_DATA_ROOT,
        PANEL_COLORS,
        apply_mpl_defaults,
        resolve_enhanced_ref_data_path,
    )
    from figure_one_plots import _format_uoa_label, _load_uoa_label_lookup
    from figure_two_regression import build_coef_df, load_regression_artifacts, pretty_label, save_figure
    from statistics_helpers import _ensure_panel, llm_female_share_tables
    from pipeline_io import read_table


PanelOrder = Iterable[str]
NON_VIOLIN_ALPHA = 0.65
MEAN_LINE_WIDTH = 1.8
PANEL_DISPLAY_LABELS = {
    "A": "Panel A:\nLife Sciences",
    "B": "Panel B:\nPhysical Sciences",
    "C": "Panel C:\nSocial Sciences",
    "D": "Panel D:\nHumanities",
}
MODEL_COLORS = ("#1B9E77", "#D95F02", "#7570B3")
DISCIPLINE_VARS = ("C(MainPanel)[T.B]", "C(MainPanel)[T.C]", "C(MainPanel)[T.D]")
UNIVERSITY_VARS = ("OxBridge", "Redbrick", "Ancient", "RussellGroup")
IMPACT_DOMAIN_VARS = (
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
)


def load_llm_tables(data_root: Path = DEFAULT_DATA_ROOT, panel_order: PanelOrder = ("A", "B", "C", "D")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ICS data and compute llm_* female share tables.

    Returns (overall_table, panel_table) where pct_female/share_of_ics are proportions.
    """
    data_root = Path(data_root)
    df_ics = read_table(resolve_enhanced_ref_data_path(data_root))
    df_ics = _ensure_panel(df_ics)
    return llm_female_share_tables(df_ics, panel_order)


def _topic_female_share_tables(
    df_ics: pd.DataFrame,
    *,
    prefix: str,
    panel_order: PanelOrder = ("A", "B", "C", "D"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall + by-panel female-share tables for binary topic columns with a shared prefix."""
    frame = df_ics.copy()
    topic_cols: List[str] = [c for c in frame.columns if c.startswith(prefix)]
    for col in topic_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    for col in ("number_female", "number_male"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    if not topic_cols:
        empty_cols = ["llm_topic", "n_cases", "share_of_ics", "female", "total_people", "pct_female"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols + ["panel"])

    panel_order = list(panel_order)

    def _rows(local_df: pd.DataFrame, extra_cols: dict | None = None) -> list[dict]:
        records = []
        for col in topic_cols:
            subset = local_df[local_df[col] > 0]
            female = subset["number_female"].sum()
            male = subset["number_male"].sum()
            total_people = female + male
            records.append(
                {
                    "llm_topic": col.replace(prefix, "").replace("_", " ").title(),
                    "n_cases": len(subset),
                    "share_of_ics": len(subset) / len(local_df) if len(local_df) else np.nan,
                    "female": female,
                    "total_people": total_people,
                    "pct_female": female / total_people if total_people else np.nan,
                    **(extra_cols or {}),
                }
            )
        return records

    overall = pd.DataFrame(_rows(frame)).sort_values("llm_topic").reset_index(drop=True)
    panel_records: List[dict] = []
    for panel in panel_order:
        panel_df = frame[frame["Panel"] == panel]
        panel_records.extend(_rows(panel_df, {"panel": panel}))
    by_panel = pd.DataFrame(panel_records).sort_values(["panel", "llm_topic"]).reset_index(drop=True)
    return overall, by_panel


def load_regex_tables(
    data_root: Path = DEFAULT_DATA_ROOT,
    panel_order: PanelOrder = ("A", "B", "C", "D"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ICS data and compute regex_* female share tables.

    Returns (overall_table, panel_table) where pct_female/share_of_ics are proportions.
    """
    data_root = Path(data_root)
    df_ics = read_table(resolve_enhanced_ref_data_path(data_root))
    df_ics = _ensure_panel(df_ics)
    return _topic_female_share_tables(df_ics, prefix="regex_", panel_order=panel_order)


def load_regression(data_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load regression artifacts (coef_df, var_order) from pickle path rooted at data_root."""
    artifacts = load_regression_artifacts(Path(data_root) / "outputs" / "models" / "regression_results.pkl")
    return artifacts["coef_df"], artifacts["var_order"]


def load_uoa_regression(data_root: Path = DEFAULT_DATA_ROOT) -> tuple[pd.DataFrame, List[str], List[str], dict[str, str]]:
    """
    Fit UoA-based regressions for Supplementary Figure 3.

    Replaces MainPanel controls with Unit-of-Assessment dummies:
    C(UoA_num_cat, Treatment(reference=1)).
    """
    data_root = Path(data_root)
    df = read_table(resolve_enhanced_ref_data_path(data_root)).copy()
    df = _ensure_panel(df)
    for col in ("number_female", "number_male"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df[(df["number_male"] + df["number_female"]) > 0].copy()
    if df.empty:
        raise ValueError("No rows with positive male/female counts available for UoA regressions.")

    df["pct_female"] = df["number_female"] / (df["number_male"] + df["number_female"])
    df["uoa_num"] = pd.to_numeric(df["Unit of assessment number"], errors="coerce")
    df = df[df["uoa_num"].notna()].copy()
    if df.empty:
        raise ValueError("No valid 'Unit of assessment number' values found for UoA regressions.")
    df["uoa_num"] = df["uoa_num"].astype(int)

    uoa_categories = sorted(df["uoa_num"].dropna().unique().tolist())
    if 1 not in uoa_categories:
        raise ValueError("UoA reference category 1 (Clinical Medicine) is missing from the regression dataset.")
    df["UoA_num_cat"] = pd.Categorical(df["uoa_num"], categories=uoa_categories, ordered=True)

    predictors_base = "C(UoA_num_cat, Treatment(reference=1))"
    predictors_m1 = predictors_base
    predictors_m2 = predictors_base + " + " + " + ".join(UNIVERSITY_VARS)
    predictors_m3 = predictors_m2 + " + " + " + ".join(IMPACT_DOMAIN_VARS)

    for col in UNIVERSITY_VARS + IMPACT_DOMAIN_VARS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    weight_col = df["number_male"] + df["number_female"]
    ols_results = []
    for name, formula in [
        ("OLS (1)", f"pct_female ~ {predictors_m1}"),
        ("OLS (2)", f"pct_female ~ {predictors_m2}"),
        ("OLS (3)", f"pct_female ~ {predictors_m3}"),
    ]:
        model = sm.WLS.from_formula(formula, data=df, weights=weight_col)
        ols_results.append((name, model.fit()))

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
        glm_results.append((name, model.fit()))

    coef_df, var_order = build_coef_df(ols_results, glm_results)
    uoa_term_prefix = "C(UoA_num_cat, Treatment(reference=1))"
    discipline_vars = [v for v in var_order if str(v).startswith(uoa_term_prefix)]

    uoa_label_lookup = _load_uoa_label_lookup()
    uoa_label_overrides: dict[str, str] = {}
    for var in discipline_vars:
        match = re.search(r"\[T\.(\d+(?:\.\d+)?)\]", str(var))
        if not match:
            continue
        uoa_num = int(float(match.group(1)))
        uoa_label = _format_uoa_label(uoa_num, uoa_label_lookup)
        uoa_label_overrides[str(var)] = uoa_label if str(uoa_label).strip() else f"UoA {uoa_num}"

    return coef_df, var_order, discipline_vars, uoa_label_overrides


def _format_percent_axes(ax: plt.Axes, max_x: float = 100):
    """Apply percent formatting and grid lines."""
    ax.set_xlim(0, max_x)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(False)
    ax.set_axisbelow(True)


def _format_llm_label(label: str) -> str:
    """Clean llm_* topic labels for display."""
    if label.lower() == "nhs":
        return "NHS"
    return label


def _model_legend_label(index: int) -> str:
    words = ["One", "Two", "Three", "Four", "Five"]
    if index < len(words):
        return f"Model {words[index]}"
    return f"Model {index + 1}"


def _label_key(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _extract_model_number(model_name: str) -> int:
    match = re.search(r"\((\d+)\)", str(model_name))
    if match:
        return int(match.group(1))
    return -1


def _ordered_variables_for_coefficients(
    coef_df: pd.DataFrame,
    var_order: Sequence[str],
    *,
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
) -> List[str]:
    """
    Order coefficient rows within each conceptual block from lower to higher female association,
    using the most fully specified OLS model as the ordering reference.
    """
    base_order = list(var_order)
    if not base_order:
        return base_order

    ols_models = coef_df.loc[coef_df["estimator"] == "OLS", "model"].dropna().astype(str).unique().tolist()
    if not ols_models:
        return base_order
    reference_model = max(ols_models, key=_extract_model_number)

    ref_rows = coef_df[(coef_df["estimator"] == "OLS") & (coef_df["model"].astype(str) == reference_model)]
    coef_map = ref_rows.drop_duplicates(subset=["variable"]).set_index("variable")["coef"].to_dict()
    original_position = {v: i for i, v in enumerate(base_order)}

    def _sort_key(v: str) -> tuple[float, int]:
        coef_val = coef_map.get(v, np.nan)
        if pd.isna(coef_val):
            return (np.inf, original_position.get(v, 10**6))
        return (float(coef_val), original_position.get(v, 10**6))

    disciplines = [v for v in base_order if v in discipline_vars]
    universities = [v for v in base_order if v in university_vars]
    impact_domains = [v for v in base_order if str(v).startswith("llm_")]

    disciplines = sorted(disciplines, key=_sort_key)
    universities = sorted(universities, key=_sort_key)
    impact_domains = sorted(impact_domains, key=_sort_key)

    seen = set(disciplines + universities + impact_domains)
    remainder = [v for v in base_order if v not in seen]
    return disciplines + universities + impact_domains + remainder


def _divider_positions(
    variables: Sequence[str],
    *,
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
) -> tuple[float | None, float | None]:
    var_to_idx = {str(v): idx for idx, v in enumerate(variables)}
    discipline_idx = [var_to_idx[v] for v in discipline_vars if v in var_to_idx]
    university_idx = [var_to_idx[v] for v in university_vars if v in var_to_idx]
    impact_idx = [idx for v, idx in var_to_idx.items() if v.startswith("llm_")]

    divider_one = None
    divider_two = None
    if discipline_idx and university_idx:
        divider_one = (max(discipline_idx) + min(university_idx)) / 2
    if university_idx and impact_idx:
        divider_two = (max(university_idx) + min(impact_idx)) / 2
    return divider_one, divider_two


def _add_coefficient_dividers(
    ax: plt.Axes,
    variables: Sequence[str],
    *,
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
) -> tuple[float | None, float | None]:
    """Add thin dotted separators between controls and topic blocks."""
    divider_one, divider_two = _divider_positions(
        variables,
        discipline_vars=discipline_vars,
        university_vars=university_vars,
    )
    for y in (divider_one, divider_two):
        if y is not None:
            ax.axhline(y=y, linestyle=":", linewidth=0.8, color="k", alpha=0.65, zorder=0)
    return divider_one, divider_two


def _draw_vertical_curly_brace(
    ax: plt.Axes,
    y_top: float,
    y_bottom: float,
    *,
    label: str,
    x_axes: float = -0.034,
    width_axes: float = 0.042,
    label_x_axes: float | None = None,
    label_rotation: float = 90,
):
    if y_bottom <= y_top:
        return
    y_mid = (y_top + y_bottom) / 2
    shoulder = (y_bottom - y_top) * 0.3
    tip_x = x_axes - (1.5 * width_axes)
    curve_ctrl_x = x_axes - (.50 * width_axes)
    verts = [
        (x_axes + width_axes, y_top),
        (x_axes, y_top),
        (curve_ctrl_x, y_top + shoulder),
        (tip_x, y_mid),
        (curve_ctrl_x, y_bottom - shoulder),
        (x_axes, y_bottom),
        (x_axes + width_axes, y_bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
    ]
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    brace = PathPatch(
        MplPath(verts, codes),
        transform=transform,
        fill=False,
        linewidth=1.1,
        edgecolor="k",
        clip_on=False,
        zorder=5,
        joinstyle="miter",
    )
    ax.add_patch(brace)
    if label_x_axes is None:
        # Keep annotation near the brace tip, nudged slightly outward.
        text_x = tip_x + (0.028 if width_axes < 0 else -0.028)
    else:
        text_x = label_x_axes
    ax.text(
        text_x,
        y_mid,
        label,
        transform=transform,
        ha="center",
        va="center",
        fontsize=12,
        rotation=label_rotation,
        rotation_mode="anchor",
        multialignment="center",
        clip_on=False,
    )


def _add_coefficient_group_braces(
    ax: plt.Axes,
    variables: Sequence[str],
    *,
    side: str = "left",
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
    discipline_label: str = "Disciplines\n(ref. Panel A)",
    university_label: str = "University\nAttributes",
    impact_label_left: str = "Impact\nDomains",
    impact_label_right: str = "Impact\nDomains",
    right_brace_scale: float = 1.0,
    label_x_override: float | None = None,
):
    divider_one, divider_two = _divider_positions(
        variables,
        discipline_vars=discipline_vars,
        university_vars=university_vars,
    )
    ylim = ax.get_ylim()
    y_top = min(ylim)
    y_bottom = max(ylim)
    if str(side).lower() == "right":
        # Shift right-side braces/labels right; keep the inner brace edge close to the axis limit.
        right_scale = float(max(right_brace_scale, 0.1))
        brace_width = -0.024 * right_scale
        brace_x = 1.0 - brace_width
        label_x = label_x_override
        label_rotation = -90
        label_disciplines = discipline_label
        label_university = university_label
        label_impact = impact_label_right
    else:
        # Figure 2b left-side braces: restore original orientation/placement.
        brace_width = 0.033
        brace_x = -brace_width
        label_x = label_x_override
        label_rotation = 90
        label_disciplines = discipline_label
        label_university = university_label
        label_impact = impact_label_left

    if divider_one is not None:
        _draw_vertical_curly_brace(
            ax,
            y_top,
            divider_one,
            label=label_disciplines,
            x_axes=brace_x,
            width_axes=brace_width,
            label_x_axes=label_x,
            label_rotation=label_rotation,
        )

    if divider_one is not None and divider_two is not None:
        _draw_vertical_curly_brace(
            ax,
            divider_one,
            divider_two,
            label=label_university,
            x_axes=brace_x,
            width_axes=brace_width,
            label_x_axes=label_x,
            label_rotation=label_rotation,
        )

    if divider_two is not None:
        _draw_vertical_curly_brace(
            ax,
            divider_two,
            y_bottom,
            label=label_impact,
            x_axes=brace_x,
            width_axes=brace_width,
            label_x_axes=label_x,
            label_rotation=label_rotation,
        )


def _plot_ols_coefficients(
    ax: plt.Axes,
    coef_df: pd.DataFrame,
    var_order: List[str],
    colors: Sequence[str],
    *,
    panel_title: str | None = "b.",
    brace_side: str = "left",
    legend_loc: str = "lower left",
    legend_title: str = "OLS Specification",
    legend_fontsize: float = 11,
    label_overrides: dict[str, str] | None = None,
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
    discipline_label: str = "Disciplines\n(ref. Panel A)",
    university_label: str = "University\nAttributes",
    impact_label_left: str = "Impact\nDomains",
    impact_label_right: str = "Impact\nDomains",
    show_braces: bool = True,
    brace_label_x_override: float | None = None,
    invert_y_axis: bool = True,
):
    """Single-panel OLS coefficient plot for Figure 2b."""
    variables = var_order
    n_vars = len(variables)
    display_labels = [
        label_overrides.get(str(v), pretty_label(v)) if label_overrides else pretty_label(v) for v in variables
    ]
    y_map = {v: i for i, v in enumerate(variables)}
    models_ols = sorted(coef_df.loc[coef_df["estimator"] == "OLS", "model"].unique())
    if not models_ols:
        raise ValueError("No OLS estimates found in regression artifacts for Figure 2.")
    offsets_ols = np.linspace(-0.2, 0.2, len(models_ols))
    for i, (offset, m, color) in enumerate(zip(offsets_ols, models_ols, colors)):
        dfm = coef_df[(coef_df["estimator"] == "OLS") & (coef_df["model"] == m) & (coef_df["variable"].isin(variables))].copy()
        dfm["y_plot"] = pd.to_numeric(dfm["variable"].map(y_map), errors="coerce")
        dfm = dfm.dropna(subset=["y_plot"]).sort_values("y_plot")
        if dfm.empty:
            continue
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        whisker_color = mcolors.to_rgba(color, NON_VIOLIN_ALPHA)
        marker_color = mcolors.to_rgba(color, 1.0)
        ax.errorbar(
            dfm["coef"],
            dfm["y_plot"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor=(0, 0, 0, 1.0),
            markerfacecolor=marker_color,
            color=marker_color,
            ecolor=whisker_color,
            capsize=5,
            label=_model_legend_label(i),
        )
    ax.axvline(0, linestyle="--", linewidth=2, color="k")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(display_labels)
    # Figure 2b request: no horizontal divider lines beneath point-ranges.
    if invert_y_axis:
        ax.invert_yaxis()
    if show_braces:
        _add_coefficient_group_braces(
            ax,
            variables,
            side=brace_side,
            discipline_vars=discipline_vars,
            university_vars=university_vars,
            discipline_label=discipline_label,
            university_label=university_label,
            impact_label_left=impact_label_left,
            impact_label_right=impact_label_right,
            label_x_override=brace_label_x_override,
        )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Associated Change in Women ICS Authors", fontsize=15)
    if panel_title is not None:
        ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(
        frameon=True,
        title=legend_title,
        edgecolor="k",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc=legend_loc,
    )
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)


def _plot_glm_coefficients(
    ax: plt.Axes,
    coef_df: pd.DataFrame,
    var_order: List[str],
    colors: Sequence[str],
    *,
    panel_title: str | None = "b.",
    brace_side: str = "left",
    legend_loc: str = "lower right",
    legend_fontsize: float = 12,
    label_overrides: dict[str, str] | None = None,
    discipline_vars: Sequence[str] = DISCIPLINE_VARS,
    university_vars: Sequence[str] = UNIVERSITY_VARS,
    discipline_label: str = "Disciplines\n(ref. Panel A)",
    university_label: str = "University\nAttributes",
    impact_label_left: str = "Impact\nDomains",
    impact_label_right: str = "Impact\nDomains",
    show_braces: bool = True,
    right_brace_scale: float = 1.0,
    brace_label_x_override: float | None = None,
    invert_y_axis: bool = True,
):
    """Single-panel GLM coefficient plot for supplementary Figure 1b."""
    variables = var_order
    n_vars = len(variables)
    display_labels = [
        label_overrides.get(str(v), pretty_label(v)) if label_overrides else pretty_label(v) for v in variables
    ]
    y_map = {v: i for i, v in enumerate(variables)}
    models_glm = sorted(coef_df.loc[coef_df["estimator"] == "GLM", "model"].unique())
    if not models_glm:
        raise ValueError("No GLM estimates found in regression artifacts for supplementary figure 1.")
    offsets_glm = np.linspace(-0.2, 0.2, len(models_glm))
    for i, (offset, m, color) in enumerate(zip(offsets_glm, models_glm, colors)):
        dfm = coef_df[(coef_df["estimator"] == "GLM") & (coef_df["model"] == m) & (coef_df["variable"].isin(variables))].copy()
        dfm["y_plot"] = pd.to_numeric(dfm["variable"].map(y_map), errors="coerce")
        dfm = dfm.dropna(subset=["y_plot"]).sort_values("y_plot")
        if dfm.empty:
            continue
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        whisker_color = mcolors.to_rgba(color, NON_VIOLIN_ALPHA)
        marker_color = mcolors.to_rgba(color, 1.0)
        ax.errorbar(
            dfm["coef"],
            dfm["y_plot"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor=(0, 0, 0, 1.0),
            markerfacecolor=marker_color,
            color=marker_color,
            ecolor=whisker_color,
            capsize=5,
            label=_model_legend_label(i),
        )
    ax.axvline(0, linestyle="--", linewidth=2, color="k")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(display_labels)
    # Supplementary Figure 2 request: no horizontal divider lines beneath point-ranges.
    if invert_y_axis:
        ax.invert_yaxis()
    if show_braces:
        _add_coefficient_group_braces(
            ax,
            variables,
            side=brace_side,
            discipline_vars=discipline_vars,
            university_vars=university_vars,
            discipline_label=discipline_label,
            university_label=university_label,
            impact_label_left=impact_label_left,
            impact_label_right=impact_label_right,
            right_brace_scale=right_brace_scale,
            label_x_override=brace_label_x_override,
        )
    ax.set_xlabel("Associated Change in Women ICS Authors (log-odds, GLM)", fontsize=15)
    if panel_title is not None:
        ax.set_title(panel_title, loc="left", fontweight="bold", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(
        frameon=True,
        title="Binomial GLM (logit)",
        edgecolor="k",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc=legend_loc,
    )
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)


def _plot_topic_female_share_panel(
    ax_panel: plt.Axes,
    topic_by_panel: pd.DataFrame,
    *,
    panel_order: Sequence[str] = ("A", "B", "C", "D"),
    title: str = "a.",
    x_max: float = 60,
    x_label: str = "% female authors in ICS",
):
    panel_plot = (
        topic_by_panel.assign(pct_female=lambda d: d["pct_female"] * 100)
        .loc[lambda d: d["panel"].isin(panel_order)]
        .copy()
    )
    panel_plot = panel_plot.loc[
        ~panel_plot["llm_topic"].astype(str).str.strip().str.lower().isin({"error", "status"})
    ].copy()
    if panel_plot.empty:
        raise ValueError("No panel-level topic rows available for plotting.")

    topic_order = (
        panel_plot.groupby("llm_topic")["pct_female"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    panel_plot["llm_topic_label"] = panel_plot["llm_topic"].map(_format_llm_label)
    topic_order_labels = [_format_llm_label(t) for t in topic_order]
    palette = [mcolors.to_rgba(PANEL_COLORS.get(p, "#999999"), NON_VIOLIN_ALPHA) for p in panel_order]
    tick_positions = np.arange(len(topic_order_labels))
    sns.barplot(
        data=panel_plot,
        y=pd.Categorical(panel_plot["llm_topic_label"], categories=topic_order_labels, ordered=True),
        x="pct_female",
        hue="panel",
        edgecolor="k",
        width=0.58,
        hue_order=list(panel_order),
        ax=ax_panel,
        palette=palette,
    )
    for patch in ax_panel.patches:
        patch.set_alpha(NON_VIOLIN_ALPHA)

    ax_panel.set_ylabel("")
    ax_panel.set_xlabel(x_label, fontsize=15)
    ax_panel.set_title(title, loc="left", fontweight="bold", fontsize=17)
    ax_panel.tick_params(axis="both", which="major", labelsize=13)
    ax_panel.yaxis.tick_left()
    ax_panel.yaxis.set_label_position("left")
    handles, labels = ax_panel.get_legend_handles_labels()
    mean_color = "k"
    mean_handle = Line2D([0], [0], color=mean_color, linestyle="-", linewidth=MEAN_LINE_WIDTH, label="Mean")
    legend_labels = [PANEL_DISPLAY_LABELS.get(lbl, lbl) for lbl in labels]
    handles.append(mean_handle)
    legend_labels.append("Mean")
    ax_panel.legend(
        handles=handles,
        labels=legend_labels,
        title="Main REF Panel",
        fontsize=8,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="upper right",
    )
    _format_percent_axes(ax_panel, max_x=x_max)
    ax_panel.set_yticks(tick_positions)
    ax_panel.set_yticklabels(topic_order_labels)
    ax_panel.tick_params(axis="y", which="both", labelright=False, labelleft=True, right=False, left=True, length=6)

    topic_means = panel_plot.groupby("llm_topic_label")["pct_female"].mean()
    y_positions = {label: i for i, label in enumerate(topic_order_labels)}
    for label, mean_val in topic_means.items():
        y = y_positions.get(label)
        if y is not None and pd.notna(mean_val):
            ax_panel.plot(
                [mean_val, mean_val],
                [y - 0.5, y + 0.5],
                linestyle="-",
                color=mean_color,
                linewidth=MEAN_LINE_WIDTH,
                clip_on=False,
            )
    ax_panel.set_xlim(0, x_max)
    ax_panel.set_axisbelow(True)
    ax_panel.xaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)
    ax_panel.yaxis.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.32)


def plot_llm_female_share_by_panel(
    llm_by_panel: pd.DataFrame,
    panel_order: Sequence[str] = ("A", "B", "C", "D"),
) -> plt.Axes:
    """
    Plot % female authors across llm_* topics split by REF panel.
    Returns the axis used.
    """
    panel_plot = (
        llm_by_panel.assign(pct_female=lambda d: d["pct_female"] * 100)
        .loc[lambda d: d["panel"].isin(panel_order)]
        .sort_values(["pct_female", "llm_topic"])
    )

    fig, ax_panel = plt.subplots(figsize=(8, 8))
    palette = [mcolors.to_rgba(PANEL_COLORS.get(p, "#999999"), NON_VIOLIN_ALPHA) for p in panel_order]
    sns.barplot(
        data=panel_plot,
        y="llm_topic",
        x="pct_female",
        hue="panel",
        hue_order=list(panel_order),
        ax=ax_panel,
        palette=palette,
    )
    ax_panel.set_ylabel("")
    ax_panel.set_xlabel("% female authors in ICS")
    ax_panel.set_title("a.", loc="left", fontweight="bold")
    for patch in ax_panel.patches:
        patch.set_edgecolor("k")
        patch.set_alpha(NON_VIOLIN_ALPHA)
    handles, labels = ax_panel.get_legend_handles_labels()
    legend_labels = [PANEL_DISPLAY_LABELS.get(lbl, lbl) for lbl in labels]
    ax_panel.legend(
        handles=handles,
        labels=legend_labels,
        title="Main REF Panel",
        fontsize=8,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="upper right",
    )
    _format_percent_axes(ax_panel, max_x=100)
    ax_panel.yaxis.tick_left()
    ax_panel.yaxis.set_label_position("left")
    ax_panel.tick_params(axis="y", labelright=False, labelleft=True)
    fig.tight_layout()
    sns.despine(fig=fig, left=True)
    return ax_panel


def plot_combined_figure(
    coef_df: pd.DataFrame,
    var_order: List[str],
    llm_by_panel: pd.DataFrame,
    panel_order: Sequence[str] = ("A", "B", "C", "D"),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Two-panel Figure 2: (a) llm_* % female by panel (left axis), (b) OLS coefficients with right-hand axis."""
    apply_mpl_defaults()
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.titleweight"] = "bold"

    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_panel, ax_coef) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    ordered_var_order = _ordered_variables_for_coefficients(coef_df, var_order)
    _plot_ols_coefficients(
        ax_coef,
        coef_df,
        ordered_var_order,
        MODEL_COLORS,
        brace_label_x_override=-0.125,
    )
    ax_coef.set_xlim(left=-0.3)
    _plot_topic_female_share_panel(
        ax_panel,
        llm_by_panel,
        panel_order=panel_order,
        title="a.",
        x_max=60,
        x_label="Share of Women",
    )

    ax_coef.yaxis.tick_right()
    ax_coef.yaxis.set_label_position("right")
    ax_coef.spines["top"].set_visible(False)
    ax_coef.spines["left"].set_visible(False)
    ax_coef.spines["right"].set_visible(True)
    ax_coef.tick_params(axis="y", which="both", labelleft=False, labelright=True, left=True, right=True, length=6)
    sns.despine(ax=ax_coef, left=True, top=True, right=False, bottom=False)
    sns.despine(ax=ax_panel, left=False, top=True, right=True, bottom=False)
    ax_panel.spines["left"].set_visible(True)
    ax_panel.spines["right"].set_visible(False)
    ax_panel.yaxis.set_ticks_position("left")
    ax_panel.tick_params(axis="y", which="both", labelright=False, labelleft=True, right=False, left=True, length=6)
    fig.tight_layout()
    return fig, (ax_coef, ax_panel)


def plot_supplementary_figure_one(
    coef_df: pd.DataFrame,
    var_order: List[str],
    regex_by_panel: pd.DataFrame,
    panel_order: Sequence[str] = ("A", "B", "C", "D"),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Two-panel supplementary figure: (a) regex_* % female by panel, (b) GLM coefficients."""
    apply_mpl_defaults()
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.titleweight"] = "bold"

    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_panel, ax_coef) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    _plot_glm_coefficients(ax_coef, coef_df, var_order, MODEL_COLORS)
    _plot_topic_female_share_panel(
        ax_panel,
        regex_by_panel,
        panel_order=panel_order,
        title="a.",
        x_max=60,
        x_label="Contributions by Women (Regex-based Classification)",
    )

    ax_coef.yaxis.tick_right()
    ax_coef.yaxis.set_label_position("right")
    ax_coef.spines["top"].set_visible(False)
    ax_coef.spines["left"].set_visible(False)
    ax_coef.spines["right"].set_visible(True)
    ax_coef.tick_params(axis="y", which="both", labelleft=False, labelright=True, left=True, right=True, length=6)
    sns.despine(ax=ax_coef, left=True, top=True, right=False, bottom=False)
    sns.despine(ax=ax_panel, left=False, top=True, right=True, bottom=False)
    ax_panel.spines["left"].set_visible(True)
    ax_panel.spines["right"].set_visible(False)
    ax_panel.yaxis.set_ticks_position("left")
    ax_panel.tick_params(axis="y", which="both", labelright=False, labelleft=True, right=False, left=True, length=6)
    fig.tight_layout()
    return fig, (ax_coef, ax_panel)


def plot_supplementary_figure_two(
    coef_df: pd.DataFrame,
    var_order: List[str],
) -> tuple[plt.Figure, plt.Axes]:
    """Single-panel supplementary figure: GLM coefficients with the same model specs as Figure 2b."""
    apply_mpl_defaults()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False

    # Single-panel GLM figure: same height as Figure 2, just slightly wider.
    fig, ax = plt.subplots(1, 1, figsize=(15, 6.3))
    ordered_var_order = _ordered_variables_for_coefficients(coef_df, var_order)
    _plot_glm_coefficients(
        ax,
        coef_df,
        ordered_var_order,
        MODEL_COLORS,
        panel_title=None,
        brace_side="right",
        legend_loc="lower left",
        legend_fontsize=12,
        right_brace_scale=0.5,
        brace_label_x_override=1.05,
    )
    ax.set_xlim(right=0.5)
    ax.tick_params(axis="y", which="both", labelleft=True, labelright=False, left=True, right=False, length=6)
    sns.despine(ax=ax, left=False, top=True, right=True, bottom=False)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    return fig, ax


def plot_supplementary_figure_three(
    coef_df: pd.DataFrame,
    var_order: List[str],
    *,
    discipline_vars: Sequence[str],
    label_overrides: dict[str, str] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Supplementary Figure 3:
    Two tall coefficient plots (OLS + GLM) where disciplines are UoA dummies
    with UoA 1 (Clinical Medicine) as reference.
    """
    apply_mpl_defaults()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False

    ordered_var_order = _ordered_variables_for_coefficients(
        coef_df,
        var_order,
        discipline_vars=discipline_vars,
        university_vars=UNIVERSITY_VARS,
    )
    discipline_label = "Disciplines\nReference group: UoA 1 (Clinical Medicine)"

    fig, (ax_ols, ax_glm) = plt.subplots(
        1,
        2,
        figsize=(12.75, 15),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    _plot_ols_coefficients(
        ax_ols,
        coef_df,
        ordered_var_order,
        MODEL_COLORS,
        panel_title="a.",
        brace_side="left",
        legend_loc="lower left",
        legend_title="OLS Specification",
        legend_fontsize=11,
        label_overrides=label_overrides,
        discipline_vars=discipline_vars,
        university_vars=UNIVERSITY_VARS,
        discipline_label=discipline_label,
        university_label="University\nAttributes",
        impact_label_left="Impact\nDomains",
        impact_label_right="Impact\nDomains",
        show_braces=False,
    )
    ax_ols.set_xlabel("Associated Change in Women\nICS Authors", fontsize=15)
    ax_ols.tick_params(axis="y", which="both", labelleft=True, labelright=False, left=True, right=False, length=6, labelsize=9)
    sns.despine(ax=ax_ols, left=False, top=True, right=True, bottom=False)

    _plot_glm_coefficients(
        ax_glm,
        coef_df,
        ordered_var_order,
        MODEL_COLORS,
        panel_title="b.",
        brace_side="right",
        legend_loc="lower left",
        legend_fontsize=11,
        label_overrides=label_overrides,
        discipline_vars=discipline_vars,
        university_vars=UNIVERSITY_VARS,
        discipline_label=discipline_label,
        university_label="University\nAttributes",
        impact_label_left="Impact\nDomains",
        impact_label_right="Impact\nDomains",
        right_brace_scale=0.9,
        brace_label_x_override=1.12,
        invert_y_axis=False,
    )
    ax_glm.set_xlabel("Associated Change in Women\nICS Authors (log-odds, GLM)", fontsize=15)
    # Lock panel b to panel a ordering when sharing y-axis.
    ax_glm.set_ylim(ax_ols.get_ylim())
    ax_glm.set_yticks(ax_ols.get_yticks())
    ax_glm.set_yticklabels([t.get_text() for t in ax_ols.get_yticklabels()])
    ax_glm.tick_params(axis="y", which="both", labelleft=False, labelright=False, left=True, right=False, length=6, labelsize=9)
    sns.despine(ax=ax_glm, left=False, top=True, right=True, bottom=False)

    fig.tight_layout(rect=(0.04, 0, 0.96, 1), w_pad=3.5)
    return fig, (ax_ols, ax_glm)
