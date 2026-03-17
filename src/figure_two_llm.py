from pathlib import Path
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

try:  # pragma: no cover
    from .figure_one_helpers import (
        DEFAULT_DATA_ROOT,
        PANEL_COLORS,
        apply_mpl_defaults,
        resolve_enhanced_ref_data_path,
    )
    from .figure_two_regression import load_regression_artifacts, pretty_label, save_figure
    from .statistics_helpers import _ensure_panel, llm_female_share_tables
    from .pipeline_io import read_table
except ImportError:  # pragma: no cover
    from figure_one_helpers import (
        DEFAULT_DATA_ROOT,
        PANEL_COLORS,
        apply_mpl_defaults,
        resolve_enhanced_ref_data_path,
    )
    from figure_two_regression import load_regression_artifacts, pretty_label, save_figure
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


def _divider_positions(display_labels: Sequence[str]) -> tuple[float | None, float | None]:
    label_to_idx = {_label_key(lbl): idx for idx, lbl in enumerate(display_labels)}

    def _between(label_a: str, label_b: str) -> float | None:
        idx_a = label_to_idx.get(_label_key(label_a))
        idx_b = label_to_idx.get(_label_key(label_b))
        if idx_a is None or idx_b is None:
            return None
        return (idx_a + idx_b) / 2

    return _between("Panel D", "OxBridge"), _between("Ancient", "Museum")


def _add_coefficient_dividers(ax: plt.Axes, display_labels: Sequence[str]) -> tuple[float | None, float | None]:
    """Add thin dotted separators between controls and topic blocks."""
    divider_one, divider_two = _divider_positions(display_labels)
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
    text_x = (tip_x - 0.034) if label_x_axes is None else label_x_axes
    ax.text(
        text_x,
        y_mid,
        label,
        transform=transform,
        ha="center",
        va="center",
        fontsize=9,
        rotation=90,
        rotation_mode="anchor",
        multialignment="center",
        clip_on=False,
    )


def _add_coefficient_group_braces(ax: plt.Axes, display_labels: Sequence[str]):
    divider_one, divider_two = _divider_positions(display_labels)
    ylim = ax.get_ylim()
    y_top = min(ylim)
    y_bottom = max(ylim)

    if divider_one is not None:
        _draw_vertical_curly_brace(
            ax,
            y_top,
            divider_one,
            label="Variables in\nAll Models",
        )

    if divider_one is not None and divider_two is not None:
        _draw_vertical_curly_brace(
            ax,
            divider_one,
            divider_two,
            label="Variables in Models\nTwo and Three",
        )

    if divider_two is not None:
        _draw_vertical_curly_brace(
            ax,
            divider_two,
            y_bottom,
            label="Variables in\nModel Three",
        )


def _plot_ols_coefficients(ax: plt.Axes, coef_df: pd.DataFrame, var_order: List[str], colors: Sequence[str]):
    """Single-panel OLS coefficient plot for Figure 2b."""
    variables = var_order
    n_vars = len(variables)
    display_labels = [pretty_label(v) for v in variables]
    models_ols = sorted(coef_df.loc[coef_df["estimator"] == "OLS", "model"].unique())
    if not models_ols:
        raise ValueError("No OLS estimates found in regression artifacts for Figure 2.")
    offsets_ols = np.linspace(-0.2, 0.2, len(models_ols))
    for i, (offset, m, color) in enumerate(zip(offsets_ols, models_ols, colors)):
        dfm = coef_df[(coef_df["estimator"] == "OLS") & (coef_df["model"] == m)]
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        plot_color = mcolors.to_rgba(color, NON_VIOLIN_ALPHA)
        ax.errorbar(
            dfm["coef"],
            dfm["y"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            color=plot_color,
            ecolor=plot_color,
            capsize=5,
            label=_model_legend_label(i),
        )
    ax.axvline(0, linestyle="--", linewidth=2, color="k")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(display_labels)
    _add_coefficient_dividers(ax, display_labels)
    ax.invert_yaxis()
    _add_coefficient_group_braces(ax, display_labels)
    ax.set_xlabel("Coefficient (proportion scale)", fontsize=15)
    ax.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        title="OLS Specification",
        edgecolor="k",
        fontsize=11,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="lower right",
    )
    ax.grid(False)


def _plot_glm_coefficients(ax: plt.Axes, coef_df: pd.DataFrame, var_order: List[str], colors: Sequence[str]):
    """Single-panel GLM coefficient plot for supplementary Figure 1b."""
    variables = var_order
    n_vars = len(variables)
    display_labels = [pretty_label(v) for v in variables]
    models_glm = sorted(coef_df.loc[coef_df["estimator"] == "GLM", "model"].unique())
    if not models_glm:
        raise ValueError("No GLM estimates found in regression artifacts for supplementary figure 1.")
    offsets_glm = np.linspace(-0.2, 0.2, len(models_glm))
    for i, (offset, m, color) in enumerate(zip(offsets_glm, models_glm, colors)):
        dfm = coef_df[(coef_df["estimator"] == "GLM") & (coef_df["model"] == m)]
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        plot_color = mcolors.to_rgba(color, NON_VIOLIN_ALPHA)
        ax.errorbar(
            dfm["coef"],
            dfm["y"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            color=plot_color,
            ecolor=plot_color,
            capsize=5,
            label=_model_legend_label(i),
        )
    ax.axvline(0, linestyle="--", linewidth=2, color="k")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(display_labels)
    _add_coefficient_dividers(ax, display_labels)
    ax.invert_yaxis()
    _add_coefficient_group_braces(ax, display_labels)
    ax.set_xlabel("Coefficient (log-odds, GLM)", fontsize=15)
    ax.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        title="Binomial GLM (logit)",
        edgecolor="k",
        fontsize=11,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="lower right",
    )
    ax.grid(False)


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
        fontsize=9,
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
        fontsize=9,
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

    coef_colors = list(PANEL_COLORS.values())
    coef_colors = [coef_colors[i] for i in (3, 2, 0)]
    _plot_ols_coefficients(ax_coef, coef_df, var_order, coef_colors)
    _plot_topic_female_share_panel(
        ax_panel,
        llm_by_panel,
        panel_order=panel_order,
        title="a.",
        x_max=60,
        x_label="Contributions by Women (LLM-based Classification)",
    )

    ax_coef.yaxis.tick_right()
    ax_coef.yaxis.set_label_position("right")
    ax_coef.invert_xaxis()
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

    coef_colors = list(PANEL_COLORS.values())
    coef_colors = [coef_colors[i] for i in (3, 2, 0)]
    _plot_glm_coefficients(ax_coef, coef_df, var_order, coef_colors)
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
    ax_coef.invert_xaxis()
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
