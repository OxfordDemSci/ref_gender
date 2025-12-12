from pathlib import Path
from typing import Iterable, Tuple, Sequence, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import numpy as np

from figure_one_helpers import DEFAULT_DATA_ROOT, PANEL_COLORS, apply_mpl_defaults
from statistics_helpers import llm_female_share_tables, _ensure_panel
from figure_two_regression import save_figure, load_regression_artifacts, pretty_label


PanelOrder = Iterable[str]


def load_llm_tables(data_root: Path = DEFAULT_DATA_ROOT, panel_order: PanelOrder = ("A", "B", "C", "D")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ICS data and compute llm_* female share tables.

    Returns (overall_table, panel_table) where pct_female/share_of_ics are proportions.
    """
    data_root = Path(data_root)
    df_ics = pd.read_csv(data_root / "final" / "enhanced_ref_data.csv")
    df_ics = _ensure_panel(df_ics)
    return llm_female_share_tables(df_ics, panel_order)


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


def _plot_glm_coefficients(ax: plt.Axes, coef_df: pd.DataFrame, var_order: List[str], colors: Sequence[str]):
    """Single-panel GLM coefficient plot (log-odds) for Figure 2a."""
    variables = var_order
    n_vars = len(variables)
    display_labels = [pretty_label(v) for v in variables]
    models_glm = sorted(coef_df.loc[coef_df["estimator"] == "GLM", "model"].unique())
    offsets_glm = np.linspace(-0.2, 0.2, len(models_glm))
    for offset, m, color in zip(offsets_glm, models_glm, colors):
        dfm = coef_df[(coef_df["estimator"] == "GLM") & (coef_df["model"] == m)]
        xerr = np.vstack([dfm["coef"] - dfm["ci_low"], dfm["ci_high"] - dfm["coef"]])
        ax.errorbar(
            dfm["coef"],
            dfm["y"] + offset,
            xerr=xerr,
            fmt="o",
            markeredgecolor="k",
            color=color,
            capsize=5,
            label=m,
        )
    ax.axvline(0, linestyle="--", linewidth=2, color="k")
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(display_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (log-odds)", fontsize=15)
    ax.set_title("a.", loc="left", fontweight="bold", fontsize=17)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        title="GLM Specification",
        edgecolor="k",
        fontsize=11,
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="lower left",
    )
    ax.grid(False)


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
    palette = [PANEL_COLORS.get(p, "#999999") for p in panel_order]
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
    ax_panel.set_title("b. llm_* topics by panel", loc="left", fontweight="bold")
    _format_percent_axes(ax_panel, max_x=100)
    ax_panel.yaxis.tick_right()
    ax_panel.yaxis.set_label_position("right")
    ax_panel.tick_params(axis="y", labelright=True, labelleft=False)
    fig.tight_layout()
    sns.despine(fig=fig, left=True)
    return ax_panel


def plot_combined_figure(
    coef_df: pd.DataFrame,
    var_order: List[str],
    llm_by_panel: pd.DataFrame,
    panel_order: Sequence[str] = ("A", "B", "C", "D"),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Two-panel Figure 2: (a) GLM coefficients, (b) llm_* % female by panel with right-hand axis."""
    apply_mpl_defaults()
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Helvetica"
    mpl.rcParams["axes.titleweight"] = "bold"

    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax_glm, ax_panel) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    glm_colors = list(PANEL_COLORS.values())
    glm_colors = [glm_colors[i] for i in (3, 2, 0)]
    _plot_glm_coefficients(ax_glm, coef_df, var_order, glm_colors)

    panel_plot = (
        llm_by_panel.assign(pct_female=lambda d: d["pct_female"] * 100)
        .loc[lambda d: d["panel"].isin(panel_order)]
        .copy()
    )
    topic_order = (
        panel_plot.groupby("llm_topic")["pct_female"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    panel_plot["llm_topic_label"] = panel_plot["llm_topic"].map(_format_llm_label)
    topic_order_labels = [_format_llm_label(t) for t in topic_order]
    palette = [PANEL_COLORS.get(p, "#999999") for p in panel_order]
    tick_positions = np.arange(len(topic_order_labels))
    sns.barplot(
        data=panel_plot,
        y=pd.Categorical(panel_plot["llm_topic_label"], categories=topic_order_labels, ordered=True),
        x="pct_female",
        hue="panel",
        edgecolor='k',
        hue_order=list(panel_order),
        ax=ax_panel,
        palette=palette,
    )
    ax_panel.set_ylabel("")
    ax_panel.set_xlabel("% female authors in ICS", fontsize=15)
    ax_panel.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax_panel.tick_params(axis="both", which="major", labelsize=13)
    handles, labels = ax_panel.get_legend_handles_labels()
    mean_handle = Line2D([0], [0], color="k", linestyle="--", label="Mean")
    handles.append(mean_handle)
    labels.append("Mean")
    ax_panel.legend(
        handles=handles,
        labels=labels,
        title="Main REF Panel",
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        loc="upper left",
    )
    _format_percent_axes(ax_panel, max_x=100)
    ax_panel.yaxis.tick_right()
    ax_panel.yaxis.set_label_position("right")
    ax_panel.set_yticks(tick_positions)
    ax_panel.set_yticklabels(topic_order_labels)
    ax_panel.tick_params(axis="y", which="both", labelright=True, labelleft=False, right=True, left=False, length=6)
    ax_panel.yaxis.set_ticks_position("right")
    ax_panel.invert_xaxis()
    ax_panel.spines["top"].set_visible(False)
    ax_panel.spines["left"].set_visible(False)
    ax_panel.spines["right"].set_visible(True)
    topic_means = panel_plot.groupby("llm_topic_label")["pct_female"].mean()
    y_positions = {label: i for i, label in enumerate(topic_order_labels)}
    for label, mean_val in topic_means.items():
        y = y_positions.get(label)
        if y is not None and pd.notna(mean_val):
            ax_panel.plot(
                [mean_val, mean_val],
                [y - 0.5, y + 0.5],
                linestyle="--",
                color="k",
                linewidth=1.1,
                clip_on=False,
            )
    ax_panel.set_xlim(60, 0)
    sns.despine(ax=ax_glm)
    sns.despine(ax=ax_panel, left=True, top=True, right=False, bottom=False)
    ax_panel.yaxis.set_ticks_position("right")
    ax_panel.tick_params(axis="y", which="both", labelright=True, labelleft=False, right=True, left=False, length=6)
    fig.tight_layout()
    return fig, (ax_glm, ax_panel)
