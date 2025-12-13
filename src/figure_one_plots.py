from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_one_helpers import (
    COLOR_IMPACT_ORANGE,
    COLOR_OUTPUT_BLUE,
    DEFAULT_UOA_CODES_PATH,
    PANEL_COLORS,
    apply_mpl_defaults,
    make_percent_formatter,
    uoa_to_panel,
)


def _ensure_panel_columns(df_uoa_m, df_ics):
    """Add Panel if missing to UoA and ICS tables."""
    if "Panel" not in df_uoa_m.columns:
        df_uoa_m["Panel"] = df_uoa_m["Unit of assessment number"].apply(uoa_to_panel)
    if "Panel" not in df_ics.columns:
        df_ics["Panel"] = df_ics["Unit of assessment number"].apply(uoa_to_panel)


def _build_layout(figsize=(14, 8.25)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[0.9, 1.5, 0.9],
        height_ratios=[2.25, 1],
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[:, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    return fig, (ax1, ax2, ax3, ax4)


def _plot_uoa_scatter(ax, df_uoa_m, show_unit_names: bool):
    uoa_panels = df_uoa_m["Panel"]
    colors_uoa = uoa_panels.map(PANEL_COLORS).fillna("grey")

    ax.scatter(
        df_uoa_m["pct_female_ics"],
        df_uoa_m["pct_female_output"],
        s=150,
        c=colors_uoa,
        edgecolor="k",
    )
    ax.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=1.5)

    mean_diff_uoa = (df_uoa_m["pct_female_ics"] - df_uoa_m["pct_female_output"]).mean()
    n_more_output_uoa = (df_uoa_m["pct_female_ics"] > df_uoa_m["pct_female_output"]).sum()

    ax.text(
        0.65,
        0.05,
        f"Mean Diff: {mean_diff_uoa:.3f}\nImpact > Output: {n_more_output_uoa}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=1),
    )

    ax.set_xlim(0.05, 0.7)
    ax.set_ylim(0.05, 0.7)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Female Fraction (Impact)", fontsize=15)
    ax.set_ylabel("Female Fraction (Outputs)", fontsize=15)
    ax.set_title("b.", loc="left", fontweight="bold", fontsize=17)

    legend_elements_uoa = [
        Patch(facecolor=PANEL_COLORS["A"], edgecolor="k", label="Panel A"),
        Patch(facecolor=PANEL_COLORS["B"], edgecolor="k", label="Panel B"),
        Patch(facecolor=PANEL_COLORS["C"], edgecolor="k", label="Panel C"),
        Patch(facecolor=PANEL_COLORS["D"], edgecolor="k", label="Panel D"),
    ]
    ax.legend(
        handles=legend_elements_uoa,
        title="Main REF Panel",
        loc="upper left",
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )

    _idx_max = df_uoa_m["pct_female_output"].idxmax()
    _idx_min = df_uoa_m["pct_female_output"].idxmin()
    _row_max = df_uoa_m.loc[[_idx_max]]
    _row_min = df_uoa_m.loc[[_idx_min]]
    _x_max, _y_max = _row_max[["pct_female_ics", "pct_female_output"]].values[0]
    _x_min, _y_min = _row_min[["pct_female_ics", "pct_female_output"]].values[0]

    if show_unit_names:
        _label_max = f"Highest female\noutput: {_row_max['Unit of assessment name'].values[0]}"
        _label_min = (
            f"Lowest female\nimpact and output:\n{_row_min['Unit of assessment name'].values[0]}"
        )
    else:
        _label_max = f"Highest female output: UoA {int(_row_max['Unit of assessment number'].values[0])}"
        _label_min = f"Lowest female output: UoA {int(_row_min['Unit of assessment number'].values[0])}"

    ax.annotate(
        _label_max,
        xy=(_x_max, _y_max),
        xytext=(0.60, 0.85),
        textcoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="k", connectionstyle="arc3,rad=0.5"),
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
    )

    ax.annotate(
        _label_min,
        xy=(_x_min, _y_min),
        xytext=(0.50, 0.185),
        textcoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="k", connectionstyle="arc3,rad=-0.25"),
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
    )


def _load_uoa_label_lookup(path: Path = DEFAULT_UOA_CODES_PATH) -> Dict[int, str]:
    """
    Optional hook: read a user-supplied UoA label map from the manual CSV.
    Looks for two columns that include either 'uoa' or 'unit of assessment' plus
    'number' (for the key) and a label column (prefers Abbrev_4). Falls back
    to any column that looks like an ID and any column that looks like a label.
    Returns an empty dict if the file or expected columns are missing.
    """
    if not Path(path).exists():
        return {}

    # Support both Excel and CSV inputs
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df_lookup = pd.read_excel(path)
    else:
        df_lookup = pd.read_csv(path)

    norm_cols = {c: c.strip().lower() for c in df_lookup.columns}

    def _find_col(preferred_terms, must_contain=None):
        must_contain = must_contain or []
        for c, lower in norm_cols.items():
            if all(term in lower for term in must_contain) and any(term in lower for term in preferred_terms):
                return c
        return None

    # Column holding the numeric UoA identifier
    num_col = _find_col(preferred_terms=["number", "id", "no"], must_contain=["uoa"])
    if not num_col:
        num_col = _find_col(preferred_terms=["number", "id", "no"], must_contain=["unit of assessment"])
    if not num_col:
        num_col = _find_col(preferred_terms=["number", "id", "no"])

    # Prefer the 4-letter abbreviation field when present
    label_col = _find_col(preferred_terms=["abbrev_4", "abbrev4", "abbrev"])
    if not label_col:
        label_col = _find_col(preferred_terms=["label", "name", "title", "desc"], must_contain=["uoa"])
    if not label_col:
        label_col = _find_col(preferred_terms=["label", "name", "title", "desc"], must_contain=["unit of assessment"])
    if not label_col:
        label_col = _find_col(preferred_terms=["label", "name", "title", "desc"])

    if not num_col or not label_col:
        return {}

    mapping = (
        df_lookup[[num_col, label_col]]
        .assign(**{num_col: pd.to_numeric(df_lookup[num_col], errors="coerce")})
        .dropna(subset=[num_col, label_col])
        .drop_duplicates(subset=[num_col])
    )
    mapping[num_col] = mapping[num_col].astype(int)
    return dict(zip(mapping[num_col], mapping[label_col].astype(str)))


def _format_uoa_label(uoa_num, label_map: Dict[int, str]) -> str:
    num_str = str(int(uoa_num)) if not pd.isna(uoa_num) else ""
    if label_map:
        label = label_map.get(int(uoa_num)) or label_map.get(num_str)
        if label:
            label_str = str(label).strip()
            if num_str and num_str in label_str:
                return label_str
            return f"{label_str} ({num_str})" if num_str else label_str
    return num_str


def _plot_uoa_percent_bars(ax, df_uoa_m, uoa_label_map: Dict[int, str]):
    """Horizontal bar chart of % female (impact) by UoA, ordered high to low."""
    df_plot = (
        df_uoa_m[
            [
                "Unit of assessment number",
                "Unit of assessment name",
                "Panel",
                "pct_female_ics",
            ]
        ]
        .dropna(subset=["pct_female_ics"])
        .copy()
    )

    df_plot["label"] = df_plot["Unit of assessment number"].apply(
        lambda num: _format_uoa_label(num, uoa_label_map)
    )
    df_plot = df_plot.sort_values("pct_female_ics", ascending=False)

    y_pos = np.arange(len(df_plot))
    colors = df_plot["Panel"].map(PANEL_COLORS).fillna("grey")
    values = df_plot["pct_female_ics"].values

    bars = ax.barh(y_pos, values, color=colors, edgecolor="k", zorder=2)
    ax.bar_label(
        bars,
        labels=[f"{val*100:.1f}%" for val in values],
        padding=-30,
        label_type="edge",
        fontsize=10,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["label"])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.invert_yaxis()  # Highest % at top

    x_max = values.max() + 0.12 if len(values) else 1.0
    ax.set_xlim(max(0.35, min(1.1, x_max)), 0)
    ax.set_xlabel("Female Fraction (Impact)", fontsize=15)
    ax.set_title("c.", loc="left", fontweight="bold", fontsize=17)
    ax.set_ylim(-0.5, len(df_plot) - 0.5)
    ax.grid(linestyle="--", color="k", alpha=0.15, axis="x", zorder=1)


def _plot_panel_violins(ax, df_uoa_m):
    panel_order_uoa = [p for p in ["A", "B", "C", "D"] if p in df_uoa_m["Panel"].dropna().unique()]
    ics_distributions = []
    output_distributions = []

    for p in panel_order_uoa:
        subset = df_uoa_m[df_uoa_m["Panel"] == p]
        ics_distributions.append(subset["pct_female_ics"].dropna().values)
        output_distributions.append(subset["pct_female_output"].dropna().values)

    ax.clear()
    y_base = np.arange(len(panel_order_uoa))[::-1]
    offset = 0.15
    width_v = 0.5
    positions_ics = y_base + offset
    positions_output = y_base - offset

    vp_ics = ax.violinplot(
        ics_distributions,
        positions=positions_ics,
        widths=width_v,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        vert=False,
    )

    vp_output = ax.violinplot(
        output_distributions,
        positions=positions_output,
        widths=width_v,
        showmeans=True,
        showmedians=False,
        showextrema=False,
        vert=False,
    )

    for body in vp_ics["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("w")
    for body in vp_output["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("w")

    if "cmeans" in vp_ics:
        vp_ics["cmeans"].set_edgecolor("k")
        vp_ics["cmeans"].set_linestyle("--")
        vp_ics["cmeans"].set_zorder(6)
    if "cmeans" in vp_output:
        vp_output["cmeans"].set_edgecolor("k")
        vp_output["cmeans"].set_linestyle("--")
        vp_output["cmeans"].set_linewidth(2.0)
        vp_output["cmeans"].set_zorder(6)

    # Manually draw mean markers/lines for broader matplotlib compatibility
    line_half_height = width_v * 0.4
    for pos, dist in zip(positions_ics, ics_distributions):
        if dist.size == 0:
            continue
        m = np.nanmean(dist)
        ax.vlines(
            m,
            pos - line_half_height,
            pos + line_half_height,
            color="k",
            linewidth=2,
            linestyles=(0,(1,1)),
            zorder=6,
        )
#        ax.scatter(m, pos, marker="o", s=35, facecolor="white", edgecolor="k", zorder=7)
    for pos, dist in zip(positions_output, output_distributions):
        if dist.size == 0:
            continue
        m = np.nanmean(dist)
        ax.vlines(
            m,
            pos - line_half_height,
            pos + line_half_height,
            color="k",
            linewidth=2,
            linestyles=(0,(1,1)),
            zorder=6,
        )
#        ax.scatter(m, pos, marker="o", s=35, facecolor="white", edgecolor="k", zorder=7)

    rng = np.random.default_rng(0)
    for i, p in enumerate(panel_order_uoa):
        color_panel = PANEL_COLORS.get(p, "grey")
        x_ics_i = ics_distributions[i]
        if x_ics_i.size > 0:
            y_ics_i = np.full_like(x_ics_i, positions_ics[i], dtype=float)
            y_ics_i += rng.normal(0, 0.03, size=x_ics_i.shape)
            ax.scatter(
                x_ics_i,
                y_ics_i,
                s=90,
                marker="o",
                facecolor=color_panel,
                edgecolor="k",
                zorder=3,
            )
        x_out_i = output_distributions[i]
        if x_out_i.size > 0:
            y_out_i = np.full_like(x_out_i, positions_output[i], dtype=float)
            y_out_i += rng.normal(0, 0.03, size=x_out_i.shape)
            ax.scatter(
                x_out_i,
                y_out_i,
                s=90,
                marker="s",
                facecolor=color_panel,
                edgecolor="k",
                linewidths=1.2,
                zorder=3,
            )

    ax.set_xlim(0.025, 0.7)
    ax.set_ylim(-0.5, 3.7)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"Panel {p}" for p in panel_order_uoa])
    ax.set_xlabel("Females at UoA Level", fontsize=15)
    ax.set_title("a.", loc="left", fontweight="bold", fontsize=17)

    legend_elements_dist = [
        Line2D([], [], linestyle="None", marker="o", markersize=7,
               markerfacecolor="white", markeredgecolor="k", label="Impact"),
        Line2D([], [], linestyle="None", marker="s", markersize=7,
               markerfacecolor="white", markeredgecolor="k", label="Outputs"),
        Line2D([], [], color="k", linestyle=(0, (1,1)), label="Mean"),
    ]
    ax.legend(
        handles=legend_elements_dist,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
        ncol=3,
        fontsize=9,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )


def _plot_ratio(ax, df_uoa_m, show_unit_names: bool, uoa_label_map: Dict[int, str]):
    ratio_series = (
        df_uoa_m.set_index("Unit of assessment number")
        .assign(ratio=lambda x: x["pct_female_ics"] / x["pct_female_output"])
        ["ratio"]
        .dropna()
        .sort_values(ascending=False)
    )

    panel_map = df_uoa_m.set_index("Unit of assessment number")["Panel"]
    dot_colors = [PANEL_COLORS.get(panel_map.get(num), "grey") for num in ratio_series.index]
    positions = np.arange(len(ratio_series))

    for x_pos, y_val in zip(positions, ratio_series.values):
        ax.plot([x_pos, x_pos], [0, y_val], color="k", lw=1.2, zorder=3)

    ax.scatter(
        positions,
        ratio_series.values,
        c=dot_colors,
        edgecolor="k",
        s=80,
        zorder=3,
    )

    mean_ratio = ratio_series.mean()
    ax.axhline(1.0, color="k", linestyle="-.", lw=1.2, label="Parity", zorder=2)
    ax.axhline(mean_ratio, color="k", linestyle="--", lw=1.5, label=f"Mean = {mean_ratio:.2f}", zorder=2)

    ax.set_title("d.", loc="left", fontweight="bold", fontsize=17)
    ax.set_ylabel("Impact/Output\nRatio", fontsize=15)
    ax.set_xlabel("Unit of Assessment", fontsize=15)
    ax.grid(linestyle="--", color="k", alpha=0.15)
    ax.set_xlim(-0.5, len(ratio_series) - 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_format_uoa_label(num, uoa_label_map) for num in ratio_series.index],
        rotation=90,
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    uoas_to_annotate = [9, 30, 18, 12, 8, 23]
    for uoa_num in uoas_to_annotate:
        if uoa_num in ratio_series.index:
            x_pos = list(ratio_series.index).index(uoa_num)
            y_val = ratio_series.loc[uoa_num]
            name = df_uoa_m.loc[df_uoa_m["Unit of assessment number"] == uoa_num, "Unit of assessment name"].values[0]

            if uoa_num in (9, 30, 18):
                xytext = (x_pos + 3, y_val + 0.2)
                rad = +0.4
            elif uoa_num == 12:
                xytext = (x_pos - 3.5, y_val + 0.4)
                rad = -0.4
            elif uoa_num == 8:
                xytext = (x_pos - 2.25, y_val + 0.6)
                rad = -0.4
            elif uoa_num == 23:
                xytext = (x_pos - 3, y_val + 0.3)
                rad = -0.4
            else:
                xytext = (x_pos, y_val)
                rad = 0

            ax.annotate(
                f"{name}" if show_unit_names else f"UoA {uoa_num}",
                xy=(x_pos + 0.025, y_val + 0.025),
                xytext=xytext,
                textcoords="data",
                fontsize=10,
                ha="center",
                va="center",
                arrowprops=dict(arrowstyle="->", lw=1.1, color="k", connectionstyle=f"arc3,rad={rad}"),
                bbox=dict(facecolor="white", edgecolor="none", alpha=1),
            )

    legend_elements_d = [
        Line2D([0], [0], color="k", lw=1.2, linestyle="-.", label="Parity"),
        Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="Mean"),
    ]
    ax.legend(
        handles=legend_elements_d,
        loc="upper right",
        frameon=True,
        ncol=1,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )


def _style_axes(fig, ax1, ax2, ax3, ax4):
    fmt = make_percent_formatter()

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=13)

    ax3.spines["right"].set_visible(True)
    ax3.spines["left"].set_visible(False)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax4.spines["left"].set_visible(True)
    ax4.spines["right"].set_visible(False)
    ax4.yaxis.tick_left()
    ax4.yaxis.set_label_position("left")
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax4.set_ylim(0.65, 1.85)

    ax1.set_xlim(0.025, 0.7)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)

    ax4.tick_params(axis="x", labelrotation=90, labelsize=9)

    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_xlabel("Females at UoA Level", fontsize=15)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.set_xlabel("Female Impact (ICS)", fontsize=15)
    ax2.set_ylabel("Female Outputs (Papers)", fontsize=15)
    ax2.set_xlim(0.05, 0.7)
    ax2.set_ylim(0.05, 0.7)
    ax3.xaxis.set_major_formatter(fmt)
    ax3.set_ylabel("Unit of Assessment", fontsize=15, labelpad=12)
    ax3.tick_params(axis="y", labelsize=10, pad=6)


def _annotate_max_impact(df_uoa_m, ax2, show_unit_names: bool):
    _idx_max_imp = df_uoa_m["pct_female_ics"].idxmax()
    _row_max_imp = df_uoa_m.loc[[_idx_max_imp]]
    _x_max_imp, _y_max_imp = _row_max_imp[["pct_female_ics", "pct_female_output"]].values[0]

    _label_max_imp = (
        "Highest female\n impact: Social Work\nand Social Policy"
        if show_unit_names
        else f"Highest female impact: UoA {int(_row_max_imp['Unit of assessment number'].values[0])}"
    )
    ax2.annotate(
        _label_max_imp,
        xy=(_x_max_imp, _y_max_imp),
        xytext=(0.85, 0.45),
        textcoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="k", connectionstyle="arc3,rad=+0.3"),
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
    )


def plot_figure_one(df_ics, df_uoa_m, show_unit_names: bool = True) -> Tuple[plt.Figure, Iterable]:
    """Create the four-panel figure and return the figure plus axes."""
    apply_mpl_defaults()
    df_uoa_m = df_uoa_m.copy()
    df_ics = df_ics.copy()
    _ensure_panel_columns(df_uoa_m, df_ics)
    uoa_label_map = _load_uoa_label_lookup()
    fig, (ax1, ax2, ax3, ax4) = _build_layout()
    _plot_uoa_scatter(ax2, df_uoa_m, show_unit_names)
    _plot_uoa_percent_bars(ax3, df_uoa_m, uoa_label_map)
    _plot_panel_violins(ax1, df_uoa_m)
    _plot_ratio(ax4, df_uoa_m, show_unit_names, uoa_label_map)
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_title(ax.get_title(), fontweight="bold")
    _style_axes(fig, ax1, ax2, ax3, ax4)
    _annotate_max_impact(df_uoa_m, ax2, show_unit_names)
    return fig, (ax1, ax2, ax3, ax4)


def save_figure(fig: plt.Figure, out_dir: Path, basename: str = "gender_output_ics_four_panel"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.svg")
    fig.savefig(out_dir / f"{basename}.png", dpi=800)
