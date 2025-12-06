from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from figure_one_helpers import (
    COLOR_IMPACT_ORANGE,
    COLOR_OUTPUT_BLUE,
    PANEL_COLORS,
    make_percent_formatter,
    size_from_value,
    uoa_to_panel,
)


def _ensure_panel_columns(df_uoa_m, df_ics):
    """Add Panel if missing to UoA and ICS tables."""
    if "Panel" not in df_uoa_m.columns:
        df_uoa_m["Panel"] = df_uoa_m["Unit of assessment number"].apply(uoa_to_panel)
    if "Panel" not in df_ics.columns:
        df_ics["Panel"] = df_ics["Unit of assessment number"].apply(uoa_to_panel)


def _build_layout(figsize=(13, 8.25)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[0.8, 1.6, 1.6],
        height_ratios=[3, 1],
    )
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 1:3])
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
        xytext=(0.55, 0.175),
        textcoords="axes fraction",
        fontsize=11,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="k", connectionstyle="arc3,rad=-0.5"),
        bbox=dict(facecolor="white", edgecolor="none", alpha=1),
    )


def _plot_panel_bubbles(ax, df_ics):
    llm_cols = [c for c in df_ics.columns if c.startswith("llm_")][:11]
    panel_order_ind = [p for p in ["A", "B", "C", "D"] if p in df_ics["Panel"].dropna().unique()]

    n_panels = len(panel_order_ind)
    n_ind = len(llm_cols)
    values = np.full((n_panels, n_ind), np.nan)
    missing_mask = np.zeros((n_panels, n_ind), dtype=bool)

    for i, panel in enumerate(panel_order_ind):
        panel_df = df_ics[df_ics["Panel"] == panel]
        for j, col in enumerate(llm_cols):
            mask = panel_df[col].fillna(0).astype(bool)
            sub = panel_df[mask]
            fem = sub["number_female"].sum()
            male = sub["number_male"].sum()
            denom = fem + male
            if denom > 0:
                values[i, j] = fem / denom
            else:
                missing_mask[i, j] = True

    val_flat = values[~np.isnan(values)]
    vmin, vmax = (float(val_flat.min()), float(val_flat.max())) if val_flat.size > 0 else (0.0, 1.0)

    s_min, s_max = 10, 500

    ax.clear()
    x_coords = np.arange(n_panels)
    y_coords = np.arange(n_ind)

    for i, panel in enumerate(panel_order_ind):
        c = PANEL_COLORS.get(panel, "grey")
        for j, col in enumerate(llm_cols):
            x = i
            y = j
            if missing_mask[i, j]:
                ax.scatter(
                    x,
                    y,
                    s=s_min * 0.4,
                    facecolors="none",
                    edgecolors="0.6",
                    linewidths=0.8,
                    zorder=1.5,
                )
                continue
            v = values[i, j]
            if np.isnan(v):
                continue
            s = size_from_value(v, vmin, vmax, s_min, s_max)
            ax.scatter(
                x,
                y,
                s=s,
                color=c,
                edgecolor="k",
                zorder=2,
            )

    ax.set_xlim(-0.5, n_panels - 0.5)
    ax.set_ylim(-0.5, n_ind - 0.5)
    ax.set_xticks(x_coords)
    ax.set_xticklabels([f"Panel {p}" for p in panel_order_ind])
    ax.set_yticks(y_coords)

    llm_labels = []
    for c in llm_cols:
        lab = c[4:].replace("_", " ").title()
        llm_labels.append("NHS" if lab == "Nhs" else lab)
    ax.set_yticklabels(llm_labels)
    ax.invert_yaxis()
    ax.set_xlabel("REF Main Panel", fontsize=15)
    ax.set_title("c.", loc="left", fontweight="bold", fontsize=17)
    ax.grid(linestyle="--", color="k", alpha=0.15, axis="x", zorder=1)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    handles_size = []
    if val_flat.size > 0:
        ref_vals = np.round(np.linspace(vmin, vmax, 3), 2)
        for rv in ref_vals:
            s = size_from_value(rv, vmin, vmax, s_min, s_max)
            handles_size.append(
                Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="o",
                    markersize=np.sqrt(s),
                    markerfacecolor="none",
                    markeredgecolor="k",
                    label=f"{rv*100:.0f}% Female",
                )
            )
    ax.legend(
        handles=handles_size,
        title="Female Impact (ICS)",
        loc="upper center",
        borderaxespad=0,
        fontsize=9,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
        ncol=3,
        borderpad=0.7,
        labelspacing=0.75,
        handlelength=1,
        markerscale=0.8,
    )


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
    if "cmeans" in vp_output:
        vp_output["cmeans"].set_edgecolor("k")
        vp_output["cmeans"].set_linestyle("--")

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
    ax.set_ylim(-0.5, 3.8)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"Panel {p}" for p in panel_order_uoa])
    ax.set_xlabel("Females at UoA Level", fontsize=15)
    ax.set_title("a.", loc="left", fontweight="bold", fontsize=17)

    legend_elements_dist = [
        Line2D([], [], linestyle="None", marker="o", markersize=7, markerfacecolor="white", markeredgecolor="k", label="Impact (ICS)"),
        Line2D([], [], linestyle="None", marker="s", markersize=7, markerfacecolor="white", markeredgecolor="k", label="Outputs (Papers)"),
        Line2D([], [], color="k", linestyle="--", label="Mean"),
    ]
    ax.legend(
        handles=legend_elements_dist,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )


def _plot_ratio(ax, df_uoa_m, show_unit_names: bool):
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
    ax.axhline(mean_ratio, color="k", linestyle="--", lw=1.5, label=f"Mean = {mean_ratio:.2f}", zorder=2)

    ax.set_title("d.", loc="left", fontweight="bold", fontsize=17)
    ax.set_ylabel("Impact/Output\nRatio", fontsize=15)
    ax.set_xlabel("Unit of Assessment", fontsize=15)
    ax.grid(linestyle="--", color="k", alpha=0.15)
    ax.set_xlim(-0.5, len(ratio_series) - 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(ratio_series.index, rotation=90, fontsize=8)
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
        Line2D([0], [0], color="k", lw=1.5, linestyle="--", label="Mean Ratio"),
    ]
    ax.legend(
        handles=legend_elements_d,
        loc="upper right",
        frameon=True,
        ncol=3,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )


def _style_axes(fig, ax1, ax2, ax3, ax4):
    fmt = make_percent_formatter()

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        if ax in (ax3, ax4):
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(False)
        else:
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)
        ax.tick_params(axis="both", which="major", labelsize=13)

    ax4.set_ylim(0.65, 1.85)
    ax3.set_ylim(-0.5, 12.25)

    ax3.add_patch(
        Rectangle(
            (1.0, 0.875),
            0.03,
            0.20,
            transform=ax3.transAxes,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
    )

    ax1.set_xlim(0.025, 0.7)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)
    ax4.tick_params(axis="x", labelrotation=0, labelsize=10)

    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_xlabel("Females at UoA Level", fontsize=15)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.set_xlabel("Female Impact (ICS)", fontsize=15)
    ax2.set_ylabel("Female Outputs (Papers)", fontsize=15)
    ax2.set_xlim(0.05, 0.7)
    ax2.set_ylim(0.05, 0.7)

    ax3.spines["right"].set_color("white")
    ax3.add_line(
        Line2D(
            [1.0, 1.0],
            [0.0, 0.875],
            transform=ax3.transAxes,
            color="k",
            linewidth=2,
            zorder=10,
        )
    )


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
    df_uoa_m = df_uoa_m.copy()
    df_ics = df_ics.copy()
    _ensure_panel_columns(df_uoa_m, df_ics)
    fig, (ax1, ax2, ax3, ax4) = _build_layout()
    _plot_uoa_scatter(ax2, df_uoa_m, show_unit_names)
    _plot_panel_bubbles(ax3, df_ics)
    _plot_panel_violins(ax1, df_uoa_m)
    _plot_ratio(ax4, df_uoa_m, show_unit_names)
    _style_axes(fig, ax1, ax2, ax3, ax4)
    _annotate_max_impact(df_uoa_m, ax2, show_unit_names)
    return fig, (ax1, ax2, ax3, ax4)


def save_figure(fig: plt.Figure, out_dir: Path, basename: str = "gender_output_ics_four_panel"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.svg")
    fig.savefig(out_dir / f"{basename}.png", dpi=800)
