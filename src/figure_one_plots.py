from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:  # pragma: no cover
    from .figure_one_helpers import (
        COLOR_IMPACT_ORANGE,
        COLOR_OUTPUT_BLUE,
        DEFAULT_UOA_CODES_PATH,
        PANEL_COLORS,
        apply_mpl_defaults,
        make_percent_formatter,
        uoa_to_panel,
    )
except ImportError:  # pragma: no cover
    from figure_one_helpers import (
        COLOR_IMPACT_ORANGE,
        COLOR_OUTPUT_BLUE,
        DEFAULT_UOA_CODES_PATH,
        PANEL_COLORS,
        apply_mpl_defaults,
        make_percent_formatter,
        uoa_to_panel,
    )

AXIS_LABEL_SIZE = 12
VIOLIN_ALPHA_IMPACT = 0.35
VIOLIN_ALPHA_OUTPUT = 0.35
NON_VIOLIN_ALPHA = 0.65
SPINE_LW = 0.8


def _ensure_panel_columns(df_uoa_m, df_ics):
    """Add Panel if missing to UoA and ICS tables."""
    if "Panel" not in df_uoa_m.columns:
        df_uoa_m["Panel"] = df_uoa_m["Unit of assessment number"].apply(uoa_to_panel)
    if "Panel" not in df_ics.columns:
        df_ics["Panel"] = df_ics["Unit of assessment number"].apply(uoa_to_panel)


def _build_layout(figsize=(14, 9.75)):
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[0.76, 1.52, 1.22],
        height_ratios=[2.25, 1],
        wspace=0.16,
        hspace=0.28,
    )
    ax3 = fig.add_subplot(gs[:, 0])     # a: full-height left
    ax2 = fig.add_subplot(gs[0, 1])     # c: middle top
    ax1 = fig.add_subplot(gs[0, 2], sharey=ax2)  # b: right top, shared y with c
    ax4 = fig.add_subplot(gs[1, 1:3])   # d: shifted right, spans two columns
    return fig, (ax1, ax2, ax3, ax4)


def _plot_impact_premium_panel(ax, df_uoa_m):
    """
    Show women's impact premium by panel:
    delta_pp = 100 * (women's share in impact - women's share in outputs).
    Positive values indicate women are more represented in impact than outputs.
    """
    df_plot = (
        df_uoa_m[["Panel", "pct_female_ics", "pct_female_output"]]
        .dropna(subset=["pct_female_ics", "pct_female_output"])
        .copy()
    )
    df_plot["delta_pp"] = 100 * (df_plot["pct_female_ics"] - df_plot["pct_female_output"])
    panel_order = [p for p in ["A", "B", "C", "D"] if p in df_plot["Panel"].unique()]
    y_base = np.arange(len(panel_order))[::-1]
    panel_to_y = {p: y for p, y in zip(panel_order, y_base)}

    abs_max = float(np.nanmax(np.abs(df_plot["delta_pp"].values))) if len(df_plot) else 1.0
    x_lim = max(6, np.ceil(abs_max / 2) * 2)


    # One compact box per panel + jittered UoA points
    rng = np.random.default_rng(42)
    for p in panel_order:
        vals = df_plot.loc[df_plot["Panel"] == p, "delta_pp"].to_numpy()
        y0 = panel_to_y[p]

        bp = ax.boxplot(
            vals,
            positions=[y0],
            widths=0.5,
            vert=False,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            zorder=2,
        )
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor="k", linewidth=1.1, zorder=2)
        for whisk in bp["whiskers"]:
            whisk.set(color="k", linewidth=1.0, zorder=2)
        for cap in bp["caps"]:
            cap.set(color="k", linewidth=1.0, zorder=2)
        for med in bp["medians"]:
            med.set(color="k", linewidth=1.5, linestyle="--", zorder=2)

        y_jitter = y0 + rng.normal(0, 0.08, size=vals.size)
        # White underlay prevents whiskers/median from showing through translucent markers.
        ax.scatter(
            vals,
            y_jitter,
            s=72,
            c="white",
            edgecolor="none",
            zorder=3.5,
        )
        ax.scatter(
            vals,
            y_jitter,
            s=58,
            c=PANEL_COLORS.get(p, "grey"),
            edgecolor="k",
            linewidth=0.8,
            alpha=NON_VIOLIN_ALPHA,
            zorder=4,
        )

    xticks = np.linspace(-x_lim, x_lim, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(v):+d}%" if v else "0%" for v in xticks], fontsize=10)
    ax.set_xlim(-x_lim, x_lim)
    ax.axvline(0, color="#d9a3a3", linestyle=(0, (2, 2)), linewidth=0.65, alpha=0.75, zorder=1.4)
    ax.set_ylim(-0.6, len(panel_order) - 0.4)
    panel_label_map = {
        "A": "Panel A:\nLife\nSciences",
        "B": "Panel B:\nPhysical\nSciences",
        "C": "Panel C:\nSocial\nSciences",
        "D": "Panel D:\nHumanities",
    }
    ax.set_yticks(y_base)
    ax.set_yticklabels([panel_label_map.get(p, f"Panel {p}") for p in panel_order], fontsize=11)
    ax.set_xlabel("Impact - Output (Women's Share)\n(At the UoA Level)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("c.", loc="left", fontweight="bold", fontsize=17)

    legend_elements_b = [
        Patch(facecolor="white", edgecolor="k", linewidth=0.9, label="IQR Box"),
        Line2D([], [], color="k", linestyle="--", linewidth=1.5, label="Median"),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor=mcolors.to_rgba(PANEL_COLORS["A"], NON_VIOLIN_ALPHA),
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            label="Panel A",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor=mcolors.to_rgba(PANEL_COLORS["B"], NON_VIOLIN_ALPHA),
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            label="Panel B",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor=mcolors.to_rgba(PANEL_COLORS["C"], NON_VIOLIN_ALPHA),
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            label="Panel C",
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=6,
            markerfacecolor=mcolors.to_rgba(PANEL_COLORS["D"], NON_VIOLIN_ALPHA),
            markeredgecolor=(0, 0, 0, NON_VIOLIN_ALPHA),
            label="Panel D",
        ),
    ]
    ax.legend(
        handles=legend_elements_b,
        loc="lower left",
        ncol=1,
        fontsize=8,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 0.95),
        framealpha=0.95,
    )

    ax.grid(axis="x", linestyle="--", color="k", alpha=0.18, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_uoa_scatter(ax, df_uoa_m, show_unit_names: bool):
    uoa_panels = df_uoa_m["Panel"]
    colors_uoa = uoa_panels.map(PANEL_COLORS).fillna("grey")

    ax.scatter(
        df_uoa_m["pct_female_ics"],
        df_uoa_m["pct_female_output"],
        s=150,
        c=colors_uoa,
        edgecolor="k",
        alpha=NON_VIOLIN_ALPHA,
    )
    ax.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=1.5)

    mean_diff_uoa = (df_uoa_m["pct_female_ics"] - df_uoa_m["pct_female_output"]).mean()
    n_more_output_uoa = (df_uoa_m["pct_female_ics"] > df_uoa_m["pct_female_output"]).sum()

    ax.text(
        0.65,
        0.05,
        f"Mean Diff: {mean_diff_uoa:.3f}\nImpact > Output: {n_more_output_uoa}",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=1),
    )

    ax.set_xlim(0.05, 0.7)
    ax.set_ylim(0.05, 0.7)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Women's Share (Impact)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Women's Share (Outputs)", fontsize=AXIS_LABEL_SIZE)
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
        fontsize=8,
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
        _label_max = f"Highest women's\nshare in outputs:\n{_row_max['Unit of assessment name'].values[0]}"
        _label_min = (
            f"Lowest women's share\nin impact and outputs:\n{_row_min['Unit of assessment name'].values[0]}"
        )
    else:
        _label_max = f"Highest women's share in outputs: UoA {int(_row_max['Unit of assessment number'].values[0])}"
        _label_min = f"Lowest women's share in outputs: UoA {int(_row_min['Unit of assessment number'].values[0])}"

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
    'number' (for the key) and a label column (prefers 8-letter abbreviation). Falls back
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

    # Prefer the 8-letter abbreviation field when present
    label_col = _find_col(preferred_terms=["8-letter abbreviation", "abbrev_8", "abbrev8"])
    if not label_col:
        label_col = _find_col(preferred_terms=["8-letter", "abbrev"], must_contain=["8"])
    if not label_col:
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
    if label_map and not pd.isna(uoa_num):
        key_int = int(uoa_num)
        key_str = str(key_int)
        label = label_map.get(key_int) or label_map.get(key_str)
        if label:
            return str(label).strip()
    return ""


def _plot_uoa_percent_bars(ax, df_uoa_m, uoa_label_map: Dict[int, str]):
    """Horizontal bar chart of % women (impact) by UoA, ordered high to low."""
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

    row_spacing = 1.08
    y_pos = np.arange(len(df_plot)) * row_spacing
    colors = df_plot["Panel"].map(PANEL_COLORS).fillna("grey")
    values = df_plot["pct_female_ics"].values

    bars = ax.barh(
        y_pos,
        values,
        height=0.7,
        color=colors,
        edgecolor="k",
        linewidth=0.9,
        alpha=NON_VIOLIN_ALPHA,
        zorder=2,
    )
    ax.bar_label(
        bars,
        labels=[f"{val * 100:.1f}%" for val in values],
        padding=4,
        label_type="edge",
        fontsize=10,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["label"])
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # Highest % at top

    x_max = values.max() + 0.10 if len(values) else 1.0
    ax.set_xlim(0, max(0.35, min(1.1, x_max)))
    ax.set_xlabel("Women's Share (Impact)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("a.", loc="left", x=-0.03, fontweight="bold", fontsize=17)
    if len(df_plot):
        ax.set_ylim(-0.6 * row_spacing, y_pos[-1] + 0.6 * row_spacing)
    else:
        ax.set_ylim(-0.5, 0.5)
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
    offset = 0.12
    width_v = 0.52
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
        body.set_facecolor(mcolors.to_rgba(COLOR_IMPACT_ORANGE, VIOLIN_ALPHA_IMPACT))
        body.set_edgecolor((0, 0, 0, 1))
        body.set_alpha(None)  # use face/edge RGBA directly (keeps black edge fully opaque)
        body.set_linewidth(1.0)
    for body in vp_output["bodies"]:
        body.set_facecolor(mcolors.to_rgba(COLOR_OUTPUT_BLUE, VIOLIN_ALPHA_OUTPUT))
        body.set_edgecolor((0, 0, 0, 1))
        body.set_alpha(None)  # use face/edge RGBA directly (keeps black edge fully opaque)
        body.set_linewidth(1.0)

    if "cmeans" in vp_ics:
        vp_ics["cmeans"].set_edgecolor(COLOR_IMPACT_ORANGE)
        vp_ics["cmeans"].set_linestyle("--")
        vp_ics["cmeans"].set_linewidth(1.5)
        vp_ics["cmeans"].set_zorder(6)
    if "cmeans" in vp_output:
        vp_output["cmeans"].set_edgecolor(COLOR_OUTPUT_BLUE)
        vp_output["cmeans"].set_linestyle("--")
        vp_output["cmeans"].set_linewidth(1.5)
        vp_output["cmeans"].set_zorder(6)

    ax.set_xlim(0.025, 0.7)
    ax.set_ylim(-0.5, 3.7)
    ax.set_xticks(np.linspace(0, 1, 6))
    panel_label_map = {
        "A": "Life\nSciences",
        "B": "Physical\nSciences",
        "C": "Social\nSciences",
        "D": "Humanities",
    }
    ax.set_yticks(y_base)
    ytick_labels = ax.set_yticklabels([panel_label_map.get(p, f"Panel {p}") for p in panel_order_uoa])
    for label in ytick_labels:
        label.set_multialignment("center")
        label.set_va("center")
        label.set_ha("left")
        label.set_rotation(90)
    ax.set_xlabel("Women's Share\n(At the UoA Level)", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("b.", loc="left", x=-0.03, fontweight="bold", fontsize=17)

    legend_elements_c = [
        Patch(
            facecolor=mcolors.to_rgba(COLOR_IMPACT_ORANGE, VIOLIN_ALPHA_IMPACT),
            edgecolor=(0, 0, 0, 0.5),
            linewidth=1.0,
            label="Impact",
        ),
        Patch(
            facecolor=mcolors.to_rgba(COLOR_OUTPUT_BLUE, VIOLIN_ALPHA_OUTPUT),
            edgecolor=(0, 0, 0, 0.5),
            linewidth=1.0,
            label="Output",
        ),
        Line2D([], [], color="k", linestyle="--", lw=1.5, label="Mean"),
    ]
    ax.legend(
        handles=legend_elements_c,
        loc="lower left",
        fontsize=8,
        frameon=True,
        edgecolor="k",
        facecolor=(1, 1, 1, 0.95),
        framealpha=0.95,
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
        ax.plot(
            [x_pos, x_pos],
            [0, y_val],
            color="k",
            lw=1.0,
            alpha=1.0,
            zorder=2,
        )

    ax.scatter(
        positions,
        ratio_series.values,
        c="white",
        edgecolor="none",
        s=98,
        zorder=3.5,
    )

    ax.scatter(
        positions,
        ratio_series.values,
        c=dot_colors,
        edgecolor="k",
        s=80,
        alpha=NON_VIOLIN_ALPHA,
        zorder=4,
    )

    mean_ratio = ratio_series.mean()
    parity_color = "#4f5964"
    mean_color = "#a74242"
    mean_dash = (0, (3, 2))
    ax.axhline(1.0, color=parity_color, linestyle="-", lw=1.8, label="Parity", zorder=2)
    ax.axhline(
        mean_ratio,
        color=mean_color,
        linestyle=mean_dash,
        lw=1.8,
        label=f"Mean = {mean_ratio:.2f}",
        zorder=2,
    )

    ax.set_title("d.", loc="left", fontweight="bold", fontsize=17)
    ax.set_ylabel("Impact/Output", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("")
    ax.grid(linestyle="--", color="k", alpha=0.15)
    ax.set_xlim(-0.5, len(ratio_series) - 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_format_uoa_label(num, uoa_label_map) for num in ratio_series.index],
        rotation=90,
        fontsize=7,
    )
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    legend_elements_d = [
        Line2D([0], [0], color=parity_color, lw=1.8, linestyle="-", label="Parity"),
        Line2D([0], [0], color=mean_color, lw=1.8, linestyle=mean_dash, label="Mean"),
    ]
    ax.legend(
        handles=legend_elements_d,
        loc="upper right",
        frameon=True,
        ncol=1,
        fontsize=8,
        edgecolor="k",
        facecolor=(1, 1, 1, 1),
        framealpha=1.0,
    )


def _style_axes(fig, ax1, ax2, ax3, ax4):
    fmt = make_percent_formatter()

    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_LW)
        ax.tick_params(axis="both", which="major", labelsize=13)

    # b (right-top): only visible y-axis for b/c pair, on the right
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(True)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis="y", labelsize=11, pad=3)
    for label in ax1.get_yticklabels():
        label.set_multialignment("center")
        label.set_ha("left")
        label.set_va("center")
        label.set_rotation(90)

    # c (middle-top): share y with b but hide its y-axis entirely
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)

    ax3.spines["left"].set_visible(True)
    ax3.spines["right"].set_visible(False)
    ax3.yaxis.tick_left()
    ax3.yaxis.set_label_position("left")
    ax4.spines["left"].set_visible(False)
    ax4.spines["right"].set_visible(True)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylim(0.65, 1.85)

    ax2.set_xlim(0.025, 0.7)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax4.grid(False)

    ax4.tick_params(axis="x", labelrotation=90, labelsize=9)

    ax2.xaxis.set_major_formatter(fmt)
    ax2.set_xlabel("Women's Share\n(At the UoA Level)", fontsize=AXIS_LABEL_SIZE)
    ax3.xaxis.set_major_formatter(fmt)
    ax3.set_ylabel("")
    ax3.tick_params(axis="y", labelsize=10, pad=6)



def _annotate_max_impact(df_uoa_m, ax2, show_unit_names: bool):
    _idx_max_imp = df_uoa_m["pct_female_ics"].idxmax()
    _row_max_imp = df_uoa_m.loc[[_idx_max_imp]]
    _x_max_imp, _y_max_imp = _row_max_imp[["pct_female_ics", "pct_female_output"]].values[0]

    _label_max_imp = (
        "Highest women's\nshare in impact:\nSocial Work and Social Policy"
        if show_unit_names
        else f"Highest women's share in impact: UoA {int(_row_max_imp['Unit of assessment number'].values[0])}"
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
    _plot_panel_violins(ax2, df_uoa_m)
    _plot_uoa_percent_bars(ax3, df_uoa_m, uoa_label_map)
    _plot_impact_premium_panel(ax1, df_uoa_m)
    _plot_ratio(ax4, df_uoa_m, show_unit_names, uoa_label_map)
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_title(ax.get_title(), fontweight="bold")
    _style_axes(fig, ax1, ax2, ax3, ax4)
    return fig, (ax1, ax2, ax3, ax4)


def save_figure(fig: plt.Figure, out_dir: Path, basename: str = "gender_output_ics_four_panel"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{basename}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{basename}.png", dpi=800, bbox_inches="tight", pad_inches=0.02)
