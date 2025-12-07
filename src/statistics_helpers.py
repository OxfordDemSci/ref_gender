from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from figure_one_helpers import DEFAULT_DATA_ROOT, uoa_to_panel
from figure_one_data import prepare_figure_one_data, UOA_MAP

ALPHA = 0.05


def load_statistics_data(data_root: Path = DEFAULT_DATA_ROOT):
    """
    Load raw ICS/output tables plus aggregated UoA/university frames.
    Returns (df_output, df_ics, df_uoa_m, df_uni_m, df_uniuoa_m).
    """
    data_root = Path(data_root)
    df_output = pd.read_csv(data_root / "dimensions_outputs/outputs_concat_with_positive_authors.csv")
    df_ics = pd.read_csv(data_root / "final/enhanced_ref_data.csv")
    df_ics, df_uoa_m, df_uni_m, df_uniuoa_m = prepare_figure_one_data(data_root)
    return df_output, df_ics, df_uoa_m, df_uni_m, df_uniuoa_m


def _safe_pct(num: float, denom: float) -> float:
    return 100 * num / denom if denom else np.nan


def _total_income(df: pd.DataFrame) -> pd.Series:
    """Return total income (cash + in-kind) for each row."""
    cash = df["tot_income"] if "tot_income" in df.columns else 0
    kind = df["tot_inc_kind"] if "tot_inc_kind" in df.columns else 0
    return pd.Series(cash).fillna(0) + pd.Series(kind).fillna(0)


def _ensure_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a Panel column present."""
    if "Panel" in df.columns:
        return df
    df = df.copy()
    if "Main panel" in df.columns:
        df["Panel"] = df["Main panel"]
    elif "Unit of assessment number" in df.columns:
        df["Panel"] = df["Unit of assessment number"].apply(uoa_to_panel)
    else:
        df["Panel"] = np.nan
    return df


def _table_to_latex(df: pd.DataFrame, path: Path, column_format: Optional[str] = None, index: bool = False):
    """Save a table to LaTeX, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_latex(
        path,
        index=index,
        column_format=column_format,
        float_format=lambda x: f"{x:.2f}",
        multicolumn_format="c",
    )


def _round_sig(x: pd.Series, sig: int = 4) -> pd.Series:
    """Round a Series to the given significant figures."""
    return x.apply(lambda v: float(f"{v:.{sig}g}") if pd.notna(v) else v)


def _panel_table(df_ics: pd.DataFrame, df_output: pd.DataFrame) -> pd.DataFrame:
    df_ics = _ensure_panel(df_ics)
    df_output = _ensure_panel(df_output)
    panel_order = ["A", "B", "C", "D"]
    total_female_ics = df_ics["number_female"].sum()
    total_female_out = df_output["number_female"].sum()

    rows = []
    for panel in panel_order:
        ics_p = df_ics[df_ics["Panel"] == panel]
        out_p = df_output[df_output["Panel"] == panel]

        n_ics = len(ics_p)
        fem_ics = ics_p["number_female"].sum()
        male_ics = ics_p["number_male"].sum()
        pct_fem_ics = _safe_pct(fem_ics, fem_ics + male_ics)
        pct_all_fem_ics = _safe_pct(fem_ics, total_female_ics)

        n_out = len(out_p)
        fem_out = out_p["number_female"].sum()
        male_out = out_p["number_male"].sum()
        pct_fem_out = _safe_pct(fem_out, fem_out + male_out)
        pct_all_fem_out = _safe_pct(fem_out, total_female_out)

        fte = ics_p["fte"].sum(min_count=1)
        doc_deg = ics_p["num_doc_degrees_total"].sum(min_count=1)
        total_inc = _total_income(ics_p).sum()

        rows.append(
            {
                "Panel": panel,
                "FTE": fte,
                "PhDs": doc_deg,
                "Total inc": total_inc,
                "Number of ICS": n_ics,
                "% Female Authors (ICS)": pct_fem_ics,
                "% of All Female Authors (ICS)": pct_all_fem_ics,
                "Number of Outputs": n_out,
                "% Female Authors (Outputs)": pct_fem_out,
                "% of All Female Authors (Outputs)": pct_all_fem_out,
            }
        )

    return pd.DataFrame(rows)


def _uoa_table(df_ics: pd.DataFrame, df_output: pd.DataFrame) -> pd.DataFrame:
    df_ics = _ensure_panel(df_ics)
    df_output = _ensure_panel(df_output)
    total_female_ics = df_ics["number_female"].sum()
    total_female_out = df_output["number_female"].sum()

    uoa_numbers = sorted(set(df_ics["Unit of assessment number"]).union(set(df_output["Unit of assessment number"])))
    rows = []
    for uoa in uoa_numbers:
        ics_u = df_ics[df_ics["Unit of assessment number"] == uoa]
        out_u = df_output[df_output["Unit of assessment number"] == uoa]
        panel = uoa_to_panel(uoa)

        n_ics = len(ics_u)
        fem_ics = ics_u["number_female"].sum()
        male_ics = ics_u["number_male"].sum()
        pct_fem_ics = _safe_pct(fem_ics, fem_ics + male_ics)
        pct_all_fem_ics = _safe_pct(fem_ics, total_female_ics)

        n_out = len(out_u)
        fem_out = out_u["number_female"].sum()
        male_out = out_u["number_male"].sum()
        pct_fem_out = _safe_pct(fem_out, fem_out + male_out)
        pct_all_fem_out = _safe_pct(fem_out, total_female_out)

        fte = ics_u["fte"].sum(min_count=1)
        doc_deg = ics_u["num_doc_degrees_total"].sum(min_count=1)
        total_inc = _total_income(ics_u).sum()

        rows.append(
            {
                "Unit of Assessment": f"{uoa} - {UOA_MAP.get(uoa, 'Unknown')}",
                "Panel": panel,
                "FTE": fte,
                "PhDs": doc_deg,
                "Total inc": total_inc,
                "Number of ICS": n_ics,
                "% Female Authors (ICS)": pct_fem_ics,
                "% of All Female Authors (ICS)": pct_all_fem_ics,
                "Number of Outputs": n_out,
                "% Female Authors (Outputs)": pct_fem_out,
                "% of All Female Authors (Outputs)": pct_all_fem_out,
            }
        )
    return pd.DataFrame(rows)


def _llm_table(df_ics: pd.DataFrame) -> pd.DataFrame:
    llm_cols = [c for c in df_ics.columns if c.startswith("llm_")]
    total_female_ics = df_ics["number_female"].sum()
    rows = []
    for col in llm_cols:
        label = col.replace("llm_", "").replace("_", " ").title()
        subset = df_ics[df_ics[col] > 0]
        n_cases = len(subset)
        fem = subset["number_female"].sum()
        male = subset["number_male"].sum()
        pct_fem = _safe_pct(fem, fem + male)
        pct_all_fem = _safe_pct(fem, total_female_ics)
        rows.append(
            {
                "Topic": label,
                "Number of ICS": n_cases,
                "% Female Authors": pct_fem,
                "% of All Female Authors": pct_all_fem,
            }
        )
    return pd.DataFrame(rows)


def _llm_panel_table(df_ics: pd.DataFrame) -> pd.DataFrame:
    df_ics = _ensure_panel(df_ics)
    llm_cols = [c for c in df_ics.columns if c.startswith("llm_")]
    panel_order = ["A", "B", "C", "D"]
    rows = []
    for panel in panel_order:
        panel_df = df_ics[df_ics["Panel"] == panel]
        total_female_panel = panel_df["number_female"].sum()
        for col in llm_cols:
            label = col.replace("llm_", "").replace("_", " ").title()
            subset = panel_df[panel_df[col] > 0]
            n_cases = len(subset)
            fem = subset["number_female"].sum()
            male = subset["number_male"].sum()
            pct_fem = _safe_pct(fem, fem + male)
            pct_all_fem = _safe_pct(fem, total_female_panel)
            rows.append(
                {
                    "Panel": panel,
                    "Topic": label,
                    "Number of ICS": n_cases,
                    "% Female Authors": pct_fem,
                    "% of All Female Authors (panel)": pct_all_fem,
                }
            )
    return pd.DataFrame(rows)


def build_and_save_summary_tables(
    df_ics: pd.DataFrame,
    df_output: pd.DataFrame,
    out_dir: Path = Path("../outputs/tables"),
) -> Dict[str, pd.DataFrame]:
    """
    Build summary tables (panels, UoA, LLM topics overall, LLM topics by panel)
    and persist them as LaTeX files. Returns the DataFrames.
    """
    out_dir = Path(out_dir)

    panel_df = _panel_table(df_ics, df_output)
    panel_df = panel_df[
        [
            "Panel",
            "FTE",
            "PhDs",
            "Total inc",
            "Number of ICS",
            "% Female Authors (ICS)",
            "% of All Female Authors (ICS)",
            "Number of Outputs",
            "% Female Authors (Outputs)",
            "% of All Female Authors (Outputs)",
        ]
    ]
    panel_df["FTE"] = panel_df["FTE"].round().astype(int)
    panel_df["PhDs (000)"] = (panel_df["PhDs"] / 1000).round().astype(int)
    panel_df["Total Income (£bn)"] = _round_sig(panel_df["Total inc"] / 1e9, sig=4)
    panel_df = panel_df.drop(columns=["PhDs", "Total inc"]).rename(
        columns={
            "Number of ICS": "N (ICS)",
            "% Female Authors (ICS)": "% Female (ICS)",
            "% of All Female Authors (ICS)": "% All Female (ICS)",
            "Number of Outputs": "N (Papers)",
            "% Female Authors (Outputs)": "% Female (Papers)",
            "% of All Female Authors (Outputs)": "% All Female (Papers)",
        }
    )
    panel_df["% Female (ICS)"] = _round_sig(panel_df["% Female (ICS)"])
    panel_df["% All Female (ICS)"] = _round_sig(panel_df["% All Female (ICS)"])
    panel_df["% Female (Papers)"] = _round_sig(panel_df["% Female (Papers)"])
    panel_df["% All Female (Papers)"] = _round_sig(panel_df["% All Female (Papers)"])
    panel_df = panel_df[
        [
            "Panel",
            "FTE",
            "PhDs (000)",
            "Total Income (£bn)",
            "N (ICS)",
            "% Female (ICS)",
            "% All Female (ICS)",
            "N (Papers)",
            "% Female (Papers)",
            "% All Female (Papers)",
        ]
    ]
    _table_to_latex(panel_df, out_dir / "panel_summary.tex", column_format="lrrrrrrrrr")

    uoa_df = _uoa_table(df_ics, df_output)
    uoa_numbers = pd.to_numeric(uoa_df["Unit of Assessment"].str.split(" - ").str[0], errors="coerce").astype(int)
    uoa_df.insert(0, "UoA", uoa_numbers)
    uoa_df["Unit of Assessment"] = uoa_df["Unit of Assessment"].str.split(" - ").str[1]
    uoa_df = uoa_df[
        [
            "UoA",
            "Unit of Assessment",
            "Panel",
            "FTE",
            "PhDs",
            "Total inc",
            "Number of ICS",
            "% Female Authors (ICS)",
            "% of All Female Authors (ICS)",
            "Number of Outputs",
            "% Female Authors (Outputs)",
            "% of All Female Authors (Outputs)",
        ]
    ]
    uoa_df["FTE"] = uoa_df["FTE"].round().astype(int)
    uoa_df["PhDs (000)"] = (uoa_df["PhDs"] / 1000).round().astype(int)
    uoa_df["Total Income (£bn)"] = _round_sig(uoa_df["Total inc"] / 1e9, sig=4)
    uoa_df = uoa_df.drop(columns=["PhDs", "Total inc"]).rename(
        columns={
            "Number of ICS": "N (ICS)",
            "% Female Authors (ICS)": "% Female (ICS)",
            "% of All Female Authors (ICS)": "% All Female (ICS)",
            "Number of Outputs": "N (Papers)",
            "% Female Authors (Outputs)": "% Female (Papers)",
            "% of All Female Authors (Outputs)": "% All Female (Papers)",
        }
    )
    uoa_df["% Female (ICS)"] = _round_sig(uoa_df["% Female (ICS)"])
    uoa_df["% All Female (ICS)"] = _round_sig(uoa_df["% All Female (ICS)"])
    uoa_df["% Female (Papers)"] = _round_sig(uoa_df["% Female (Papers)"])
    uoa_df["% All Female (Papers)"] = _round_sig(uoa_df["% All Female (Papers)"])
    uoa_df = uoa_df[
        [
            "UoA",
            "Unit of Assessment",
            "Panel",
            "FTE",
            "PhDs (000)",
            "Total Income (£bn)",
            "N (ICS)",
            "% Female (ICS)",
            "% All Female (ICS)",
            "N (Papers)",
            "% Female (Papers)",
            "% All Female (Papers)",
        ]
    ]
    _table_to_latex(uoa_df, out_dir / "uoa_summary.tex", column_format="rllrrrrrrrrr")

    llm_df = _llm_table(df_ics)
    _table_to_latex(llm_df, out_dir / "llm_summary.tex", column_format="lrrr")

    llm_panel_df = _llm_panel_table(df_ics)
    _table_to_latex(llm_panel_df, out_dir / "llm_panel_summary.tex", column_format="llrrr")

    return {
        "panel": panel_df,
        "uoa": uoa_df,
        "llm": llm_df,
        "llm_panel": llm_panel_df,
    }


def llm_female_share_tables(df_ics: pd.DataFrame, panel_order: Iterable[str] = ("A", "B", "C", "D")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two tables with female percentages for each llm_* topic:
      - overall across all ICS
      - split by REF panel.
    """
    llm_cols: List[str] = [c for c in df_ics.columns if c.startswith("llm_")]
    panel_order = list(panel_order)

    def _rows(frame: pd.DataFrame, extra_cols: Dict | None = None):
        records = []
        for col in llm_cols:
            subset = frame[frame[col] > 0]
            female = subset["number_female"].sum()
            male = subset["number_male"].sum()
            total_people = female + male
            records.append(
                {
                    "llm_topic": col.replace("llm_", "").replace("_", " ").title(),
                    "n_cases": len(subset),
                    "share_of_ics": len(subset) / len(frame) if len(frame) else np.nan,
                    "female": female,
                    "total_people": total_people,
                    "pct_female": female / total_people if total_people else np.nan,
                    **(extra_cols or {}),
                }
            )
        return records

    overall = pd.DataFrame(_rows(df_ics))
    overall = overall.sort_values("llm_topic").reset_index(drop=True)

    panel_records: List[Dict] = []
    for panel in panel_order:
        panel_df = df_ics[df_ics["Panel"] == panel]
        panel_records.extend(_rows(panel_df, {"panel": panel}))
    by_panel = pd.DataFrame(panel_records)
    by_panel = by_panel.sort_values(["panel", "llm_topic"]).reset_index(drop=True)

    return overall, by_panel


def build_descriptive_summary(df_ics: pd.DataFrame, df_uoa_m: pd.DataFrame, df_uni_m: pd.DataFrame, df_output: pd.DataFrame) -> str:
    """Generate a multiline descriptive summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("DESCRIPTIVE SUMMARY OF FEMALE REPRESENTATION IN ICS & OUTPUTS")
    lines.append("=" * 60 + "\n")

    panel_order = ["A", "B", "C", "D"]
    llm_overall, llm_by_panel = llm_female_share_tables(df_ics, panel_order)

    n_fem_out = df_output["number_female"].sum()
    n_male_out = df_output["number_male"].sum()
    n_fem_ics = df_ics["number_female"].sum()
    n_male_ics = df_ics["number_male"].sum()
    pct_fem_out = _safe_pct(n_fem_out, n_fem_out + n_male_out)
    pct_fem_ics = _safe_pct(n_fem_ics, n_fem_ics + n_male_ics)

    lines.append("Overall female share:")
    lines.append(f"  • Outputs: {n_fem_out:,} women / {n_fem_out + n_male_out:,} total = {pct_fem_out:.2f}% female")
    lines.append(f"  • ICS:     {n_fem_ics:,} women / {n_fem_ics + n_male_ics:,} total = {pct_fem_ics:.2f}% female\n")

    # All-women ICS
    ics_all_female = df_ics[(df_ics["number_male"] == 0) & (df_ics["number_female"] > 0)]
    n_all_fem_ics = len(ics_all_female)
    pct_all_fem_ics = _safe_pct(n_all_fem_ics, len(df_ics))
    lines.append(f"All-female ICS submissions (excluding unknowns): {n_all_fem_ics:,} ({pct_all_fem_ics:.2f}% of all ICS cases)\n")

    # LLM-tagged topics (ICS only)
    lines.append("LLM-tagged ICS cases (aggregate across all panels):")
    if not llm_overall.empty:
        lines.append(
            llm_overall.assign(
                share_of_ics=lambda d: d["share_of_ics"] * 100,
                pct_female=lambda d: d["pct_female"] * 100,
            )
            .rename(columns={"share_of_ics": "% of ICS", "pct_female": "% female"})
            .to_string(
                index=False,
                formatters={
                    "% of ICS": "{:.2f}".format,
                    "% female": "{:.2f}".format,
                },
            )
        )
    else:
        lines.append("  No llm_* topic flags found.")

    lines.append("\nLLM-tagged ICS cases by panel:")
    if llm_by_panel.empty:
        lines.append("  No llm_* topic flags found.")
    else:
        panel_fmt = (
            llm_by_panel.assign(
                share_of_ics=lambda d: d["share_of_ics"] * 100,
                pct_female=lambda d: d["pct_female"] * 100,
            )
            .rename(columns={"share_of_ics": "% of panel ICS", "pct_female": "% female"})
            .to_string(
                index=False,
                formatters={
                    "% of panel ICS": "{:.2f}".format,
                    "% female": "{:.2f}".format,
                },
            )
        )
        lines.append(panel_fmt)
    lines.append("")  # spacer

    # Panel-level aggregates
    panel_counts = (
        df_uoa_m.groupby("Panel")[["number_female_y", "number_male_y", "number_female_x", "number_male_x"]]
        .sum(min_count=1)
        .rename(
            columns={
                "number_female_y": "female_ics",
                "number_male_y": "male_ics",
                "number_female_x": "female_output",
                "number_male_x": "male_output",
            }
        )
        .reindex(panel_order)
        .fillna(0)
    )
    lines.append("Female share by REF main panel (aggregated across UoAs):")
    for panel in panel_order:
        row = panel_counts.loc[panel]
        pct_panel_out = _safe_pct(row["female_output"], row["female_output"] + row["male_output"])
        pct_panel_ics = _safe_pct(row["female_ics"], row["female_ics"] + row["male_ics"])
        lines.append(
            f"  Panel {panel}: Outputs {int(row['female_output']):,}/{int(row['female_output'] + row['male_output']):,} "
            f"= {pct_panel_out:.2f}% | ICS {int(row['female_ics']):,}/{int(row['female_ics'] + row['male_ics']):,} = {pct_panel_ics:.2f}%"
        )

    all_female_panel = ics_all_female["Panel"].value_counts().reindex(panel_order, fill_value=0)
    total_panel_cases = df_ics["Panel"].value_counts().reindex(panel_order, fill_value=0)
    lines.append("\nAll-female ICS submissions by panel:")
    for panel in panel_order:
        num = all_female_panel.loc[panel]
        denom = total_panel_cases.loc[panel]
        pct = _safe_pct(num, denom)
        lines.append(f"  Panel {panel}: {num:,} ({pct:.2f}% of {denom:,} ICS cases)")

    # By University
    top_uni = df_uni_m[["Institution name", "pct_female_ics", "pct_female_output"]].dropna().sort_values("pct_female_ics", ascending=False)
    bottom_uni = top_uni.sort_values("pct_female_ics", ascending=True)

    lines.append("Universities with highest female Impact (ICS) proportions:")
    lines.append(top_uni.head(5).to_string(index=False))
    lines.append("\nUniversities with lowest female Impact (ICS) proportions:")
    lines.append(bottom_uni.head(5).to_string(index=False))

    df_uni_m = df_uni_m.copy()
    df_uni_m["diff_ics_output"] = df_uni_m["pct_female_ics"] - df_uni_m["pct_female_output"]
    lines.append("\nUniversities with largest positive difference (ICS − Output):")
    lines.append(
        df_uni_m[["Institution name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    lines.append("\nUniversities with largest negative difference (ICS − Output):")
    lines.append(
        df_uni_m[["Institution name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    # By UoA
    df_uoa_m = df_uoa_m.copy()
    df_uoa_m["diff_ics_output"] = df_uoa_m["pct_female_ics"] - df_uoa_m["pct_female_output"]
    top_uoa = df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output"]].dropna().sort_values("pct_female_ics", ascending=False)
    bottom_uoa = top_uoa.sort_values("pct_female_ics", ascending=True)

    lines.append("\nUoAs with highest female Impact (ICS) proportions:")
    lines.append(top_uoa.head(5).to_string(index=False))
    lines.append("\nUoAs with lowest female Impact (ICS) proportions:")
    lines.append(bottom_uoa.head(5).to_string(index=False))
    lines.append("\nUoAs with largest positive difference (ICS − Output):")
    lines.append(
        df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=False)
        .head(5)
        .to_string(index=False)
    )
    lines.append("\nUoAs with largest negative difference (ICS − Output):")
    lines.append(
        df_uoa_m[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
        .sort_values("diff_ics_output", ascending=True)
        .head(5)
        .to_string(index=False)
    )

    # Panel-specific UoA breakdowns
    lines.append("\nPanel-specific UoA breakdowns:")
    for panel in panel_order:
        panel_uoa = df_uoa_m[df_uoa_m["Panel"] == panel].copy()
        lines.append(f"\nPanel {panel}:")
        if panel_uoa.empty:
            lines.append("  No UoA data available for this panel.")
            continue

        panel_uoa["diff_ics_output"] = panel_uoa["pct_female_ics"] - panel_uoa["pct_female_output"]
        panel_top = panel_uoa[["Unit of assessment name", "pct_female_ics", "pct_female_output"]].dropna().sort_values("pct_female_ics", ascending=False)
        panel_bottom = panel_top.sort_values("pct_female_ics", ascending=True)

        lines.append("  UoAs with highest female Impact (ICS) proportions:")
        lines.append(panel_top.head(5).to_string(index=False))
        lines.append("  UoAs with lowest female Impact (ICS) proportions:")
        lines.append(panel_bottom.head(5).to_string(index=False))

        lines.append("  UoAs with largest positive difference (ICS − Output):")
        lines.append(
            panel_uoa[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
            .sort_values("diff_ics_output", ascending=False)
            .head(5)
            .to_string(index=False)
        )
        lines.append("  UoAs with largest negative difference (ICS − Output):")
        lines.append(
            panel_uoa[["Unit of assessment name", "pct_female_ics", "pct_female_output", "diff_ics_output"]]
            .sort_values("diff_ics_output", ascending=True)
            .head(5)
            .to_string(index=False)
        )

        lines.append("  Summary statistics (Impact) for pct female across UoAs:")
        lines.append(panel_uoa["pct_female_ics"].describe().round(3).to_string())
        lines.append("  Summary statistics (Outputs) for pct female across UoAs:")
        lines.append(panel_uoa["pct_female_output"].describe().round(3).to_string())

    # Discipline groups (STEM vs SHAPE)
    disc = (
        df_uoa_m[["Discipline_group", "pct_female_ics", "pct_female_output"]]
        .dropna(subset=["Discipline_group", "pct_female_ics", "pct_female_output"])
        .groupby("Discipline_group")[["pct_female_ics", "pct_female_output"]]
        .mean()
    )
    disc["diff"] = disc["pct_female_ics"] - disc["pct_female_output"]
    lines.append("\nAverage female share by Discipline group (STEM vs SHAPE):")
    lines.append(disc.to_string(float_format=lambda x: f"{x:.3f}"))

    # Distribution summaries
    lines.append("\nSummary statistics of female proportions across Universities and UoAs:")
    lines.append("Universities (Impact):")
    lines.append(df_uni_m["pct_female_ics"].describe().round(3).to_string())
    lines.append("\nUoAs (Impact):")
    lines.append(df_uoa_m["pct_female_ics"].describe().round(3).to_string())
    lines.append("\nUniversities (Outputs):")
    lines.append(df_uni_m["pct_female_output"].describe().round(3).to_string())
    lines.append("\nUoAs (Outputs):")
    lines.append(df_uoa_m["pct_female_output"].describe().round(3).to_string())

    return "\n".join(lines)


def _wald_ci_proportion(p, n, alpha=ALPHA):
    if n <= 0 or np.isnan(p):
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p * (1 - p) / n)
    return (p - z * se, p + z * se)


def _wald_ci_diff(p1, n1, p2, n2, alpha=ALPHA):
    if min(n1, n2) <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return (diff - z * se, diff + z * se)


def _two_prop_ztest(count1, nobs1, count2, nobs2, alternative="larger", alpha=ALPHA) -> Dict:
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2))
    z = (p1 - p2) / se
    if alternative == "two-sided":
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        alt_text = "p1 ≠ p2"
    elif alternative == "larger":
        pval = 1 - stats.norm.cdf(z)
        alt_text = "p1 > p2"
    else:
        pval = stats.norm.cdf(z)
        alt_text = "p1 < p2"
    ci_lo, ci_hi = _wald_ci_diff(p1, nobs1, p2, nobs2, alpha=alpha)
    signif = pval < alpha
    return {
        "z": z,
        "p": pval,
        "diff": p1 - p2,
        "ci": (ci_lo, ci_hi),
        "alt_text": alt_text,
        "significant": signif,
    }


def _describe_prop(name: str, p: float, n: int, alpha=ALPHA) -> str:
    lo, hi = _wald_ci_proportion(p, n, alpha)
    return f"{name}: p̂ = {p:.4f}  (95% CI [{lo:.4f}, {hi:.4f}]), n = {n}"


def _paired_suite(label: str, x_ics: pd.Series, x_out: pd.Series, alpha=ALPHA) -> str:
    df = pd.DataFrame({"ics": x_ics, "out": x_out}).dropna()
    d = (df["ics"] - df["out"]).to_numpy()
    n = d.size
    if n == 0:
        return f"{label}: no paired observations."
    mean_d = d.mean()
    sd_d = d.std(ddof=1) if n > 1 else np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    se = (sd_d / np.sqrt(n)) if n > 1 else np.nan
    ci = (mean_d - z * se, mean_d + z * se) if n > 1 else (np.nan, np.nan)
    dz = mean_d / sd_d if (n > 1 and sd_d > 0) else np.nan

    lines = []
    lines.append(f"\n— {label}: paired analysis for Δ = (ICS − Output)")
    lines.append(f"  Descriptives: n = {n}, mean(Δ) = {mean_d:.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}]), sd = {sd_d:.4f}, Cohen's dz = {dz:.3f}")

    if n >= 2:
        t_stat, p_t = stats.ttest_1samp(d, 0.0, alternative="greater")
        signif_t = "significant" if p_t < alpha else "not significant"
        direction_t = "positive" if mean_d > 0 else ("negative" if mean_d < 0 else "zero")
        lines.append(f"  t-test (mean Δ > 0): t = {t_stat:.3f}, p = {p_t:.3g} → {signif_t} at α={alpha} (mean Δ {direction_t}).")
    else:
        lines.append("  t-test: not applicable (n<2).")

    d_nz = d[d != 0]
    if d_nz.size > 0:
        try:
            w_stat, p_w = stats.wilcoxon(d_nz, alternative="greater", zero_method="wilcox", correction=False)
            med = np.median(d)
            signif_w = "significant" if p_w < alpha else "not significant"
            direction_w = "positive" if med > 0 else ("negative" if med < 0 else "zero")
            lines.append(f"  Wilcoxon (median Δ > 0): W = {w_stat:.3f}, p = {p_w:.3g} → {signif_w} (median Δ {direction_w}).")
        except Exception as e:
            lines.append(f"  Wilcoxon: not computed ({e}).")
    else:
        lines.append("  Wilcoxon: all differences are zero; test not applicable.")

    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_eff = n_pos + n_neg
    if n_eff > 0:
        p_bin = stats.binomtest(n_pos, n_eff, p=0.5, alternative="greater").pvalue
        frac_pos = n_pos / n_eff
        signif_s = "significant" if p_bin < alpha else "not significant"
        lines.append(f"  Sign test (P(Δ>0) > 0.5): positives = {n_pos}/{n_eff} ({frac_pos:.3%}), p = {p_bin:.3g} → {signif_s}.")
    else:
        lines.append("  Sign test: no nonzero differences; not applicable.")
    return "\n".join(lines)


def build_inference_summary(
    df_ics: pd.DataFrame,
    df_output: pd.DataFrame,
    df_uoa_m: pd.DataFrame,
    df_uni_m: pd.DataFrame,
    df_uniuoa_m: Optional[pd.DataFrame] = None,
    alpha=ALPHA,
) -> str:
    """
    One-sided tests asking whether female share in ICS exceeds Outputs at multiple levels.
    If the institution×UoA frame is not provided, it is re-computed from disk.
    """
    if df_uniuoa_m is None:
        _, _, _, _, df_uniuoa_m = load_statistics_data()

    lines = []
    lines.append("Hypothesis across all levels: female proportion in ICS exceeds Outputs (one-sided tests).\n")

    x_ics = int(df_ics["number_female"].sum())
    n_ics = int((df_ics["number_female"] + df_ics["number_male"]).sum())
    x_out = int(df_output["number_female"].sum())
    n_out = int((df_output["number_female"] + df_output["number_male"]).sum())

    p_ics = x_ics / n_ics if n_ics > 0 else np.nan
    p_out = x_out / n_out if n_out > 0 else np.nan

    lines.append("RAW pooled female shares:")
    lines.append(_describe_prop("  ICS   ", p_ics, n_ics, alpha=alpha))
    lines.append(_describe_prop("  Output", p_out, n_out, alpha=alpha))

    res = _two_prop_ztest(x_ics, n_ics, x_out, n_out, alternative="larger", alpha=alpha)
    lines.append("\nTwo-proportion z-test (RAW):")
    lines.append("  H0: p_ICS = p_Output   vs   H1: p_ICS > p_Output")
    ci_lo, ci_hi = res["ci"]
    signif_text = "statistically significant" if res["significant"] else "not statistically significant"
    direction = "positive" if res["diff"] > 0 else ("negative" if res["diff"] < 0 else "zero")
    lines.append(
        f"  z = {res['z']:.3f}, p = {res['p']:.3g} (H0: p1 = p2 vs H1: {res['alt_text']}). "
        f"Observed difference p1−p2 = {res['diff']:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}] "
        f"(95% CI, Wald, unpooled). Result is {signif_text} at α={alpha}; the estimated difference is {direction}."
    )
    lines.append("  Interpretation: This tests the overall female share across all observations. A significant result supports higher ICS share.\n")

    uni = df_uni_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("University level", uni["pct_female_ics"], uni["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: Positive, significant results indicate universities tend to have higher female shares in ICS than in Outputs.\n")

    uoa = df_uoa_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("Unit of Assessment (UoA) level", uoa["pct_female_ics"], uoa["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: Positive, significant results indicate disciplines (UoAs) tend to have higher female shares in ICS.\n")

    inst_uoa = df_uniuoa_m[["pct_female_ics", "pct_female_output"]].dropna()
    lines.append(_paired_suite("Institution × UoA level", inst_uoa["pct_female_ics"], inst_uoa["pct_female_output"], alpha=alpha))
    lines.append("  Interpretation: At the cell level, a significant positive Δ indicates the tendency persists at finer granularity.\n")

    lines.append("RECAP:")
    lines.append("  • RAW z-test asks: Is the *overall* female share higher in ICS than in Outputs?")
    lines.append("  • Paired tests (University, UoA, and Inst×UoA) ask: Within each unit, is ICS share higher than Outputs (Δ>0)?")
    lines.append("  • Convergent significance across multiple tests provides robust evidence that female ICS contribution exceeds Output.")

    return "\n".join(lines)
