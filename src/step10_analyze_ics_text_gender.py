"""
Step 10: Word-level association analysis for ICS text and case-level gender share.

This step uses the same free-text fields as thematic indicators:
    - 1. Summary of the impact
    - 4. Details of the impact

It estimates which words are statistically associated with the case-level female
share outcome (default: number_female / (number_female + number_male)).
Outputs:
    - CSV with all tested words and association statistics
    - CSVs with top positive / negative associated words
    - publication-style figure (pdf/svg/png)
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import FuncFormatter
from statsmodels.stats.multitest import multipletests

try:  # pragma: no cover
    from .figure_one_helpers import PANEL_COLORS, apply_mpl_defaults, resolve_enhanced_ref_data_path
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import atomic_write_csv, read_table
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from figure_one_helpers import PANEL_COLORS, apply_mpl_defaults, resolve_enhanced_ref_data_path
    from pipeline_config import load_config_and_paths
    from pipeline_io import atomic_write_csv, read_table
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs


DEFAULT_TEXT_COLUMNS = ("1. Summary of the impact", "4. Details of the impact")
DEFAULT_ID_COL = "REF impact case study identifier"
DEFAULT_OUTCOME_COL = "pct_female"

# Keep this explicit so results are reproducible across environments.
DEFAULT_STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "also",
    "although",
    "always",
    "am",
    "among",
    "an",
    "and",
    "another",
    "any",
    "anyone",
    "anything",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "case",
    "cases",
    "could",
    "data",
    "details",
    "did",
    "do",
    "does",
    "done",
    "down",
    "during",
    "each",
    "either",
    "enough",
    "even",
    "every",
    "for",
    "form",
    "found",
    "from",
    "further",
    "get",
    "getting",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "if",
    "impact",
    "impacts",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "less",
    "made",
    "make",
    "many",
    "may",
    "more",
    "most",
    "much",
    "must",
    "my",
    "near",
    "need",
    "new",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "per",
    "put",
    "rather",
    "ref",
    "research",
    "result",
    "results",
    "same",
    "she",
    "should",
    "since",
    "so",
    "some",
    "such",
    "summary",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "toward",
    "under",
    "up",
    "upon",
    "use",
    "used",
    "using",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "within",
    "without",
    "work",
    "worked",
    "working",
    "works",
    "would",
    "year",
    "years",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 10: analyse word-level relationships between ICS text and case-level female share."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--input", type=str, default=None, help="Override enhanced_ref_data path.")
    parser.add_argument(
        "--text-columns",
        nargs="+",
        default=list(DEFAULT_TEXT_COLUMNS),
        help="Text columns to concatenate before tokenisation.",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default=DEFAULT_OUTCOME_COL,
        help=(
            "Outcome column to analyse. If 'pct_female', it is computed as "
            "number_female/(number_female+number_male) when needed."
        ),
    )
    parser.add_argument("--id-col", type=str, default=DEFAULT_ID_COL, help="Identifier column for reporting rows.")
    parser.add_argument("--min-token-len", type=int, default=3, help="Minimum token length.")
    parser.add_argument(
        "--min-doc-freq",
        type=int,
        default=40,
        help="Minimum number of documents a token must appear in.",
    )
    parser.add_argument(
        "--max-doc-prop",
        type=float,
        default=0.85,
        help="Maximum document prevalence allowed for a token (0-1).",
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=5000,
        help="Maximum number of vocabulary terms retained after filtering.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of strongest positive/negative words to export and show.",
    )
    parser.add_argument(
        "--stopwords-file",
        type=str,
        default=None,
        help="Optional newline-delimited extra stopwords file.",
    )
    parser.add_argument(
        "--keep-default-stopwords",
        action="store_true",
        help="Do not remove default stopwords.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="supplementary_figure_3",
        help="Base filename for outputs.",
    )
    return parser.parse_args(argv)


def _normalise_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace("[\u2012\u2013\u2014\u2015]", "-", regex=True)
    s = s.str.replace(r"[^a-zA-Z\s'-]", " ", regex=True)
    s = s.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _load_stopwords(path: Path | None, keep_defaults: bool) -> set[str]:
    out = set() if keep_defaults else set(DEFAULT_STOPWORDS)
    if path and not path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {path}")
    if path and path.exists():
        extra = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        out |= extra
    return out


def _prepare_outcome(df: pd.DataFrame, outcome_col: str) -> pd.Series:
    if outcome_col in df.columns:
        return pd.to_numeric(df[outcome_col], errors="coerce")
    if outcome_col == "pct_female":
        required = {"number_female", "number_male"}
        if not required.issubset(df.columns):
            missing = sorted(required.difference(df.columns))
            raise ValueError(
                f"Cannot compute pct_female; missing required columns: {missing}"
            )
        female = pd.to_numeric(df["number_female"], errors="coerce")
        male = pd.to_numeric(df["number_male"], errors="coerce")
        denom = female + male
        out = female / denom
        out = out.where(denom > 0)
        return out
    raise ValueError(f"Outcome column '{outcome_col}' was not found in input data.")


def build_analysis_frame(
    df: pd.DataFrame,
    *,
    text_columns: Iterable[str],
    outcome_col: str,
    id_col: str,
) -> tuple[pd.DataFrame, float]:
    text_columns = list(text_columns)
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required text columns: {missing}")

    text = _normalise_text(df[text_columns[0]])
    for col in text_columns[1:]:
        text = (text + " " + _normalise_text(df[col])).str.strip()

    outcome = _prepare_outcome(df, outcome_col=outcome_col)
    if outcome.notna().sum() == 0:
        raise ValueError(f"Outcome '{outcome_col}' has no non-missing values after parsing.")

    if id_col in df.columns:
        ids = df[id_col]
    else:
        ids = pd.Series(df.index.astype(str), index=df.index, name=id_col)

    out = pd.DataFrame(
        {
            "row_id": ids.astype(str),
            "text": text,
            "outcome": outcome,
        },
        index=df.index,
    )
    out = out.loc[(out["text"] != "") & out["outcome"].notna()].copy()
    if out.empty:
        raise ValueError("No analysable rows: text and outcome overlap is empty.")

    # Convert to percentage-point scale in outputs when the underlying outcome is a fraction.
    y_min = float(out["outcome"].min())
    y_max = float(out["outcome"].max())
    scale = 100.0 if (y_min >= 0.0 and y_max <= 1.0) else 1.0
    return out, scale


def _tokenise_document(text: str, token_re: re.Pattern[str], stopwords: set[str]) -> set[str]:
    tokens = set(token_re.findall(text))
    if stopwords:
        tokens = {t for t in tokens if t not in stopwords}
    return tokens


def build_document_tokens(
    text: pd.Series,
    *,
    min_token_len: int,
    stopwords: set[str],
) -> tuple[list[set[str]], Counter]:
    token_re = re.compile(rf"[a-z]{{{int(min_token_len)},}}")
    doc_tokens: list[set[str]] = []
    doc_freq: Counter = Counter()
    for doc in text.astype(str).tolist():
        tokens = _tokenise_document(doc, token_re=token_re, stopwords=stopwords)
        doc_tokens.append(tokens)
        doc_freq.update(tokens)
    return doc_tokens, doc_freq


def select_vocabulary(
    doc_freq: Counter,
    *,
    n_docs: int,
    min_doc_freq: int,
    max_doc_prop: float,
    max_vocab: int,
) -> list[str]:
    rows = []
    for token, n in doc_freq.items():
        prop = n / n_docs if n_docs else 0.0
        if n < min_doc_freq:
            continue
        if prop > max_doc_prop:
            continue
        rows.append((token, n, prop))
    if not rows:
        return []
    rows.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab > 0:
        rows = rows[:max_vocab]
    return [r[0] for r in rows]


def build_token_to_doc_index(doc_tokens: list[set[str]], vocabulary: set[str]) -> dict[str, list[int]]:
    token_to_docs: dict[str, list[int]] = defaultdict(list)
    for i, tokset in enumerate(doc_tokens):
        for token in tokset:
            if token in vocabulary:
                token_to_docs[token].append(i)
    return dict(token_to_docs)


def _welch_from_summaries(
    n1: int,
    n0: int,
    mean1: float,
    mean0: float,
    var1: float,
    var0: float,
) -> tuple[float, float, float, float, float, float]:
    delta = mean1 - mean0
    se_term = (var1 / n1) + (var0 / n0)
    if not np.isfinite(se_term) or se_term <= 0:
        return delta, np.nan, 1.0, np.nan, np.nan, np.nan
    se = math.sqrt(se_term)
    t_stat = delta / se
    num = se_term**2
    den = ((var1 / n1) ** 2) / (n1 - 1) + ((var0 / n0) ** 2) / (n0 - 1)
    df_welch = num / den if den > 0 else np.nan
    if np.isfinite(df_welch) and df_welch > 0:
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df_welch))
        t_crit = float(stats.t.ppf(0.975, df_welch))
    else:
        p_value = 1.0
        t_crit = np.nan
    ci_low = delta - t_crit * se if np.isfinite(t_crit) else np.nan
    ci_high = delta + t_crit * se if np.isfinite(t_crit) else np.nan
    return delta, t_stat, p_value, df_welch, ci_low, ci_high


def compute_associations(
    y: np.ndarray,
    token_to_docs: dict[str, list[int]],
    *,
    outcome_scale: float,
) -> pd.DataFrame:
    n_docs = int(len(y))
    y_sum = float(y.sum())
    y_sumsq = float(np.square(y).sum())
    y_std = float(np.std(y, ddof=1)) if n_docs > 1 else np.nan

    rows: list[dict[str, object]] = []
    for token, idx_list in token_to_docs.items():
        n1 = int(len(idx_list))
        n0 = int(n_docs - n1)
        if n1 < 2 or n0 < 2:
            continue

        idx = np.asarray(idx_list, dtype=int)
        y_present = y[idx]
        sum1 = float(y_present.sum())
        sumsq1 = float(np.square(y_present).sum())
        sum0 = y_sum - sum1
        sumsq0 = y_sumsq - sumsq1
        mean1 = sum1 / n1
        mean0 = sum0 / n0

        var1 = (sumsq1 - n1 * mean1 * mean1) / (n1 - 1)
        var0 = (sumsq0 - n0 * mean0 * mean0) / (n0 - 1)
        var1 = float(max(var1, 0.0))
        var0 = float(max(var0, 0.0))

        delta, t_stat, p_value, df_welch, ci_low, ci_high = _welch_from_summaries(
            n1=n1,
            n0=n0,
            mean1=mean1,
            mean0=mean0,
            var1=var1,
            var0=var0,
        )
        prevalence = n1 / n_docs if n_docs else np.nan

        if np.isfinite(y_std) and y_std > 0 and np.isfinite(prevalence) and 0 < prevalence < 1:
            r_point_biserial = float(delta * math.sqrt(prevalence * (1.0 - prevalence)) / y_std)
        else:
            r_point_biserial = np.nan

        rows.append(
            {
                "token": token,
                "n_docs_present": n1,
                "n_docs_absent": n0,
                "prevalence": prevalence,
                "prevalence_pct": prevalence * 100.0,
                "mean_outcome_present": mean1,
                "mean_outcome_absent": mean0,
                "delta_outcome": delta,
                "delta_pct_points": delta * outcome_scale,
                "ci_low_outcome": ci_low,
                "ci_high_outcome": ci_high,
                "ci_low_pct_points": ci_low * outcome_scale if np.isfinite(ci_low) else np.nan,
                "ci_high_pct_points": ci_high * outcome_scale if np.isfinite(ci_high) else np.nan,
                "t_stat": t_stat,
                "df_welch": df_welch,
                "p_value": p_value,
                "r_point_biserial": r_point_biserial,
            }
        )

    if not rows:
        return pd.DataFrame()

    assoc = pd.DataFrame(rows)
    valid = assoc["p_value"].notna() & np.isfinite(assoc["p_value"].to_numpy())
    q_values = np.full(len(assoc), np.nan, dtype=float)
    if bool(valid.any()):
        q_values[valid.to_numpy()] = multipletests(
            assoc.loc[valid, "p_value"].to_numpy(),
            method="fdr_bh",
        )[1]
    assoc["q_value"] = q_values
    assoc["significant_fdr_05"] = assoc["q_value"] <= 0.05
    assoc["abs_delta_pct_points"] = assoc["delta_pct_points"].abs()
    assoc = assoc.sort_values(
        ["q_value", "p_value", "abs_delta_pct_points", "n_docs_present"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return assoc


def build_top_tables(assoc: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = assoc[assoc["q_value"].notna()].copy()
    if base.empty:
        base = assoc.copy()

    top_pos = (
        base[base["delta_pct_points"] > 0]
        .sort_values(["q_value", "delta_pct_points", "n_docs_present"], ascending=[True, False, False], na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )
    top_neg = (
        base[base["delta_pct_points"] < 0]
        .sort_values(["q_value", "delta_pct_points", "n_docs_present"], ascending=[True, True, False], na_position="last")
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_pos, top_neg


def _prep_pub_style() -> None:
    apply_mpl_defaults()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.unicode_minus"] = False


def plot_word_association_figure(
    assoc: pd.DataFrame,
    *,
    top_n: int,
    outcome_label: str,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    if assoc.empty:
        raise ValueError("Association table is empty; cannot plot.")

    _prep_pub_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 8))

    top_pos, top_neg = build_top_tables(assoc, top_n=top_n)
    panel_df = pd.concat([top_neg, top_pos], ignore_index=True)
    if panel_df.empty:
        panel_df = assoc.head(top_n).copy()
    panel_df = panel_df.sort_values("delta_pct_points", ascending=True).reset_index(drop=True)

    neg_color = PANEL_COLORS.get("B", "#0072B2")
    pos_color = PANEL_COLORS.get("C", "#E76F00")
    y = np.arange(len(panel_df))
    colors = [
        pos_color if d >= 0 else neg_color
        for d in panel_df["delta_pct_points"].to_numpy()
    ]
    ax_a.axvline(0.0, color="k", linestyle="--", linewidth=1.2, zorder=1)
    for i, row in panel_df.iterrows():
        ax_a.hlines(
            i,
            row["ci_low_pct_points"],
            row["ci_high_pct_points"],
            color="0.35",
            linewidth=1.2,
            zorder=2,
        )
    ax_a.scatter(
        panel_df["delta_pct_points"],
        y,
        s=60,
        c=colors,
        edgecolor="k",
        linewidth=0.35,
        zorder=3,
    )

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(panel_df["token"].tolist())
    ax_a.set_xlabel(f"Difference in {outcome_label}", fontsize=13)
    ax_a.set_title("a.", loc="left", fontsize=17, fontweight="bold")
    ax_a.tick_params(axis="both", labelsize=11)
    ax_a.set_axisbelow(True)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=pos_color,
            markeredgecolor="k",
            markeredgewidth=0.35,
            markersize=7,
            label="Higher female share",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=neg_color,
            markeredgecolor="k",
            markeredgewidth=0.35,
            markersize=7,
            label="Lower female share",
        ),
    ]
    ax_a.legend(handles=legend_handles, frameon=True, edgecolor="k", fontsize=10, loc="lower right")

    scatter_df = assoc.copy()
    q_for_color = scatter_df["q_value"].fillna(1.0).clip(lower=1e-12)
    color_values = -np.log10(q_for_color)
    marker_size = np.clip(24 + scatter_df["n_docs_present"].to_numpy() / 8.0, 24, 180)
    sc = ax_b.scatter(
        scatter_df["prevalence_pct"],
        scatter_df["delta_pct_points"],
        c=color_values,
        cmap="viridis",
        s=marker_size,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.2,
        zorder=2,
    )
    ax_b.axhline(0.0, color="k", linestyle="--", linewidth=1.2, zorder=1)
    ax_b.set_xlabel("Word Prevalence Across ICS Texts", fontsize=13)
    ax_b.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    ax_b.set_ylabel(f"Difference in {outcome_label} (pp)", fontsize=13)
    ax_b.set_title("b.", loc="left", fontsize=17, fontweight="bold")
    ax_b.tick_params(axis="both", labelsize=11)
    ax_b.set_axisbelow(True)

    label_df = (
        scatter_df.dropna(subset=["q_value"])
        .sort_values(["q_value", "abs_delta_pct_points"], ascending=[True, False])
        .head(15)
    )
    for _, row in label_df.iterrows():
        ax_b.annotate(
            str(row["token"]),
            xy=(row["prevalence_pct"], row["delta_pct_points"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=9,
            color="0.15",
        )

    cbar = fig.colorbar(sc, ax=ax_b, fraction=0.065, pad=0.02)
    cbar.set_label("-log10(FDR q-value)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    for ax in (ax_a, ax_b):
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    return fig, (ax_a, ax_b)


def save_figure_triplet(fig: plt.Figure, stem: Path) -> dict[str, Path]:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return {"pdf": pdf, "png": png, "svg": svg}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("[step10] Starting ICS text-word gender association analysis...")
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    print(f"[step10] Project root: {project_root}")
    _config, paths = load_config_and_paths(
        config_path=Path(args.config) if args.config else None,
        project_root=project_root,
    )
    ensure_core_dirs(paths)

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts: dict[str, int] = {}

    tables_dir = paths.outputs_dir / "tables"
    figures_dir = paths.outputs_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    basename = str(args.basename)
    table_all_path = tables_dir / f"{basename}_all.csv"
    table_top_pos_path = tables_dir / f"{basename}_top_positive.csv"
    table_top_neg_path = tables_dir / f"{basename}_top_negative.csv"
    fig_stem = figures_dir / basename

    output_paths = {
        "table_all": table_all_path,
        "table_top_positive": table_top_pos_path,
        "table_top_negative": table_top_neg_path,
        "figure_pdf": fig_stem.with_suffix(".pdf"),
        "figure_png": fig_stem.with_suffix(".png"),
        "figure_svg": fig_stem.with_suffix(".svg"),
    }

    input_path = Path(args.input) if args.input else resolve_enhanced_ref_data_path(paths.data_dir)
    input_path = input_path.resolve()
    print(f"[step10] Input data: {input_path}")

    try:
        df = read_table(input_path)
        print(f"[step10] Loaded rows={len(df):,}, cols={len(df.columns):,}")

        analysis_df, outcome_scale = build_analysis_frame(
            df,
            text_columns=args.text_columns,
            outcome_col=str(args.outcome_col),
            id_col=str(args.id_col),
        )
        print(
            "[step10] Analysable rows after text/outcome filtering: "
            f"{len(analysis_df):,}"
        )

        stopwords = _load_stopwords(
            Path(args.stopwords_file).resolve() if args.stopwords_file else None,
            keep_defaults=bool(args.keep_default_stopwords),
        )
        print(f"[step10] Stopwords loaded: {len(stopwords):,}")

        doc_tokens, doc_freq = build_document_tokens(
            analysis_df["text"],
            min_token_len=int(args.min_token_len),
            stopwords=stopwords,
        )
        vocab = select_vocabulary(
            doc_freq,
            n_docs=len(doc_tokens),
            min_doc_freq=int(args.min_doc_freq),
            max_doc_prop=float(args.max_doc_prop),
            max_vocab=int(args.max_vocab),
        )
        if not vocab:
            raise ValueError(
                "No tokens remained after frequency/prevalence filtering. "
                "Lower --min-doc-freq or increase --max-doc-prop."
            )
        print(f"[step10] Vocabulary size after filtering: {len(vocab):,}")

        token_to_docs = build_token_to_doc_index(doc_tokens, set(vocab))
        y = analysis_df["outcome"].to_numpy(dtype=float)
        assoc = compute_associations(y, token_to_docs, outcome_scale=outcome_scale)
        if assoc.empty:
            raise ValueError("No token associations were computed.")
        print(f"[step10] Computed associations for {len(assoc):,} words.")

        top_pos, top_neg = build_top_tables(assoc, top_n=int(args.top_n))
        print(
            "[step10] Top tables: "
            f"positive={len(top_pos):,}, negative={len(top_neg):,}"
        )

        atomic_write_csv(assoc, table_all_path)
        atomic_write_csv(top_pos, table_top_pos_path)
        atomic_write_csv(top_neg, table_top_neg_path)
        print(f"[step10] Wrote tables to {tables_dir}")

        outcome_label = (
            "Female Share"
            if str(args.outcome_col) == "pct_female"
            else str(args.outcome_col)
        )
        fig, _axes = plot_word_association_figure(
            assoc,
            top_n=int(args.top_n),
            outcome_label=outcome_label,
        )
        save_figure_triplet(fig, fig_stem)
        plt.close(fig)
        print(f"[step10] Wrote figure triplet to {fig_stem.parent} with stem '{fig_stem.name}'.")

        row_counts = {
            "input_rows": int(len(df)),
            "analysis_rows": int(len(analysis_df)),
            "vocab_size": int(len(vocab)),
            "n_words_tested": int(len(assoc)),
            "n_significant_fdr_05": int(assoc["significant_fdr_05"].fillna(False).sum()),
        }
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        print(f"[step10] Failed: {exc}")
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step10_analyze_ics_text_gender",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={
                "input": str(input_path),
                "text_columns": list(args.text_columns),
                "outcome_col": str(args.outcome_col),
                "min_token_len": int(args.min_token_len),
                "min_doc_freq": int(args.min_doc_freq),
                "max_doc_prop": float(args.max_doc_prop),
                "max_vocab": int(args.max_vocab),
                "top_n": int(args.top_n),
                "keep_default_stopwords": bool(args.keep_default_stopwords),
                "stopwords_file": str(args.stopwords_file) if args.stopwords_file else None,
                "basename": basename,
            },
            input_paths={"enhanced_ref_data": input_path},
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )

    print("[step10] Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
