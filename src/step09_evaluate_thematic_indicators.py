from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

try:  # pragma: no cover
    from .figure_one_helpers import PANEL_COLORS, apply_mpl_defaults, resolve_enhanced_ref_data_path
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import atomic_write_csv, read_table
except ImportError:  # pragma: no cover
    from figure_one_helpers import PANEL_COLORS, apply_mpl_defaults, resolve_enhanced_ref_data_path
    from pipeline_config import load_config_and_paths
    from pipeline_io import atomic_write_csv, read_table


TRUSTED_LLM_STATUSES = {
    "ok",
    "cached",
    "ok_schema_fallback",
    "ok_prompt_cache_fallback",
    "ok_schema_prompt_cache_fallback",
}
BAD_LLM_STATUSES = {"missing", "disabled", "not_run", "error", "parse_error"}

METHOD_PREFIXES = ("regex", "llmmini", "llm51", "llm54")
METHOD_DISPLAY = {
    "regex": "Regex",
    "llmmini": "GPT-5-mini",
    "llm51": "GPT-5.1",
    "llm54": "GPT-5.4",
}

PAIR_SPECS: dict[str, tuple[str, str, str]] = {
    "regex_vs_mini": ("Regex vs GPT-5-mini", "regex", "llmmini"),
    "regex_vs_51": ("Regex vs GPT-5.1", "regex", "llm51"),
    "regex_vs_54": ("Regex vs GPT-5.4", "regex", "llm54"),
    "mini_vs_51": ("GPT-5-mini vs GPT-5.1", "llmmini", "llm51"),
    "mini_vs_54": ("GPT-5-mini vs GPT-5.4", "llmmini", "llm54"),
    "51_vs_54": ("GPT-5.1 vs GPT-5.4", "llm51", "llm54"),
}
PAIR_ORDER = list(PAIR_SPECS.keys())

PAIR_COLORS = {
    "regex_vs_mini": "#4D4D4D",
    "regex_vs_51": PANEL_COLORS.get("B", "#0072B2"),
    "regex_vs_54": PANEL_COLORS.get("C", "#E76F00"),
    "mini_vs_51": "#4E79A7",
    "mini_vs_54": "#59A14F",
    "51_vs_54": PANEL_COLORS.get("D", "#B2182B"),
}


def _normalise_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace("[\u2012\u2013\u2014\u2015]", "-", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _make_cache_key(text: str, *, model: str, prompt_version: str) -> str:
    basis = f"{prompt_version}\n{model}\n{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _to_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    if isinstance(value, (int, np.integer)):
        return float(int(value != 0))
    if isinstance(value, float):
        if np.isnan(value):
            return np.nan
        return float(int(value != 0.0))
    txt = str(value).strip().lower()
    if txt in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if txt in {"0", "false", "f", "no", "n"}:
        return 0.0
    return np.nan


def _format_topic_label(topic: str) -> str:
    if str(topic).strip().lower() == "nhs":
        return "NHS"
    return str(topic).replace("_", " ").title()


def discover_topics(df: pd.DataFrame) -> list[str]:
    regex_topics = {c.replace("regex_", "") for c in df.columns if c.startswith("regex_")}
    llm_topics = {
        c.replace("llm_", "")
        for c in df.columns
        if c.startswith("llm_") and c not in {"llm_status", "llm_error"}
    }
    if regex_topics and llm_topics:
        return sorted(regex_topics & llm_topics)
    return sorted(regex_topics or llm_topics)


def add_hybrid_columns(df: pd.DataFrame, topics: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    has_status = "llm_status" in out.columns
    trusted = out["llm_status"].astype(str).isin(TRUSTED_LLM_STATUSES) if has_status else None
    for topic in topics:
        regex_col = f"regex_{topic}"
        llm_col = f"llm_{topic}"
        hybrid_col = f"hybrid_{topic}"
        if regex_col not in out.columns:
            out[regex_col] = 0
        if llm_col not in out.columns:
            out[llm_col] = out[regex_col]
        if trusted is None:
            out[hybrid_col] = out[llm_col]
        else:
            out[hybrid_col] = np.where(trusted, out[llm_col], out[regex_col])
        out[regex_col] = pd.to_numeric(out[regex_col], errors="coerce").fillna(0).astype("int8")
        out[llm_col] = pd.to_numeric(out[llm_col], errors="coerce").fillna(0).astype("int8")
        out[hybrid_col] = pd.to_numeric(out[hybrid_col], errors="coerce").fillna(0).astype("int8")
    return out


def attach_model_predictions_from_cache(
    df: pd.DataFrame,
    topics: list[str],
    categories_path: Path,
    *,
    model_mini: str = "gpt-5-nano",
    prompt_mini: str = "v2",
    model_51: str = "gpt-5.1",
    prompt_51: str = "v2",
    model_54: str = "gpt-5.4",
    prompt_54: str = "v2",
) -> pd.DataFrame:
    if not Path(categories_path).exists():
        raise FileNotFoundError(f"Missing OpenAI cache file: {categories_path}")

    cache_df = pd.read_csv(categories_path).copy()
    for col in ["cache_key", "text", "model", "prompt_version", "llm_status", "llm_error"]:
        if col not in cache_df.columns:
            cache_df[col] = ""
    for topic in topics:
        if topic not in cache_df.columns:
            cache_df[topic] = 0
        cache_df[topic] = pd.to_numeric(cache_df[topic], errors="coerce").fillna(0).astype("int8")

    cache_df["cache_key"] = cache_df["cache_key"].fillna("").astype(str)
    cache_df["text"] = cache_df["text"].fillna("").astype(str)
    cache_df["model"] = cache_df["model"].fillna("").astype(str)
    cache_df["prompt_version"] = cache_df["prompt_version"].fillna("").astype(str)
    cache_df = cache_df.drop_duplicates(subset=["cache_key"], keep="last")

    method_specs = {
        "llmmini": {"model": str(model_mini), "prompt": str(prompt_mini)},
        "llm51": {"model": str(model_51), "prompt": str(prompt_51)},
        "llm54": {"model": str(model_54), "prompt": str(prompt_54)},
    }

    def _subset_for(model: str, prompt: str) -> pd.DataFrame:
        sub = cache_df[(cache_df["model"] == model) & (cache_df["prompt_version"] == prompt)].copy()
        return sub.drop_duplicates(subset=["cache_key"], keep="last")

    maps: dict[str, dict[str, dict[str, object]]] = {}
    text_maps: dict[str, dict[str, dict[str, object]]] = {}
    for prefix, spec in method_specs.items():
        sub = _subset_for(spec["model"], spec["prompt"])
        maps[prefix] = sub.set_index("cache_key").to_dict("index") if not sub.empty else {}
        text_maps[prefix] = {
            str(r["text"]): r for _, r in sub.iterrows() if str(r.get("text", "")).strip()
        }

    out = df.copy()
    text_cols = ["1. Summary of the impact", "4. Details of the impact"]
    missing_text = [c for c in text_cols if c not in out.columns]
    if missing_text:
        raise ValueError(f"Input table missing text columns required for cache linkage: {missing_text}")

    impact_text = (_normalise_text(out[text_cols[0]]) + " " + _normalise_text(out[text_cols[1]])).str.strip()

    for topic in topics:
        for prefix in method_specs:
            out[f"{prefix}_{topic}"] = np.nan
    for prefix in method_specs:
        out[f"{prefix}_status"] = "missing"

    for idx, text in impact_text.items():
        text = str(text)
        if not text:
            for prefix in method_specs:
                out.at[idx, f"{prefix}_status"] = "empty_text"
                for topic in topics:
                    out.at[idx, f"{prefix}_{topic}"] = 0
            continue

        for prefix, spec in method_specs.items():
            key = _make_cache_key(text, model=spec["model"], prompt_version=spec["prompt"])
            row = maps[prefix].get(key)
            if row is None:
                row = text_maps[prefix].get(text)
            if row is not None:
                out.at[idx, f"{prefix}_status"] = str(row.get("llm_status", "ok") or "ok")
                for topic in topics:
                    out.at[idx, f"{prefix}_{topic}"] = int(bool(row.get(topic, 0)))

    return out


def validate_method_coverage(df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    errors: list[str] = []

    if "1. Summary of the impact" in df.columns and "4. Details of the impact" in df.columns:
        text = (_normalise_text(df["1. Summary of the impact"]) + " " + _normalise_text(df["4. Details of the impact"])).str.strip()
        non_empty_mask = text != ""
    else:
        non_empty_mask = pd.Series(True, index=df.index)

    for prefix in METHOD_PREFIXES:
        cols = [f"{prefix}_{topic}" for topic in topics if f"{prefix}_{topic}" in df.columns]
        if not cols:
            errors.append(f"{METHOD_DISPLAY.get(prefix, prefix)}: no topic columns found.")
            continue

        raw = df[cols].apply(pd.to_numeric, errors="coerce")
        filled = raw.fillna(0).astype("int8")

        row = {
            "method": prefix,
            "method_label": METHOD_DISPLAY.get(prefix, prefix),
            "n_rows": int(len(df)),
            "n_rows_nonempty_text": int(non_empty_mask.sum()),
            "n_missing_cells_all_rows": int(raw.isna().sum().sum()),
            "n_rows_any_positive": int((filled.sum(axis=1) > 0).sum()),
            "total_positive_flags": int(filled.to_numpy().sum()),
            "bad_status_rows_nonempty": 0,
            "disabled_rows_nonempty": 0,
            "missing_status_rows_nonempty": 0,
            "status_counts_nonempty": "",
        }

        if row["total_positive_flags"] == 0:
            errors.append(f"{row['method_label']}: all topic flags are zero.")

        if prefix == "regex":
            rows.append(row)
            continue

        status_col = f"{prefix}_status"
        if status_col not in df.columns:
            errors.append(f"{row['method_label']}: missing status column {status_col}.")
            rows.append(row)
            continue

        status = df[status_col].fillna("missing").astype(str).str.strip().str.lower()
        status_non_empty = status[non_empty_mask]
        raw_non_empty = raw.loc[non_empty_mask]

        bad_status_rows = int(status_non_empty.isin(BAD_LLM_STATUSES).sum())
        disabled_rows = int((status_non_empty == "disabled").sum())
        missing_status_rows = int((status_non_empty == "missing").sum())
        missing_cells_non_empty = int(raw_non_empty.isna().sum().sum())

        row["bad_status_rows_nonempty"] = bad_status_rows
        row["disabled_rows_nonempty"] = disabled_rows
        row["missing_status_rows_nonempty"] = missing_status_rows
        row["n_missing_cells_nonempty"] = missing_cells_non_empty
        row["status_counts_nonempty"] = "; ".join(
            [f"{k}:{v}" for k, v in status_non_empty.value_counts(dropna=False).to_dict().items()]
        )

        if bad_status_rows > 0:
            errors.append(
                f"{row['method_label']}: found bad statuses in non-empty rows "
                f"(disabled/missing/not_run/error/parse_error) = {bad_status_rows}."
            )
        if missing_cells_non_empty > 0:
            errors.append(f"{row['method_label']}: found missing prediction cells in non-empty rows: {missing_cells_non_empty}.")

        ok_like = status_non_empty.eq("cached") | status_non_empty.str.startswith("ok") | status_non_empty.eq("empty_text")
        if int(ok_like.sum()) == 0:
            errors.append(f"{row['method_label']}: no usable completed statuses found.")

        rows.append(row)

    summary = pd.DataFrame(rows)
    if errors:
        detail = "\n".join(f"- {e}" for e in errors)
        summary_txt = summary.to_string(index=False) if not summary.empty else "<empty summary>"
        raise ValueError(
            "Classification completeness checks failed.\n"
            "Resolve these before analysis:\n"
            f"{detail}\n\n"
            "Coverage summary:\n"
            f"{summary_txt}"
        )
    return summary


def _pair_metrics(a: pd.Series, b: pd.Series) -> dict[str, float]:
    a_bin = a.map(_to_binary)
    b_bin = b.map(_to_binary)
    valid = a_bin.isin([0.0, 1.0]) & b_bin.isin([0.0, 1.0])
    if not valid.any():
        return {
            "n_valid": 0,
            "n_agree": 0,
            "n_disagree": 0,
            "agreement_rate": np.nan,
            "disagreement_rate": np.nan,
            "n_agree_positive": 0,
            "n_agree_negative": 0,
            "agree_positive_share": np.nan,
        }

    a_i = a_bin[valid].astype(int)
    b_i = b_bin[valid].astype(int)
    agree = a_i == b_i
    n_valid = int(valid.sum())
    n_agree = int(agree.sum())
    n_disagree = int((~agree).sum())
    n_agree_positive = int(((agree) & (a_i == 1)).sum())
    n_agree_negative = int(((agree) & (a_i == 0)).sum())

    return {
        "n_valid": n_valid,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": (n_agree / n_valid) if n_valid else np.nan,
        "disagreement_rate": (n_disagree / n_valid) if n_valid else np.nan,
        "n_agree_positive": n_agree_positive,
        "n_agree_negative": n_agree_negative,
        "agree_positive_share": (n_agree_positive / n_agree) if n_agree else np.nan,
    }


def build_pairwise_agreement_tables(df: pd.DataFrame, topics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for pair_id, (pair_label, left_prefix, right_prefix) in PAIR_SPECS.items():
        for topic in topics:
            left_col = f"{left_prefix}_{topic}"
            right_col = f"{right_prefix}_{topic}"
            if left_col not in df.columns or right_col not in df.columns:
                continue
            m = _pair_metrics(df[left_col], df[right_col])
            rows.append(
                {
                    "pair_id": pair_id,
                    "pair": pair_label,
                    "left_prefix": left_prefix,
                    "right_prefix": right_prefix,
                    "topic": topic,
                    "topic_label": _format_topic_label(topic),
                    **m,
                }
            )

    by_topic = pd.DataFrame(rows)
    if by_topic.empty:
        summary = pd.DataFrame(
            columns=[
                "pair_id",
                "pair",
                "n_topic_rows",
                "n_valid_micro",
                "n_agree_micro",
                "n_disagree_micro",
                "agreement_rate_micro",
                "agreement_rate_macro",
                "agree_positive_share_micro",
                "agree_positive_share_macro",
            ]
        )
        return by_topic, summary

    by_topic["pair_id"] = pd.Categorical(by_topic["pair_id"], categories=PAIR_ORDER, ordered=True)
    by_topic = by_topic.sort_values(["pair_id", "agreement_rate", "topic"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for pair_id in PAIR_ORDER:
        grp = by_topic.loc[by_topic["pair_id"] == pair_id]
        if grp.empty:
            continue
        pair_label = str(grp["pair"].iloc[0])
        n_valid_micro = int(grp["n_valid"].sum())
        n_agree_micro = int(grp["n_agree"].sum())
        n_disagree_micro = int(grp["n_disagree"].sum())
        n_agree_positive_micro = int(grp["n_agree_positive"].sum())
        summary_rows.append(
            {
                "pair_id": pair_id,
                "pair": pair_label,
                "n_topic_rows": int(len(grp)),
                "n_valid_micro": n_valid_micro,
                "n_agree_micro": n_agree_micro,
                "n_disagree_micro": n_disagree_micro,
                "agreement_rate_micro": (n_agree_micro / n_valid_micro) if n_valid_micro else np.nan,
                "agreement_rate_macro": float(grp["agreement_rate"].mean()),
                "agree_positive_share_micro": (n_agree_positive_micro / n_agree_micro) if n_agree_micro else np.nan,
                "agree_positive_share_macro": float(grp["agree_positive_share"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return by_topic, summary


def build_topic_positive_rates(df: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in METHOD_PREFIXES:
        for topic in topics:
            col = f"{source}_{topic}"
            if col not in df.columns:
                continue
            pred = df[col].map(_to_binary)
            valid = pred.isin([0.0, 1.0])
            if not valid.any():
                continue
            p = pred[valid].astype(int)
            rows.append(
                {
                    "source": source,
                    "source_label": METHOD_DISPLAY[source],
                    "topic": topic,
                    "topic_label": _format_topic_label(topic),
                    "n_valid": int(valid.sum()),
                    "n_positive": int((p == 1).sum()),
                    "positive_rate": float((p == 1).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_all_methods_agree_sample_df(
    df: pd.DataFrame,
    topics: list[str],
    sample_size: int,
    random_seed: int,
    id_col: str,
    methods: tuple[str, ...] = METHOD_PREFIXES,
) -> pd.DataFrame:
    dedup = df.drop_duplicates(subset=[id_col]) if id_col in df.columns else df.copy()

    agree_positive_any = pd.Series(False, index=dedup.index)
    for topic in topics:
        cols = [f"{m}_{topic}" for m in methods]
        if any(c not in dedup.columns for c in cols):
            continue

        mapped = [dedup[c].map(_to_binary) for c in cols]
        valid = pd.Series(True, index=dedup.index)
        for arr in mapped:
            valid = valid & arr.isin([0.0, 1.0])

        agree = pd.Series(True, index=dedup.index)
        base = mapped[0]
        for arr in mapped[1:]:
            agree = agree & (arr == base)

        agree_pos = valid & agree & (base == 1.0)
        agree_positive_any = agree_positive_any | agree_pos

    eligible = dedup.loc[agree_positive_any].copy()
    if eligible.empty:
        eligible = dedup.copy()

    sample_size = min(max(int(sample_size), 0), len(eligible))
    if sample_size == 0:
        sample = eligible.iloc[:0].copy()
    else:
        rng = np.random.default_rng(random_seed)
        sample_idx = sorted(rng.choice(eligible.index.to_numpy(), size=sample_size, replace=False).tolist())
        sample = eligible.loc[sample_idx].copy()

    base_cols = [
        id_col,
        "Institution name",
        "Unit of assessment number",
        "Main Panel",
        "Title",
        "1. Summary of the impact",
        "4. Details of the impact",
        "llm_status",
        "llm_error",
        "llmmini_status",
        "llm51_status",
        "llm54_status",
    ]
    pred_cols = [f"{prefix}_{topic}" for prefix in (*methods, "hybrid") for topic in topics]
    keep = [c for c in base_cols + pred_cols if c in sample.columns]
    sample = sample[keep].copy()

    for topic in topics:
        sample[f"gold_{topic}"] = pd.NA
    sample["adjudication_notes"] = ""
    return sample


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_on_all_methods_agree_only(
    adjudicated: pd.DataFrame,
    topics: list[str],
    methods: tuple[str, ...] = METHOD_PREFIXES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for topic in topics:
        gold_col = f"gold_{topic}"
        pred_cols = {m: f"{m}_{topic}" for m in methods}
        if gold_col not in adjudicated.columns:
            continue
        if any(c not in adjudicated.columns for c in pred_cols.values()):
            continue

        y_true_raw = adjudicated[gold_col].map(_to_binary)
        pred = {m: adjudicated[c].map(_to_binary) for m, c in pred_cols.items()}

        valid_gold = y_true_raw.isin([0.0, 1.0])
        valid_preds = pd.Series(True, index=adjudicated.index)
        for m in methods:
            valid_preds = valid_preds & pred[m].isin([0.0, 1.0])

        agree_mask = pd.Series(True, index=adjudicated.index)
        first_method = methods[0]
        for m in methods[1:]:
            agree_mask = agree_mask & (pred[m] == pred[first_method])

        eval_mask = valid_gold & valid_preds & agree_mask
        if not eval_mask.any():
            continue

        y_true = y_true_raw[eval_mask].astype(int).to_numpy()
        for m in methods:
            y_pred = pred[m][eval_mask].fillna(0.0).astype(int).to_numpy()
            metrics = _binary_metrics(y_true, y_pred)
            rows.append(
                {
                    "method": m,
                    "method_label": METHOD_DISPLAY.get(m, m),
                    "topic": topic,
                    "n_labeled_agree": int(eval_mask.sum()),
                    **metrics,
                }
            )

    by_topic = pd.DataFrame(rows)
    if by_topic.empty:
        return by_topic, pd.DataFrame(
            columns=[
                "method",
                "method_label",
                "n_topic_rows",
                "n_labeled_micro",
                "precision_micro",
                "recall_micro",
                "f1_micro",
                "accuracy_micro",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "accuracy_macro",
            ]
        )

    summary_rows: list[dict[str, object]] = []
    for method, grp in by_topic.groupby("method", sort=True):
        tp = int(grp["tp"].sum())
        fp = int(grp["fp"].sum())
        fn = int(grp["fn"].sum())
        tn = int(grp["tn"].sum())
        precision_micro = tp / (tp + fp) if (tp + fp) else 0.0
        recall_micro = tp / (tp + fn) if (tp + fn) else 0.0
        f1_micro = (
            2 * precision_micro * recall_micro / (precision_micro + recall_micro)
            if (precision_micro + recall_micro)
            else 0.0
        )
        accuracy_micro = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_DISPLAY.get(method, method),
                "n_topic_rows": int(len(grp)),
                "n_labeled_micro": int(grp["n_labeled_agree"].sum()),
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "accuracy_micro": accuracy_micro,
                "precision_macro": float(grp["precision"].mean()),
                "recall_macro": float(grp["recall"].mean()),
                "f1_macro": float(grp["f1"].mean()),
                "accuracy_macro": float(grp["accuracy"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    by_topic = by_topic.sort_values(["method", "topic"]).reset_index(drop=True)
    return by_topic, summary


def _prep_pub_style():
    apply_mpl_defaults()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.unicode_minus"] = False


def plot_model_comparison_figure(
    by_topic: pd.DataFrame,
    summary: pd.DataFrame,
    df: pd.DataFrame,
    topics: list[str],
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    if by_topic.empty:
        raise ValueError("Pairwise agreement table is empty; cannot draw comparison figure.")

    _prep_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # a) ECDF of thematic breadth (moved from distribution figure)
    breadth_rows = []
    for source in METHOD_PREFIXES:
        cols = [f"{source}_{t}" for t in topics if f"{source}_{t}" in df.columns]
        if not cols:
            continue
        mat = df[cols].apply(lambda s: s.map(_to_binary))
        valid_rows = mat.notna().all(axis=1)
        if not valid_rows.any():
            continue
        counts = mat.loc[valid_rows].astype(int).sum(axis=1)
        breadth_rows.append(pd.DataFrame({"source": METHOD_DISPLAY[source], "breadth": counts}))
    if not breadth_rows:
        raise ValueError("No valid source/topic columns for ECDF panel.")
    breadth = pd.concat(breadth_rows, ignore_index=True)
    breadth_color_map = {
        "Regex": "#4D4D4D",
        "GPT-5-mini": PANEL_COLORS.get("B", "#0072B2"),
        "GPT-5.1": PANEL_COLORS.get("C", "#E76F00"),
        "GPT-5.4": PANEL_COLORS.get("D", "#B2182B"),
    }
    for src, grp in breadth.groupby("source", sort=False):
        sns.ecdfplot(
            data=grp,
            x="breadth",
            linewidth=2.0,
            color=breadth_color_map.get(src, "#333333"),
            label=src,
            ax=ax_a,
        )
    ax_a.set_title("a.", loc="left", fontweight="bold", fontsize=17)
    ax_a.set_xlabel("Thematic Breadth per ICS (Number of Positive Topics)", fontsize=14)
    ax_a.set_ylabel("ECDF", fontsize=14)
    ax_a.tick_params(axis="both", labelsize=12)
    ax_a.legend(frameon=True, edgecolor="k", fontsize=10, loc="lower right")
    ax_a.grid(False)

    # b) Topic-level pairwise agreement (point-range style)
    bdf = by_topic.copy()
    bdf["agreement_pct"] = bdf["agreement_rate"] * 100.0
    topic_order = (
        bdf.groupby("topic_label", as_index=False)["agreement_pct"]
        .mean()
        .sort_values("agreement_pct", ascending=True)["topic_label"]
        .tolist()
    )
    y_positions = {label: i for i, label in enumerate(topic_order)}
    for label in topic_order:
        row = bdf[bdf["topic_label"] == label]
        vals = row["agreement_pct"].tolist()
        y = y_positions[label]
        if len(vals) >= 2:
            ax_b.plot([min(vals), max(vals)], [y, y], color="0.80", linewidth=1.1, zorder=1)
        for pair_id, pair_row in row.set_index("pair_id").iterrows():
            ax_b.scatter(
                pair_row["agreement_pct"],
                y,
                s=74,
                color=PAIR_COLORS.get(pair_id, "#333333"),
                edgecolor="k",
                linewidth=0.35,
                zorder=3,
                label=PAIR_SPECS[pair_id][0],
            )
    handles, labels_l = ax_b.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels_l):
        if l not in uniq:
            uniq[l] = h
    ax_b.legend(
        uniq.values(),
        uniq.keys(),
        frameon=True,
        edgecolor="k",
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.2,
    )
    ax_b.set_yticks(range(len(topic_order)))
    ax_b.set_yticklabels(topic_order)
    ax_b.set_xlabel("Topic-Level Agreement (%)", fontsize=14)
    ax_b.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax_b.tick_params(axis="both", labelsize=11)
    ax_b.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    x_lo = max(0.0, float(bdf["agreement_pct"].min()) - 5.0)
    x_hi = min(100.0, float(bdf["agreement_pct"].max()) + 2.0)
    ax_b.set_xlim(x_lo, x_hi)
    ax_b.grid(False)

    # c) Overall pairwise agreement summary (horizontal bars)
    cdf = summary.loc[summary["pair_id"].isin(PAIR_ORDER), ["pair_id", "pair", "agreement_rate_micro"]].copy()
    if cdf.empty:
        raise ValueError("Pairwise summary is empty; cannot draw panel c.")
    cdf["agreement_pct"] = cdf["agreement_rate_micro"] * 100.0
    cdf["pair_id"] = pd.Categorical(cdf["pair_id"], categories=PAIR_ORDER, ordered=True)
    cdf = cdf.sort_values(["agreement_pct", "pair_id"], ascending=[True, True]).reset_index(drop=True)
    cdf["pair_multiline"] = cdf["pair"].astype(str).str.replace(" vs ", "\nvs.\n", regex=False)

    y = np.arange(len(cdf))
    x_vals = cdf["agreement_pct"].to_numpy()
    x_min = float(np.floor(np.nanmin(x_vals) * 2.0) / 2.0)
    x_max = float(np.ceil(np.nanmax(x_vals) * 2.0) / 2.0)
    if x_max <= x_min:
        x_max = x_min + 1.0

    bar_colors = [PAIR_COLORS.get(str(pid), "#333333") for pid in cdf["pair_id"].tolist()]
    ax_c.barh(y, x_vals, height=0.62, color=bar_colors, edgecolor="k", linewidth=0.35, zorder=2)
    for i, row in cdf.iterrows():
        val = float(row["agreement_pct"])
        ax_c.text(val + 0.12, i, f"{val:.1f}%", va="center", ha="left", fontsize=9)

    ax_c.set_yticks(y)
    ax_c.set_yticklabels(cdf["pair_multiline"].tolist())
    ax_c.set_xlabel("Overall Pairwise Agreement (%)", fontsize=14)
    ax_c.set_title("c.", loc="left", fontweight="bold", fontsize=17)
    ax_c.tick_params(axis="both", labelsize=11)
    ax_c.set_xlim(x_min, x_max + 0.8)
    ax_c.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_c.grid(axis="x", color="0.9", linewidth=0.8, zorder=1)

    # d) Disagreement profile by topic (line chart)
    ddf = bdf.pivot_table(
        index="topic_label",
        columns="pair_id",
        values="disagreement_rate",
        aggfunc="mean",
        observed=False,
    )
    ddf = ddf.reindex(topic_order)
    x = np.arange(len(ddf.index))
    for pair_id in PAIR_ORDER:
        if pair_id not in ddf.columns:
            continue
        ax_d.plot(
            x,
            ddf[pair_id].to_numpy() * 100.0,
            marker="o",
            markersize=6.0,
            linewidth=1.4,
            color=PAIR_COLORS.get(pair_id, "#333333"),
            markeredgecolor="k",
            markeredgewidth=0.35,
            label=PAIR_SPECS[pair_id][0],
        )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(ddf.index.tolist(), rotation=45, ha="right")
    ax_d.set_ylabel("Disagreement Rate (%)", fontsize=14)
    ax_d.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_d.set_title("d.", loc="left", fontweight="bold", fontsize=17)
    ax_d.tick_params(axis="both", labelsize=10)
    ax_d.legend(frameon=True, edgecolor="k", fontsize=9, loc="upper right")

    # Consistent major gridlines across all panels.
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_axisbelow(True)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.35)

    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, (ax_a, ax_b, ax_c, ax_d)


def plot_distribution_figure(df: pd.DataFrame, topics: list[str]) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    _prep_pub_style()

    breadth_rows = []
    for source in METHOD_PREFIXES:
        cols = [f"{source}_{t}" for t in topics if f"{source}_{t}" in df.columns]
        if not cols:
            continue
        mat = df[cols].apply(lambda s: s.map(_to_binary))
        valid_rows = mat.notna().all(axis=1)
        if not valid_rows.any():
            continue
        counts = mat.loc[valid_rows].astype(int).sum(axis=1)
        breadth_rows.append(pd.DataFrame({"source": METHOD_DISPLAY[source], "breadth": counts}))

    if not breadth_rows:
        raise ValueError("No valid source/topic columns for distribution figure.")

    breadth = pd.concat(breadth_rows, ignore_index=True)
    color_map = {
        "Regex": "#4D4D4D",
        "GPT-5-mini": PANEL_COLORS.get("B", "#0072B2"),
        "GPT-5.1": PANEL_COLORS.get("C", "#E76F00"),
        "GPT-5.4": PANEL_COLORS.get("D", "#B2182B"),
    }

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6))

    for src, grp in breadth.groupby("source", sort=False):
        sns.kdeplot(
            data=grp,
            x="breadth",
            fill=False,
            linewidth=2.0,
            color=color_map.get(src, "#333333"),
            label=src,
            ax=ax_a,
            bw_adjust=1.1,
            warn_singular=False,
        )
    ax_a.set_title("a.", loc="left", fontweight="bold", fontsize=17)
    ax_a.set_xlabel("Thematic Breadth per ICS (Number of Positive Topics)", fontsize=14)
    ax_a.set_ylabel("Density", fontsize=14)
    ax_a.tick_params(axis="both", labelsize=11)
    ax_a.legend(frameon=True, edgecolor="k", fontsize=10)
    ax_a.grid(False)

    for src, grp in breadth.groupby("source", sort=False):
        sns.ecdfplot(
            data=grp,
            x="breadth",
            linewidth=2.0,
            color=color_map.get(src, "#333333"),
            label=src,
            ax=ax_b,
        )
    ax_b.set_title("b.", loc="left", fontweight="bold", fontsize=17)
    ax_b.set_xlabel("Thematic Breadth per ICS (Number of Positive Topics)", fontsize=14)
    ax_b.set_ylabel("ECDF", fontsize=14)
    ax_b.tick_params(axis="both", labelsize=11)
    ax_b.legend(frameon=True, edgecolor="k", fontsize=10)
    ax_b.grid(False)

    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, (ax_a, ax_b)


def save_figure_triplet(fig: plt.Figure, stem: Path):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=800)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-compare thematic indicators for Regex, GPT-5-mini, GPT-5.1, and GPT-5.4 with publication-ready outputs."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--input", type=str, default=None, help="Override path to enhanced_ref_data table.")
    parser.add_argument("--categories", type=str, default=None, help="Override path to data/openai/categories.csv.")

    parser.add_argument("--model-mini", type=str, default="gpt-5-nano")
    parser.add_argument("--prompt-mini", type=str, default="v2")
    parser.add_argument("--model-51", type=str, default="gpt-5.1")
    parser.add_argument("--prompt-51", type=str, default="v2")
    parser.add_argument("--model-54", type=str, default="gpt-5.4")
    parser.add_argument("--prompt-54", type=str, default="v2")

    parser.add_argument(
        "--health-check-output",
        type=str,
        default="outputs/tables/thematic_model_health_checks.csv",
        help="Coverage/status check table output path.",
    )
    parser.add_argument(
        "--pairwise-by-topic-output",
        type=str,
        default="outputs/tables/thematic_pairwise_agreement_by_topic.csv",
        help="Pairwise agreement metrics by topic output path.",
    )
    parser.add_argument(
        "--pairwise-summary-output",
        type=str,
        default="outputs/tables/thematic_pairwise_agreement_summary.csv",
        help="Pairwise agreement summary output path.",
    )
    parser.add_argument(
        "--topic-positive-rates-output",
        type=str,
        default="outputs/tables/thematic_topic_positive_rates.csv",
        help="Topic-level positive-rate table output path.",
    )

    parser.add_argument(
        "--comparison-figure-stem",
        type=str,
        default="outputs/figures/supplementary_figure_2",
        help="Output stem for model comparison figure (writes pdf/svg/png).",
    )
    parser.add_argument(
        "--distribution-figure-stem",
        type=str,
        default="outputs/figures/thematic_model_distributions",
        help="Deprecated (distribution figure has been folded into comparison figure panel a.).",
    )

    parser.add_argument(
        "--sample-output",
        type=str,
        default="outputs/tables/thematic_adjudication_sample.csv",
        help="Where to write adjudication sample CSV (all-methods-agree-positive focused).",
    )
    parser.add_argument("--sample-size", type=int, default=400, help="Number of rows for adjudication sample.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--id-col",
        type=str,
        default="REF impact case study identifier",
        help="Primary key column used to merge adjudicated rows with predictions.",
    )

    parser.add_argument(
        "--adjudicated",
        type=str,
        default=None,
        help="Optional adjudicated CSV containing gold_* columns.",
    )
    parser.add_argument(
        "--metrics-by-topic-output",
        type=str,
        default="outputs/tables/thematic_eval_by_topic.csv",
        help="Per-topic metrics output path (all-methods-agree-only evaluation).",
    )
    parser.add_argument(
        "--metrics-summary-output",
        type=str,
        default="outputs/tables/thematic_eval_summary.csv",
        help="Aggregate metrics output path (all-methods-agree-only evaluation).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    _config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)

    input_path = Path(args.input) if args.input else resolve_enhanced_ref_data_path(paths.data_dir)
    categories_path = Path(args.categories) if args.categories else (paths.data_dir / "openai" / "categories.csv")

    df = read_table(input_path)
    topics = discover_topics(df)
    if not topics:
        raise ValueError(f"No thematic columns found in {input_path}. Expected regex_* and/or llm_* columns.")

    # Keep existing hybrid for downstream compatibility where needed.
    df = add_hybrid_columns(df, topics)

    # Attach explicit model predictions from cache for direct 4-way comparison.
    df_cmp = attach_model_predictions_from_cache(
        df,
        topics,
        categories_path=categories_path,
        model_mini=args.model_mini,
        prompt_mini=args.prompt_mini,
        model_51=args.model_51,
        prompt_51=args.prompt_51,
        model_54=args.model_54,
        prompt_54=args.prompt_54,
    )

    health = validate_method_coverage(df_cmp, topics)
    health_output = Path(args.health_check_output)
    atomic_write_csv(health, health_output)

    pair_by_topic, pair_summary = build_pairwise_agreement_tables(df_cmp, topics)
    topic_positive = build_topic_positive_rates(df_cmp, topics)

    pair_by_topic_output = Path(args.pairwise_by_topic_output)
    pair_summary_output = Path(args.pairwise_summary_output)
    topic_positive_output = Path(args.topic_positive_rates_output)
    atomic_write_csv(pair_by_topic, pair_by_topic_output)
    atomic_write_csv(pair_summary, pair_summary_output)
    atomic_write_csv(topic_positive, topic_positive_output)

    fig_cmp, _ = plot_model_comparison_figure(pair_by_topic, pair_summary, df_cmp, topics)
    save_figure_triplet(fig_cmp, Path(args.comparison_figure_stem))
    plt.close(fig_cmp)

    print(f"Wrote model health checks:     {health_output}")
    print(f"Wrote pairwise-by-topic table: {pair_by_topic_output}")
    print(f"Wrote pairwise summary:        {pair_summary_output}")
    print(f"Wrote topic positive rates:    {topic_positive_output}")
    print(f"Wrote comparison figure stem:  {args.comparison_figure_stem} (pdf/svg/png)")

    if args.sample_size > 0:
        sample = build_all_methods_agree_sample_df(
            df=df_cmp,
            topics=topics,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            id_col=args.id_col,
        )
        sample_output = Path(args.sample_output)
        atomic_write_csv(sample, sample_output)
        print(f"Wrote all-methods-agree sample ({len(sample)} rows): {sample_output}")

    if not args.adjudicated:
        print("No --adjudicated file provided; four-method model-comparison analysis complete.")
        return 0

    adjudicated = read_table(Path(args.adjudicated))

    required_pred_cols = [f"{prefix}_{topic}" for prefix in METHOD_PREFIXES for topic in topics]
    missing_pred_cols = [c for c in required_pred_cols if c not in adjudicated.columns]
    if missing_pred_cols:
        if args.id_col not in adjudicated.columns or args.id_col not in df_cmp.columns:
            raise ValueError(
                "Adjudicated file is missing prediction columns and cannot be merged back because id column is absent."
            )
        merge_cols = [args.id_col] + required_pred_cols
        merge_df = df_cmp[merge_cols].drop_duplicates(subset=[args.id_col])
        adjudicated = adjudicated.merge(merge_df, how="left", on=args.id_col, validate="many_to_one")

    by_topic_eval, summary_eval = evaluate_on_all_methods_agree_only(adjudicated=adjudicated, topics=topics)
    if by_topic_eval.empty:
        raise ValueError(
            "No evaluable rows found on all-methods-agree subset. "
            "Ensure adjudicated file has gold_* columns with 0/1 or true/false labels."
        )

    metrics_by_topic_output = Path(args.metrics_by_topic_output)
    metrics_summary_output = Path(args.metrics_summary_output)
    atomic_write_csv(by_topic_eval, metrics_by_topic_output)
    atomic_write_csv(summary_eval, metrics_summary_output)

    print(f"Wrote per-topic metrics (all-methods-agree-only): {metrics_by_topic_output}")
    print(f"Wrote summary metrics (all-methods-agree-only):  {metrics_summary_output}")
    print("\nSummary:")
    print(summary_eval.to_string(index=False))
    return 0


# Backward-compatible aliases used in earlier notebook/script drafts.
def build_all_three_agree_sample_df(
    df: pd.DataFrame,
    topics: list[str],
    sample_size: int,
    random_seed: int,
    id_col: str,
) -> pd.DataFrame:
    return build_all_methods_agree_sample_df(
        df=df,
        topics=topics,
        sample_size=sample_size,
        random_seed=random_seed,
        id_col=id_col,
        methods=("regex", "llm51", "llm54"),
    )


def evaluate_on_all_three_agree_only(
    adjudicated: pd.DataFrame,
    topics: list[str],
    methods: tuple[str, ...] = ("regex", "llm51", "llm54"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_on_all_methods_agree_only(adjudicated=adjudicated, topics=topics, methods=methods)


if __name__ == "__main__":
    raise SystemExit(main())
