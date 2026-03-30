from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:  # pragma: no cover
    from .figure_one_helpers import resolve_enhanced_ref_data_path
    from .figure_two_llm import (
        load_llm_tables,
        load_regression,
        load_uoa_regression,
        plot_combined_figure,
        plot_supplementary_figure_three,
        plot_supplementary_figure_two,
        save_figure,
    )
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import read_table
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from figure_one_helpers import resolve_enhanced_ref_data_path
    from figure_two_llm import (
        load_llm_tables,
        load_regression,
        load_uoa_regression,
        plot_combined_figure,
        plot_supplementary_figure_three,
        plot_supplementary_figure_two,
        save_figure,
    )
    from pipeline_config import load_config_and_paths
    from pipeline_io import read_table
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 2 from regression artifacts and LLM tables.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    return parser.parse_args(argv)


_BLOCKING_LLM_STATUSES = {"not_run", "disabled", "error", "parse_error", "missing_cache_regex_fallback"}


def _normalise_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace("[\u2012\u2013\u2014\u2015]", "-", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _make_cache_key(text: str, *, model: str, prompt_version: str) -> str:
    basis = f"{prompt_version}\n{model}\n{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _validate_gpt54_thematic_indicators(paths, *, model: str, prompt_version: str) -> dict[str, int]:
    if model != "gpt-5.4":
        raise ValueError(
            f"step06_make_figure_two requires openai.model='gpt-5.4' for thematic indicators; found '{model}'."
        )

    enhanced_path = resolve_enhanced_ref_data_path(paths.data_dir)
    df_ics = read_table(enhanced_path)

    llm_cols = [c for c in df_ics.columns if c.startswith("llm_") and c not in {"llm_status", "llm_error"}]
    if not llm_cols:
        raise ValueError("No llm_* thematic columns found in enhanced_ref_data; run step01 with --with-llm first.")
    if "llm_status" not in df_ics.columns:
        raise ValueError("enhanced_ref_data is missing llm_status; cannot verify GPT-5.4 thematic indicators.")

    status_series = df_ics["llm_status"].astype(str).str.strip().str.lower()
    blocking_mask = status_series.isin(_BLOCKING_LLM_STATUSES)
    if blocking_mask.any():
        counts = df_ics.loc[blocking_mask, "llm_status"].value_counts(dropna=False).to_dict()
        raise ValueError(
            "Figure 2 requires successful GPT-5.4 thematic indicators. "
            f"Found blocking llm_status values: {counts}. "
            "Re-run step01 with --with-llm after confirming OPENAI_API_KEY."
        )

    text_cols = ["1. Summary of the impact", "4. Details of the impact"]
    missing_text_cols = [c for c in text_cols if c not in df_ics.columns]
    if missing_text_cols:
        raise ValueError(f"enhanced_ref_data missing text columns needed for cache verification: {missing_text_cols}")
    text_norm = (_normalise_text(df_ics[text_cols[0]]) + " " + _normalise_text(df_ics[text_cols[1]])).str.strip()
    nonempty_text = text_norm != ""
    expected_keys = text_norm.loc[nonempty_text].map(lambda t: _make_cache_key(t, model=model, prompt_version=prompt_version))
    expected_unique = pd.Index(expected_keys.unique())

    categories_path = paths.data_dir / "openai" / "categories.csv"
    if not categories_path.exists():
        raise ValueError(f"Missing LLM cache file: {categories_path}. Run step01 with --with-llm.")
    cache_df = pd.read_csv(categories_path)
    required_cache_cols = {"cache_key", "model", "prompt_version", "llm_status"}
    if not required_cache_cols.issubset(set(cache_df.columns)):
        raise ValueError(
            "LLM cache file does not contain required columns "
            f"{sorted(required_cache_cols)}; found {sorted(cache_df.columns)}."
        )
    cache_df["cache_key"] = cache_df["cache_key"].astype(str)
    cache_df = cache_df.drop_duplicates(subset=["cache_key"], keep="last")
    target_cache = cache_df[
        (cache_df["model"].astype(str) == str(model))
        & (cache_df["prompt_version"].astype(str) == str(prompt_version))
    ].set_index("cache_key", drop=False)

    missing_keys = expected_unique.difference(pd.Index(target_cache.index))
    if len(missing_keys) > 0:
        raise ValueError(
            "Figure 2 expects GPT-5.4 thematic cache entries for all non-empty case texts. "
            f"Missing {len(missing_keys)} cache keys for model={model}, prompt_version={prompt_version}. "
            "Re-run step01 with --with-llm."
        )

    target_status = target_cache.loc[expected_unique, "llm_status"].astype(str).str.strip().str.lower()
    blocked_cache = target_status[target_status.isin(_BLOCKING_LLM_STATUSES)]
    if not blocked_cache.empty:
        bad_counts = blocked_cache.value_counts(dropna=False).to_dict()
        raise ValueError(
            "GPT-5.4 thematic cache contains blocking statuses for Figure 2: "
            f"{bad_counts}. Re-run step01 with --with-llm."
        )

    return {
        "llm_verified_keys": int(len(expected_unique)),
        "llm_verified_rows": int(nonempty_text.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("[step06] Starting Figure 2 build...")
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    print(f"[step06] Project root: {project_root}")
    config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    fig_dir = paths.outputs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts = {}
    output_paths = {
        "figure_pdf": fig_dir / "figure_two.pdf",
        "figure_png": fig_dir / "figure_two.png",
        "figure_svg": fig_dir / "figure_two.svg",
        "supplementary_figure_2_pdf": fig_dir / "supplementary_figure_2.pdf",
        "supplementary_figure_2_png": fig_dir / "supplementary_figure_2.png",
        "supplementary_figure_2_svg": fig_dir / "supplementary_figure_2.svg",
        "supplementary_figure_3_pdf": fig_dir / "supplementary_figure_3.pdf",
        "supplementary_figure_3_png": fig_dir / "supplementary_figure_3.png",
        "supplementary_figure_3_svg": fig_dir / "supplementary_figure_3.svg",
    }
    try:
        openai_cfg = config.get("openai", {})
        print("[step06] Validating thematic indicators are GPT-5.4-derived...")
        llm_validation_counts = _validate_gpt54_thematic_indicators(
            paths,
            model=str(openai_cfg.get("model", "gpt-5.4")),
            prompt_version=str(openai_cfg.get("prompt_version", "v2")),
        )
        print(
            "[step06] LLM validation passed: "
            f"verified_keys={llm_validation_counts['llm_verified_keys']}, "
            f"verified_rows={llm_validation_counts['llm_verified_rows']}"
        )
        print("[step06] Loading regression artifacts...")
        coef_df, var_order = load_regression(paths.project_root)
        print(f"[step06] Regression rows: {len(coef_df)}")
        print("[step06] Loading LLM summary tables...")
        _llm_overall, llm_by_panel = load_llm_tables(paths.data_dir)
        print(f"[step06] LLM panel rows: {len(llm_by_panel)}")
        print("[step06] Rendering Figure 2...")
        fig, _axes = plot_combined_figure(coef_df, var_order, llm_by_panel)
        print("[step06] Writing Figure 2 outputs (pdf/svg/png)...")
        save_figure(fig, fig_dir, basename="figure_two")
        plt.close(fig)
        print("[step06] Rendering supplementary figure 2 (GLM coefficients only)...")
        supp_fig, _supp_ax = plot_supplementary_figure_two(coef_df, var_order)
        print("[step06] Writing supplementary figure 2 outputs (pdf/svg/png)...")
        save_figure(supp_fig, fig_dir, basename="supplementary_figure_2")
        plt.close(supp_fig)
        print("[step06] Fitting UoA-based regressions for supplementary figure 3...")
        uoa_coef_df, uoa_var_order, uoa_discipline_vars, uoa_label_overrides = load_uoa_regression(paths.data_dir)
        print(f"[step06] UoA regression rows: {len(uoa_coef_df)}")
        print("[step06] Rendering supplementary figure 3 (UoA controls; OLS + GLM)...")
        supp3_fig, _supp3_axes = plot_supplementary_figure_three(
            uoa_coef_df,
            uoa_var_order,
            discipline_vars=uoa_discipline_vars,
            label_overrides=uoa_label_overrides,
        )
        print("[step06] Writing supplementary figure 3 outputs (pdf/svg/png)...")
        save_figure(supp3_fig, fig_dir, basename="supplementary_figure_3")
        plt.close(supp3_fig)
        row_counts = {
            "coef_rows": int(len(coef_df)),
            "llm_panel_rows": int(len(llm_by_panel)),
            "uoa_coef_rows": int(len(uoa_coef_df)),
            "uoa_discipline_terms": int(len(uoa_discipline_vars)),
            **llm_validation_counts,
        }
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        print(f"[step06] Failed: {exc}")
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        print("[step06] Recording manifest row...")
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step06_make_figure_two",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={},
            input_paths={},
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )
    print(f"[step06] Saved Figure 2 to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
