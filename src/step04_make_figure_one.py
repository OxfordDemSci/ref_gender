from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover
    from .figure_one_data import prepare_figure_one_data, save_wrangled
    from .figure_one_plots import plot_figure_one, save_figure
    from .pipeline_config import load_config_and_paths
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from figure_one_data import prepare_figure_one_data, save_wrangled
    from figure_one_plots import plot_figure_one, save_figure
    from pipeline_config import load_config_and_paths
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1 from prepared data.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--show-unit-names", action="store_true", help="Annotate UoA points with names.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("[step04] Starting Figure 1 build...")
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    print(f"[step04] Project root: {project_root}")
    _config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    fig_dir = paths.outputs_dir / "figures"
    wrangled_dir = paths.silver_dir / "wrangled"
    legacy_wrangled_dir = paths.data_dir / "wrangled"
    fig_dir.mkdir(parents=True, exist_ok=True)
    wrangled_dir.mkdir(parents=True, exist_ok=True)
    legacy_wrangled_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts = {}
    output_paths = {
        "figure_pdf": fig_dir / "figure_one.pdf",
        "figure_png": fig_dir / "figure_one.png",
        "figure_svg": fig_dir / "figure_one.svg",
    }
    try:
        print("[step04] Preparing Figure 1 input tables...")
        df_ics, df_uoa_m, df_uni_m, df_uniuoa_m = prepare_figure_one_data(paths.data_dir)
        row_counts = {
            "df_ics_rows": int(len(df_ics)),
            "df_uoa_rows": int(len(df_uoa_m)),
            "df_uni_rows": int(len(df_uni_m)),
            "df_uniuoa_rows": int(len(df_uniuoa_m)),
        }
        print(
            "[step04] Input rows: "
            f"ics={row_counts['df_ics_rows']}, "
            f"uoa={row_counts['df_uoa_rows']}, "
            f"uni={row_counts['df_uni_rows']}, "
            f"uniuoa={row_counts['df_uniuoa_rows']}"
        )
        print("[step04] Saving wrangled Figure 1 tables...")
        save_wrangled(wrangled_dir, df_uoa_m, df_uni_m, df_uniuoa_m)
        save_wrangled(legacy_wrangled_dir, df_uoa_m, df_uni_m, df_uniuoa_m)

        print("[step04] Rendering Figure 1...")
        fig, _axes = plot_figure_one(df_ics, df_uoa_m, show_unit_names=args.show_unit_names)
        print("[step04] Writing Figure 1 outputs (pdf/svg/png)...")
        save_figure(fig, fig_dir, basename="figure_one")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        print(f"[step04] Failed: {exc}")
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        print("[step04] Recording manifest row...")
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step04_make_figure_one",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={"show_unit_names": bool(args.show_unit_names)},
            input_paths={},
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )
    print(f"[step04] Saved Figure 1 to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
