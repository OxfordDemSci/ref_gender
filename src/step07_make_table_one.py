from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover
    from .figure_two_regression import load_regression_artifacts, save_latex
    from .pipeline_config import load_config_and_paths
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from figure_two_regression import load_regression_artifacts, save_latex
    from pipeline_config import load_config_and_paths
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regression LaTeX table (Table 1).")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--results-path", type=str, default=None, help="Path to regression artifacts pickle.")
    parser.add_argument("--out", type=str, default=None, help="Output LaTeX path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    _config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    results_path = Path(args.results_path).resolve() if args.results_path else (paths.outputs_dir / "models" / "regression_results.pkl")
    table_path = Path(args.out).resolve() if args.out else (paths.outputs_dir / "tables" / "regression_results.tex")

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts = {}
    try:
        artifacts = load_regression_artifacts(results_path)
        save_latex(artifacts["latex_str"], table_path)
        row_counts = {"coef_rows": int(len(artifacts["coef_df"]))}
        print(f"Saved Table 1 LaTeX to: {table_path}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step07_make_table_one",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={"results_path": str(results_path), "table_path": str(table_path)},
            input_paths={"regression_results": results_path},
            output_paths={"table_one_latex": table_path},
            row_counts=row_counts,
            notes=notes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

