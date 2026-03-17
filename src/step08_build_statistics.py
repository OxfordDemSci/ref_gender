from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover
    from .pipeline_config import load_config_and_paths
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
    from .statistics_helpers import (
        build_and_save_summary_tables,
        build_descriptive_summary,
        build_inference_summary,
        load_statistics_data,
    )
except ImportError:  # pragma: no cover
    from pipeline_config import load_config_and_paths
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs
    from statistics_helpers import (
        build_and_save_summary_tables,
        build_descriptive_summary,
        build_inference_summary,
        load_statistics_data,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build descriptive/inference statistics outputs.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--report-out", type=str, default=None, help="Optional plain-text summary output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    _config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    table_out_dir = paths.outputs_dir / "tables"
    report_path = Path(args.report_out).resolve() if args.report_out else (table_out_dir / "statistics_report.txt")
    table_out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts = {}
    try:
        df_output, df_ics, df_uoa_m, df_uni_m, df_uniuoa_m = load_statistics_data(paths.data_dir)
        descriptive = build_descriptive_summary(df_ics, df_uoa_m, df_uni_m, df_output)
        inference = build_inference_summary(df_ics, df_output, df_uoa_m, df_uni_m, df_uniuoa_m)
        tables = build_and_save_summary_tables(df_ics, df_output, out_dir=table_out_dir)

        report_text = descriptive + "\n\n" + inference + "\n"
        report_path.write_text(report_text, encoding="utf-8")

        row_counts = {
            "ics_rows": int(len(df_ics)),
            "output_rows": int(len(df_output)),
            "table_panel_rows": int(len(tables["panel"])),
            "table_uoa_rows": int(len(tables["uoa"])),
        }
        print(f"Saved statistics report to: {report_path}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step08_build_statistics",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={"report_out": str(report_path)},
            input_paths={},
            output_paths={"statistics_report": report_path},
            row_counts=row_counts,
            notes=notes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

