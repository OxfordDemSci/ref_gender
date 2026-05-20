from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    keys_dir: Path
    source_dir: Path
    working_dir: Path
    analysis_dir: Path
    legacy_raw_dir: Path
    legacy_edit_dir: Path
    legacy_final_dir: Path
    legacy_dimensions_dir: Path
    manual_dir: Path
    manifests_dir: Path
    manifest_csv: Path


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_relative_path(path: Path | str, project_root: Path | str | None = None) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    root = Path(project_root).resolve() if project_root else default_project_root()
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)


def normalize_path_values(value: Any, project_root: Path | str | None = None, *, key: str | None = None) -> Any:
    if isinstance(value, Path):
        return format_relative_path(value, project_root)
    if isinstance(value, dict):
        return {str(k): normalize_path_values(v, project_root, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_path_values(item, project_root, key=key) for item in value]
    if isinstance(value, tuple):
        return [normalize_path_values(item, project_root, key=key) for item in value]
    if isinstance(value, str) and key and key.endswith("_path") and Path(value).is_absolute():
        return format_relative_path(value, project_root)
    return value


def build_paths(
    project_root: Path | None = None,
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    keys_dir: str = "keys",
) -> PipelinePaths:
    root = Path(project_root).resolve() if project_root else default_project_root()
    data = (root / data_dir).resolve()
    outputs = (root / outputs_dir).resolve()
    keys = (root / keys_dir).resolve()

    source = data / "source"
    working = data / "working"
    analysis = data / "analysis"

    manifests_dir = outputs / "manifests"

    return PipelinePaths(
        project_root=root,
        data_dir=data,
        outputs_dir=outputs,
        keys_dir=keys,
        source_dir=source,
        working_dir=working,
        analysis_dir=analysis,
        legacy_raw_dir=data / "raw",
        legacy_edit_dir=data / "edit",
        legacy_final_dir=data / "final",
        legacy_dimensions_dir=data / "dimensions_outputs",
        manual_dir=data / "manual",
        manifests_dir=manifests_dir,
        manifest_csv=manifests_dir / "pipeline_manifest.csv",
    )


def ensure_core_dirs(paths: PipelinePaths) -> None:
    for path in (
        paths.data_dir,
        paths.outputs_dir,
        paths.keys_dir,
        paths.source_dir,
        paths.working_dir,
        paths.analysis_dir,
        paths.legacy_raw_dir,
        paths.legacy_edit_dir,
        paths.legacy_final_dir,
        paths.legacy_dimensions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_first_existing(candidates: Sequence[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Looked in: {joined}")


def resolve_enhanced_ref_data_path(paths: PipelinePaths, must_exist: bool = True) -> Path:
    preferred = paths.analysis_dir / "enhanced_ref_data.parquet"
    if not must_exist:
        return preferred
    return resolve_first_existing(
        (
            preferred,
            paths.analysis_dir / "enhanced_ref_data.csv",
            paths.legacy_final_dir / "enhanced_ref_data.csv",
            paths.legacy_final_dir / "enhanced_ref_data.zip",
        ),
        label="enhanced_ref_data",
    )


def resolve_outputs_concat_path(paths: PipelinePaths, must_exist: bool = True) -> Path:
    preferred = paths.analysis_dir / "outputs_concat_with_positive_authors.parquet"
    if not must_exist:
        return preferred
    return resolve_first_existing(
        (
            preferred,
            paths.analysis_dir / "outputs_concat_with_positive_authors.csv",
            paths.legacy_dimensions_dir / "outputs_concat_with_positive_authors.csv",
        ),
        label="outputs_concat_with_positive_authors",
    )


def as_abs_paths(paths: Iterable[Path]) -> list[Path]:
    return [p.resolve() for p in paths]
