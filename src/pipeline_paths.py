from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    data_dir: Path
    outputs_dir: Path
    keys_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    legacy_raw_dir: Path
    legacy_edit_dir: Path
    legacy_final_dir: Path
    legacy_dimensions_dir: Path
    manual_dir: Path
    manifests_dir: Path
    manifest_csv: Path


def build_paths(
    project_root: Path | None = None,
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    keys_dir: str = "keys",
) -> PipelinePaths:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    data = (root / data_dir).resolve()
    outputs = (root / outputs_dir).resolve()
    keys = (root / keys_dir).resolve()

    bronze = data / "bronze"
    silver = data / "silver"
    gold = data / "gold"

    manifests_dir = outputs / "manifests"

    return PipelinePaths(
        project_root=root,
        data_dir=data,
        outputs_dir=outputs,
        keys_dir=keys,
        bronze_dir=bronze,
        silver_dir=silver,
        gold_dir=gold,
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
        paths.bronze_dir,
        paths.silver_dir,
        paths.gold_dir,
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
    preferred = paths.gold_dir / "enhanced_ref_data.parquet"
    if not must_exist:
        return preferred
    return resolve_first_existing(
        (
            preferred,
            paths.gold_dir / "enhanced_ref_data.csv",
            paths.legacy_final_dir / "enhanced_ref_data.csv",
            paths.legacy_final_dir / "enhanced_ref_data.zip",
        ),
        label="enhanced_ref_data",
    )


def resolve_outputs_concat_path(paths: PipelinePaths, must_exist: bool = True) -> Path:
    preferred = paths.gold_dir / "outputs_concat_with_positive_authors.parquet"
    if not must_exist:
        return preferred
    return resolve_first_existing(
        (
            preferred,
            paths.gold_dir / "outputs_concat_with_positive_authors.csv",
            paths.legacy_dimensions_dir / "outputs_concat_with_positive_authors.csv",
        ),
        label="outputs_concat_with_positive_authors",
    )


def as_abs_paths(paths: Iterable[Path]) -> list[Path]:
    return [p.resolve() for p in paths]
