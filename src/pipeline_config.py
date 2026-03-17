from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from .pipeline_paths import PipelinePaths, build_paths
except ImportError:  # pragma: no cover
    from pipeline_paths import PipelinePaths, build_paths

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "paths": {
        "data_dir": "data",
        "outputs_dir": "outputs",
        "keys_dir": "keys",
    },
    "http": {
        "timeout_seconds": 60,
        "max_retries": 5,
        "backoff_factor": 1.5,
    },
    "openai": {
        "enabled": True,
        "model": "gpt-5.4",
        "service_tier": "flex",
        "prompt_version": "v2",
        "thematic_batch_size": 12,
        "prompt_cache_key": "thematic_indicators_v2",
        "prompt_cache_retention": "24h",
        "key_env_var": "OPENAI_API_KEY",
        "key_file": "keys/OPENAI_API_KEY",
    },
    "dimensions": {
        "enabled": True,
        "key_env_var": "DIMENSIONS_API_KEY",
        "key_file": "keys/dimensions_apikey.txt",
        "endpoint": "https://app.dimensions.ai/api/dsl/v2",
    },
    "drift_checks": {
        "enhanced_ref_data": {
            "row_count_min": 1000,
            "row_count_max": 20000,
            "max_null_rate": {
                "Institution name": 0.01,
                "Unit of assessment number": 0.01,
            },
        },
        "outputs_concat": {
            "row_count_any_min": 50000,
            "row_count_any_max": 500000,
            "row_count_positive_min": 10000,
            "row_count_positive_max": 500000,
            "zero_author_rows_min": 1,
            "max_null_rate": {
                "Institution name": 0.01,
            },
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_config(config_path: Path | None = None, project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    cfg_path = Path(config_path).resolve() if config_path else root / "pipeline.yaml"
    if not cfg_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required to read pipeline.yaml. Install dependencies from requirements.txt."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Invalid config in {cfg_path}: expected a mapping/object.")
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def load_config_and_paths(config_path: Path | None = None, project_root: Path | None = None) -> tuple[dict[str, Any], PipelinePaths]:
    config = load_pipeline_config(config_path=config_path, project_root=project_root)
    paths_cfg = config.get("paths", {})
    paths = build_paths(
        project_root=project_root,
        data_dir=paths_cfg.get("data_dir", "data"),
        outputs_dir=paths_cfg.get("outputs_dir", "outputs"),
        keys_dir=paths_cfg.get("keys_dir", "keys"),
    )
    return config, paths
