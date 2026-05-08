from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_manifest_row(
    manifest_path: Path,
    *,
    step: str,
    status: str,
    started_at_utc: str,
    finished_at_utc: str,
    duration_seconds: float,
    parameters: dict[str, Any] | None = None,
    input_paths: dict[str, Path] | None = None,
    output_paths: dict[str, Path] | None = None,
    row_counts: dict[str, Any] | None = None,
    notes: str = "",
) -> None:
    """
    Append a lightweight CSV manifest row unless REF_SKIP_MANIFEST=1.
    """
    if os.environ.get("REF_SKIP_MANIFEST", "").strip().lower() in {"1", "true", "yes", "y"}:
        return

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "parameters",
        "input_paths",
        "output_paths",
        "row_counts",
        "notes",
    ]

    def _jsonable(value: Any) -> str:
        if isinstance(value, dict):
            value = {str(k): str(v) for k, v in value.items()}
        return json.dumps(value if value is not None else {}, sort_keys=True)

    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "status": status,
                "started_at_utc": started_at_utc,
                "finished_at_utc": finished_at_utc,
                "duration_seconds": duration_seconds,
                "parameters": _jsonable(parameters),
                "input_paths": _jsonable(input_paths),
                "output_paths": _jsonable(output_paths),
                "row_counts": _jsonable(row_counts),
                "notes": notes,
            }
        )
