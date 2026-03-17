from __future__ import annotations

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
    Deprecated compatibility shim.

    Manifest logging has been intentionally disabled to keep the script workflow
    simple. Step scripts still call this function, but it does nothing.
    """
    _ = (
        manifest_path,
        step,
        status,
        started_at_utc,
        finished_at_utc,
        duration_seconds,
        parameters,
        input_paths,
        output_paths,
        row_counts,
        notes,
    )
    return
