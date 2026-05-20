from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from .pipeline_paths import normalize_path_values
except ImportError:  # pragma: no cover
    from pipeline_paths import normalize_path_values

PENDING_BATCH_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
FAILED_BATCH_STATUSES = {"failed", "expired", "cancelled"}


class OpenAIBatchPending(RuntimeError):
    """Raised when an async OpenAI batch has been submitted or is still running."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batch_to_dict(batch: Any) -> dict[str, Any]:
    if hasattr(batch, "model_dump"):
        return batch.model_dump(mode="python")
    if isinstance(batch, dict):
        return batch
    return {
        "id": getattr(batch, "id", ""),
        "status": getattr(batch, "status", ""),
        "input_file_id": getattr(batch, "input_file_id", ""),
        "output_file_id": getattr(batch, "output_file_id", ""),
        "error_file_id": getattr(batch, "error_file_id", ""),
        "endpoint": getattr(batch, "endpoint", ""),
    }


def batch_status(batch: Any) -> str:
    return str(batch_to_dict(batch).get("status", "")).strip().lower()


def _file_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(content, "content", None)
    if isinstance(body, bytes):
        return body.decode("utf-8")
    if isinstance(body, str):
        return body
    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    return str(content)


def create_or_retrieve_batch(
    client: Any,
    *,
    project_root: Path | None = None,
    manifest_path: Path,
    jsonl_path: Path,
    output_path: Path,
    error_path: Path,
    endpoint: str,
    requests: list[dict[str, Any]],
    metadata: dict[str, str],
    wait: bool,
    poll_interval_seconds: float,
) -> tuple[str, dict[str, Any]]:
    """
    Return (state, manifest), where state is one of:
      - completed: output_path is ready
      - pending: job exists but is not complete
      - submitted: new job submitted
    """
    manifest_path = Path(manifest_path)
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)
    error_path = Path(error_path)
    manifest = load_json(manifest_path)

    if output_path.exists():
        manifest["status"] = "completed"
        manifest["output_path"] = output_path
        manifest = normalize_path_values(manifest, project_root)
        atomic_write_json(manifest_path, manifest)
        return "completed", manifest

    batch = None
    if manifest.get("batch_id"):
        batch = client.batches.retrieve(str(manifest["batch_id"]))
    else:
        write_jsonl(jsonl_path, requests)
        with jsonl_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        input_file_id = getattr(uploaded, "id", None) or uploaded["id"]
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window="24h",
            metadata=metadata,
        )
        batch_dict = batch_to_dict(batch)
        manifest = {
            "batch_id": batch_dict.get("id", ""),
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "status": batch_dict.get("status", ""),
            "metadata": metadata,
            "request_count": len(requests),
            "manifest_path": manifest_path,
            "jsonl_path": jsonl_path,
            "output_path": output_path,
            "error_path": error_path,
            "submitted_at_utc": utc_now_iso(),
            "batch": batch_dict,
        }
        manifest = normalize_path_values(manifest, project_root)
        atomic_write_json(manifest_path, manifest)
        if not wait:
            return "submitted", manifest

    while True:
        batch_dict = batch_to_dict(batch)
        status = str(batch_dict.get("status", "")).strip().lower()
        manifest.update(
            {
                "status": status,
                "batch": batch_dict,
                "last_checked_at_utc": utc_now_iso(),
            }
        )
        manifest = normalize_path_values(manifest, project_root)
        atomic_write_json(manifest_path, manifest)

        if status == "completed":
            output_file_id = batch_dict.get("output_file_id")
            if not output_file_id:
                raise RuntimeError(f"Completed batch {manifest.get('batch_id')} has no output_file_id.")
            output_text = _file_content_text(client.files.content(str(output_file_id)))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            error_file_id = batch_dict.get("error_file_id")
            if error_file_id:
                error_text = _file_content_text(client.files.content(str(error_file_id)))
                error_path.write_text(error_text, encoding="utf-8")
            manifest["output_file_id"] = output_file_id
            manifest["output_path"] = output_path
            manifest["error_path"] = error_path if error_path.exists() else ""
            manifest = normalize_path_values(manifest, project_root)
            atomic_write_json(manifest_path, manifest)
            return "completed", manifest
        if status in FAILED_BATCH_STATUSES:
            raise RuntimeError(f"OpenAI batch {manifest.get('batch_id')} ended with status={status}.")
        if not wait:
            return "pending", manifest

        time.sleep(max(5.0, float(poll_interval_seconds)))
