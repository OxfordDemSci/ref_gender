from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_retry_session(max_retries: int = 5, backoff_factor: float = 1.5) -> requests.Session:
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_file(url: str, out_path: Path, session: requests.Session, timeout_seconds: int = 60) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(url, allow_redirects=True, timeout=timeout_seconds)
    resp.raise_for_status()
    atomic_write_bytes(resp.content, out_path)
    return out_path


def atomic_write_bytes(data: bytes, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", delete=False, dir=str(out_path.parent), prefix=".tmp_") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out_path)


def atomic_write_text(data: str, out_path: Path, encoding: str = "utf-8") -> None:
    atomic_write_bytes(data.encode(encoding), out_path)


def atomic_write_csv(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", delete=False, dir=str(out_path.parent), prefix=".tmp_", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_csv(tmp_path, index=False, **kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path, **kwargs) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", delete=False, dir=str(out_path.parent), prefix=".tmp_") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False, **kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_table(path: Path | str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".zip":
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported table format for {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_secret(env_var: str, file_path: Path | str | None = None, required: bool = True) -> str | None:
    value = os.getenv(env_var, "").strip()
    if value:
        return value

    if file_path:
        fp = Path(file_path)
        if fp.exists():
            txt = fp.read_text(encoding="utf-8").strip()
            if txt:
                return txt

    if required:
        src = f"environment variable {env_var}"
        if file_path:
            src += f" or file {Path(file_path)}"
        raise FileNotFoundError(f"Missing secret: expected {src}.")
    return None
