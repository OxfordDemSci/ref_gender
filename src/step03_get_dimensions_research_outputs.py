from __future__ import annotations

import argparse
import ast
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dimcli
import numpy as np
import pandas as pd
from tqdm import tqdm

try:  # pragma: no cover
    from .pipeline_config import load_config_and_paths
    from .pipeline_drift import apply_outputs_drift_checks
    from .pipeline_io import atomic_write_csv, atomic_write_parquet, read_secret, read_table
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
    from .pipeline_schema import validate_outputs_concat
except ImportError:  # pragma: no cover
    from pipeline_config import load_config_and_paths
    from pipeline_drift import apply_outputs_drift_checks
    from pipeline_io import atomic_write_csv, atomic_write_parquet, read_secret, read_table
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs
    from pipeline_schema import validate_outputs_concat


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and assemble Dimensions-based REF output authorship data.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--limit", type=int, default=100, help="Dimensions query chunk size.")
    parser.add_argument("--max-retries", type=int, default=8, help="Per-chunk retry count for Dimensions API calls.")
    parser.add_argument("--skip-api", action="store_true", help="Skip API collection and only build from existing raw chunks.")
    parser.add_argument("--force", action="store_true", help="Overwrite final output files.")
    return parser.parse_args(argv)


def _normalise_doi(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _normalise_isbn(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    keep = "".join(ch for ch in str(value) if ch.isalnum())
    return keep.lower()


def _parse_authors(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [v for v in parsed if isinstance(v, dict)]
        except Exception:  # noqa: BLE001
            continue
    return []


def _map_gender(label: str) -> str:
    mapping = {
        "male": "male",
        "mostly_male": "male",
        "female": "female",
        "mostly_female": "female",
        "andy": "unknown",
        "unknown": "unknown",
    }
    return mapping.get((label or "").strip().lower(), "unknown")


def _build_gender_detector():
    import gender_guesser.detector as gender

    return gender.Detector(case_sensitive=False)


def _infer_gender_list(forenames: list[str], detector) -> list[str]:
    out = []
    for name in forenames:
        first = str(name).strip().split()[0] if str(name).strip() else ""
        if not first:
            out.append("unknown")
            continue
        out.append(_map_gender(detector.get_gender(first)))
    return out


def _login_dimcli(key: str, endpoint: str):
    dimcli.login(key=key, endpoint=endpoint)
    return dimcli.Dsl()


def _query_publications(dsl, field: str, values: list[Any], limit: int) -> pd.DataFrame:
    string_representation = json.dumps(values)
    query = f"""search publications
    where {field} in {string_representation}
    return publications[authors + authors_count + category_for_2020 + dimensions_url + doi + isbn + id + year] limit {limit}"""
    result = dsl.query(query)
    return result.as_dataframe()


def _collect_raw_dimensions(
    dsl,
    outputs_df: pd.DataFrame,
    out_root: Path,
    limit: int,
    max_retries: int,
    logger: logging.Logger,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m")

    for field in ("doi", "isbn"):
        field_values = (
            outputs_df["DOI"].dropna().tolist()
            if field == "doi"
            else outputs_df["ISBN"].dropna().tolist()
        )
        field_dir = out_root / "raw" / field / timestamp
        field_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(0, len(field_values), limit), desc=f"Dimensions chunks ({field})"):
            chunk = [int(x) if isinstance(x, np.integer) else x for x in field_values[i : i + limit]]
            if not chunk:
                continue
            fpath = field_dir / f"df_{i}_to_{i + limit}.csv"
            if fpath.exists():
                continue
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    df_chunk = _query_publications(dsl, field=field, values=chunk, limit=limit)
                    atomic_write_csv(df_chunk, fpath)
                    logger.info("Saved %s (%d rows)", fpath, len(df_chunk))
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning("Chunk failed (%s, attempt %d/%d): %s", field, attempt, max_retries, exc)
            if last_exc is not None:
                logger.error("Skipping chunk %s after %d retries: %s", fpath.name, max_retries, last_exc)


def _load_dimensions_raw(raw_root: Path) -> pd.DataFrame:
    csvs = sorted(raw_root.glob("raw/*/*/df_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No Dimensions raw chunk CSVs found under {raw_root / 'raw'}.")
    frames = [pd.read_csv(p) for p in csvs]
    dim_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"], keep="first")
    if "doi" in dim_df.columns:
        dim_df["doi_norm"] = dim_df["doi"].map(_normalise_doi)
    else:
        dim_df["doi_norm"] = ""
    if "isbn" in dim_df.columns:
        dim_df["isbn_norm"] = dim_df["isbn"].map(_normalise_isbn)
    else:
        dim_df["isbn_norm"] = ""
    return dim_df


def _sync_existing_outputs(
    gold_any_parquet: Path,
    gold_pos_parquet: Path,
    gold_any_csv: Path,
    gold_pos_csv: Path,
    legacy_any_path: Path,
    legacy_pos_path: Path,
    *,
    strict_contract: bool = True,
) -> bool:
    """
    If outputs already exist in either gold or legacy paths, ensure both locations are populated.
    Returns True when existing artifacts were used (and no rebuild is required).
    """
    source_any = (
        gold_any_parquet
        if gold_any_parquet.exists()
        else gold_any_csv
        if gold_any_csv.exists()
        else legacy_any_path
        if legacy_any_path.exists()
        else None
    )
    source_pos = (
        gold_pos_parquet
        if gold_pos_parquet.exists()
        else gold_pos_csv
        if gold_pos_csv.exists()
        else legacy_pos_path
        if legacy_pos_path.exists()
        else None
    )
    if not source_any or not source_pos:
        return False

    any_df = read_table(source_any)
    pos_df = read_table(source_pos)
    if strict_contract:
        _validate_outputs_pair_contract(any_df, pos_df)
    else:
        try:
            _validate_outputs_pair_contract(any_df, pos_df)
        except ValueError as exc:
            warnings.warn(
                "Proceeding in skip-api mode with existing outputs despite strict pair-contract "
                f"mismatch: {exc}"
            )
    any_df = validate_outputs_concat(any_df)
    pos_df = validate_outputs_concat(pos_df)
    atomic_write_parquet(any_df, gold_any_parquet)
    atomic_write_parquet(pos_df, gold_pos_parquet)
    atomic_write_csv(any_df, gold_any_csv)
    atomic_write_csv(pos_df, gold_pos_csv)
    atomic_write_csv(any_df, legacy_any_path)
    atomic_write_csv(pos_df, legacy_pos_path)
    return True


def _validate_outputs_pair_contract(any_df: pd.DataFrame, positive_df: pd.DataFrame) -> None:
    """
    Enforce strict contracts between any-author and positive-author outputs tables.
    """
    if len(any_df) <= len(positive_df):
        raise ValueError(
            f"Invalid outputs pair: any table must have more rows than positive table "
            f"(got any={len(any_df)}, positive={len(positive_df)})."
        )

    if "number_people" not in any_df.columns or "number_people" not in positive_df.columns:
        raise ValueError("Both outputs tables must include 'number_people'.")

    any_people = pd.to_numeric(any_df["number_people"], errors="coerce").fillna(0)
    pos_people = pd.to_numeric(positive_df["number_people"], errors="coerce").fillna(0)
    if (any_people < 0).any() or (pos_people < 0).any():
        raise ValueError("number_people must be non-negative in both outputs tables.")
    if (pos_people <= 0).any():
        raise ValueError("Positive outputs table contains rows with number_people <= 0.")
    if int((any_people == 0).sum()) < 1:
        raise ValueError("Expected at least one zero-author row in any-author outputs table.")

    key_col = "REF2ID" if "REF2ID" in any_df.columns and "REF2ID" in positive_df.columns else None
    if key_col is None:
        raise ValueError("Both outputs tables must include REF2ID for strict contract checks.")

    any_ids = set(any_df[key_col].astype(str))
    pos_ids = set(positive_df[key_col].astype(str))
    if not pos_ids.issubset(any_ids):
        raise ValueError("Positive outputs contains REF2ID values not present in any outputs.")


def _assemble_outputs_with_authors(raw_outputs_path: Path, dim_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(raw_outputs_path, skiprows=4)
    raw = raw.copy()
    raw["doi_norm"] = raw["DOI"].map(_normalise_doi)
    raw["isbn_norm"] = raw["ISBN"].map(_normalise_isbn)

    dim_by_doi = dim_df[dim_df["doi_norm"] != ""].drop_duplicates("doi_norm", keep="first").set_index("doi_norm")
    dim_by_isbn = dim_df[dim_df["isbn_norm"] != ""].drop_duplicates("isbn_norm", keep="first").set_index("isbn_norm")

    detector = _build_gender_detector()

    author_records: list[dict[str, Any]] = []
    for idx, row in raw.iterrows():
        dim_row = None
        doi_key = row.get("doi_norm", "")
        isbn_key = row.get("isbn_norm", "")
        if doi_key and doi_key in dim_by_doi.index:
            dim_row = dim_by_doi.loc[doi_key]
        elif isbn_key and isbn_key in dim_by_isbn.index:
            dim_row = dim_by_isbn.loc[isbn_key]

        authors = _parse_authors(dim_row["authors"]) if dim_row is not None and "authors" in dim_row else []
        forenames = [str(a.get("first_name", "")).strip() for a in authors if str(a.get("first_name", "")).strip()]
        genders = _infer_gender_list(forenames, detector)
        author_records.append(
            {
                "authors": authors if authors else np.nan,
                "category_for_2020": dim_row.get("category_for_2020", np.nan) if dim_row is not None else np.nan,
                "year": dim_row.get("year", np.nan) if dim_row is not None else np.nan,
                "doi": dim_row.get("doi", row.get("DOI", np.nan)) if dim_row is not None else row.get("DOI", np.nan),
                "isbn": dim_row.get("isbn", row.get("ISBN", np.nan)) if dim_row is not None else row.get("ISBN", np.nan),
                "authors_count": len(forenames),
                "author_forenames": forenames,
                "author_genders": genders,
                "number_male": int(sum(1 for g in genders if g == "male")),
                "number_female": int(sum(1 for g in genders if g == "female")),
                "number_unknown": int(sum(1 for g in genders if g == "unknown")),
                "number_people": int(len(genders)),
            }
        )

    enrich = pd.DataFrame(author_records)
    merged = pd.concat([raw.reset_index(drop=True), enrich], axis=1)

    any_authors = merged[merged["number_people"] >= 0].copy()
    positive_authors = merged[merged["number_people"] > 0].copy()
    return any_authors, positive_authors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    bronze_dim_root = paths.bronze_dir / "dimensions_api"
    gold_any_parquet = paths.gold_dir / "outputs_concat_with_any_number_authors.parquet"
    gold_pos_parquet = paths.gold_dir / "outputs_concat_with_positive_authors.parquet"
    gold_any_csv = paths.gold_dir / "outputs_concat_with_any_number_authors.csv"
    gold_pos_csv = paths.gold_dir / "outputs_concat_with_positive_authors.csv"
    legacy_any_path = paths.legacy_dimensions_dir / "outputs_concat_with_any_number_authors.csv"
    legacy_pos_path = paths.legacy_dimensions_dir / "outputs_concat_with_positive_authors.csv"
    raw_outputs_path = paths.bronze_dir / "raw_ref_outputs_data.xlsx"
    if not raw_outputs_path.exists():
        raw_outputs_path = paths.legacy_raw_dir / "raw_ref_outputs_data.xlsx"

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = ""
    row_counts: dict[str, Any] = {}

    input_paths = {
        "raw_outputs_xlsx": raw_outputs_path,
        "dimensions_raw_root": bronze_dim_root,
    }
    output_paths = {
        "gold_any_parquet": gold_any_parquet,
        "gold_positive_parquet": gold_pos_parquet,
        "gold_any_csv": gold_any_csv,
        "gold_positive_csv": gold_pos_csv,
        "legacy_any": legacy_any_path,
        "legacy_positive": legacy_pos_path,
    }

    logger = logging.getLogger("dimensions_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    (paths.outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(paths.outputs_dir / "logs" / "dimensions_pipeline.log")
    logger.addHandler(fh)

    strict_existing_contract = not bool(args.skip_api)

    try:
        if not args.force and _sync_existing_outputs(
            gold_any_parquet,
            gold_pos_parquet,
            gold_any_csv,
            gold_pos_csv,
            legacy_any_path,
            legacy_pos_path,
            strict_contract=strict_existing_contract,
        ):
            print(f"Outputs already exist; synced canonical and legacy locations.")
            row_counts = {
                "outputs_any_rows": int(len(read_table(gold_any_parquet))),
                "outputs_positive_rows": int(len(read_table(gold_pos_parquet))),
            }
            return 0

        if not raw_outputs_path.exists():
            raise FileNotFoundError(
                f"Missing raw outputs workbook at {raw_outputs_path}. Run step01_make_enhanced_data first."
            )

        if not args.skip_api:
            dim_cfg = config.get("dimensions", {})
            try:
                api_key = read_secret(
                    env_var=str(dim_cfg.get("key_env_var", "DIMENSIONS_API_KEY")),
                    file_path=paths.project_root / str(dim_cfg.get("key_file", "keys/dimensions_apikey.txt")),
                    required=True,
                )
                dsl = _login_dimcli(api_key, endpoint=str(dim_cfg.get("endpoint", "https://app.dimensions.ai/api/dsl/v2")))
                raw_outputs = pd.read_excel(raw_outputs_path, skiprows=4)
                _collect_raw_dimensions(
                    dsl=dsl,
                    outputs_df=raw_outputs,
                    out_root=bronze_dim_root,
                    limit=int(args.limit),
                    max_retries=int(args.max_retries),
                    logger=logger,
                )
            except Exception as exc:  # noqa: BLE001
                fallback_note = f"dimensions_api_failed_fallback_to_existing: {exc}"
                if _sync_existing_outputs(
                    gold_any_parquet,
                    gold_pos_parquet,
                    gold_any_csv,
                    gold_pos_csv,
                    legacy_any_path,
                    legacy_pos_path,
                    strict_contract=strict_existing_contract,
                ):
                    notes = fallback_note
                    row_counts = {
                        "outputs_any_rows": int(len(read_table(gold_any_parquet))),
                        "outputs_positive_rows": int(len(read_table(gold_pos_parquet))),
                    }
                    print(fallback_note)
                    return 0
                raise

        dim_df = None
        try:
            dim_df = _load_dimensions_raw(bronze_dim_root)
        except FileNotFoundError:
            dim_df = None

        if dim_df is None:
            if _sync_existing_outputs(
                gold_any_parquet,
                gold_pos_parquet,
                gold_any_csv,
                gold_pos_csv,
                legacy_any_path,
                legacy_pos_path,
                strict_contract=strict_existing_contract,
            ):
                notes = "used_existing_outputs_due_to_missing_raw_chunks"
                row_counts = {
                    "outputs_any_rows": int(len(read_table(gold_any_parquet))),
                    "outputs_positive_rows": int(len(read_table(gold_pos_parquet))),
                }
                print("No Dimensions raw chunks found; used existing outputs datasets.")
                return 0
            raise FileNotFoundError(
                f"No Dimensions raw chunks found under {bronze_dim_root / 'raw'} and no existing outputs datasets available."
            )

        any_authors, positive_authors = _assemble_outputs_with_authors(raw_outputs_path, dim_df)
        _validate_outputs_pair_contract(any_authors, positive_authors)
        positive_authors = validate_outputs_concat(positive_authors)
        any_authors = validate_outputs_concat(any_authors)
        apply_outputs_drift_checks(any_authors, positive_authors, config.get("drift_checks", {}))

        atomic_write_parquet(any_authors, gold_any_parquet)
        atomic_write_parquet(positive_authors, gold_pos_parquet)
        atomic_write_csv(any_authors, gold_any_csv)
        atomic_write_csv(positive_authors, gold_pos_csv)
        atomic_write_csv(any_authors, legacy_any_path)
        atomic_write_csv(positive_authors, legacy_pos_path)

        row_counts = {
            "outputs_any_rows": int(len(any_authors)),
            "outputs_positive_rows": int(len(positive_authors)),
        }
        print(f"Saved outputs with positive authors to: {gold_pos_parquet}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step03_get_dimensions_research_outputs",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={
                "skip_api": bool(args.skip_api),
                "limit": int(args.limit),
                "max_retries": int(args.max_retries),
            },
            input_paths=input_paths,
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
