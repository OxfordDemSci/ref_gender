# ref_staff.py
# -*- coding: utf-8 -*-
"""
REF staff extraction pipeline with OpenAI integration.

Reads the API key from:  ../keys/OPENAI_API_KEY
Requires:
    - pandas, requests, tqdm, pdfminer.six, PyMuPDF (fitz)
    - openai
    - gender-guesser
    - gender-detector   # added as offline fallback
"""

from __future__ import annotations

import argparse
import io
import os
import re
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

try:  # pragma: no cover
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import atomic_write_csv, build_retry_session, read_secret
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from pipeline_config import load_config_and_paths
    from pipeline_io import atomic_write_csv, build_retry_session, read_secret
    from pipeline_manifest import append_manifest_row
    from pipeline_paths import ensure_core_dirs

# =========================
# 0) OPENAI INITIALISATION
# =========================

def _load_openai_client(key_env_var: str, key_file: Path | str | None) -> OpenAI:
    key = read_secret(env_var=key_env_var, file_path=key_file, required=True)
    if not (key.startswith("sk-") or key.startswith("proj-") or key.startswith("sk-proj-")):
        raise ValueError("OpenAI key appears invalid (expected sk-/proj- prefix).")
    return OpenAI(api_key=key)

# =========================
# 1) PDF TEXT EXTRACTION
# =========================

def _extract_text_pdfminer(pdf_bytes: bytes) -> str:
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    from pdfminer.high_level import extract_text
    buf = io.BytesIO(pdf_bytes)
    import contextlib
    sink = io.StringIO()
    with contextlib.redirect_stderr(sink):
        try:
            t = extract_text(buf) or ""
        except Exception:
            t = ""
    return t.strip()

def _extract_text_pymupdf(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    return (text or "").strip()

def extract_text_safe(pdf_bytes: bytes) -> str:
    try:
        return _extract_text_pymupdf(pdf_bytes)
    except Exception:
        return _extract_text_pdfminer(pdf_bytes)

# =========================
# 2) STAFF BLOCK ISOLATION
# =========================

DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # hyphen, non-breaking hyphen, figure dash, en/em dash, minus
RE_DASHES      = re.compile(f"[{DASHES}]")
RE_PAGE        = re.compile(r"\n?Page\s+\d+\s*\n", flags=re.I)
RE_MULTI_SPACE = re.compile(r"[ \t]+")

def _norm_text(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.replace("\r", "")
    s = RE_DASHES.sub("-", s)
    s = RE_PAGE.sub("\n", s)
    return re.sub(r"\n{3,}", "\n\n", s)

# Header punctuation set
_PUNCT = r"[:\-–—\uFF1A]"

# -------- Strict header variants (robust but expect punctuation) --------
PAT_NAMES_STRICT   = re.compile(rf"(?mi)^\s*Name(?:\s*\(\s*s\s*\))?s?\s*{_PUNCT}")
PAT_ROLES_STRICT   = re.compile(
    rf"(?mi)^\s*(?:Role|Position|Job\s*title)(?:\s*\(\s*s\s*\))?s?"
    rf"(?:\s*\(\s*e\.?\s*g\.?\s*job\s*title\s*\))?\s*{_PUNCT}"
)
PAT_PERIODS_STRICT = re.compile(
    rf"(?mi)^\s*(?:Period|Date|Dates|Employment\s*period)(?:\s*\(\s*s\s*\))?s?"
    rf"(?:\s+(?:employed|of\s*employment|in\s*post))?"
    rf"(?:[^\n]{0,200})?\s*{_PUNCT}"
)

# -------- Flex header variants (tolerate missing punctuation / brackets) --------
PAT_NAMES_FLEX   = re.compile(r"(?mi)^\s*Name(?:\s*\[\s*s\s*\])?s?\b\s*:?")
PAT_ROLES_FLEX   = re.compile(r"(?mi)^\s*(?:Role|Position|Job\s*title)(?:\s*\[\s*s\s*\])?s?\b\s*:?")
PAT_PERIODS_FLEX = re.compile(
    r"(?mi)^\s*(?:Period|Date|Dates|Employment\s*period)(?:\s*\[\s*s\s*\])?s?"
    r"(?:\s+(?:employed|of\s*employment|in\s*post))?\b\s*:?"
)

# Next-section sentinels that terminate the staff block
NEXT_SECTION_MARKERS = [
    re.compile(r"(?mi)^\s*Period\s*when\s*the\s*claimed\s*impact\s*occurred(?:\s*[:\-–—])?"),
    re.compile(r"(?mi)^\s*\d+\.\s*Summary\s*of\s*the\s*impact"),
    re.compile(r"(?mi)^\s*\d+\.\s*Underpinning\s*research"),
    re.compile(r"(?mi)^\s*\d+\.\s*References\s*to\s*the\s*research"),
    re.compile(r"(?mi)^\s*\d+\.\s*Details\s*of\s*the\s*impact"),
    re.compile(r"(?mi)^\s*Sources\s*to\s*corroborate"),
    re.compile(r"(?mi)^\s*Further\s*information"),
]

def _first_hit(text: str, patterns: List[re.Pattern], pos: int = 0) -> Optional[re.Match]:
    hits = []
    for p in patterns:
        m = p.search(text, pos)
        if m:
            hits.append(m)
    if not hits:
        return None
    return min(hits, key=lambda m: m.start())

def _canonicalise_headers(block: str) -> str:
    """Rewrite header variants to canonical labels for downstream stability."""
    out = PAT_NAMES_STRICT.sub("Name(s):", block)
    out = PAT_NAMES_FLEX.sub("Name(s):", out)
    out = PAT_ROLES_STRICT.sub("Role(s):", out)
    out = PAT_ROLES_FLEX.sub("Role(s):", out)
    out = PAT_PERIODS_STRICT.sub("Period(s) employed by submitting HEI:", out)
    out = PAT_PERIODS_FLEX.sub("Period(s) employed by submitting HEI:", out)
    return out

def isolate_staff_names_block_with_status(
    text: Optional[str],
    service_mode: str = "auto",  # "strict" | "flex" | "auto"
) -> Tuple[Optional[str], str]:
    """
    Return (block_text | None, extraction_status) where status ∈ {"strict","flex","none"}.
    - strict: found using STRICT patterns
    - flex:   found only via FLEX patterns
    - none:   not found
    """
    if not isinstance(text, str) or not text.strip():
        return None, "none"

    txt = _norm_text(text)
    if not txt:
        return None, "none"

    # Choose pattern sets per mode
    if service_mode == "strict":
        status_order = ["strict"]
    elif service_mode == "flex":
        status_order = ["flex"]
    else:  # "auto": try strict first, then flex
        status_order = ["strict", "flex"]

    # Attempt in order
    for status_label in status_order:
        if status_label == "strict":
            start_m = _first_hit(txt, [PAT_NAMES_STRICT]) or _first_hit(txt, [PAT_ROLES_STRICT, PAT_PERIODS_STRICT])
        else:  # flex
            start_m = _first_hit(txt, [PAT_NAMES_FLEX]) or _first_hit(txt, [PAT_ROLES_FLEX, PAT_PERIODS_FLEX])
        if not start_m:
            continue

        start = start_m.start()
        next_hits = [pat.search(txt, pos=start) for pat in NEXT_SECTION_MARKERS]
        ends = [m.start() for m in next_hits if m]
        end = min(ends) if ends else len(txt)

        block = txt[start:end].strip()
        if not block:
            continue

        block = _canonicalise_headers(block)

        # Collapse wrapped lines within paragraphs but preserve paragraph breaks
        paras = [
            RE_MULTI_SPACE.sub(" ", " ".join(p.strip() for p in para.splitlines())).strip()
            for para in re.split(r"(?:\n\s*){2,}", block)
        ]
        out = "\n\n".join(p for p in paras if p)
        if out.strip():
            return out, status_label

    # Not found
    return None, "none"

# =========================
# 3) NAME NORMALISATION
# =========================

TITLE_PREFIXES = [r"professor", r"prof", r"dr", r"sir", r"dame", r"mr", r"mrs", r"ms", r"miss"]
TITLE_SUFFIXES = [r"phd", r"dphil", r"md", r"frs", r"frse", r"freng", r"obe", r"cbe", r"mbe"]

RE_TITLE_PREFIX = re.compile(rf"^({'|'.join(TITLE_PREFIXES)})\b\.?\s+", flags=re.I)
RE_TITLE_SUFFIX = re.compile(rf"\b,?\s+({'|'.join(TITLE_SUFFIXES)})\.?\b\.?", flags=re.I)

def strip_titles(name: str) -> str:
    if not isinstance(name, str): return name
    n = name.strip()
    changed = True
    while changed:
        changed = False
        n2 = RE_TITLE_PREFIX.sub("", n)
        n2 = RE_TITLE_SUFFIX.sub("", n2).strip(" ,")
        if n2 != n:
            n = n2
            changed = True
    return n

def normalize_name(name: str) -> str:
    if not isinstance(name, str): return name
    t = name.strip()
    if not t: return t
    tokens = [w[:1].upper() + w[1:] if len(w) > 1 else w.upper() for w in t.split()]
    return " ".join(tokens)

def extract_given_name(name_no_titles: str) -> str:
    if not isinstance(name_no_titles, str) or not name_no_titles.strip():
        return ""
    toks = re.split(r"[ \-]+", name_no_titles.strip())
    for t in toks:
        if re.fullmatch(r"[A-Za-z]\.?([A-Za-z]\.)?", t):  # initials
            continue
        if t.lower() in {"van", "von", "de", "del", "du", "da"}:
            continue
        return t
    return toks[0] if toks else ""

# =========================
# 4) LLM PARSING
# =========================

_SYSTEM_MSG = (
    "You are given REF 'Details of staff' blocks whose headers have been canonicalised to "
    "'Name(s):', 'Role(s):', and 'Period(s) employed by submitting HEI:'. "
    "The sections are PARALLEL LISTS. We are only interested in extracting author FORENAMES."
    "Return JSON {'people': [{'name': ..., 'roles': [...]}]}."
)
_STAFF_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_staff",
        "description": "Emit extracted staff objects aligned across Name(s) and Role(s).",
        "parameters": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {"type": "string"},
                            "roles": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "required": ["people"]
        }
    }
}

_SYSTEM_MSG_BATCH = (
    "You are given multiple REF 'Details of staff' blocks. "
    "Each block has a case_id and canonical headers: "
    "'Name(s):', 'Role(s):', and 'Period(s) employed by submitting HEI:'. "
    "Treat each case independently and extract author FORENAMES only. "
    "Return JSON {'cases': [{'case_id': ..., 'people': [{'name': ..., 'roles': [...]}]}]}."
)
_STAFF_BATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_staff_batch",
        "description": "Emit extracted staff objects per case_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["case_id", "people"],
                        "properties": {
                            "case_id": {"type": "string"},
                            "people": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "roles": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                            },
                        },
                    },
                }
            },
            "required": ["cases"],
        },
    },
}


def parse_staff_with_llm(
    client: OpenAI,
    block_text: str,
    model: str = "gpt-5.4",
    service_tier: str = "flex",
) -> List[Dict[str, Any]]:
    if not isinstance(block_text, str) or not block_text.strip():
        return []
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_MSG},
                  {"role": "user", "content": block_text}],
        tools=[_STAFF_TOOL],
        service_tier=service_tier,
        temperature=0,
    )
    ch = resp.choices[0]
    if getattr(ch.message, "tool_calls", None):
        try:
            data = json.loads(ch.message.tool_calls[0].function.arguments)
            return data.get("people", []) or []
        except Exception:
            return []
    try:
        data = json.loads(ch.message.content or "{}")
        return data.get("people", []) or []
    except Exception:
        return []


def parse_staff_with_llm_batch(
    client: OpenAI,
    batch_items: List[Tuple[str, str]],
    model: str = "gpt-5.4",
    service_tier: str = "flex",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse multiple staff blocks in a single LLM request.

    Returns mapping: case_id -> list of people dicts.
    """
    if not batch_items:
        return {}

    case_ids = [str(case_id).strip() for case_id, _ in batch_items]
    payload_cases = [
        {"case_id": str(case_id).strip(), "staff_block": block_text}
        for case_id, block_text in batch_items
        if isinstance(block_text, str) and block_text.strip()
    ]
    if not payload_cases:
        return {cid: [] for cid in case_ids}

    user_payload = json.dumps({"cases": payload_cases}, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_MSG_BATCH}, {"role": "user", "content": user_payload}],
        tools=[_STAFF_BATCH_TOOL],
        service_tier=service_tier,
        temperature=0,
    )
    ch = resp.choices[0]

    parsed_cases: List[Dict[str, Any]] = []
    if getattr(ch.message, "tool_calls", None):
        try:
            data = json.loads(ch.message.tool_calls[0].function.arguments)
            parsed_cases = data.get("cases", []) or []
        except Exception:
            parsed_cases = []
    else:
        try:
            data = json.loads(ch.message.content or "{}")
            parsed_cases = data.get("cases", []) or []
        except Exception:
            parsed_cases = []

    out: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in case_ids}
    for item in parsed_cases:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("case_id", "")).strip()
        if cid not in out:
            continue
        people = item.get("people", [])
        out[cid] = people if isinstance(people, list) else []
    return out

# =========================
# 5) OFFLINE GENDER (with fallback)
# =========================

import functools
import gender_guesser.detector as gender
_detector = gender.Detector(case_sensitive=False)

# Try to import gender-detector as an optional fallback (UK locale for your domain)
try:
    from gender_detector.gender_detector import GenderDetector
    _detector2 = GenderDetector('uk')
    _has_detector2 = True
except Exception as e:
    print(e)
    _detector2 = None
    _has_detector2 = False

def _map_gender_guesser(label: Optional[str]) -> str:
    mapping = {
        "male": "male",
        "mostly_male": "male",
        "female": "female",
        "mostly_female": "female",
        "andy": "unknown",
        "unknown": "unknown",
    }
    return mapping.get((label or "").strip().lower(), "unknown")

def _map_gender_detector(label: Optional[str]) -> str:
    if not isinstance(label, str):
        return "unknown"
    l = label.strip().lower()
    if l in {"male", "female"}:
        return l
    return "unknown"

@functools.lru_cache(maxsize=8192)
def infer_gender_offline(name: Optional[str]) -> str:
    """
    Deterministic offline gender inference with a strict precedence rule:

        1) gender-guesser (primary)
        2) gender-detector (secondary; only if primary returns 'unknown' and available)

    Returns one of {'male','female','unknown'}.
    """
    if not isinstance(name, str) or not name.strip():
        return "unknown"

    first = name.strip().split()[0]

    # Primary: gender-guesser
    gg_raw = _detector.get_gender(first)
    gg = _map_gender_guesser(gg_raw)
    if gg != "unknown":
        return gg

    # Secondary: gender-detector (optional)
    if _has_detector2:
        try:
            # some versions use .guess, others .get_gender
            if hasattr(_detector2, "guess"):
                gd_raw = _detector2.guess(first)
            else:
                gd_raw = _detector2.get_gender(first)  # type: ignore[attr-defined]
        except Exception:
            gd_raw = None
        gd = _map_gender_detector(gd_raw)
        if gd != "unknown":
            return gd

    return "unknown"

# =========================
# 6) PIPELINE ENTRY POINT
# =========================

def get_staff_rows(
    input_data_path: str | Path,
    out_dir: str | Path,
    session: requests.Session,
    base_url="https://results2021.ref.ac.uk/impact",
    model_staff="gpt-5.1",
    service_tier: str = "default",
    llm_enabled: bool = True,
    client: OpenAI | None = None,
    timeout_seconds: int = 60,
    sleep_between_calls=0.03,
    service_mode: str = "flex",  # "strict" | "flex" | "auto"
    llm_batch_size: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline.
    Produces:
      - ../data/ics_staff_rows/ref_text_and_staff_blocks.csv (unified file with Extracted Text + staff_block + extraction_status)
      - ../data/ics_staff_rows/ref_staff_rows.csv           (flattened people rows from LLM)
      - ../data/ics_staff_rows/ref_case_level.csv           (aggregated by case; includes staff_block + extraction_status)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_data_path = Path(input_data_path)
    if not input_data_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_data_path}")
    if input_data_path.suffix.lower() in {".xlsx", ".xls"}:
        df_ids = pd.read_excel(input_data_path)
    else:
        df_ids = pd.read_csv(input_data_path)
    if "REF impact case study identifier" not in df_ids.columns:
        raise ValueError("Input file must contain 'REF impact case study identifier'.")
    ids = (
        df_ids["REF impact case study identifier"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    # Explicit empty-input fast path so no network/API work is attempted and
    # output schemas remain stable.
    if not ids:
        df_master = pd.DataFrame(
            columns=["REF impact case study identifier", "Extracted Text", "staff_block", "extraction_status"]
        )
        atomic_write_csv(df_master, out_dir / "ref_text_and_staff_blocks.csv")

        df_staff_rows = pd.DataFrame(
            columns=["REF impact case study identifier", "given_name", "role", "offline_gender"]
        )
        atomic_write_csv(df_staff_rows, out_dir / "ref_staff_rows.csv")

        ref_case_level = pd.DataFrame(
            columns=[
                "REF impact case study identifier",
                "staff_block",
                "extraction_status",
                "given_names",
                "roles",
                "genders",
                "number_people",
                "number_male",
                "number_female",
                "number_unknown",
            ]
        )
        atomic_write_csv(ref_case_level, out_dir / "ref_case_level.csv")
        return df_staff_rows, ref_case_level

    # 1) Download & extract PDFs
    all_texts: Dict[str, Optional[str]] = {}
    for ics in tqdm(ids, desc="Downloading & extracting PDFs"):
        target = f"{base_url}/{ics}/pdf"
        try:
            r = session.get(target, timeout=timeout_seconds)
            r.raise_for_status()
            text = extract_text_safe(r.content)
            all_texts[ics] = text
        except Exception:
            all_texts[ics] = None

    # 2) Build a single master file with extracted text + staff block + extraction status
    master_rows: List[Tuple[str, Optional[str], Optional[str], str]] = []
    for ics in tqdm(ids, desc="Isolating staff blocks"):
        text = all_texts.get(ics)
        try:
            blk, status = isolate_staff_names_block_with_status(text, service_mode=service_mode)
        except Exception:
            blk, status = None, "none"
        master_rows.append((ics, text, blk, status))

    df_master = pd.DataFrame(
        master_rows,
        columns=["REF impact case study identifier", "Extracted Text", "staff_block", "extraction_status"]
    )
    out_master_path = out_dir / "ref_text_and_staff_blocks.csv"
    atomic_write_csv(df_master, out_master_path)

    # 3) LLM extraction over non-null blocks
    df_valid_blocks = df_master.dropna(subset=["staff_block"]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    valid_items: List[Tuple[str, str]] = [
        (str(r["REF impact case study identifier"]).strip(), r["staff_block"])
        for _, r in df_valid_blocks.iterrows()
    ]
    batch_size = max(1, int(llm_batch_size))

    for start in tqdm(range(0, len(valid_items), batch_size), desc="Extracting staff with LLM"):
        batch = valid_items[start : start + batch_size]
        batch_people: Dict[str, List[Dict[str, Any]]] = {ics_id: [] for ics_id, _ in batch}
        if llm_enabled and client is not None:
            if len(batch) == 1:
                ics_id, block = batch[0]
                try:
                    batch_people[ics_id] = parse_staff_with_llm(
                        client,
                        block,
                        model=model_staff,
                        service_tier=service_tier,
                    )
                except Exception as e:
                    print(e)
                    batch_people[ics_id] = [{"name": None, "roles": [], "error": str(e)}]
            else:
                try:
                    batch_people = parse_staff_with_llm_batch(
                        client,
                        batch,
                        model=model_staff,
                        service_tier=service_tier,
                    )
                except Exception as e:
                    print(e)
                    batch_people = {}
                # Robust fallback if batch parse missed any case.
                for ics_id, block in batch:
                    if ics_id in batch_people:
                        continue
                    try:
                        batch_people[ics_id] = parse_staff_with_llm(
                            client,
                            block,
                            model=model_staff,
                            service_tier=service_tier,
                        )
                    except Exception as e:
                        print(e)
                        batch_people[ics_id] = [{"name": None, "roles": [], "error": str(e)}]

        for ics_id, _block in batch:
            people = batch_people.get(ics_id, [])
            for person in people:
                raw_name = (person.get("name") or "").strip()
                name_norm = normalize_name(raw_name)
                name_no_titles = strip_titles(name_norm)
                given_name = extract_given_name(name_no_titles)
                roles = [x.strip() for x in (person.get("roles") or []) if x.strip()]
                records.append({
                    "REF impact case study identifier": ics_id,
                    "given_name": given_name or None,
                    "role": "; ".join(roles) if roles else None
                })
        time.sleep(sleep_between_calls)

    df_staff_rows = pd.DataFrame.from_records(records, columns=[
        "REF impact case study identifier", "given_name", "role"
    ])
    if not df_staff_rows.empty:
        df_staff_rows["offline_gender"] = df_staff_rows["given_name"].apply(infer_gender_offline)
    else:
        df_staff_rows["offline_gender"] = pd.Series(dtype="object")

    atomic_write_csv(df_staff_rows, out_dir / "ref_staff_rows.csv")

    # 4) Aggregate to case-study level with guaranteed columns and full coverage of IDs
    index_all = pd.Index(ids, name="REF impact case study identifier")

    if df_staff_rows.empty:
        ref_case_level = pd.DataFrame({
            "REF impact case study identifier": ids,
            "given_names": [[] for _ in ids],
            "roles": [[] for _ in ids],
            "genders": [[] for _ in ids],
            "number_people": 0,
            "number_male": 0,
            "number_female": 0,
            "number_unknown": 0,
        })
    else:
        df = df_staff_rows.copy().fillna("")
        grouped = (
            df.groupby("REF impact case study identifier")
              .agg(
                  given_names=("given_name", list),
                  roles=("role", list),
                  genders=("offline_gender", list)
              )
              .reindex(index_all)
        )
        for col in ["given_names", "roles", "genders"]:
            grouped[col] = grouped[col].apply(lambda x: x if isinstance(x, list) else [])

        counts_raw = (
            df.groupby("REF impact case study identifier")["offline_gender"]
              .value_counts()
              .unstack(fill_value=0)
              .reindex(index_all, fill_value=0)
        )
        for col in ["male", "female", "unknown"]:
            if col not in counts_raw.columns:
                counts_raw[col] = 0

        counts = counts_raw[["male", "female", "unknown"]].rename(columns={
            "male": "number_male",
            "female": "number_female",
            "unknown": "number_unknown"
        })

        ref_case_level = grouped.join(counts, how="left").reset_index()
        for c in ["number_male", "number_female", "number_unknown"]:
            ref_case_level[c] = ref_case_level[c].fillna(0).astype(int)
        ref_case_level["number_people"] = (
            ref_case_level[["number_male", "number_female", "number_unknown"]]
            .sum(axis=1)
            .astype(int)
        )

    # --- merge staff_block + extraction_status into final output ---
    df_master_subset = df_master[["REF impact case study identifier", "staff_block", "extraction_status"]].copy()
    df_master_subset["REF impact case study identifier"] = (
        df_master_subset["REF impact case study identifier"].astype(str)
    )
    ref_case_level = ref_case_level.copy()
    ref_case_level["REF impact case study identifier"] = (
        ref_case_level["REF impact case study identifier"].astype(str)
    )
    ref_case_level = (
        ref_case_level
        .merge(df_master_subset, on="REF impact case study identifier", how="left")
        [[
            "REF impact case study identifier",
            "staff_block", "extraction_status",
            "given_names", "roles", "genders",
            "number_people", "number_male", "number_female", "number_unknown"
        ]]
    )

    atomic_write_csv(ref_case_level, out_dir / "ref_case_level.csv")

    return df_staff_rows, ref_case_level

# =========================
# 7) MAIN
# =========================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract REF staff rows from case-study PDFs.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root (defaults to repo root).")
    parser.add_argument("--input", type=str, default=None, help="Input CSV/XLSX containing REF case IDs.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for extracted staff files.")
    parser.add_argument("--service-mode", type=str, default="flex", choices=["strict", "flex", "auto"])
    parser.add_argument("--llm-batch-size", type=int, default=None, help="Number of ICS blocks per LLM API call.")
    parser.add_argument("--with-llm", action="store_true", help="Force-enable LLM extraction.")
    parser.add_argument("--without-llm", action="store_true", help="Disable LLM extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    input_path = Path(args.input).resolve() if args.input else (paths.gold_dir / "enhanced_ref_data.csv")
    if not input_path.exists():
        # Fallback to the raw ICS workbook so this step can run from scratch.
        input_path = paths.bronze_dir / "raw_ref_ics_data.xlsx"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (paths.data_dir / "ics_staff_rows")

    openai_cfg = config.get("openai", {})
    llm_enabled = bool(openai_cfg.get("enabled", True))
    if args.with_llm:
        llm_enabled = True
    if args.without_llm:
        llm_enabled = False

    client = None
    llm_note = ""
    if llm_enabled:
        try:
            client = _load_openai_client(
                key_env_var=str(openai_cfg.get("key_env_var", "OPENAI_API_KEY")),
                key_file=paths.project_root / str(openai_cfg.get("key_file", "keys/OPENAI_API_KEY")),
            )
        except Exception as exc:  # noqa: BLE001
            llm_enabled = False
            llm_note = f"LLM disabled (missing/invalid key): {exc}"
            print(llm_note)

    http_cfg = config.get("http", {})
    session = build_retry_session(
        max_retries=int(http_cfg.get("max_retries", 5)),
        backoff_factor=float(http_cfg.get("backoff_factor", 1.5)),
    )
    timeout_seconds = int(http_cfg.get("timeout_seconds", 60))

    started_at = datetime.now(timezone.utc)
    status = "success"
    notes = llm_note
    row_counts: dict[str, Any] = {}
    input_paths = {"input_case_ids": input_path}
    output_paths = {
        "master": out_dir / "ref_text_and_staff_blocks.csv",
        "staff_rows": out_dir / "ref_staff_rows.csv",
        "case_level": out_dir / "ref_case_level.csv",
    }

    try:
        llm_batch_size = int(
            args.llm_batch_size
            if args.llm_batch_size is not None
            else openai_cfg.get("staff_batch_size", 8)
        )
        if llm_batch_size < 1:
            llm_batch_size = 1
        service_tier = "flex" if llm_enabled else str(openai_cfg.get("service_tier", "flex"))
        if llm_enabled and str(openai_cfg.get("service_tier", "flex")).lower() != "flex":
            print("[step02] Overriding configured service_tier to 'flex' for staff extraction.")

        rows, cases = get_staff_rows(
            input_data_path=input_path,
            out_dir=out_dir,
            session=session,
            model_staff=str(openai_cfg.get("model", "gpt-5.1")),
            service_tier=service_tier,
            llm_enabled=llm_enabled,
            client=client,
            timeout_seconds=timeout_seconds,
            service_mode=args.service_mode,
            llm_batch_size=llm_batch_size,
        )
        row_counts = {"staff_rows": int(len(rows)), "case_level_rows": int(len(cases))}
        print(f"Saved staff rows: {len(rows)}; case-level rows: {len(cases)}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        notes = str(exc)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        append_manifest_row(
            manifest_path=paths.manifest_csv,
            step="step02_make_ref_staff",
            status=status,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            parameters={
                "llm_enabled": llm_enabled,
                "service_mode": args.service_mode,
                "service_tier": service_tier if 'service_tier' in locals() else str(openai_cfg.get("service_tier", "flex")),
                "llm_batch_size": llm_batch_size if 'llm_batch_size' in locals() else None,
                "input_path": str(input_path),
            },
            input_paths=input_paths,
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
