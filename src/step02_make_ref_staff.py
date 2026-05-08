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
import hashlib
import io
import os
import re
import json
import time
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

try:  # pragma: no cover
    from .openai_batch import OpenAIBatchPending, create_or_retrieve_batch, read_jsonl
    from .pipeline_config import load_config_and_paths
    from .pipeline_io import atomic_write_csv, build_retry_session, read_secret
    from .pipeline_manifest import append_manifest_row
    from .pipeline_paths import ensure_core_dirs
except ImportError:  # pragma: no cover
    from openai_batch import OpenAIBatchPending, create_or_retrieve_batch, read_jsonl
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
    candidates: list[str] = []
    try:
        candidates.append(_extract_text_pymupdf(pdf_bytes))
    except Exception:
        pass
    try:
        candidates.append(_extract_text_pdfminer(pdf_bytes))
    except Exception:
        pass
    candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    if not candidates:
        return ""

    def _candidate_score(candidate: str) -> tuple[int, int]:
        try:
            block, status = isolate_staff_names_block_with_status(candidate, service_mode="flex")
            has_staff_signal = int(bool(block) and status != "none")
        except Exception:
            has_staff_signal = 0
        return has_staff_signal, len(candidate)

    return max(candidates, key=_candidate_score)

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
_FIELD_PREFIX = r"(?:\d+\s*[A-Za-z]?\s*)?"

# -------- Strict header variants (robust but expect punctuation) --------
PAT_NAMES_STRICT   = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Name(?:\s*\(\s*s\s*\))?s?\s*{_PUNCT}")
PAT_ROLES_STRICT   = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}(?:Role|Position|Job\s*title)(?:\s*\(\s*s\s*\))?s?"
    rf"(?:\s*\(\s*e\.?\s*g\.?\s*job\s*title\s*\))?\s*{_PUNCT}"
)
PAT_PERIODS_STRICT = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}(?:Period|Date|Dates|Employment\s*period)(?:\s*\(\s*s\s*\))?s?"
    rf"(?:\s+(?:employed|of\s*employment|in\s*post))?"
    rf"(?:[^\n]{0,200})?\s*{_PUNCT}"
)

# -------- Flex header variants (tolerate missing punctuation / brackets) --------
PAT_NAMES_FLEX   = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Name(?:\s*\[\s*s\s*\])?s?\b\s*:?")
PAT_ROLES_FLEX   = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}(?:Role|Position|Job\s*title)(?:\s*\[\s*s\s*\])?s?\b\s*:?")
PAT_PERIODS_FLEX = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}(?:Period|Date|Dates|Employment\s*period)(?:\s*\[\s*s\s*\])?s?"
    r"(?:\s+(?:employed|of\s*employment|in\s*post))?\b\s*:?"
)

PAT_STAFF_SECTION = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}Details\s+of\s+staff\s+conducting\s+the\s+underpinning\s+research"
    rf"(?:\s+from\s+the\s+submitting\s+unit)?\s*{_PUNCT}?"
)
PAT_AUTHOR_STRICT = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Author(?:\s*\(\s*s\s*\))?s?\s*{_PUNCT}")
PAT_AUTHOR_FLEX = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Author(?:\s*[\(\[]\s*s\s*[\)\]])?s?\b\s*:?")

PAT_NAMES_HEADER = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Name(?:\s*[\(\[]\s*s\s*[\)\]])?s?\s*{_PUNCT}?")
PAT_AUTHOR_HEADER = re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Author(?:\s*[\(\[]\s*s\s*[\)\]])?s?\s*{_PUNCT}?")
PAT_ROLES_HEADER = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}(?:Role|Position|Job\s*title)(?:\s*[\(\[]\s*s\s*[\)\]])?s?"
    rf"(?:\s*\(\s*e\.?\s*g\.?\s*job\s*title\s*\))?\s*{_PUNCT}?"
)
PAT_PERIODS_HEADER = re.compile(
    rf"(?mi)^\s*{_FIELD_PREFIX}(?:Period|Date|Dates|Employment\s*period)(?:\s*[\(\[]\s*s\s*[\)\]])?s?"
    rf"(?:\s+(?:employed|of\s*employment|in\s*post))?"
    rf"(?:[^\n]{{0,200}})?\s*{_PUNCT}?"
)

# Next-section sentinels that terminate the staff block
NEXT_SECTION_MARKERS = [
    re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}Period\s*when\s*the\s*claimed\s*impact\s*occurred(?:\s*[:\-–—])?"),
    re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}\d+\.\s*Summary\s*of\s*the\s*impact"),
    re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}\d+\.\s*Underpinning\s*research"),
    re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}\d+\.\s*References\s*to\s*the\s*research"),
    re.compile(rf"(?mi)^\s*{_FIELD_PREFIX}\d+\.\s*Details\s*of\s*the\s*impact"),
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
    out = PAT_AUTHOR_HEADER.sub("Name(s):\n", block)
    out = PAT_NAMES_HEADER.sub("Name(s):\n", out)
    out = PAT_ROLES_HEADER.sub("Role(s):\n", out)
    out = PAT_PERIODS_HEADER.sub("Period(s) employed by submitting HEI:\n", out)
    out = re.sub(
        r"(?i)Period\(s\) employed by submitting HEI:\s*submitting HEI\s*:",
        "Period(s) employed by submitting HEI:",
        out,
    )
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
            start_m = _first_hit(txt, [PAT_STAFF_SECTION, PAT_AUTHOR_STRICT, PAT_NAMES_STRICT, PAT_ROLES_STRICT])
        else:  # flex
            start_m = _first_hit(txt, [PAT_STAFF_SECTION, PAT_AUTHOR_FLEX, PAT_NAMES_FLEX, PAT_ROLES_FLEX])
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

        lines = [RE_MULTI_SPACE.sub(" ", line).strip() for line in block.splitlines()]
        out = "\n".join(line for line in lines if line).strip()
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


CASE_STUDY_TEXT_FIELDS = [
    "1. Summary of the impact",
    "2. Underpinning research",
    "3. References to the research",
    "4. Details of the impact",
    "5. Sources to corroborate the impact",
]

_LOCAL_NAME_STOPWORDS = {
    "name",
    "names",
    "role",
    "roles",
    "period",
    "periods",
    "employed",
    "submitting",
    "hei",
    "professor",
    "reader",
    "lecturer",
    "senior",
    "research",
    "fellow",
    "chair",
}


def _normalise_person_name(name: object) -> str:
    raw = re.sub(r"\s+", " ", str(name or "")).strip(" ;,")
    if not raw:
        return ""
    return strip_titles(normalize_name(raw)).strip(" ;,")


def _valid_person_name(name: str) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    cleaned = name.strip()
    if len(cleaned) < 2 or not re.search(r"[A-Za-z]", cleaned):
        return False
    lower_tokens = {t.lower().strip(".") for t in re.split(r"\s+", cleaned)}
    if lower_tokens and lower_tokens.issubset(_LOCAL_NAME_STOPWORDS):
        return False
    return True


def _split_name_candidates(names_text: str) -> list[str]:
    text = re.sub(r"\([^)]{0,80}\)", " ", str(names_text or ""))
    text = text.replace("•", "\n")
    text = re.sub(r"\s+(?:and|&)\s+", "\n", text, flags=re.I)
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip(" ;,")
        if not line:
            continue
        line_parts = re.split(r"\s*;\s*|\s*,\s*(?=(?:Prof|Professor|Dr|Sir|Dame|Mr|Mrs|Ms|Miss)\b)", line, flags=re.I)
        parts.extend(p for p in line_parts if p.strip())
    return parts


def parse_staff_block_locally(block_text: str) -> list[dict[str, Any]]:
    """
    Deterministic fallback for canonical REF staff tables.

    This deliberately only parses explicit Name(s)/Author(s) sections; it does
    not try to infer researchers from prose, which is left to the LLM fallback.
    """
    if not isinstance(block_text, str) or not block_text.strip():
        return []
    block = _canonicalise_headers(_norm_text(block_text))
    name_match = re.search(r"(?mi)^\s*Name\(s\):\s*", block)
    if not name_match:
        return []

    end_candidates = []
    for marker in (r"(?mi)^\s*Role\(s\):", r"(?mi)^\s*Period\(s\) employed by submitting HEI:"):
        m = re.search(marker, block[name_match.end() :])
        if m:
            end_candidates.append(name_match.end() + m.start())
    end = min(end_candidates) if end_candidates else len(block)
    names_text = block[name_match.end() : end]

    people: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in _split_name_candidates(names_text):
        name = _normalise_person_name(candidate)
        if not _valid_person_name(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        people.append({"name": name, "roles": []})
    return people


def _case_text_fallback(row: pd.Series) -> str:
    chunks: list[str] = []
    title = row.get("Title", "")
    if isinstance(title, str) and title.strip():
        chunks.append(f"Title: {title.strip()}")
    for col in CASE_STUDY_TEXT_FIELDS:
        value = row.get(col, "")
        if isinstance(value, str) and value.strip():
            chunks.append(f"{col}\n{value.strip()}")
    if not chunks:
        return ""
    return (
        "Fallback source: full REF impact case-study text. Extract the named researchers/authors "
        "who conducted the underpinning research for this case study. Do not extract cited-paper "
        "authors, beneficiaries, funders, external partners, or people mentioned only as impact users.\n\n"
        + "\n\n".join(chunks)
    )


def _safe_case_id_filename(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", case_id.strip())


def _get_pdf_bytes(
    case_id: str,
    target_url: str,
    session: requests.Session,
    timeout_seconds: int,
    pdf_cache_dir: Path | None,
) -> tuple[bytes | None, str]:
    """
    Return (pdf_bytes, source) where source ∈ {'cache','download','failed'}.
    """
    cache_path = None
    if pdf_cache_dir is not None:
        cache_path = pdf_cache_dir / f"{_safe_case_id_filename(case_id)}.pdf"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                return cache_path.read_bytes(), "cache"
            except Exception:
                pass

    try:
        r = session.get(target_url, timeout=timeout_seconds)
        r.raise_for_status()
        pdf_bytes = r.content
    except Exception:
        return None, "failed"

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pdf_bytes)
        except Exception:
            # Caching failure should never fail extraction.
            pass
    return pdf_bytes, "download"

# =========================
# 4) LLM PARSING
# =========================

_SYSTEM_MSG = (
    "You are extracting named researchers/authors from REF impact case-study source text. "
    "Inputs may be a canonical 'Details of staff' table, an Author(s) block, or a full case-study "
    "text fallback. Prefer explicit Name(s)/Author(s) entries. For full text fallbacks, extract "
    "only researchers who conducted the underpinning research for the case study. Do not extract "
    "cited-paper authors, beneficiaries, funders, external partners, or people mentioned only as "
    "impact users. Return JSON {'people': [{'name': ..., 'roles': [...]}]}."
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
    "You are given multiple REF impact case-study staff/name sources. "
    "Each item has a case_id and either a canonical staff table, an Author(s) block, or full "
    "case-study fallback text. Treat each case independently. Extract only named researchers/authors "
    "who conducted the underpinning research for that case. Do not extract cited-paper authors, "
    "beneficiaries, funders, external partners, or people mentioned only as impact users. "
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


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 429} or status_code >= 500:
            return True
        if status_code in {400, 401, 403, 404}:
            return False

    msg = str(exc).lower()
    retry_terms = (
        "connection",
        "connect error",
        "disconnect",
        "connection reset",
        "reset before headers",
        "connection termination",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "rate limit",
        "server error",
        "internal error",
    )
    return any(term in msg for term in retry_terms)


def _create_staff_completion_with_retries(
    client: OpenAI,
    *,
    max_retries: int,
    retry_base_sleep: float,
    **kwargs: Any,
) -> Any:
    attempts = max(1, int(max_retries) + 1)
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts - 1 or not _is_retryable_openai_error(exc):
                raise
            base_delay = min(90.0, max(0.1, float(retry_base_sleep)) * (2 ** attempt))
            delay = base_delay + random.uniform(0.0, min(3.0, base_delay * 0.25))
            logging.getLogger(__name__).warning(
                "Transient OpenAI staff extraction error; retrying in %.1fs (%s/%s): %s",
                delay,
                attempt + 1,
                attempts - 1,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("OpenAI staff extraction retry loop exhausted")


def parse_staff_with_llm(
    client: OpenAI,
    block_text: str,
    model: str = "gpt-5.5",
    service_tier: str = "flex",
    max_retries: int = 5,
    retry_base_sleep: float = 1.0,
) -> List[Dict[str, Any]]:
    if not isinstance(block_text, str) or not block_text.strip():
        return []
    resp = _create_staff_completion_with_retries(
        client,
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_MSG},
                  {"role": "user", "content": block_text}],
        tools=[_STAFF_TOOL],
        service_tier=service_tier,
        max_retries=max_retries,
        retry_base_sleep=retry_base_sleep,
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
    model: str = "gpt-5.5",
    service_tier: str = "flex",
    max_retries: int = 5,
    retry_base_sleep: float = 1.0,
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
    resp = _create_staff_completion_with_retries(
        client,
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_MSG_BATCH}, {"role": "user", "content": user_payload}],
        tools=[_STAFF_BATCH_TOOL],
        service_tier=service_tier,
        max_retries=max_retries,
        retry_base_sleep=retry_base_sleep,
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


def _parse_staff_batch_chat_body(body: dict[str, Any], expected_case_ids: list[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {cid: [] for cid in expected_case_ids}
    try:
        choices = body.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
    except Exception:
        return out

    parsed_cases: List[Dict[str, Any]] = []
    try:
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "{}")
            data = json.loads(arguments or "{}")
            parsed_cases = data.get("cases", []) or []
        else:
            data = json.loads(message.get("content") or "{}")
            parsed_cases = data.get("cases", []) or []
    except Exception:
        parsed_cases = []

    for item in parsed_cases:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("case_id", "")).strip()
        if cid not in out:
            continue
        people = item.get("people", [])
        out[cid] = people if isinstance(people, list) else []
    return out


def _run_staff_openai_batch(
    client: OpenAI,
    *,
    llm_items: List[Tuple[str, str]],
    model_staff: str,
    llm_batch_size: int,
    batch_dir: Path,
    batch_wait: bool,
    batch_poll_interval_seconds: float,
) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    people_by_case: Dict[str, List[Dict[str, Any]]] = {case_id: [] for case_id, _ in llm_items}
    errors_by_case: Dict[str, str] = {}
    if not llm_items:
        return people_by_case, errors_by_case

    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_staff))
    basis = "\n".join(
        f"{case_id}\t{hashlib.sha256(str(block).encode('utf-8')).hexdigest()}"
        for case_id, block in llm_items
    )
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    stem = f"staff_{safe_model}_{digest}"
    manifest_path = batch_dir / f"{stem}.manifest.json"
    jsonl_path = batch_dir / f"{stem}.input.jsonl"
    output_path = batch_dir / f"{stem}.output.jsonl"
    error_path = batch_dir / f"{stem}.errors.jsonl"
    index_path = batch_dir / f"{stem}.index.json"

    requests: list[dict[str, Any]] = []
    custom_index: dict[str, list[dict[str, str]]] = {}
    for request_number, start in enumerate(range(0, len(llm_items), max(1, int(llm_batch_size))), start=1):
        batch = llm_items[start : start + max(1, int(llm_batch_size))]
        case_ids = [case_id for case_id, _ in batch]
        payload_cases = [{"case_id": case_id, "staff_block": block} for case_id, block in batch]
        user_payload = json.dumps({"cases": payload_cases}, ensure_ascii=False)
        body = {
            "model": model_staff,
            "messages": [
                {"role": "system", "content": _SYSTEM_MSG_BATCH},
                {"role": "user", "content": user_payload},
            ],
            "tools": [_STAFF_BATCH_TOOL],
        }
        custom_id = f"staff-{request_number:06d}-{hashlib.sha256('|'.join(case_ids).encode('utf-8')).hexdigest()[:12]}"
        custom_index[custom_id] = [{"case_id": case_id, "staff_block": block} for case_id, block in batch]
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

    if not index_path.exists():
        index_path.write_text(json.dumps(custom_index, indent=2, sort_keys=True), encoding="utf-8")
    else:
        custom_index = json.loads(index_path.read_text(encoding="utf-8"))

    state, manifest = create_or_retrieve_batch(
        client,
        manifest_path=manifest_path,
        jsonl_path=jsonl_path,
        output_path=output_path,
        error_path=error_path,
        endpoint="/v1/chat/completions",
        requests=requests,
        metadata={
            "project": "ref_gender",
            "task": "staff_extraction",
            "model": str(model_staff),
        },
        wait=bool(batch_wait),
        poll_interval_seconds=float(batch_poll_interval_seconds),
    )
    if state != "completed":
        raise OpenAIBatchPending(
            "OpenAI staff extraction batch is pending. "
            f"batch_id={manifest.get('batch_id')} status={manifest.get('status')} "
            f"manifest={manifest_path}. Re-run the same pipeline command later to collect it."
        )

    output_lines = read_jsonl(output_path)
    seen_custom_ids: set[str] = set()
    for line in output_lines:
        custom_id = str(line.get("custom_id", ""))
        seen_custom_ids.add(custom_id)
        batch_items = custom_index.get(custom_id, [])
        case_ids = [str(item["case_id"]) for item in batch_items]
        response = line.get("response") or {}
        status_code = int(response.get("status_code") or 0)
        request_error = line.get("error")
        if request_error or status_code >= 400:
            error_text = json.dumps(request_error or response.get("body") or response, ensure_ascii=False)
            for cid in case_ids:
                errors_by_case[cid] = error_text
            continue
        parsed = _parse_staff_batch_chat_body(response.get("body") or {}, case_ids)
        people_by_case.update(parsed)

    missing_custom_ids = set(custom_index) - seen_custom_ids
    for custom_id in sorted(missing_custom_ids):
        for item in custom_index.get(custom_id, []):
            errors_by_case[str(item["case_id"])] = "missing_batch_output_line"
    return people_by_case, errors_by_case

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
    model_staff="gpt-5.5",
    service_tier: str = "default",
    llm_enabled: bool = True,
    client: OpenAI | None = None,
    timeout_seconds: int = 60,
    sleep_between_calls=0.03,
    service_mode: str = "flex",  # "strict" | "flex" | "auto"
    llm_batch_size: int = 8,
    llm_max_retries: int = 5,
    llm_retry_base_sleep: float = 1.0,
    pdf_cache_dir: Path | None = None,
    require_people: bool = False,
    local_first: bool = True,
    openai_processing_mode: str = "sync",
    batch_wait: bool = False,
    batch_poll_interval_seconds: float = 60.0,
    batch_dir: Path | None = None,
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
    df_ids_by_id = (
        df_ids.assign(**{"REF impact case study identifier": df_ids["REF impact case study identifier"].astype(str).str.strip()})
        .drop_duplicates(subset=["REF impact case study identifier"], keep="first")
        .set_index("REF impact case study identifier", drop=False)
    )

    # Explicit empty-input fast path so no network/API work is attempted and
    # output schemas remain stable.
    if not ids:
        df_master = pd.DataFrame(
            columns=["REF impact case study identifier", "Extracted Text", "staff_block", "extraction_status"]
        )
        atomic_write_csv(df_master, out_dir / "ref_text_and_staff_blocks.csv")

        df_staff_rows = pd.DataFrame(
            columns=["REF impact case study identifier", "name", "given_name", "role", "offline_gender"]
        )
        atomic_write_csv(df_staff_rows, out_dir / "ref_staff_rows.csv")

        ref_case_level = pd.DataFrame(
            columns=[
                "REF impact case study identifier",
                "staff_block",
                "extraction_status",
                "names",
                "given_names",
                "roles",
                "genders",
                "number_people",
                "number_male",
                "number_female",
                "number_unknown",
                "staff_extraction_status",
                "staff_extraction_error",
            ]
        )
        atomic_write_csv(ref_case_level, out_dir / "ref_case_level.csv")
        return df_staff_rows, ref_case_level

    if pdf_cache_dir is not None:
        pdf_cache_dir = Path(pdf_cache_dir)
        pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    prior_text_by_id: Dict[str, str] = {}
    prior_master_path = out_dir / "ref_text_and_staff_blocks.csv"
    if prior_master_path.exists():
        try:
            prior_master = pd.read_csv(prior_master_path, usecols=["REF impact case study identifier", "Extracted Text"])
            for _, row in prior_master.iterrows():
                cid = str(row["REF impact case study identifier"]).strip()
                txt = row["Extracted Text"]
                if isinstance(txt, str) and txt.strip():
                    prior_text_by_id[cid] = txt
        except Exception:
            prior_text_by_id = {}

    # 1) Download & extract PDFs
    all_texts: Dict[str, Optional[str]] = {}
    pdf_sources: Dict[str, str] = {}
    for ics in tqdm(ids, desc="Downloading & extracting PDFs"):
        target = f"{base_url}/{ics}/pdf"
        try:
            pdf_bytes, src = _get_pdf_bytes(
                case_id=ics,
                target_url=target,
                session=session,
                timeout_seconds=timeout_seconds,
                pdf_cache_dir=pdf_cache_dir,
            )
            if pdf_bytes is None:
                fallback_text = prior_text_by_id.get(ics)
                if isinstance(fallback_text, str) and fallback_text.strip():
                    all_texts[ics] = fallback_text
                    pdf_sources[ics] = "previous_text"
                else:
                    all_texts[ics] = None
                    pdf_sources[ics] = src
                continue
            text = extract_text_safe(pdf_bytes)
            all_texts[ics] = text
            pdf_sources[ics] = src
        except Exception:
            all_texts[ics] = None
            pdf_sources[ics] = "failed"

    # 2) Build a single master file with extracted text + staff block + extraction status
    master_rows: List[Tuple[str, Optional[str], Optional[str], str]] = []
    for ics in tqdm(ids, desc="Isolating staff blocks"):
        text = all_texts.get(ics)
        try:
            blk, status = isolate_staff_names_block_with_status(text, service_mode=service_mode)
        except Exception:
            blk, status = None, "none"
        if not blk:
            try:
                fallback_text = _case_text_fallback(df_ids_by_id.loc[ics]) if ics in df_ids_by_id.index else ""
            except Exception:
                fallback_text = ""
            if fallback_text:
                blk, status = fallback_text, "case_text_fallback"
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
    llm_items: List[Tuple[str, str]] = []
    batch_size = max(1, int(llm_batch_size))
    case_status: Dict[str, str] = {ics: "not_attempted" for ics in ids}
    case_error: Dict[str, str] = {ics: "" for ics in ids}
    unresolved_ids: set[str] = set()

    def _add_people_records(ics_id: str, people: List[Dict[str, Any]]) -> None:
        for person in people:
            raw_name = (person.get("name") or "").strip()
            if not raw_name:
                continue
            name_norm = normalize_name(raw_name)
            name_no_titles = strip_titles(name_norm)
            given_name = extract_given_name(name_no_titles)
            roles = [x.strip() for x in (person.get("roles") or []) if x.strip()]
            records.append({
                "REF impact case study identifier": ics_id,
                "name": name_no_titles or None,
                "given_name": given_name or None,
                "role": "; ".join(roles) if roles else None
            })

    for _, row in df_master.iterrows():
        cid = str(row["REF impact case study identifier"]).strip()
        block = row.get("staff_block")
        if not isinstance(block, str) or not block.strip():
            case_status[cid] = "missing_staff_block"
            case_error[cid] = "No staff block or case-text fallback was available."
            unresolved_ids.add(cid)

    for ics_id, block in valid_items:
        local_people = parse_staff_block_locally(block) if local_first else []
        if local_people:
            case_status[ics_id] = "local_first"
            case_error[ics_id] = ""
            _add_people_records(ics_id, local_people)
        else:
            llm_items.append((ics_id, block))

    batch_mode_people: Dict[str, List[Dict[str, Any]]] | None = None
    batch_mode_errors: Dict[str, str] = {}
    if (
        llm_items
        and llm_enabled
        and client is not None
        and str(openai_processing_mode).strip().lower() == "batch"
    ):
        batch_mode_people, batch_mode_errors = _run_staff_openai_batch(
            client,
            llm_items=llm_items,
            model_staff=model_staff,
            llm_batch_size=batch_size,
            batch_dir=Path(batch_dir) if batch_dir is not None else out_dir / "batches",
            batch_wait=bool(batch_wait),
            batch_poll_interval_seconds=float(batch_poll_interval_seconds),
        )

    for start in tqdm(range(0, len(llm_items), batch_size), desc="Extracting staff with LLM"):
        batch = llm_items[start : start + batch_size]
        batch_people: Dict[str, List[Dict[str, Any]]] = {ics_id: [] for ics_id, _ in batch}
        batch_errors: Dict[str, str] = {}
        if batch_mode_people is not None:
            batch_people = {ics_id: batch_mode_people.get(ics_id, []) for ics_id, _ in batch}
            batch_errors = {ics_id: batch_mode_errors[ics_id] for ics_id, _ in batch if ics_id in batch_mode_errors}
        elif llm_enabled and client is not None:
            if len(batch) == 1:
                ics_id, block = batch[0]
                try:
                    batch_people[ics_id] = parse_staff_with_llm(
                        client,
                        block,
                        model=model_staff,
                        service_tier=service_tier,
                        max_retries=llm_max_retries,
                        retry_base_sleep=llm_retry_base_sleep,
                    )
                except Exception as e:
                    logging.getLogger(__name__).warning("Staff LLM extraction failed for %s: %s", ics_id, e)
                    batch_people[ics_id] = []
                    batch_errors[ics_id] = str(e)
            else:
                try:
                    batch_people = parse_staff_with_llm_batch(
                        client,
                        batch,
                        model=model_staff,
                        service_tier=service_tier,
                        max_retries=llm_max_retries,
                        retry_base_sleep=llm_retry_base_sleep,
                    )
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Staff batch LLM extraction failed for batch starting %s: %s",
                        batch[0][0] if batch else "unknown",
                        e,
                    )
                    batch_people = {}
                    batch_errors.update({ics_id: str(e) for ics_id, _ in batch})
                # Robust fallback if batch parse missed any case.
                for ics_id, block in batch:
                    if batch_people.get(ics_id):
                        continue
                    try:
                        batch_people[ics_id] = parse_staff_with_llm(
                            client,
                            block,
                            model=model_staff,
                            service_tier=service_tier,
                            max_retries=llm_max_retries,
                            retry_base_sleep=llm_retry_base_sleep,
                        )
                        if batch_people.get(ics_id):
                            batch_errors.pop(ics_id, None)
                    except Exception as e:
                        logging.getLogger(__name__).warning("Staff single-case LLM retry failed for %s: %s", ics_id, e)
                        batch_people[ics_id] = []
                        batch_errors[ics_id] = str(e)

        for ics_id, _block in batch:
            people = batch_people.get(ics_id, [])
            if not people:
                local_people = parse_staff_block_locally(_block)
                if local_people:
                    people = local_people
                    if batch_errors.get(ics_id):
                        case_status[ics_id] = "local_fallback_after_llm_error"
                        case_error[ics_id] = batch_errors[ics_id]
                    elif llm_enabled and client is not None:
                        case_status[ics_id] = "local_fallback_after_empty_llm"
                    else:
                        case_status[ics_id] = "local_only"
                else:
                    unresolved_ids.add(ics_id)
                    if batch_errors.get(ics_id):
                        case_status[ics_id] = "llm_failed"
                        case_error[ics_id] = batch_errors[ics_id]
                    elif llm_enabled and client is not None:
                        case_status[ics_id] = "unresolved_empty_llm"
                    else:
                        case_status[ics_id] = "unresolved_no_llm"
            else:
                case_status[ics_id] = "llm_ok"
                case_error[ics_id] = batch_errors.get(ics_id, "")

            _add_people_records(ics_id, people)
        time.sleep(sleep_between_calls)

    df_staff_rows = pd.DataFrame.from_records(records, columns=[
        "REF impact case study identifier", "name", "given_name", "role"
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
            "names": [[] for _ in ids],
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
                  names=("name", list),
                  given_names=("given_name", list),
                  roles=("role", list),
                  genders=("offline_gender", list)
              )
              .reindex(index_all)
        )
        for col in ["names", "given_names", "roles", "genders"]:
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

    unresolved_mask = ref_case_level["REF impact case study identifier"].astype(str).isin(unresolved_ids)
    if bool(unresolved_mask.any()):
        for col in ["names", "given_names", "roles", "genders"]:
            ref_case_level.loc[unresolved_mask, col] = pd.NA
        for col in ["number_people", "number_male", "number_female", "number_unknown"]:
            ref_case_level.loc[unresolved_mask, col] = pd.NA

    case_status_df = pd.DataFrame(
        {
            "REF impact case study identifier": ids,
            "staff_extraction_status": [case_status.get(ics, "not_attempted") for ics in ids],
            "staff_extraction_error": [case_error.get(ics, "") for ics in ids],
        }
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
        .merge(case_status_df, on="REF impact case study identifier", how="left")
        [[
            "REF impact case study identifier",
            "staff_block", "extraction_status",
            "staff_extraction_status", "staff_extraction_error",
            "names", "given_names", "roles", "genders",
            "number_people", "number_male", "number_female", "number_unknown"
        ]]
    )

    atomic_write_csv(ref_case_level, out_dir / "ref_case_level.csv")

    # 5) Case-level extraction audit for diagnostics
    master_audit = df_master[["REF impact case study identifier", "Extracted Text", "staff_block", "extraction_status"]].copy()
    master_audit["has_extracted_text"] = ~(
        master_audit["Extracted Text"].isna() | (master_audit["Extracted Text"].astype(str).str.strip() == "")
    )
    master_audit["has_staff_block"] = ~(
        master_audit["staff_block"].isna() | (master_audit["staff_block"].astype(str).str.strip() == "")
    )
    master_audit["pdf_source"] = master_audit["REF impact case study identifier"].astype(str).map(
        lambda cid: pdf_sources.get(cid, "unknown")
    )

    case_audit = ref_case_level[
        [
            "REF impact case study identifier",
            "number_people",
            "staff_extraction_status",
            "staff_extraction_error",
        ]
    ].copy()
    case_audit["has_people"] = case_audit["number_people"].fillna(0).astype(float) > 0
    case_audit["is_unresolved"] = case_audit["number_people"].isna()
    audit = master_audit.merge(case_audit, on="REF impact case study identifier", how="left")
    atomic_write_csv(audit, out_dir / "ref_staff_extraction_audit.csv")

    if llm_enabled and require_people:
        missing_people = audit.loc[~audit["has_people"].fillna(False), "REF impact case study identifier"].astype(str).tolist()
        if missing_people:
            preview = ", ".join(missing_people[:20])
            raise ValueError(
                f"Staff extraction produced zero people for {len(missing_people)} case(s). "
                f"First missing case IDs: {preview}. See {out_dir / 'ref_staff_extraction_audit.csv'}."
            )

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
    parser.add_argument("--llm-max-retries", type=int, default=None, help="Retries per staff LLM request after transient API failures.")
    parser.add_argument("--pdf-cache-dir", type=str, default=None, help="Directory for cached REF ICS PDFs.")
    parser.add_argument("--no-pdf-cache", action="store_true", help="Disable on-disk PDF caching.")
    parser.add_argument("--with-llm", action="store_true", help="Force-enable LLM extraction.")
    parser.add_argument("--without-llm", action="store_true", help="Disable LLM extraction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    config, paths = load_config_and_paths(config_path=Path(args.config) if args.config else None, project_root=project_root)
    ensure_core_dirs(paths)

    input_path = Path(args.input).resolve() if args.input else (paths.analysis_dir / "enhanced_ref_data.csv")
    if not input_path.exists():
        # Fallback to the raw ICS workbook so this step can run from scratch.
        input_path = paths.source_dir / "raw_ref_ics_data.xlsx"
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
            if args.with_llm:
                raise RuntimeError(
                    "step02 was run with --with-llm, but the OpenAI key could not be loaded. "
                    "Set OPENAI_API_KEY or keys/OPENAI_API_KEY before running the rebuild."
                ) from exc
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
    exit_code = 0
    row_counts: dict[str, Any] = {}
    input_paths = {"input_case_ids": input_path}
    output_paths = {
        "master": out_dir / "ref_text_and_staff_blocks.csv",
        "staff_rows": out_dir / "ref_staff_rows.csv",
        "case_level": out_dir / "ref_case_level.csv",
        "audit": out_dir / "ref_staff_extraction_audit.csv",
    }

    try:
        llm_batch_size = int(
            args.llm_batch_size
            if args.llm_batch_size is not None
            else openai_cfg.get("staff_batch_size", 8)
        )
        if llm_batch_size < 1:
            llm_batch_size = 1
        llm_max_retries = int(
            args.llm_max_retries
            if args.llm_max_retries is not None
            else openai_cfg.get("staff_max_retries", 5)
        )
        if llm_max_retries < 0:
            llm_max_retries = 0
        llm_retry_base_sleep = float(openai_cfg.get("staff_retry_base_sleep", 1.0))
        if llm_retry_base_sleep <= 0:
            llm_retry_base_sleep = 1.0
        service_tier = "flex" if llm_enabled else str(openai_cfg.get("service_tier", "flex"))
        if llm_enabled and str(openai_cfg.get("service_tier", "flex")).lower() != "flex":
            print("[step02] Overriding configured service_tier to 'flex' for staff extraction.")

        step02_cfg = config.get("step02", {})
        openai_processing_mode = str(openai_cfg.get("processing_mode", "sync")).strip().lower()
        batch_wait = bool(openai_cfg.get("batch_wait", False))
        batch_poll_interval_seconds = float(openai_cfg.get("batch_poll_interval_seconds", 60))
        cache_enabled = bool(step02_cfg.get("pdf_cache_enabled", True))
        if args.no_pdf_cache:
            cache_enabled = False
        if args.pdf_cache_dir:
            pdf_cache_dir = Path(args.pdf_cache_dir).resolve()
        else:
            pdf_cache_rel = str(step02_cfg.get("pdf_cache_dir", "cache/ref_pdfs"))
            pdf_cache_dir = (paths.data_dir / pdf_cache_rel).resolve()
        if not cache_enabled:
            pdf_cache_dir = None

        rows, cases = get_staff_rows(
            input_data_path=input_path,
            out_dir=out_dir,
            session=session,
            model_staff=str(openai_cfg.get("model", "gpt-5.5")),
            service_tier=service_tier,
            llm_enabled=llm_enabled,
            client=client,
            timeout_seconds=timeout_seconds,
            service_mode=args.service_mode,
            llm_batch_size=llm_batch_size,
            llm_max_retries=llm_max_retries,
            llm_retry_base_sleep=llm_retry_base_sleep,
            pdf_cache_dir=pdf_cache_dir,
            require_people=bool(step02_cfg.get("require_people", True)),
            local_first=bool(step02_cfg.get("staff_local_first", True)),
            openai_processing_mode=openai_processing_mode,
            batch_wait=batch_wait,
            batch_poll_interval_seconds=batch_poll_interval_seconds,
            batch_dir=paths.data_dir / "openai" / "batches",
        )
        row_counts = {"staff_rows": int(len(rows)), "case_level_rows": int(len(cases))}
        print(f"Saved staff rows: {len(rows)}; case-level rows: {len(cases)}")
    except OpenAIBatchPending as exc:
        status = "pending"
        notes = str(exc)
        print(str(exc))
        exit_code = 75
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
                "llm_max_retries": llm_max_retries if 'llm_max_retries' in locals() else None,
                "llm_retry_base_sleep": llm_retry_base_sleep if 'llm_retry_base_sleep' in locals() else None,
                "input_path": str(input_path),
                "pdf_cache_enabled": cache_enabled if 'cache_enabled' in locals() else None,
                "pdf_cache_dir": str(pdf_cache_dir) if 'pdf_cache_dir' in locals() and pdf_cache_dir is not None else None,
                "staff_local_first": bool(step02_cfg.get("staff_local_first", True)) if 'step02_cfg' in locals() else None,
                "openai_processing_mode": openai_processing_mode if 'openai_processing_mode' in locals() else None,
                "batch_wait": batch_wait if 'batch_wait' in locals() else None,
            },
            input_paths=input_paths,
            output_paths=output_paths,
            row_counts=row_counts,
            notes=notes,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
