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

import io
import os
import re
import json
import time
import logging
import requests
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI

# =========================
# 0) OPENAI INITIALISATION
# =========================

def _load_openai_client() -> OpenAI:
    key_path = os.path.join(os.path.dirname(__file__), "../keys/OPENAI_API_KEY")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"OpenAI key not found at {key_path}")
    with open(key_path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not (key.startswith("sk-") or key.startswith("proj-") or key.startswith("sk-proj-")):
        raise ValueError("The file ../keys/OPENAI_API_KEY does not contain a valid OpenAI API key.")
    return OpenAI(api_key=key)

client = _load_openai_client()

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

def parse_staff_with_llm(block_text: str, model: str = "gpt-4.1") -> List[Dict[str, Any]]:
    if not isinstance(block_text, str) or not block_text.strip():
        return []
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM_MSG},
                  {"role": "user", "content": block_text}],
        tools=[_STAFF_TOOL],
        service_tier="default",
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
    input_csv_path="../data/final/enhanced_ref_data.csv",
    out_dir="../data/ics_staff_rows",
    base_url="https://results2021.ref.ac.uk/impact",
    model_staff="gpt-5.1",
    sleep_between_calls=0.03,
    service_mode: str = "flex"  # "strict" | "flex" | "auto"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline.
    Produces:
      - ../data/ics_staff_rows/ref_text_and_staff_blocks.csv (unified file with Extracted Text + staff_block + extraction_status)
      - ../data/ics_staff_rows/ref_staff_rows.csv           (flattened people rows from LLM)
      - ../data/ics_staff_rows/ref_case_level.csv           (aggregated by case; includes staff_block + extraction_status)
    """
    os.makedirs(out_dir, exist_ok=True)

    df_ids = pd.read_csv(input_csv_path)
    ids = df_ids["REF impact case study identifier"].astype(str).tolist()

    # 1) Download & extract PDFs
    all_texts: Dict[str, Optional[str]] = {}
    for ics in tqdm(ids, desc="Downloading & extracting PDFs"):
        target = f"{base_url}/{ics}/pdf"
        try:
            r = requests.get(target, timeout=60)
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
    out_master_path = os.path.join(out_dir, "ref_text_and_staff_blocks.csv")
    df_master.to_csv(out_master_path, index=False)

    # 3) LLM extraction over non-null blocks
    df_valid_blocks = df_master.dropna(subset=["staff_block"]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []
    for _, r in tqdm(df_valid_blocks.iterrows(), total=len(df_valid_blocks), desc="Extracting staff with LLM"):
        ics_id, block = r["REF impact case study identifier"], r["staff_block"]
        try:
            people = parse_staff_with_llm(block, model=model_staff)
        except Exception as e:
            print(e)
            people = [{"name": None, "roles": [], "error": str(e)}]

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

    df_staff_rows.to_csv(os.path.join(out_dir, "ref_staff_rows.csv"), index=False)

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
    df_master_subset = df_master[["REF impact case study identifier", "staff_block", "extraction_status"]]
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

    ref_case_level.to_csv(os.path.join(out_dir, "ref_case_level.csv"), index=False)

    return df_staff_rows, ref_case_level

# =========================
# 7) MAIN
# =========================

if __name__ == "__main__":
    # service_mode: "strict" | "flex" | "auto"
    rows, cases = get_staff_rows(service_mode="flex")
    print(len(rows))
    print(len(cases))
