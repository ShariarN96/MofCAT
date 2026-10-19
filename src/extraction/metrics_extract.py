"""LLM extraction of reaction-specific performance metrics from saved Elsevier full texts.

Per paper: load data/elsevier_fulltext/<REACTION>/text/<doi>.txt, keep tables, captions
and the most relevant paragraphs within a character budget, and ask the model for the
reaction's structured records (see reaction_schemas.py). Results are cached in SQLite
by (model, prompt version, reaction, doi, text hash), so reruns are free.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..env import ensure_loaded
from ..paths import EXTRACT_CACHE
from .reaction_schemas import (
    HUMAN_TEMPLATE,
    PROMPT_VERSION,
    build_paper_model,
    metric_names,
    system_prompt,
)
from .vocabulary import normalize_linker, normalize_metal, normalize_mof_name

DEFAULT_CACHE = EXTRACT_CACHE

MAX_PROMPT_CHARS = 90_000  # ~22k tokens; most papers fit entirely
ALWAYS_KEEP_FIRST_BLOCKS = 6  # title, keywords, abstract/highlights

_BASE_PATTERN = (
    r"overpotential|tafel|mV\b|current density|mA\s*cm|RHE|SHE|Ag/AgCl|electrolyte|"
    r"stability|LSV|chronoamper|chronopotent|faradaic|selectivity|yield|\bTOF\b|"
    r"mass activity|Cdl|ECSA|\bMOF\b|metal.organic|linker|catalyst|"
    r"BET|surface area|morpholog|nanosheet|nanorod|SEM\b|TEM\b|XRD|crystalline|amorphous|solvothermal|hydrothermal"
)
_REACTION_PATTERN = {
    "HER": r"hydrogen evolution|\bHER\b|Pt/C|H_?2\b|water splitting|exchange current",
    "OER": r"oxygen evolution|\bOER\b|RuO_?2|IrO_?2|water splitting|O_?2\b|reconstruct",
    "ORR": r"oxygen reduction|\bORR\b|half.wave|onset potential|electron transfer|H_?2O_?2|"
           r"RRDE|methanol|Zn.air|Pt/C|limiting current|kinetic current",
    "N2RR": r"nitrogen reduction|\bNRR\b|N_?2RR|NH_?3|ammonia|15N|isotope|indophenol|Nessler|"
            r"N_?2H_?4|hydrazine|yield rate",
    "NO3RR": r"nitrate|NO_?3|NO3RR|NH_?3|NH_?4|ammonia|nitrite|NO_?2|15N|isotope|indophenol|"
             r"conversion|yield rate|Zn.nitrate",
}


def _score_pattern(reaction: str) -> re.Pattern:
    return re.compile(f"{_BASE_PATTERN}|{_REACTION_PATTERN[reaction]}", re.I)


def select_text(text: str, reaction: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Fit the paper into the budget: keep tables, captions, headings, abstract; fill with relevant paragraphs."""
    if len(text) <= max_chars:
        return text
    pattern = _score_pattern(reaction)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    keep = [False] * len(blocks)
    used = 0
    for i, b in enumerate(blocks):
        must = (
            i < ALWAYS_KEEP_FIRST_BLOCKS
            or b.startswith("[TABLE")
            or b.startswith("[FIGURE")
            or b.startswith("#")
        )
        if must:
            keep[i] = True
            used += len(b) + 2
    ranked = sorted(
        (i for i in range(len(blocks)) if not keep[i]),
        key=lambda i: (-len(pattern.findall(blocks[i])), i),
    )
    for i in ranked:
        if used + len(blocks[i]) + 2 > max_chars:
            continue
        keep[i] = True
        used += len(blocks[i]) + 2
    return "\n\n".join(b for b, k in zip(blocks, keep) if k)


def _init_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS extractions (
            cache_key TEXT PRIMARY KEY, reaction TEXT, doi TEXT, model TEXT,
            prompt_ver TEXT, response TEXT, created_at TEXT)"""
    )
    conn.commit()
    return conn


def _cache_key(model: str, reaction: str, doi: str, text_hash: str) -> str:
    raw = f"{model}||{PROMPT_VERSION}||{reaction}||{doi}||{text_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def build_extractor(reaction: str, model: str):
    ensure_loaded()
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=12000, timeout=300, max_retries=10)
    return llm.with_structured_output(build_paper_model(reaction))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Controlled-vocabulary metals/linkers, keeping the linker names as the authors wrote them."""
    rec = dict(rec)
    rec["node_metals"] = _dedupe([m for m in (normalize_metal(x) for x in rec.get("node_metals") or []) if m])
    raw_linkers = [x for x in (rec.get("organic_linkers") or []) if isinstance(x, str) and x.strip()]
    rec["organic_linkers_raw"] = "|".join(raw_linkers)
    linkers = _dedupe([l for l in (normalize_linker(x) for x in raw_linkers) if l])
    source = "stated" if linkers else ""
    if not linkers and rec.get("mof_or_precursor_name"):
        _, _, default_linker = normalize_mof_name(rec["mof_or_precursor_name"])
        if default_linker:
            linkers, source = [default_linker], "inferred_from_mof_name"
    rec["organic_linkers"] = linkers
    rec["linker_source"] = source
    rec["n_metals"] = len(rec["node_metals"])
    return rec


def extract_paper(
    row: dict[str, Any],
    reaction: str,
    *,
    model: str,
    chain=None,
    cache_path: Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    """Extract one paper; returns {'paper': {...}, 'records': [...], 'status': ...}."""
    from langchain_core.messages import HumanMessage, SystemMessage

    doi = str(row["doi"])
    text = Path(row["text_path"]).read_text(encoding="utf-8", errors="ignore")
    prompt_text = select_text(text, reaction)
    text_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    key = _cache_key(model, reaction, doi, text_hash)

    conn = _init_cache(cache_path)
    cached = conn.execute("SELECT response FROM extractions WHERE cache_key = ?", (key,)).fetchone()
    if cached:
        data = json.loads(cached[0])
        status = "cached"
    else:
        chain = chain or build_extractor(reaction, model)
        messages = [
            SystemMessage(content=system_prompt(reaction)),
            HumanMessage(content=HUMAN_TEMPLATE.format(
                title=row.get("title", ""), doi=doi, year=row.get("year", ""), text=prompt_text)),
        ]
        result = chain.invoke(messages)
        data = result.model_dump()
        conn.execute(
            "INSERT OR REPLACE INTO extractions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, reaction, doi, model, PROMPT_VERSION, json.dumps(data, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        status = "ok"
    conn.close()

    records = [_normalize_record(r) for r in data.pop("records", [])]
    paper = {
        "reaction": reaction,
        "doi": doi,
        "title": row.get("title"),
        "year": row.get("year"),
        "journal": row.get("journal"),
        "status": status,
        "model": model,
        "n_chars_full": len(text),
        "n_chars_prompt": len(prompt_text),
        "n_tables": row.get("n_tables"),
        "n_records": len(records),
        **{k: (v if not isinstance(v, list) else "|".join(map(str, v))) for k, v in data.items()},
    }
    return {"paper": paper, "records": records, "status": status}


def flatten_records(paper: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, rec in enumerate(records, start=1):
        flat = {
            "reaction": paper["reaction"],
            "doi": paper["doi"],
            "title": paper["title"],
            "year": paper["year"],
            "journal": paper["journal"],
            "record_idx": i,
        }
        for k, v in rec.items():
            flat[k] = "|".join(map(str, v)) if isinstance(v, list) else v
        rows.append(flat)
    return rows


def run_reaction(
    manifest_path: Path,
    reaction: str,
    *,
    model: str,
    limit: int | None = None,
    cache_path: Path = DEFAULT_CACHE,
    verification_path: Path | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract every full-text paper in the manifest.

    If *verification_path* is given, only papers whose MOF verification status is
    'verified' are sent to the LLM (the rest are skipped and reported).
    """
    from concurrent.futures import ThreadPoolExecutor

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()
    if verification_path is not None:
        ver = pd.read_csv(verification_path)
        ok = set(ver.loc[ver["verification_status"] == "verified", "doi"])
        skipped = manifest[~manifest["doi"].isin(ok)]
        print(f"[{reaction}] MOF verification: {len(ok & set(manifest['doi']))} verified, "
              f"{len(skipped)} skipped (not verified)")
        manifest = manifest[manifest["doi"].isin(ok)].copy()
    if limit:
        manifest = manifest.head(limit)

    chain = build_extractor(reaction, model) if len(manifest) else None
    rows = [r.to_dict() for _, r in manifest.iterrows()]

    def work(item):
        i, row = item
        try:
            out = extract_paper(row, reaction, model=model, chain=chain, cache_path=cache_path)
            main = [r for r in out["records"] if r.get("role") == "main"]
            print(f"[{reaction} {i}/{len(rows)}] {row['doi']}  {out['status']}: {len(out['records'])} records "
                  f"({len(main)} main)", flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[{reaction} {i}/{len(rows)}] {row['doi']}  ERROR: {exc}", flush=True)
            return {"paper": {"reaction": reaction, "doi": row["doi"], "title": row.get("title"),
                              "status": "error", "notes": str(exc)[:500]}, "records": []}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, enumerate(rows, start=1)))

    papers = [r["paper"] for r in results]
    all_records = [flat for r in results if r["records"] for flat in flatten_records(r["paper"], r["records"])]
    return pd.DataFrame(papers), pd.DataFrame(all_records)


def coverage_summary(records: pd.DataFrame, reaction: str) -> dict[str, int]:
    """How many records populate each reaction metric (a quick quality check)."""
    if records.empty:
        return {}
    return {m: int(records[m].notna().sum()) for m in metric_names(reaction) if m in records.columns}
