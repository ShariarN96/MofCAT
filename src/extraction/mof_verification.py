"""Full-text fact-check: is this paper genuinely about a MOF electrocatalyst for the sheet's reaction?

Scopus matched papers on title/abstract/keywords, so a paper can be in the corpus because it
mentions MOFs in passing, studies a COF, a theory-only model, or a different reaction.
This module reads the saved full text and returns, per paper:

  heuristics   MOF/COF term counts overall, in the abstract, in the experimental section, in the title
  LLM verdict  what the studied catalyst is, MOF role, whether a MOF is synthesised/used,
               whether electrochemical data is reported, whether the reaction matches,
               plus verbatim evidence quotes
  status       verified | needs_review | rejected_not_mof | rejected_mof_background_only |
               rejected_not_experimental | rejected_wrong_reaction

Only 'verified' papers go on to metric extraction and figures (see elsevier_extract --only-verified).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..env import ensure_loaded
from ..paths import VERIFY_CACHE

DEFAULT_CACHE = VERIFY_CACHE
PROMPT_VERSION = "mofcheck-v1"

REACTION_NAMES = {
    "HER": "hydrogen evolution reaction (HER)",
    "OER": "oxygen evolution reaction (OER)",
    "ORR": "oxygen reduction reaction (ORR)",
    "N2RR": "electrochemical nitrogen reduction to ammonia (N2RR / NRR)",
    "NO3RR": "electrochemical nitrate reduction to ammonia (NO3RR)",
}

# Case-insensitive prose terms and case-sensitive family codes (ZIF-8, UiO-66, MIL-101, ...).
_MOF_PROSE = re.compile(r"metal[\s\-–]*organic[\s\-–]*frameworks?|coordination polymers?|\bMOFs?\b", re.I)
_MOF_CODES = re.compile(r"\b(?:ZIF|UiO|MIL|HKUST|PCN|IRMOF|CPO|DUT|CAU|MUF|ZJU|JUC|NU|MOF|PBA)[-\s]?\d+[A-Za-z\-]*|Prussian blue")
_COF_TERMS = re.compile(r"covalent[\s\-–]*organic[\s\-–]*frameworks?|\bCOFs?\b", re.I)
_SYNTH_TERMS = re.compile(r"solvothermal|hydrothermal|synthes|prepar|pyroly|calcin|carboni[sz]|phosphid|sulfid|annealed", re.I)
_EXPERIMENTAL_HEADING = re.compile(r"experiment|method|synthesis|preparation|material", re.I)
_END_HEADING = re.compile(r"result|discussion|conclusion|acknowledg", re.I)


# --------------------------------------------------------------------------- #
# Heuristic evidence
# --------------------------------------------------------------------------- #
def _count_mof(text: str) -> int:
    return len(_MOF_PROSE.findall(text)) + len(_MOF_CODES.findall(text))


def heuristic_features(text: str) -> dict[str, Any]:
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    title = blocks[0] if blocks else ""
    # Abstract: blocks before the first section heading that looks like Introduction.
    abstract_blocks, exp_blocks = [], []
    in_exp = False
    seen_intro = False
    for b in blocks[1:]:
        if b.startswith("#"):
            if re.search(r"introduction", b, re.I):
                seen_intro = True
            in_exp = bool(_EXPERIMENTAL_HEADING.search(b)) and not _END_HEADING.search(b)
            continue
        if not seen_intro:
            abstract_blocks.append(b)
        if in_exp:
            exp_blocks.append(b)
    abstract = " ".join(abstract_blocks[:6])
    experimental = " ".join(exp_blocks)
    n_total = _count_mof(text)
    return {
        "mof_terms_total": n_total,
        "mof_terms_per_10k_chars": round(10000 * n_total / max(len(text), 1), 1),
        "mof_in_title": _count_mof(title) > 0,
        "mof_terms_abstract": _count_mof(abstract),
        "mof_terms_experimental": _count_mof(experimental),
        "cof_terms_total": len(_COF_TERMS.findall(text)),
        "experimental_chars": len(experimental),
    }


def select_verification_text(text: str, max_chars: int = 36000) -> str:
    """Abstract, headings and the paragraphs richest in MOF / synthesis evidence."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    keep = [i < 7 or b.startswith("#") for i, b in enumerate(blocks)]
    used = sum(len(b) + 2 for b, k in zip(blocks, keep) if k)

    def score(b: str) -> int:
        return 3 * _count_mof(b) + 2 * len(_COF_TERMS.findall(b)) + len(_SYNTH_TERMS.findall(b))

    ranked = sorted((i for i, k in enumerate(keep) if not k), key=lambda i: (-score(blocks[i]), i))
    for i in ranked:
        if score(blocks[i]) == 0 or used + len(blocks[i]) + 2 > max_chars:
            continue
        keep[i] = True
        used += len(blocks[i]) + 2
    return "\n\n".join(b for b, k in zip(blocks, keep) if k)


# --------------------------------------------------------------------------- #
# LLM schema and prompt
# --------------------------------------------------------------------------- #
class MofVerification(BaseModel):
    catalyst_class: Literal["MOF", "MOF_composite", "MOF_derived", "COF", "other_framework", "non_framework"] = Field(
        description="What the electrocatalyst actually studied is. MOF = as-synthesised metal-organic framework "
                    "(incl. MOF nanosheets on a substrate); MOF_composite = MOF combined with another component; "
                    "MOF_derived = made by pyrolysis/phosphidation/oxidation etc. of a MOF precursor; COF = covalent "
                    "organic framework (NOT a MOF); other_framework = LDH, zeolite, POP, etc.; non_framework = no framework.")
    mof_role: Literal["studied_catalyst_is_mof", "studied_catalyst_is_mof_derived", "mof_background_only", "no_mof"] = Field(
        description="studied_catalyst_is_mof: a MOF or MOF composite is the measured catalyst. "
                    "studied_catalyst_is_mof_derived: a MOF is the precursor/template of the measured catalyst. "
                    "mof_background_only: MOFs appear only in introduction, comparison tables or discussion of "
                    "other work. no_mof: no MOF involvement.")
    mof_synthesized_or_used_in_experimental: bool = Field(
        description="True only if the experimental/synthesis section actually prepares or uses a MOF "
                    "(or MOF precursor) for the studied catalyst.")
    experimental_electrochemistry_reported: bool = Field(
        description="True if the paper reports NEW measured electrochemical data for its own catalyst "
                    "(False for reviews, perspectives, or purely computational work).")
    reaction_studied: str = Field(description="The electrochemical process actually studied, in a few words.")
    matches_expected_reaction: bool = Field(description="True if reaction_studied is the expected reaction.")
    mof_identity: list[str] = Field(default_factory=list, description="MOF names/codes used (e.g. ZIF-67, Ni-BDC).")
    synthesis_evidence_quote: str | None = Field(
        None, description="Verbatim snippet (<=220 chars) showing the MOF being synthesised or used as precursor.")
    catalyst_evidence_quote: str | None = Field(
        None, description="Verbatim snippet (<=220 chars) identifying the studied electrocatalyst.")
    reason: str = Field(description="One sentence justifying the verdict.")
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM = """You are a careful materials-chemistry reviewer fact-checking whether a journal article is genuinely \
about a metal-organic framework (MOF) based electrocatalyst. Judge only from the supplied full text.

Be strict:
- A covalent organic framework (COF), LDH, zeolite, polymer, or plain metal oxide/sulfide is NOT a MOF, \
unless the text states it is made from a MOF precursor.
- MOFs mentioned only in the introduction, literature-comparison tables, or as other groups' work are \
'mof_background_only'.
- Pyrolysed/phosphidated/oxidised materials count as MOF-derived only if the text says a MOF was the precursor.
- Prussian blue analogues and coordination polymers count as MOFs only when the paper calls them MOFs or \
frameworks used as such.
- Papers on a different process (e.g. NH3-SCR, NO oxidation, supercapacitors, batteries, photocatalysis) do not \
match the expected electrochemical reaction.
- Quotes must be verbatim from the text."""

HUMAN = """EXPECTED REACTION: {reaction}

TITLE: {title}
DOI: {doi}

HEURISTIC COUNTS (for context): MOF terms in full text={mof_total}, in abstract={mof_abs}, in experimental section={mof_exp}, \
COF terms={cof}.

FULL-TEXT EXCERPTS (abstract, headings and the most MOF/synthesis-relevant paragraphs):
{text}

Return the verification."""


def build_verifier(model: str):
    ensure_loaded()
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=1500, timeout=180, max_retries=10)
    return llm.with_structured_output(MofVerification)


def decide(llm: dict[str, Any], heur: dict[str, Any]) -> tuple[str, str]:
    """Combine the LLM verdict with keyword evidence into a status and a note."""
    role = llm["mof_role"]
    has_mof_catalyst = role in ("studied_catalyst_is_mof", "studied_catalyst_is_mof_derived")
    note = ""
    if not has_mof_catalyst:
        status = "rejected_mof_background_only" if role == "mof_background_only" else "rejected_not_mof"
        if heur["mof_in_title"] or heur["mof_terms_abstract"] >= 3:
            return "needs_review", "LLM rejects but MOF is in title/abstract; check manually"
        return status, note
    if not llm["matches_expected_reaction"]:
        return "rejected_wrong_reaction", note
    if not llm["experimental_electrochemistry_reported"]:
        return "rejected_not_experimental", note
    if not llm["mof_synthesized_or_used_in_experimental"]:
        return "needs_review", "LLM calls the catalyst MOF-based but no MOF synthesis/use is confirmed"
    if llm["confidence"] < 0.6:
        return "needs_review", "low LLM confidence"
    if heur["mof_terms_total"] < 5:
        return "needs_review", "very few MOF terms in the full text"
    return "verified", note


def _init_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS verifications (
            cache_key TEXT PRIMARY KEY, reaction TEXT, doi TEXT, model TEXT,
            prompt_ver TEXT, response TEXT, created_at TEXT)"""
    )
    conn.commit()
    return conn


def verify_paper(row: dict[str, Any], reaction: str, *, model: str, chain=None,
                 cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    doi = str(row["doi"])
    text = Path(row["text_path"]).read_text(encoding="utf-8", errors="ignore")
    heur = heuristic_features(text)
    excerpt = select_verification_text(text)
    key = hashlib.sha256(
        f"{model}||{PROMPT_VERSION}||{reaction}||{doi}||{hashlib.sha256(excerpt.encode()).hexdigest()[:16]}".encode()
    ).hexdigest()[:32]

    conn = _init_cache(cache_path)
    hit = conn.execute("SELECT response FROM verifications WHERE cache_key = ?", (key,)).fetchone()
    if hit:
        llm = json.loads(hit[0])
        cached = True
    else:
        from langchain_core.messages import HumanMessage, SystemMessage

        chain = chain or build_verifier(model)
        first = excerpt.split("\n\n", 1)[0].replace("TITLE: ", "")
        msg = [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=HUMAN.format(
                reaction=REACTION_NAMES[reaction], title=row.get("title") or first, doi=doi,
                mof_total=heur["mof_terms_total"], mof_abs=heur["mof_terms_abstract"],
                mof_exp=heur["mof_terms_experimental"], cof=heur["cof_terms_total"], text=excerpt)),
        ]
        llm = chain.invoke(msg).model_dump()
        conn.execute("INSERT OR REPLACE INTO verifications VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (key, reaction, doi, model, PROMPT_VERSION, json.dumps(llm, ensure_ascii=False),
                      datetime.now(timezone.utc).isoformat()))
        conn.commit()
        cached = False
    conn.close()

    status, note = decide(llm, heur)
    llm = {**llm, "mof_identity": "|".join(llm.get("mof_identity") or [])}
    return {
        "reaction": reaction, "doi": doi, "title": row.get("title"), "year": row.get("year"),
        "journal": row.get("journal"), "verification_status": status, "review_note": note,
        **heur, **llm, "model": model, "from_cache": cached,
    }


def verify_reaction(manifest_path: Path, reaction: str, *, model: str, limit: int | None = None,
                    workers: int = 4, cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"]
    if limit:
        manifest = manifest.head(limit)
    rows = [r.to_dict() for _, r in manifest.iterrows()]
    chain = build_verifier(model) if rows else None

    def work(item):
        i, row = item
        try:
            out = verify_paper(row, reaction, model=model, chain=chain, cache_path=cache_path)
            print(f"[{reaction} {i}/{len(rows)}] {row['doi']}  {out['verification_status']}"
                  f"  ({out['catalyst_class']}, {out['mof_role']})", flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"[{reaction} {i}/{len(rows)}] {row['doi']}  ERROR: {exc}", flush=True)
            return {"reaction": reaction, "doi": row["doi"], "title": row.get("title"),
                    "verification_status": "error", "review_note": str(exc)[:300]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, enumerate(rows, start=1)))
    return pd.DataFrame(results)
