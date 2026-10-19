"""Second-pass, high-precision resolution of metals and organic linkers per catalyst.

The first extraction pass fills linker/metal fields as a side task. This pass does only that one job:
for every catalyst of a verified paper it reads the synthesis section(s) and returns

  - the organic linker(s) that build the MOF (or the precursor MOF of a MOF-derived catalyst),
    as written by the authors, with a verbatim evidence quote;
  - the metal elements of the MOF nodes and of the final catalyst.

Each answer is then fact-checked mechanically against the saved full text:
  linker_check = text_confirmed   linker name/abbreviation AND the quote are found in the paper
                 quote_not_found  linker found in the paper but the quote is not verbatim
                 not_in_text      linker string does not occur anywhere in the paper (suspect)
                 implied_by_code  not named; standard linker of a named MOF code (ZIF-8, MIL-101 ...)
                 none             no organic linker (e.g. not a MOF, or authors never say)
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
from ..paths import LINKER_CACHE
from .vocabulary import LINKER_ALIASES, normalize_linker, normalize_metal, normalize_mof_name

DEFAULT_CACHE = LINKER_CACHE
PROMPT_VERSION = "linker-v1"

_LINKER_HINT = re.compile(
    r"ligand|linker|H\d?[A-Z]{2,}|carboxyl|imidazol|terephthal|benzene-?1,?[345]|trimesic|\bBDC\b|\bBTC\b|"
    r"bipyridyl|pyridyl|organic|tricarboxyl|dicarboxyl|nitrogen-?containing|2-?MIM|\bMOF\b|ZIF|MIL-|UiO|HKUST|"
    r"solvothermal|hydrothermal|precursor", re.I)
_EXP_HEADING = re.compile(r"experiment|method|synthesis|preparation|materials?\b|fabrication|chemicals", re.I)
_END_HEADING = re.compile(r"result|discussion|conclusion|acknowledg|reference", re.I)


class Linker(BaseModel):
    name_as_written: str = Field(description="Linker/ligand name exactly as the authors write it "
                                             "(e.g. '2-methylimidazole', 'H2BDC', 'terephthalic acid').")
    abbreviation: str | None = Field(None, description="Abbreviation used for it in the paper, if any (BDC, BTC, 2-MIM, ...).")


class CatalystMaterial(BaseModel):
    catalyst_name: str = Field(description="Copy EXACTLY one of the catalyst names given in the request.")
    precursor_mof: str | None = Field(None, description="Name of the MOF that this catalyst is, contains, or was "
                                                         "derived from (e.g. 'ZIF-67', 'Ni-BDC'); null if none.")
    mof_node_metals: list[str] = Field(default_factory=list, description="Element symbols of the metal nodes of the (precursor) MOF.")
    catalyst_metals: list[str] = Field(default_factory=list, description="Element symbols of all metal elements that are part of the "
                                                                         "final catalyst's active composition (exclude the substrate such as Ni foam / "
                                                                         "carbon cloth / glassy carbon, and excluding Nafion etc.).")
    organic_linkers: list[Linker] = Field(default_factory=list, description="Organic linker(s) that build the framework of the "
                                          "(precursor) MOF. NOT solvents (DMF, ethanol), modulators (acetic acid), surfactants (PVP, CTAB), "
                                          "later-added dopant sources (melamine, thiourea, NaH2PO2) or the electrolyte.")
    linker_basis: Literal["stated", "implied_by_mof_code", "none"] = Field(
        description="stated: the text names the linker for this MOF. implied_by_mof_code: not named, but the MOF has a standard "
                    "code (ZIF-8, ZIF-67, MIL-101, UiO-66, HKUST-1...) -> give the standard linker. none: no framework linker "
                    "(not a MOF, control, or authors never say and no standard code).")
    linker_evidence_quote: str | None = Field(None, description="Verbatim sentence fragment (<=250 chars) from the text where the "
                                              "linker is named for this MOF. Null when basis is not 'stated'.")


class PaperMaterials(BaseModel):
    materials: list[CatalystMaterial]


SYSTEM = """You are an expert metal-organic-framework chemist. For each catalyst listed, determine from the paper's text \
(mainly the synthesis section) which ORGANIC LINKER builds the MOF the catalyst is based on, and which metals it contains.

Rules for accuracy:
- Report only what the text supports. Copy linker names exactly as written. Do NOT guess a linker that the text does not name \
unless the MOF has a standard code (ZIF-8/ZIF-67 -> 2-methylimidazole, MIL-53/MIL-101/UiO-66/MOF-5 -> terephthalic acid, \
HKUST-1/Cu-BTC -> trimesic acid) and then set linker_basis='implied_by_mof_code'.
- For MOF-derived materials (pyrolysed, phosphidated, sulfided, oxidised, etched) report the linker of the PRECURSOR MOF.
- Linkers are the organic ligands bridging the metal nodes. Solvents (DMF, DMAc, methanol), modulators/capping agents \
(acetic acid, PVP, CTAB, triethylamine), pyrolysis additives (melamine, urea, thiourea) and phosphorus/sulfur sources are NOT linkers \
unless the text explicitly says they are the framework ligand.
- If a MOF is built from several linkers (mixed-linker or pillared MOF), list all of them.
- Controls, commercial benchmarks (Pt/C, RuO2) and bare substrates: no precursor_mof, no linkers, basis 'none'.
- Metals: element symbols only. 'mof_node_metals' = metals of the MOF nodes; 'catalyst_metals' = all metals in the final active \
catalyst (e.g. Ru-doped Co3O4 from ZIF-67 -> Co, Ru), excluding the supporting substrate.
- The quote must be copied verbatim from the text."""

HUMAN = """PAPER TITLE: {title}
DOI: {doi}

CATALYSTS TO RESOLVE (copy these names exactly):
{catalysts}

FIRST-PASS HINTS (may be wrong, verify against the text): {hints}

TEXT (abstract, synthesis/experimental section and linker-related paragraphs):
{text}

Return one entry per catalyst listed."""


def select_text(text: str, max_chars: int = 40000) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    keep = [i < 7 or b.startswith("#") for i, b in enumerate(blocks)]
    in_exp = False
    for i, b in enumerate(blocks):
        if b.startswith("#"):
            in_exp = bool(_EXP_HEADING.search(b)) and not _END_HEADING.search(b)
        elif in_exp:
            keep[i] = True
    used = sum(len(b) + 2 for b, k in zip(blocks, keep) if k)
    if used > max_chars:  # experimental section alone is too long: keep earliest blocks
        out, tot = [], 0
        for b, k in zip(blocks, keep):
            if k and tot + len(b) + 2 <= max_chars:
                out.append(b)
                tot += len(b) + 2
        return "\n\n".join(out)
    ranked = sorted((i for i, k in enumerate(keep) if not k), key=lambda i: (-len(_LINKER_HINT.findall(blocks[i])), i))
    for i in ranked:
        if not _LINKER_HINT.search(blocks[i]) or used + len(blocks[i]) + 2 > max_chars:
            continue
        keep[i] = True
        used += len(blocks[i]) + 2
    return "\n\n".join(b for b, k in zip(blocks, keep) if k)


def _norm(s: str) -> str:
    s = s.lower().replace("−", "-").replace("–", "-").replace("‐", "-").replace("′", "'")
    s = re.sub(r"[_^{}]", "", s)
    return re.sub(r"\s+", " ", s)


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(s))


def check_linker(linker: dict[str, Any], quote: str | None, basis: str, full_alnum: str, mof_name: str | None) -> str:
    """text_confirmed | linker_in_text | not_in_text | implied_by_code | none (see module docstring)."""
    if basis == "none":
        return "none"
    if basis == "implied_by_mof_code":
        return "implied_by_code"
    names = [_alnum(n) for n in (linker.get("name_as_written"), linker.get("abbreviation")) if n]
    names = [n for n in names if len(n) >= 3]
    if not any(n in full_alnum for n in names):
        return "not_in_text"
    if quote:
        q = _alnum(quote.strip(" .…"))
        if q[:90] in full_alnum and any(n in q for n in names):
            return "text_confirmed"
    return "linker_in_text"


# Synonym families -> one canonical label, so that 'benzene-1,3,5-tricarboxylic acid', 'H3BTC' and
# 'trimesic acid' are counted as the same linker. First match wins.
_CANON = [
    (r"dihydroxyterephth|dihydroxy-?1,?4-benzenedicarb|2,?5-dihydroxy|dobdc|dhtp|dhbdc", "DOBDC (2,5-dihydroxyterephthalic acid)"),
    (r"amino-?terephth|nh_?2-?h?_?2?bdc|h_?2bdc-?nh|bdc-?nh_?2|2-?aminobenzene-?1,?4", "NH2-BDC (2-aminoterephthalic acid)"),
    (r"nitro-?terephth|bdc-?no_?2|no_?2-?bdc", "NO2-BDC (2-nitroterephthalic acid)"),
    (r"terephthal|terephtal|1,?\s?4-?\s?(benzene)?-?\s?dicarboxy|benzene-?1,?4-?dicarb|1,?4-?dicarboxybenzene|"
     r"1,?4-?benzenedicarb|\bh_?2?bdc\b|\bbdc\b|\btpa\b|p-?phthalic", "BDC (terephthalic acid)"),
    (r"trimes|1,?\s?3,?\s?5-?\s?benzene-?\s?tricarb|benzene-?1,?3,?5-?\s?tricarb|\bh_?3?btc\b|\bbtc\b", "BTC (trimesic acid)"),
    (r"tetrakis\(?4-carboxyphenyl\)?porphyrin|\btcpp\b", "TCPP (porphyrin)"),
    (r"2-?\s?methyl-?\s?imi?z?a?d?i?zol|2-?mim|\bmeim\b|2-?mei?m|hmim|methylimizadol", "2-MIM (2-methylimidazole)"),
    (r"benzimidazol", "benzimidazole"),
    (r"^imidazol(e|ate)?s?$", "imidazole"),
    (r"4,?4['′’]?-?\s?(bipyridine|dipyridyl|bipyridyl)|\bbpy\b|bipyridine|dipyridyl|bipyridyl", "bipyridine (bpy)"),
    (r"1,?10-?phenanthroline|\bphen\b", "1,10-phenanthroline"),
    (r"oxalic|oxalate", "oxalic acid"),
    (r"ferrocenedicarboxylic|\bfcda\b", "ferrocenedicarboxylic acid"),
    (r"fumaric|fumarate", "fumaric acid"),
    (r"2,?6-?naphthalenedicarb|\bndc\b", "NDC (2,6-naphthalenedicarboxylic acid)"),
    (r"biphenyl-?4,?4['′’]?-?dicarb|\bbpdc\b", "BPDC (biphenyldicarboxylic acid)"),
    (r"hexamethylenetetramine|\bhmt\b", "hexamethylenetetramine"),
    (r"triethylenediamine|dabco", "DABCO (triethylenediamine)"),
    (r"pyrazole-?3,?5-?dicarb", "pyrazole-3,5-dicarboxylic acid"),
    (r"iminodiacetic|\bida\b", "iminodiacetic acid"),
    (r"1,?3,?5-?tris\(4-carboxyphenyl\)benzene|\bbtb\b|h_?3?btb", "BTB"),
    (r"4,?4['′’]?-?oxybis|oxydibenzo", "4,4'-oxybis(benzoic acid)"),
    (r"2-?aminoimidazol|purine|adenine", "purine/adenine-type"),
]
_NOT_LINKER = re.compile(r"^(zif|mil|uio|hkust|mof|cpo|pcn|nu|irmof|dut|cau)[-\s]?\d|^(dmf|dmso|dmac|ethanol|methanol|water|"
                         r"urea|melamine|thiourea|pvp|ctab|nafion)$", re.I)


def canonical_linker(name: str | None, abbrev: str | None = None) -> str | None:
    """Map a linker string (+ optional abbreviation) to a canonical label; None if it is not a linker."""
    cand = " ".join(x for x in (name, abbrev) if x).strip()
    if not cand or _NOT_LINKER.match((name or "").strip()):
        return None
    low = cand.lower().replace("−", "-").replace("–", "-").replace("‐", "-")
    for pat, label in _CANON:
        if re.search(pat, low):
            return label
    # exact vocabulary alias only (normalize_linker's substring matching is too permissive here)
    fam = LINKER_ALIASES.get((name or "").strip().lower()) or LINKER_ALIASES.get((abbrev or "").strip().lower())
    return fam or re.sub(r"\s+", " ", (name or abbrev or "").strip())[:60]


def canon_list(raw: str | None) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    out = list(dict.fromkeys(c for c in (canonical_linker(x) for x in raw.split("|")) if c))
    return "|".join(out) or None


def build_resolver(model: str):
    ensure_loaded()
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0, max_tokens=6000, timeout=240, max_retries=10)
    return llm.with_structured_output(PaperMaterials)


def _init_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=120)
    conn.execute("""CREATE TABLE IF NOT EXISTS resolved (cache_key TEXT PRIMARY KEY, doi TEXT, model TEXT,
                    prompt_ver TEXT, response TEXT, created_at TEXT)""")
    conn.commit()
    return conn


def resolve_paper(doi: str, title: str, text: str, catalysts: list[str], hints: str, *, model: str, chain=None,
                  cache_path: Path = DEFAULT_CACHE) -> list[dict[str, Any]]:
    excerpt = select_text(text)
    key = hashlib.sha256(
        f"{model}||{PROMPT_VERSION}||{doi}||{'|'.join(sorted(catalysts))}||"
        f"{hashlib.sha256(excerpt.encode()).hexdigest()[:16]}".encode()).hexdigest()[:32]
    conn = _init_cache(cache_path)
    hit = conn.execute("SELECT response FROM resolved WHERE cache_key=?", (key,)).fetchone()
    if hit:
        mats = json.loads(hit[0])["materials"]
    else:
        from langchain_core.messages import HumanMessage, SystemMessage

        chain = chain or build_resolver(model)
        msg = [SystemMessage(content=SYSTEM),
               HumanMessage(content=HUMAN.format(title=title, doi=doi,
                                                 catalysts="\n".join(f"- {c}" for c in catalysts),
                                                 hints=hints or "none", text=excerpt))]
        out = chain.invoke(msg).model_dump()
        mats = out["materials"]
        conn.execute("INSERT OR REPLACE INTO resolved VALUES (?,?,?,?,?,?)",
                     (key, doi, model, PROMPT_VERSION, json.dumps(out, ensure_ascii=False),
                      datetime.now(timezone.utc).isoformat()))
        conn.commit()
    conn.close()

    full_norm = _alnum(text)
    rows = []
    for m in mats:
        checks = [check_linker(l, m.get("linker_evidence_quote"), m["linker_basis"], full_norm, m.get("precursor_mof"))
                  for l in m["organic_linkers"]] or ["none"]
        rank = ["not_in_text", "linker_in_text", "implied_by_code", "text_confirmed", "none"]
        worst = min(checks, key=rank.index)
        basis = m["linker_basis"]
        raw = [l["name_as_written"] for l in m["organic_linkers"]]
        abbr = [l.get("abbreviation") or "" for l in m["organic_linkers"]]
        if worst == "not_in_text":
            # the claimed linker occurs nowhere in the paper: fall back to the standard linker of a named MOF code
            code_linker = None
            for nm in (m.get("precursor_mof"), m["catalyst_name"]):
                if nm and normalize_mof_name(nm)[2]:
                    code_linker = normalize_mof_name(nm)[2]
                    break
            if code_linker:
                raw, abbr, basis, worst = [code_linker], [""], "implied_by_mof_code", "implied_by_code"
            else:
                raw, abbr, basis, worst = [], [], "none", "unsupported_removed"
        canon = list(dict.fromkeys(c for c in (canonical_linker(n, a) for n, a in zip(raw, abbr)) if c))
        rows.append({
            "catalyst_name": m["catalyst_name"],
            "precursor_mof": m.get("precursor_mof"),
            "mof_node_metals": "|".join(dict.fromkeys(x for x in (normalize_metal(e) for e in m["mof_node_metals"]) if x)),
            "catalyst_metals": "|".join(dict.fromkeys(x for x in (normalize_metal(e) for e in m["catalyst_metals"]) if x)),
            "linker_names_raw": "|".join(raw),
            "linker_abbrev": "|".join(a for a in abbr if a),
            "linker_family": "|".join(canon),
            "linker_basis": basis if canon else "none",
            "linker_evidence_quote": m.get("linker_evidence_quote"),
            "linker_check": worst,
        })
    return rows


def resolve_reaction(reaction: str, records_csv: Path, manifest_csv: Path, *, model: str, workers: int = 4,
                     cache_path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    """Return records with resolved descriptors merged in (first-pass values kept as *_pass1)."""
    from concurrent.futures import ThreadPoolExecutor

    rec = pd.read_csv(records_csv)
    man = pd.read_csv(manifest_csv).set_index("doi")
    chain = build_resolver(model)
    papers = list(rec.groupby("doi"))

    def work(item):
        i, (doi, g) = item
        cats = [c for c in dict.fromkeys(g["catalyst_name"].dropna()) if isinstance(c, str)]
        hints = "; ".join(f"{r.catalyst_name}: metals={r.node_metals}, linkers={r.organic_linkers_raw}"
                          for r in g.drop_duplicates("catalyst_name").itertuples() if isinstance(r.catalyst_name, str))
        try:
            text = Path(man.loc[doi, "text_path"]).read_text(encoding="utf-8", errors="ignore")
            rows = resolve_paper(doi, str(g["title"].iloc[0]), text, cats, hints[:2500], model=model, chain=chain,
                                 cache_path=cache_path)
            print(f"[{reaction} {i}/{len(papers)}] {doi}: {len(rows)} catalysts resolved", flush=True)
            return doi, rows
        except Exception as exc:  # noqa: BLE001
            print(f"[{reaction} {i}/{len(papers)}] {doi}: ERROR {exc}", flush=True)
            return doi, []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(work, enumerate(papers, start=1)))

    res = []
    for doi, rows in results:
        for r in rows:
            res.append({"doi": doi, **r})
    res = pd.DataFrame(res)
    if res.empty:
        return rec
    # match catalysts on a punctuation/case-insensitive key (LLM may alter sub/superscript markup)
    res["_key"] = res["catalyst_name"].astype(str).map(_alnum)
    res = res.drop(columns="catalyst_name").drop_duplicates(["doi", "_key"])
    rec = rec.copy()
    rec["_key"] = rec["catalyst_name"].astype(str).map(_alnum)
    out = rec.merge(res, on=["doi", "_key"], how="left").drop(columns="_key")

    # promote resolved values over first-pass values where the resolver gave an answer
    out["node_metals_pass1"] = out["node_metals"]
    out["organic_linkers_pass1"] = out["organic_linkers_raw"]
    has_m = out["catalyst_metals"].fillna("") != ""
    out.loc[has_m, "node_metals"] = out.loc[has_m, "catalyst_metals"]
    has_l = out["linker_basis"].isin(["stated", "implied_by_mof_code"]) & (out["linker_family"].fillna("") != "")
    out.loc[has_l, "organic_linkers_raw"] = out.loc[has_l, "linker_names_raw"]
    out.loc[has_l, "organic_linkers"] = out.loc[has_l, "linker_family"]
    out.loc[has_l, "linker_source"] = out.loc[has_l, "linker_basis"]
    # resolver says there is no linker -> clear first-pass guesses that the text cannot back up
    none_l = (out["linker_basis"] == "none") & ~has_l
    out.loc[none_l, ["organic_linkers", "organic_linkers_raw", "linker_source"]] = [None, None, None]
    # catalysts the resolver did not return: keep first-pass linkers (canonicalised) but flag them
    unres = out["linker_check"].isna()
    out.loc[unres, "organic_linkers"] = out.loc[unres, "organic_linkers_raw"].map(canon_list)
    out.loc[unres & out["organic_linkers"].notna(), "linker_check"] = "first_pass_only"
    out.loc[unres & out["organic_linkers"].isna(), "linker_check"] = "none"
    out["organic_linkers"] = out["organic_linkers"].map(canon_list)
    out["n_metals"] = out["node_metals"].fillna("").map(lambda s: len([m for m in s.split("|") if m]))
    return out
