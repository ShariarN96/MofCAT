"""Elsevier full-text retrieval (ScienceDirect Article Retrieval API).

For each DOI we save, locally:
  xml/<doi>.xml   the raw API response (kept so parsing can be redone offline)
  text/<doi>.txt  cleaned plain text: abstract, headings, body paragraphs, tables
                  (pipe-separated rows) and figure captions; references removed

API: GET https://api.elsevier.com/content/article/doi/{doi}?view=FULL
Auth: X-ELS-APIKey header (ELSEVIER_API_KEY in .env), optional X-ELS-Insttoken
(ELSEVIER_INSTTOKEN). Full body text needs the entitlement your institution has
(typically UIC campus network / VPN or an institutional token).

Downloaded text is for your own text-mining research under Elsevier's TDM terms.
Do not redistribute it or commit it to a public repository.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

from .queries import REACTIONS as REACTION_SHEETS  # one sheet per reaction in scopus_results_ALL.xlsx
from .wiley_tdm import doi_to_filename, normalize_doi

ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/"

MIN_PARAGRAPHS_FOR_FULLTEXT = 10
MIN_CHARS_FOR_FULLTEXT = 6000


# --------------------------------------------------------------------------- #
# Input selection
# --------------------------------------------------------------------------- #
def load_elsevier_records(
    xlsx_path: Path,
    sheet: str,
    *,
    exclude_reviews: bool = True,
) -> pd.DataFrame:
    """Rows of one reaction sheet that are Elsevier journal articles with a DOI."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df["doi_clean"] = df["doi"].map(normalize_doi)
    mask = df["publisher_family"].astype(str).str.strip().str.lower() == "elsevier"
    mask &= df["doi_clean"].str.len() > 0
    if exclude_reviews and "document_type" in df.columns:
        mask &= ~df["document_type"].astype(str).str.contains("review", case=False)
    out = df[mask].drop_duplicates(subset=["doi_clean"]).reset_index(drop=True)
    out["reaction"] = sheet
    return out


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class ElsevierRateLimiter:
    """Space requests; Elsevier allows several per second, we stay well under."""

    def __init__(self, min_interval_sec: float = 0.5) -> None:
        self.min_interval_sec = min_interval_sec
        self._last: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            gap = now - self._last
            if gap < self.min_interval_sec:
                time.sleep(self.min_interval_sec - gap)
        self._last = time.monotonic()


@dataclass
class FetchResult:
    doi: str
    status: str  # ok | error
    http_status: int | None = None
    message: str = ""
    content: bytes = b""


def fetch_article_xml(
    doi: str,
    api_key: str,
    *,
    insttoken: str | None = None,
    limiter: ElsevierRateLimiter | None = None,
    timeout: int = 90,
    max_retries: int = 3,
) -> FetchResult:
    doi = normalize_doi(doi)
    if not doi:
        return FetchResult(doi, "error", message="empty DOI")
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "text/xml",
        "User-Agent": "mof-literature-mining/1.0",
    }
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken

    last = ""
    for attempt in range(max_retries):
        if limiter:
            limiter.wait()
        try:
            resp = requests.get(
                ELSEVIER_ARTICLE_URL + doi,
                params={"view": "FULL"},
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last = str(exc)
            time.sleep(5 * (attempt + 1))
            continue

        if resp.status_code == 200:
            return FetchResult(doi, "ok", 200, "downloaded", resp.content)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30 * (attempt + 1)))
            last = f"429 rate limited; waited {wait}s"
            time.sleep(min(wait, 300))
            continue
        if resp.status_code in (401, 403):
            return FetchResult(doi, "error", resp.status_code,
                               "401/403 not entitled or invalid key for this article")
        if resp.status_code == 404:
            return FetchResult(doi, "error", 404, "404 article not found in ScienceDirect")
        if resp.status_code == 400 and "view" in (resp.headers.get("X-ELS-Status") or "").lower():
            # FULL view is refused when the request is not entitled (off the campus network / VPN, no insttoken)
            return FetchResult(doi, "error", 400,
                               "400 FULL view refused: not entitled (connect to the university network/VPN "
                               "or set ELSEVIER_INSTTOKEN)")
        last = f"HTTP {resp.status_code}"
        time.sleep(3 * (attempt + 1))
    return FetchResult(doi, "error", None, last or "failed")


# --------------------------------------------------------------------------- #
# XML -> text
# --------------------------------------------------------------------------- #
_SKIP_BLOCKS = {
    "bibliography", "bib-reference", "acknowledgment", "author-group",
    "grant-sponsor", "grant-number", "grant-sponsor-group", "footnote",
    "biography", "date-received", "date-accepted", "date-revised",
    "copyright", "article-footnote", "correspondence",
}
_NOT_INLINE = {"table", "figure", "display", "float-anchor", "textbox", "e-component"}


def _ln(el) -> str:
    tag = el.tag
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _inline(el) -> str:
    """Text of an element; sup/inf rendered as ^x / _x so units survive (cm^-2)."""
    out: list[str] = []
    if el.text:
        out.append(el.text)
    for child in el:
        ln = _ln(child)
        if ln == "cross-ref" and str(child.get("refid", "")).startswith("bib"):
            pass  # citation marker, noise
        elif ln in _NOT_INLINE:
            pass
        elif ln == "sup":
            out.append("^" + _inline(child))
        elif ln == "inf":
            out.append("_" + _inline(child))
        else:
            out.append(_inline(child))
        if child.tail:
            out.append(child.tail)
    return "".join(out)


def _table_block(tbl) -> str:
    label = ""
    caption = ""
    for child in tbl:
        ln = _ln(child)
        if ln == "label":
            label = _clean(_inline(child))
        elif ln == "caption":
            caption = _clean(" ".join(_inline(p) for p in child.iter() if _ln(p) in ("simple-para", "para")) or _inline(child))
    rows: list[str] = []
    for row in tbl.iter():
        if _ln(row) != "row":
            continue
        cells = [_clean(_inline(e)) for e in row if _ln(e) == "entry"]
        if any(cells):
            rows.append(" | ".join(cells))
    notes = [
        _clean(_inline(n))
        for n in tbl.iter()
        if _ln(n) in ("table-footnote", "legend") and _clean(_inline(n))
    ]
    header = f"[TABLE {label}] {caption}".strip()
    parts = [header] + rows + ([f"Table notes: {' '.join(notes)}"] if notes else [])
    return "\n".join(parts)


def _figure_block(fig) -> str:
    label = ""
    caption = ""
    for child in fig:
        ln = _ln(child)
        if ln == "label":
            label = _clean(_inline(child))
        elif ln == "caption":
            caption = _clean(" ".join(_inline(p) for p in child.iter() if _ln(p) in ("simple-para", "para")) or _inline(child))
    return f"[FIGURE {label}] {caption}".strip() if caption else ""


def _walk(el, blocks: list[str], stats: dict, depth: int = 0) -> None:
    ln = _ln(el)
    if ln in _SKIP_BLOCKS:
        return
    if ln == "section-title":
        title = _clean(_inline(el))
        if title:
            blocks.append("#" * min(depth + 1, 4) + " " + title)
        return
    if ln in ("para", "simple-para"):
        text = _clean(_inline(el))
        if text:
            blocks.append(text)
            stats["paragraphs"] += 1
        for child in el.iter():  # tables/figures nested in a paragraph
            if child is el:
                continue
            cl = _ln(child)
            if cl == "table":
                blocks.append(_table_block(child))
                stats["tables"] += 1
            elif cl == "figure":
                fb = _figure_block(child)
                if fb:
                    blocks.append(fb)
                    stats["figures"] += 1
        return
    if ln == "table":
        blocks.append(_table_block(el))
        stats["tables"] += 1
        return
    if ln == "figure":
        fb = _figure_block(el)
        if fb:
            blocks.append(fb)
            stats["figures"] += 1
        return
    if ln == "keywords":
        words = [_clean(_inline(k)) for k in el.iter() if _ln(k) == "keyword"]
        words = [w for w in words if w]
        if words:
            blocks.append("Keywords: " + "; ".join(words))
        return
    next_depth = depth + 1 if ln in ("section", "abstract-sec") else depth
    for child in el:
        _walk(child, blocks, stats, next_depth)


@dataclass
class ParsedArticle:
    title: str = ""
    text: str = ""
    pii: str = ""
    openaccess: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def has_fulltext(self) -> bool:
        return (
            self.stats.get("paragraphs", 0) >= MIN_PARAGRAPHS_FOR_FULLTEXT
            and len(self.text) >= MIN_CHARS_FOR_FULLTEXT
        )


def parse_article_xml(xml_bytes: bytes) -> ParsedArticle:
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
    root = etree.fromstring(xml_bytes, parser=parser)
    stats = {"paragraphs": 0, "tables": 0, "figures": 0}

    def first_text(name: str) -> str:
        for el in root.iter():
            if _ln(el) == name:
                return _clean(_inline(el))
        return ""

    title = first_text("title")
    pii = first_text("pii")
    openaccess = first_text("openaccessArticle")

    article = next((el for el in root.iter() if _ln(el) == "article"), None)
    blocks: list[str] = []
    if article is not None:
        _walk(article, blocks, stats)
    else:  # fall back to coredata abstract only
        desc = first_text("description")
        if desc:
            blocks.append(desc)

    text = f"TITLE: {title}\n\n" + "\n\n".join(b for b in blocks if b)
    stats["chars"] = len(text)
    return ParsedArticle(title=title, text=text, pii=pii, openaccess=openaccess, stats=stats)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def reaction_dir(root: Path, reaction: str) -> Path:
    return root / reaction


def xml_path(root: Path, reaction: str, doi: str) -> Path:
    return reaction_dir(root, reaction) / "xml" / doi_to_filename(doi, ".xml")


def text_path(root: Path, reaction: str, doi: str) -> Path:
    return reaction_dir(root, reaction) / "text" / doi_to_filename(doi, ".txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
