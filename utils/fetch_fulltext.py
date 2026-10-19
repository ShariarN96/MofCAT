"""Download and parse Elsevier full texts for the reaction sheets in data/raw/scopus/scopus_results_ALL.xlsx.

Saves per reaction under data/fulltext/<REACTION>/:
  xml/            raw API responses
  text/           cleaned full text (body, tables, figure captions; no references)
  manifest.csv    one row per attempted DOI (status, size, tables, paths)

Usage:
    python scripts/02_fetch_fulltext.py --test-key
    python scripts/02_fetch_fulltext.py --per-sheet 10          # pilot: 10 per reaction
    python scripts/02_fetch_fulltext.py --reactions NO3RR --per-sheet 10
    python scripts/02_fetch_fulltext.py --all                   # every Elsevier article
    python scripts/02_fetch_fulltext.py --dry-run

Pilot mode picks a reproducible random sample (seed 42) of Elsevier research
articles (reviews excluded) and keeps drawing until N papers with real full text
are saved, so papers you are not entitled to do not shrink the pilot.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mofcat.ingestion.fulltext import (  # noqa: E402
    ElsevierRateLimiter,
    fetch_article_xml,
    load_elsevier_records,
    parse_article_xml,
    reaction_dir,
    text_path,
    utc_now,
    xml_path,
)
from mofcat.env import ensure_loaded  # noqa: E402

from mofcat.ingestion.queries import REACTIONS as REACTION_SHEETS  # noqa: E402
from mofcat.paths import FULLTEXT_DIR, SCOPUS_XLSX  # noqa: E402

DEFAULT_XLSX = SCOPUS_XLSX
DEFAULT_OUT = FULLTEXT_DIR
TEST_DOI = "10.1016/j.apcatb.2022.121805"


def process_doi(row, reaction, out_root, api_key, insttoken, limiter, refetch):
    doi = row["doi_clean"]
    xp = xml_path(out_root, reaction, doi)
    tp = text_path(out_root, reaction, doi)
    base = {
        "reaction": reaction,
        "doi": doi,
        "title": row.get("title"),
        "year": row.get("year"),
        "journal": row.get("journal"),
        "cited_by": row.get("cited_by"),
        "document_type": row.get("document_type"),
    }

    if xp.exists() and not refetch:
        xml_bytes, http, msg = xp.read_bytes(), 200, "cached xml"
    else:
        res = fetch_article_xml(doi, api_key, insttoken=insttoken, limiter=limiter)
        if res.status != "ok":
            return {**base, "status": "error", "http_status": res.http_status,
                    "message": res.message, "fetched_at": utc_now()}
        xp.parent.mkdir(parents=True, exist_ok=True)
        xp.write_bytes(res.content)
        xml_bytes, http, msg = res.content, res.http_status, "downloaded"

    parsed = parse_article_xml(xml_bytes)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(parsed.text, encoding="utf-8")
    status = "ok" if parsed.has_fulltext else "abstract_only"
    return {
        **base,
        "status": status,
        "http_status": http,
        "message": msg,
        "pii": parsed.pii,
        "openaccess": parsed.openaccess,
        "xml_path": str(xp),
        "text_path": str(tp),
        "n_chars": parsed.stats.get("chars"),
        "n_paragraphs": parsed.stats.get("paragraphs"),
        "n_tables": parsed.stats.get("tables"),
        "n_figures": parsed.stats.get("figures"),
        "fetched_at": utc_now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--reactions", nargs="+", default=REACTION_SHEETS)
    ap.add_argument("--per-sheet", type=int, default=None,
                    help="stop after N full-text papers per reaction (pilot)")
    ap.add_argument("--all", action="store_true", help="fetch every Elsevier article")
    ap.add_argument("--include-reviews", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refetch", action="store_true", help="ignore cached XML")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new batch: archive manifest.csv to manifest_prev.csv (first time only) and "
                         "never re-select any DOI listed in an earlier manifest")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-key", action="store_true")
    args = ap.parse_args()

    ensure_loaded()
    api_key = os.environ.get("ELSEVIER_API_KEY", "")
    insttoken = os.environ.get("ELSEVIER_INSTTOKEN") or None
    limiter = ElsevierRateLimiter(args.interval)
    out_root = Path(args.output)

    if args.test_key:
        if not api_key:
            print("ELSEVIER_API_KEY not set in .env")
            return 1
        res = fetch_article_xml(TEST_DOI, api_key, insttoken=insttoken, limiter=limiter)
        print(f"HTTP {res.http_status}: {res.message}")
        if res.status == "ok":
            p = parse_article_xml(res.content)
            print(f"paragraphs={p.stats['paragraphs']} tables={p.stats['tables']} "
                  f"figures={p.stats['figures']} chars={p.stats['chars']} "
                  f"full_text={p.has_fulltext}")
        return 0 if res.status == "ok" else 1

    if not args.all and args.per_sheet is None:
        print("Specify --per-sheet N (pilot) or --all")
        return 1
    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        print(f"Workbook not found: {xlsx}")
        return 1
    if not api_key and not args.dry_run:
        print("ELSEVIER_API_KEY not set in .env")
        return 1

    for reaction in args.reactions:
        df = load_elsevier_records(xlsx, reaction, exclude_reviews=not args.include_reviews)
        print(f"\n=== {reaction}: {len(df)} Elsevier articles eligible ===")
        if args.dry_run:
            continue
        if not args.all:
            df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        manifest_path = reaction_dir(out_root, reaction) / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            prev_path = manifest_path.with_name("manifest_prev.csv")
            if manifest_path.exists() and not prev_path.exists():
                manifest_path.rename(prev_path)
            seen = set()
            for name in ("manifest_prev.csv", "manifest_all_fetched.csv"):
                p = manifest_path.with_name(name)
                if p.exists():
                    seen |= set(pd.read_csv(p)["doi"].dropna())
            df = df[~df["doi_clean"].isin(seen)].reset_index(drop=True)
            print(f"  --fresh: excluded {len(seen)} previously used DOIs; {len(df)} candidates left")
        try:
            existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        done = {r["doi"]: r for r in existing.to_dict("records")} if len(existing) else {}
        rows = dict(done)

        n_ok = sum(1 for r in rows.values() if r.get("status") == "ok")
        bad_view = 0
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if not args.all and n_ok >= args.per_sheet:
                break
            doi = row["doi_clean"]
            if doi in rows and rows[doi].get("status") in ("ok", "abstract_only") and not args.refetch:
                continue
            rec = process_doi(row, reaction, out_root, api_key, insttoken, limiter, args.refetch)
            rows[doi] = rec
            if rec["status"] == "ok":
                n_ok += 1
                bad_view = 0
            elif rec.get("http_status") == 400 and "not entitled" in str(rec.get("message")):
                bad_view += 1
                if bad_view >= 5:
                    pd.DataFrame(rows.values()).to_csv(manifest_path, index=False)
                    print("\nSTOPPING: Elsevier refuses full text (5 x 400 'FULL view'). "
                          "Connect to the university network/VPN, then rerun.")
                    return 2
            print(f"[{reaction} {n_ok}/{args.per_sheet or '-'}] {doi} -> {rec['status']} "
                  f"({rec.get('n_paragraphs')} paras, {rec.get('n_tables')} tables) {rec.get('message', '')}")
            pd.DataFrame(rows.values()).to_csv(manifest_path, index=False)

        m = pd.DataFrame(rows.values(), columns=None if rows else ["reaction", "doi", "title", "year", "journal",
                                                                  "cited_by", "document_type", "status"])
        m.to_csv(manifest_path, index=False)
        if m.empty:
            print(f"{reaction}: no unused Elsevier articles left -> 0 new papers")
            continue
        print(f"{reaction}: {m['status'].value_counts().to_dict()}  -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

