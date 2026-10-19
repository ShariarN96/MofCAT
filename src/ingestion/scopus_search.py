"""Stage 1: search Scopus for every reaction query and save the records.

    python utils/search_scopus.py

Writes to data/raw/scopus/: per-reaction CSVs, scopus_results_ALL.csv/.xlsx and summary tables.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from ..env import ensure_loaded
from ..paths import SCOPUS_DIR
from .queries import (
    FETCH_ALL,
    MAX_RETRIES,
    PAGE_SIZE,
    SCOPUS_QUERIES,
    SCOPUS_URL,
    SERIAL_URL,
    SLEEP_SECONDS,
)

# ============================================================
# Setup
# ============================================================

ensure_loaded()

API_KEY = os.getenv("ELSEVIER_API_KEY")

if not API_KEY:
    raise ValueError("ELSEVIER_API_KEY not found. Check your .env file.")

headers = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/json",
}

OUTPUT_DIR = SCOPUS_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parallelism knobs — keep low; Scopus rate-limits aggressive bursts.
# Reactions run sequentially; only pages / ISSNs are parallelized.
PAGE_WORKERS = 3
ISSN_WORKERS = 4
# Global spacing between API calls (seconds). ~3 req/s is usually safe.
MIN_REQUEST_INTERVAL = 0.35

publisher_cache = {}
publisher_cache_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request_time = 0.0


def _throttle():
    """Ensure a minimum gap between outbound API calls across all threads."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = MIN_REQUEST_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


# ============================================================
# API helper functions
# ============================================================


def request_with_retries(url, params):
    for attempt in range(MAX_RETRIES):
        _throttle()
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            return response.json()

        if response.status_code in [429, 500, 502, 503, 504]:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = float(retry_after)
            else:
                wait = SLEEP_SECONDS * (2**attempt)
            print(
                f"Temporary error {response.status_code} "
                f"(attempt {attempt + 1}/{MAX_RETRIES}). Waiting {wait:.1f}s..."
            )
            time.sleep(wait)
            continue

        print("Permanent error:")
        print("Status:", response.status_code)
        print("Body:", response.text)
        return None

    return None


def clean_issn(issn):
    if not issn:
        return None

    return issn.replace("-", "").strip()


def get_publisher_by_issn(issn):
    issn = clean_issn(issn)

    if not issn:
        return "N/A"

    with publisher_cache_lock:
        if issn in publisher_cache:
            return publisher_cache[issn]

    params = {
        "issn": issn,
        "view": "STANDARD",
    }

    data = request_with_retries(SERIAL_URL, params)

    if data is None:
        publisher = "N/A"
    else:
        entries = data.get("serial-metadata-response", {}).get("entry", [])
        if not entries:
            publisher = "N/A"
        else:
            publisher = entries[0].get("dc:publisher", "N/A")

    with publisher_cache_lock:
        publisher_cache[issn] = publisher

    return publisher


def fetch_scopus_page(query, start):
    params = {
        "query": query,
        "count": PAGE_SIZE,
        "start": start,
        "field": (
            "dc:title,"
            "dc:creator,"
            "prism:publicationName,"
            "prism:issn,"
            "prism:eIssn,"
            "prism:coverDate,"
            "prism:doi,"
            "prism:aggregationType,"
            "subtypeDescription,"
            "citedby-count"
        ),
        "sort": "-date",
    }

    return request_with_retries(SCOPUS_URL, params)


# ============================================================
# Publisher family classification (for text-mining routing)
# ============================================================


def classify_publisher_family(publisher="", doi="", journal=""):
    """Map a record to ACS / Wiley / RSC / Elsevier / Other."""
    publisher_l = (publisher or "").lower()
    journal_l = (journal or "").lower()
    doi_l = (doi or "").lower().strip()

    # DOI prefixes are the most reliable signal when present
    if doi_l.startswith("10.1021"):
        return "ACS"
    if doi_l.startswith("10.1002"):
        return "Wiley"
    if doi_l.startswith("10.1039"):
        return "RSC"
    if doi_l.startswith("10.1016"):
        return "Elsevier"

    # Publisher name patterns
    if (
        "american chemical society" in publisher_l
        or publisher_l.strip() == "acs"
        or "acs publications" in publisher_l
    ):
        return "ACS"

    if "wiley" in publisher_l:
        return "Wiley"

    if "royal society of chemistry" in publisher_l or publisher_l.strip() == "rsc":
        return "RSC"

    if (
        "elsevier" in publisher_l
        or "academic press" in publisher_l
        or "pergamon" in publisher_l
    ):
        return "Elsevier"

    # Journal / imprint fallbacks
    if journal_l.startswith("acs ") or " acs " in f" {journal_l} ":
        return "ACS"
    if "wiley" in journal_l or "angewandte" in journal_l:
        return "Wiley"
    if any(
        token in journal_l
        for token in (
            "chemcomm",
            "chemical communications",
            "chemical science",
            "dalton",
            "rsc ",
            " royal society",
        )
    ):
        return "RSC"

    return "Other"


# ============================================================
# Data cleaning / formatting
# ============================================================


def scopus_item_to_row(item, reaction_name, application, query):
    title = item.get("dc:title", "N/A")
    creator = item.get("dc:creator", "N/A")
    journal = item.get("prism:publicationName", "N/A")
    date = item.get("prism:coverDate", "N/A")
    doi = item.get("prism:doi", "N/A")
    doc_type = item.get("subtypeDescription", "N/A")
    aggregation_type = item.get("prism:aggregationType", "N/A")
    cited_by = item.get("citedby-count", "N/A")

    issn_raw = item.get("prism:issn") or item.get("prism:eIssn")
    eissn_raw = item.get("prism:eIssn")
    issn_clean = clean_issn(issn_raw)

    with publisher_cache_lock:
        publisher = publisher_cache.get(issn_clean, "N/A") if issn_clean else "N/A"

    publisher_family = classify_publisher_family(
        publisher=publisher, doi=doi, journal=journal
    )

    return {
        "reaction": reaction_name,
        "application": application,
        "title": title,
        "creator": creator,
        "publisher": publisher,
        "publisher_family": publisher_family,
        "journal": journal,
        "publication_date": date,
        "year": date[:4] if date and date != "N/A" else "N/A",
        "doi": doi,
        "issn": issn_raw if issn_raw else "N/A",
        "eissn": eissn_raw if eissn_raw else "N/A",
        "document_type": doc_type,
        "aggregation_type": aggregation_type,
        "cited_by": cited_by,
        "query": query,
        "source": "Scopus API",
    }


# ============================================================
# Fetch one reaction
# ============================================================


def fetch_reaction(reaction_name, reaction_config):
    query = reaction_config["query"]
    application = reaction_config["application"]

    print("\n" + "=" * 100)
    print(f"FETCHING REACTION: {reaction_name}")
    print(f"APPLICATION: {application}")
    print("=" * 100)

    all_entries = []

    # First page: discover total_results, then fetch remaining pages in parallel
    print(f"[{reaction_name}] Fetching results 1 to {PAGE_SIZE}...")
    first_data = fetch_scopus_page(query, 0)

    if first_data is None:
        print(f"[{reaction_name}] Stopping because first request failed.")
        return pd.DataFrame()

    search_results = first_data.get("search-results", {})
    total_results = int(search_results.get("opensearch:totalResults", 0))
    print(f"[{reaction_name}] Total results available: {total_results}")

    first_entries = search_results.get("entry", [])
    if not first_entries:
        print(f"[{reaction_name}] No results.")
        return pd.DataFrame()

    all_entries.extend(first_entries)
    print(
        f"[{reaction_name}] Collected so far: {len(all_entries)} / {total_results}"
    )

    if FETCH_ALL and len(first_entries) >= PAGE_SIZE and total_results > PAGE_SIZE:
        starts = list(range(PAGE_SIZE, total_results, PAGE_SIZE))

        def _fetch_page(start):
            print(
                f"[{reaction_name}] Fetching results "
                f"{start + 1} to {start + PAGE_SIZE}..."
            )
            data = fetch_scopus_page(query, start)
            if data is None:
                return start, []
            entries = data.get("search-results", {}).get("entry", [])
            return start, entries or []

        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as executor:
            futures = [executor.submit(_fetch_page, start) for start in starts]
            page_results = []
            for future in as_completed(futures):
                start, entries = future.result()
                page_results.append((start, entries))
                print(
                    f"[{reaction_name}] Page start={start}: "
                    f"got {len(entries)} entries"
                )

        # Keep original order by start offset
        for _, entries in sorted(page_results, key=lambda x: x[0]):
            all_entries.extend(entries)
    elif not FETCH_ALL:
        print(f"[{reaction_name}] FETCH_ALL is False. Stopping after first page.")

    print(f"[{reaction_name}] Total entries collected: {len(all_entries)}")

    # --------------------------------------------------------
    # Fetch publisher names only once per ISSN
    # --------------------------------------------------------

    unique_issns = set()

    for item in all_entries:
        issn_raw = item.get("prism:issn") or item.get("prism:eIssn")
        issn_clean = clean_issn(issn_raw)

        if issn_clean:
            unique_issns.add(issn_clean)

    # Skip ISSNs already resolved by another reaction worker
    with publisher_cache_lock:
        missing_issns = [i for i in unique_issns if i not in publisher_cache]

    print(
        f"[{reaction_name}] Unique ISSNs: {len(unique_issns)} "
        f"(need fetch: {len(missing_issns)})"
    )

    def _fetch_one(issn):
        result = get_publisher_by_issn(issn)
        return issn, result

    if missing_issns:
        with ThreadPoolExecutor(max_workers=ISSN_WORKERS) as executor:
            futures = {
                executor.submit(_fetch_one, issn): issn for issn in missing_issns
            }
            for idx, future in enumerate(as_completed(futures), 1):
                issn, publisher = future.result()
                print(
                    f"[{reaction_name}] Fetched publisher "
                    f"{idx}/{len(missing_issns)} for ISSN {issn}: {publisher}"
                )

    # --------------------------------------------------------
    # Convert Scopus entries into rows
    # --------------------------------------------------------

    rows = []

    for item in all_entries:
        row = scopus_item_to_row(
            item=item, reaction_name=reaction_name, application=application, query=query
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save per-reaction CSV
    # --------------------------------------------------------

    reaction_csv = OUTPUT_DIR / f"scopus_results_{reaction_name}.csv"
    df.to_csv(reaction_csv, index=False, encoding="utf-8-sig")

    print(f"[{reaction_name}] Saved CSV: {reaction_csv}")
    if not df.empty and "publisher_family" in df.columns:
        print(
            f"[{reaction_name}] publisher_family counts:\n"
            f"{df['publisher_family'].value_counts().to_string()}"
        )

    return df


# ============================================================
# Main
# ============================================================


def main():
    all_dfs = []

    # Reactions are sequential to avoid rate-limit storms.
    # Pages and ISSN lookups inside each reaction still use a thread pool.
    for reaction_name, reaction_config in SCOPUS_QUERIES.items():
        try:
            df_reaction = fetch_reaction(reaction_name, reaction_config)
            all_dfs.append(df_reaction)
        except Exception as exc:
            print(f"[{reaction_name}] Failed with error: {exc}")

    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        print("No results fetched. Exiting.")
        return
    combined_df = pd.concat(non_empty, ignore_index=True)

    combined_csv = OUTPUT_DIR / "scopus_results_ALL.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    excel_path = OUTPUT_DIR / "scopus_results_ALL.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="ALL", index=False)

        for reaction_name in SCOPUS_QUERIES.keys():
            reaction_df = combined_df[combined_df["reaction"] == reaction_name]
            reaction_df.to_excel(writer, sheet_name=reaction_name, index=False)

    summary = (
        combined_df.groupby("reaction")
        .agg(
            total_records=("title", "count"),
            unique_dois=("doi", "nunique"),
            unique_journals=("journal", "nunique"),
            total_citations=(
                "cited_by",
                lambda x: pd.to_numeric(x, errors="coerce").sum(),
            ),
        )
        .reset_index()
    )

    summary_csv = OUTPUT_DIR / "scopus_summary_by_reaction.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    family_summary = (
        combined_df.groupby("publisher_family")
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    family_csv = OUTPUT_DIR / "scopus_summary_by_publisher_family.csv"
    family_summary.to_csv(family_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    print(f"Combined CSV saved: {combined_csv}")
    print(f"Excel workbook saved: {excel_path}")
    print(f"Summary CSV saved: {summary_csv}")
    print(f"Publisher-family summary saved: {family_csv}")

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nPublisher family:")
    print(family_summary.to_string(index=False))


if __name__ == "__main__":
    main()
