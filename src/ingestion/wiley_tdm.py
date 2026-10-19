"""Wiley Text & Data Mining (TDM) PDF download client.

Follows Wiley rate limits:
  - up to 3 requests/second (we use 10 s spacing to stay conservative)
  - up to 60 requests per 10 minutes

API docs: https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

WILEY_TDM_BASE = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"
DOI_RESOLVER = "https://doi.org/"

WILEY_PUBLISHER_PATTERN = re.compile(r"wiley", re.I)
WILEY_DOI_PREFIXES = ("10.1002/", "10.1111/")


@dataclass
class DownloadResult:
    doi: str
    status: str  # ok | skipped | error
    http_status: int | None = None
    message: str = ""
    pdf_path: Path | None = None
    text_path: Path | None = None
    bytes_written: int = 0


class WileyRateLimiter:
    """Enforce Wiley TDM pacing: >=10 s between calls, <=60 per 10 min."""

    def __init__(
        self,
        min_interval_sec: float = 10.0,
        max_per_window: int = 60,
        window_sec: float = 600.0,
    ) -> None:
        self.min_interval_sec = min_interval_sec
        self.max_per_window = max_per_window
        self.window_sec = window_sec
        self._last_request_at: float | None = None
        self._request_times: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()

        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self.min_interval_sec:
                time.sleep(self.min_interval_sec - elapsed)
                now = time.monotonic()

        # Drop timestamps outside the rolling window.
        while self._request_times and now - self._request_times[0] > self.window_sec:
            self._request_times.popleft()

        if len(self._request_times) >= self.max_per_window:
            sleep_for = self.window_sec - (now - self._request_times[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] > self.window_sec:
                self._request_times.popleft()

        self._last_request_at = time.monotonic()
        self._request_times.append(self._last_request_at)


def normalize_doi(doi: str | None) -> str:
    if not doi or (isinstance(doi, float) and str(doi) == "nan"):
        return ""
    text = str(doi).strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    return text.strip().lower()


def doi_to_filename(doi: str, suffix: str = ".pdf") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doi.strip())
    return safe + suffix


def is_wiley_publisher(publisher: str | None) -> bool:
    return bool(publisher and WILEY_PUBLISHER_PATTERN.search(str(publisher)))


def is_likely_wiley_doi(doi: str) -> bool:
    doi = normalize_doi(doi)
    return any(doi.startswith(prefix) for prefix in WILEY_DOI_PREFIXES)


def doi_resolves_to_wiley(doi: str, timeout: int = 20) -> bool:
    """HEAD-resolve DOI and check final host is onlinelibrary.wiley.com."""
    doi = normalize_doi(doi)
    if not doi:
        return False
    try:
        response = requests.head(
            DOI_RESOLVER + doi,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "mof-wiley-tdm/1.0"},
        )
        host = urlparse(response.url).netloc.lower()
        return "onlinelibrary.wiley.com" in host or "wiley.com" in host
    except requests.RequestException:
        return False


def filter_wiley_records(
    df,
    *,
    verify_doi: bool = False,
    publisher_only: bool = False,
) -> Any:
    """Return rows that belong to Wiley-hosted articles."""
    import pandas as pd

    work = df.copy()
    work["doi_clean"] = work["doi"].map(normalize_doi)
    work = work[work["doi_clean"].astype(str).str.len() > 0]

    if publisher_only:
        mask = work["publisher"].map(is_wiley_publisher)
    else:
        pub_mask = work["publisher"].map(is_wiley_publisher)
        doi_mask = work["doi_clean"].map(is_likely_wiley_doi)
        mask = pub_mask | doi_mask

    work = work[mask].copy()

    if verify_doi:
        verified = []
        for _, row in work.iterrows():
            if doi_resolves_to_wiley(row["doi_clean"]):
                verified.append(row)
            time.sleep(0.2)  # light pacing for resolver checks
        work = pd.DataFrame(verified) if verified else work.iloc[0:0]

    return work.drop_duplicates(subset=["doi_clean"]).reset_index(drop=True)


def wiley_tdm_url(doi: str) -> str:
    return f"{WILEY_TDM_BASE}/{normalize_doi(doi)}"


def download_wiley_pdf(
    doi: str,
    dest_path: Path,
    token: str,
    *,
    limiter: WileyRateLimiter | None = None,
    timeout: int = 180,
    max_retries: int = 3,
) -> DownloadResult:
    """Download one PDF from Wiley TDM to *dest_path*."""
    doi = normalize_doi(doi)
    if not doi:
        return DownloadResult(doi=doi, status="error", message="empty DOI")
    if not token:
        return DownloadResult(doi=doi, status="error", message="missing Wiley TDM token")

    headers = {
        "Wiley-TDM-Client-Token": token,
        "Accept": "application/pdf",
        "User-Agent": "mof-wiley-tdm/1.0",
    }
    url = wiley_tdm_url(doi)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    for attempt in range(max_retries):
        if limiter:
            limiter.wait()
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            ) as response:
                status = response.status_code
                if status == 429:
                    last_error = "429 Too many requests — backing off"
                    time.sleep(30 * (attempt + 1))
                    continue
                if status == 400:
                    return DownloadResult(
                        doi=doi, status="error", http_status=400,
                        message="400 No TDM Client Token in request",
                    )
                if status == 403:
                    return DownloadResult(
                        doi=doi, status="error", http_status=403,
                        message="403 Invalid or unregistered TDM Client Token",
                    )
                if status == 404:
                    return DownloadResult(
                        doi=doi, status="error", http_status=404,
                        message="404 No full-text access for this DOI",
                    )
                response.raise_for_status()

                tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
                nbytes = 0
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            handle.write(chunk)
                            nbytes += len(chunk)
                tmp_path.replace(dest_path)

                return DownloadResult(
                    doi=doi,
                    status="ok",
                    http_status=status,
                    message="downloaded",
                    pdf_path=dest_path,
                    bytes_written=nbytes,
                )
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(5 * (attempt + 1))

    return DownloadResult(doi=doi, status="error", message=last_error or "download failed")


def extract_pdf_text(pdf_path: Path, text_path: Path) -> bool:
    """Extract plain text from a PDF (pypdf, then pymupdf fallback)."""
    text_path.parent.mkdir(parents=True, exist_ok=True)
    pages: list[str] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception:
        pages = []

    if not any(p.strip() for p in pages):
        try:
            import fitz  # pymupdf

            doc = fitz.open(pdf_path)
            pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
            doc.close()
        except Exception:
            return False

    full_text = "\n\n".join(p.strip() for p in pages if p and p.strip())
    if not full_text:
        return False
    text_path.write_text(full_text, encoding="utf-8")
    return True


def optional_upload_azure(pdf_path: Path, doi: str, connection_string: str, container: str) -> str:
    """Stream-upload an existing local PDF to Azure Blob Storage."""
    from azure.storage.blob import BlobClient

    blob_name = doi_to_filename(doi, ".pdf")
    client = BlobClient.from_connection_string(
        conn_str=connection_string,
        container_name=container,
        blob_name=blob_name,
    )
    with open(pdf_path, "rb") as handle:
        client.upload_blob(handle, overwrite=True, content_type="application/pdf")
    return client.url
