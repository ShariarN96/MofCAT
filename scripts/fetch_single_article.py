"""Download one article's full-text XML from Elsevier and convert it to readable text / Markdown.

A quick way to check your API key, institutional entitlement, or the XML converters on a single DOI.

    python scripts/fetch_single_article.py 10.1016/j.electacta.2020.137384

Saves:
    data/raw/elsevier_xml/<doi>.xml
    data/raw/elsevier_xml/<doi>_readable.txt
    data/processed/elsevier_md/<doi>.md
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import requests  # noqa: E402

from mofcat.env import ensure_loaded  # noqa: E402
from mofcat.ingestion.xml_to_markdown import convert_file  # noqa: E402
from mofcat.ingestion.xml_to_text import default_readable_output_path, xml_to_readable_text  # noqa: E402
from mofcat.paths import MARKDOWN_DIR, XML_DIR  # noqa: E402

BASE_URL = "https://api.elsevier.com/content/article/doi"


def article_url(doi: str) -> str:
    """Encode DOI so slashes are safe in the path (e.g. 10.1016/j.xxx)."""
    return f"{BASE_URL}/{quote(doi, safe='')}"


def has_full_article_xml(text: str) -> bool:
    """Elsevier article payloads include this root; entitlement checks do not."""
    return "<full-text-retrieval-response" in text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doi", help="e.g. 10.1016/j.electacta.2020.137384")
    args = ap.parse_args()
    doi = args.doi.strip()

    ensure_loaded()
    api_key = os.getenv("ELSEVIER_API_KEY")
    inst_token = os.getenv("ELSEVIER_INST_TOKEN")
    if not api_key:
        print("Set ELSEVIER_API_KEY in .env", file=sys.stderr)
        return 1

    headers = {"X-ELS-APIKey": api_key, "Accept": "application/xml"}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    # view=ENTITLED returns only <document-entitlement> (no body). Prefer real article XML.
    response: requests.Response | None = None
    for params in ({}, {"view": "FULL"}, {"view": "ENTITLED"}):
        try:
            r = requests.get(article_url(doi), headers=headers, params=params, timeout=60)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return 1
        response = r
        if r.status_code != 200 or "<service-error>" in r.text:
            continue
        if has_full_article_xml(r.text):
            break

    if response is None:
        print("No response received.")
        return 1

    print(f"Status: {response.status_code}\nDOI: {doi}\nPreview:\n{response.text[:1000]}")

    if response.status_code != 200:
        print("Not saving files (non-200 response).")
        return 1
    if "<service-error>" in response.text:
        print("API returned a service-error XML; not saving as successful full text.")
        return 1
    if not has_full_article_xml(response.text):
        print("Response is not full-text article XML (e.g. entitlement-only). "
              "Not saving; check API key, DOI, and institutional token.")
        return 1

    XML_DIR.mkdir(parents=True, exist_ok=True)
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    xml_file = XML_DIR / f"{doi.replace('/', '_')}.xml"
    xml_file.write_text(response.text, encoding="utf-8")
    print(f"Saved XML to {xml_file}")

    try:
        readable = default_readable_output_path(xml_file)
        readable.write_text(xml_to_readable_text(response.text), encoding="utf-8")
        print(f"Saved readable text to {readable}")
        print(f"Saved Markdown to {convert_file(xml_file, MARKDOWN_DIR)}")
    except ET.ParseError as exc:
        print(f"Could not parse XML for conversion: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
