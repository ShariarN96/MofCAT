"""Convert Elsevier full-text XML files to LLM-friendly Markdown.

    python scripts/convert_xml.py                           # data/raw/elsevier_xml/ -> data/processed/elsevier_md/
    python scripts/convert_xml.py in.xml [more.xml] -o out_dir
    python scripts/convert_xml.py data/raw/elsevier_xml -o data/processed/elsevier_md --no-refs

Thin wrapper around mofcat.ingestion.xml_to_markdown.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:  # lets you run without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from mofcat.ingestion.xml_to_markdown import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
