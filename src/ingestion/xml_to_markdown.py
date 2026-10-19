"""Convert Elsevier full-text API XML files to LLM-friendly Markdown.

Usage:
    python scripts/convert_xml.py                         # data/raw/elsevier_xml/ -> data/processed/elsevier_md/
    python scripts/convert_xml.py in.xml [more.xml] -o out_dir
    python scripts/convert_xml.py data/raw/elsevier_xml -o data/processed/elsevier_md --no-refs
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ..paths import MARKDOWN_DIR, XML_DIR

DC = "{http://purl.org/dc/elements/1.1/}"
PRISM = "{http://prismstandard.org/namespaces/basic/2.0/}"
DCTERMS = "{http://purl.org/dc/terms/}"

SKIP_INLINE = {"label", "float-anchor", "alt-text", "attachment", "link"}


def ln(el: ET.Element) -> str:
    t = el.tag
    return t.split("}", 1)[1] if isinstance(t, str) and t.startswith("{") else t


def ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def inline(el: ET.Element) -> str:
    """Render mixed-content element to Markdown inline text."""
    name = ln(el)
    if name == "math":
        return "".join(el.itertext()).strip()
    parts = [el.text or ""]
    for ch in el:
        c = ln(ch)
        if c in SKIP_INLINE:
            inner = ""
        elif c in ("italic", "emphasis"):
            s = inline(ch).strip()
            inner = f"*{s}*" if s else ""
        elif c == "bold":
            s = inline(ch).strip()
            inner = f"**{s}**" if s else ""
        elif c == "sup":
            inner = f"^{inline(ch).strip()}" if inline(ch).strip() else ""
        elif c == "inf":
            inner = f"_{inline(ch).strip()}" if inline(ch).strip() else ""
        elif c == "br":
            inner = " "
        elif c == "display" or c == "formula":
            inner = f" {inline(ch).strip()} "
        else:
            inner = inline(ch)
        parts.append(inner)
        parts.append(ch.tail or "")
    return ws("".join(parts)) if name in ("para", "simple-para", "title", "section-title") else "".join(parts)


def text_of(el: ET.Element | None) -> str:
    return ws(inline(el)) if el is not None else ""


def child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if ln(c) == name:
            return c
    return None


def table_md(tbl: ET.Element) -> list[str]:
    rows: list[list[str]] = []
    header_rows = 0
    for sec in tbl.iter():
        if ln(sec) in ("thead", "tbody"):
            for row in sec:
                if ln(row) != "row":
                    continue
                rows.append([text_of(e).replace("|", "\\|") for e in row if ln(e) == "entry"])
                if ln(sec) == "thead":
                    header_rows += 1
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if not header_rows:
        rows.insert(0, [""] * width)
    head, body = rows[0] if header_rows <= 1 else [
        " ".join(filter(None, col)) for col in zip(*rows[:header_rows])
    ], rows[max(header_rows, 1):]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def block(el: ET.Element, level: int, out: list[str]) -> None:
    """Append Markdown blocks for a body element."""
    name = ln(el)
    if name == "section":
        label = text_of(child(el, "label"))
        title = text_of(child(el, "section-title"))
        head = " ".join(filter(None, [label, title]))
        if head:
            out.append(f"{'#' * min(level, 6)} {head}")
        for c in el:
            block(c, level + 1, out)
    elif name in ("para", "simple-para"):
        t = text_of(el)
        if t:
            out.append(t)
        for c in el:  # lists/figures/tables nested inside paragraphs
            if ln(c) in ("list", "figure", "table", "display"):
                block(c, level, out)
    elif name == "list":
        for item in el:
            if ln(item) != "list-item":
                continue
            label = text_of(child(item, "label"))
            body = " ".join(text_of(p) for p in item if ln(p) in ("para", "simple-para"))
            out.append(f"- {label} {body}".replace("-  ", "- ") if label else f"- {body}")
    elif name == "figure":
        label = text_of(child(el, "label"))
        cap = child(el, "caption")
        txt = text_of(cap)
        if label or txt:
            out.append(f"*{label}: {txt}*".replace(": *", "*") if txt else f"*{label}*")
    elif name == "table":
        label = text_of(child(el, "label"))
        cap = text_of(child(el, "caption"))
        if label or cap:
            out.append(f"**{' '.join(filter(None, [label, cap]))}**")
        out.extend(["\n".join(table_md(el))] if table_md(el) else [])
    elif name in ("display", "formula"):
        t = ws("".join(el.itertext()))
        if t:
            out.append(t)
    elif name in ("sections", "tables", "floats", "table-wrap", "textbox", "acknowledgment", "appendices", "appendix"):
        if name == "acknowledgment":
            out.append(f"{'#' * min(level, 6)} Acknowledgements")
        for c in el:
            block(c, level, out)


def find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in root.iter() if ln(e) == name]


def largest(root: ET.Element, name: str) -> ET.Element | None:
    cands = find_all(root, name)
    return max(cands, key=lambda e: len("".join(e.itertext())), default=None)


def metadata(root: ET.Element) -> dict:
    core = next((e for e in root.iter() if ln(e) == "coredata"), root)
    m: dict = {"authors": [], "keywords": []}
    for e in core:
        t = ws(e.text or "")
        if not t:
            continue
        if e.tag == DC + "title":
            m["title"] = t
        elif e.tag == DC + "creator":
            m["authors"].append(t)
        elif e.tag == PRISM + "publicationName":
            m["journal"] = t
        elif e.tag == PRISM + "coverDate":
            m["date"] = t
        elif e.tag == PRISM + "doi":
            m["doi"] = t
        elif e.tag == PRISM + "volume":
            m["volume"] = t
        elif e.tag == DCTERMS + "subject":
            m["keywords"].append(t)
        elif e.tag == DC + "description":
            m["abstract"] = ws(inline(e))
    return m


def references(root: ET.Element) -> list[str]:
    refs = []
    for br in find_all(root, "bib-reference"):
        label = text_of(child(br, "label"))
        src = next((e for e in br.iter() if ln(e) == "source-text"), None)
        if src is not None and ws("".join(src.itertext())):
            body = ws("".join(src.itertext()))
        else:
            ref = next((e for e in br.iter() if ln(e) == "reference"), None)
            body = ws(" ".join("".join(ref.itertext()).split())) if ref is not None else ""
        if body:
            refs.append(f"{label} {body}".strip())
    return refs


def xml_to_markdown(raw: str | bytes, include_refs: bool = True) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if "<document-entitlement" in raw and "<full-text-retrieval-response" not in raw:
        raise ValueError("XML is an entitlement check only (no article content)")
    root = ET.fromstring(raw)
    m = metadata(root)

    out: list[str] = []
    out.append(f"# {m.get('title', 'Untitled')}")
    meta_lines = []
    if m["authors"]:
        meta_lines.append(f"- **Authors:** {'; '.join(m['authors'])}")
    if m.get("journal"):
        meta_lines.append(f"- **Journal:** {m['journal']}" + (f", vol. {m['volume']}" if m.get("volume") else ""))
    if m.get("date"):
        meta_lines.append(f"- **Date:** {m['date']}")
    if m.get("doi"):
        meta_lines.append(f"- **DOI:** {m['doi']}")
    if m["keywords"]:
        meta_lines.append(f"- **Keywords:** {', '.join(m['keywords'])}")
    if meta_lines:
        out.append("\n".join(meta_lines))
    if m.get("abstract"):
        out += ["## Abstract", m["abstract"]]

    body = largest(root, "body")
    if body is not None:
        out_body: list[str] = []
        for c in body:
            block(c, 2, out_body)
        if not out_body:
            out_body = [ws("".join(body.itertext()))]
        out += out_body
    else:
        out.append("_No full-text body in this XML (metadata only)._")

    floats = largest(root, "floats")
    if floats is not None:
        fl: list[str] = []
        block(floats, 3, fl)
        if fl:
            out += ["## Figures and Tables"] + fl

    if include_refs:
        refs = references(root)
        if refs:
            out.append("## References")
            out.append("\n".join(f"- {r}" for r in refs))

    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(out)).strip() + "\n"


def convert_file(src: Path, out_dir: Path, include_refs: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{src.stem}.md"
    dest.write_text(xml_to_markdown(src.read_bytes(), include_refs), encoding="utf-8")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("inputs", nargs="*", default=[str(XML_DIR)], help="XML files or folders")
    ap.add_argument("-o", "--out", default=str(MARKDOWN_DIR), help="output folder")
    ap.add_argument("--no-refs", action="store_true", help="omit the reference list")
    args = ap.parse_args()

    files: list[Path] = []
    for p in map(Path, args.inputs):
        files += sorted(p.glob("*.xml")) if p.is_dir() else [p]
    rc = 0
    for f in files:
        try:
            print(f"{f} -> {convert_file(f, Path(args.out), not args.no_refs)}")
        except Exception as e:  # keep going on bad files
            print(f"FAILED {f}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
