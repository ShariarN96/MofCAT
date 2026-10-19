"""Utility helpers: Elsevier full-text API XML → readable plain text (import only)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


DC = "{http://purl.org/dc/elements/1.1/}"
PRISM = "{http://prismstandard.org/namespaces/basic/2.0/}"


def localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_coredata_fields(root: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for el in root.iter():
        tag = el.tag
        if tag == DC + "title" and (el.text or "").strip():
            out.setdefault("title", (el.text or "").strip())
        elif tag == DC + "description" and (el.text or "").strip():
            out.setdefault("abstract", normalize_ws(el.text or ""))
        elif tag == DC + "identifier" and (el.text or "").strip():
            if (el.text or "").lower().startswith("doi:"):
                out.setdefault("doi", (el.text or "").strip())
        elif tag == PRISM + "doi" and (el.text or "").strip():
            out.setdefault("prism_doi", (el.text or "").strip())
    return out


def find_largest_body(root: ET.Element) -> ET.Element | None:
    """Elsevier responses may include multiple body-like nodes; take the richest one."""
    best: ET.Element | None = None
    best_len = 0
    for el in root.iter():
        if localname(el.tag) != "body":
            continue
        raw = "".join(el.itertext())
        if len(raw) > best_len:
            best_len = len(raw)
            best = el
    return best


def section_to_lines(elem: ET.Element, depth: int = 0) -> list[str]:
    """Walk ce:section tree: headings + paragraph text (namespace-agnostic)."""
    indent = "  " * depth
    name = localname(elem.tag)

    if name == "section":
        label_el = None
        title_el = None
        for child in elem:
            ln = localname(child.tag)
            if ln == "label":
                label_el = child
            elif ln == "section-title":
                title_el = child
        label = "".join(label_el.itertext()).strip() if label_el is not None else ""
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        if label or title:
            head = f"{label}. {title}".strip(". ")
            lines: list[str] = [f"{indent}{head}"]
        else:
            lines = []

        for child in elem:
            ln = localname(child.tag)
            if ln == "section":
                lines.extend(section_to_lines(child, depth + 1))
            elif ln in ("para", "simple-para"):
                raw = "".join(child.itertext())
                t = normalize_ws(re.sub(r"\s*\n\s*", " ", raw))
                if t:
                    lines.append(f"{indent}  {t}")
        return lines

    return []


def is_elsevier_entitlement_only(raw: str) -> bool:
    """Elsevier returns this for view=ENTITLED — not article XML."""
    return (
        "<document-entitlement" in raw
        and "<full-text-retrieval-response" not in raw
    )


def extract_body_readable(body: ET.Element) -> str:
    for child in body:
        if localname(child.tag) == "sections":
            block_lines: list[str] = []
            for sec in child:
                if localname(sec.tag) == "section":
                    block_lines.extend(section_to_lines(sec, depth=0))
            if block_lines:
                return "\n\n".join(block_lines)
    return normalize_ws("".join(body.itertext()))


def xml_to_readable_text(xml_bytes: str | bytes) -> str:
    raw = (
        xml_bytes.decode("utf-8")
        if isinstance(xml_bytes, (bytes, bytearray))
        else xml_bytes
    )
    if is_elsevier_entitlement_only(raw):
        return (
            "This XML is an Elsevier entitlement check only (no article content).\n"
            "Re-fetch with no view= parameter, or view=FULL — not view=ENTITLED.\n"
        )
    root = ET.fromstring(raw)
    meta = extract_coredata_fields(root)
    body = find_largest_body(root)

    parts: list[str] = []
    if meta.get("title"):
        parts.append(meta["title"])
    doi = meta.get("doi") or meta.get("prism_doi")
    if doi:
        parts.append(f"DOI: {doi.replace('doi:', '').strip()}")
    if meta.get("abstract"):
        parts.append("Abstract\n" + meta["abstract"])
    if body is not None:
        parts.append("Body\n" + extract_body_readable(body))
    else:
        parts.append("(No <body> found in XML — may be metadata-only response.)")

    return "\n\n".join(parts) + "\n"


def default_readable_output_path(xml_path: Path) -> Path:
    """e.g. test_output.xml -> test_output_readable.txt"""
    return xml_path.parent / f"{xml_path.stem}_readable.txt"


def convert_xml_file_to_text(
    xml_path: Path, output_path: Path | None = None
) -> Path:
    """Read Elsevier XML from disk and write plain text."""
    out = output_path or default_readable_output_path(xml_path)
    text = xml_to_readable_text(xml_path.read_text(encoding="utf-8"))
    out.write_text(text, encoding="utf-8")
    return out
