"""Structure-aware chunking for OCR'd Markdown pages (Marathi + English).

Strategy (see README "Chunking"):
  * parent = one page (what the LLM reads; keeps [doc, page] citations exact)
  * children = heading sections within the page, split further only if too long
  * tables are never split; sentence splits respect । ॥ ۔ . ? !
  * small adjacent sections are merged up to the budget
  * every child gets a free breadcrumb header: document > module > heading
  * no overlap (Nalanda-Lite sweep and Jan-2026 study both favoured 0%)

Token counting is pluggable: pass the embedder's tokenizer as ``count``;
the default is a character-based estimate so this module has no dependencies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_LINE = re.compile(r"^\s*\|")
SENT_END = re.compile(r"(?<=[।॥\.\?\!۔])\s+")


def approx_tokens(text: str) -> int:
    """Rough XLM-R/bge-m3 estimate: Devanagari ~1 token per 2.5 chars, Latin ~1 per 4."""
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    return int(dev / 2.5 + (len(text) - dev) / 4) + 1


@dataclass
class Page:
    doc_id: str
    filename: str
    module: str
    language: str
    page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    language: str
    heading_path: list[str]
    breadcrumb: str          # prepended for embedding/keyword search
    content: str             # original text only
    parent_id: str           # = f"{doc_id}#p{page}"
    meta: dict = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        return f"{self.breadcrumb}\n{self.content}"


IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)|\bimg-\d+\.(?:jpe?g|png)\b")


def clean_ocr(text: str) -> str:
    """Drop OCR image placeholders and empty table rows."""
    text = IMG.sub("", text)
    text = re.sub(r"^\s*(\|\s*)+\|?\s*$", "", text, flags=re.M)       # | | | rows with no content
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_pages(path: str | Path, include_junk: bool = False) -> list[Page]:
    pages = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("is_junk") and not include_junk:
            continue
        text = clean_ocr(r["text"])
        if len(text) < 30 and not include_junk:
            continue
        pages.append(Page(r["doc_id"], r["filename"], r.get("module") or "", r["language"], r["page"], text))
    return pages


def _blocks(text: str) -> list[tuple[list[str], str]]:
    """Split a page into (heading_path, body) sections, keeping tables intact."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            level, title = len(m[1]), m[2]
            path = [p for p in path if p[0] < level] + [(level, title)]
            sections.append(([t for _, t in path], []))
        else:
            sections[-1][1].append(line)
    return [(hp, "\n".join(body).strip()) for hp, body in sections if "\n".join(body).strip() or hp]


def _split_long(body: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Split an over-budget section: paragraphs -> sentences; table runs stay whole."""
    units, buf, in_table = [], [], False
    for line in body.splitlines():
        is_t = bool(TABLE_LINE.match(line))
        if is_t != in_table and buf:
            units.append("\n".join(buf)); buf = []
        in_table = is_t
        if not is_t and not line.strip():
            if buf: units.append("\n".join(buf)); buf = []
            continue
        buf.append(line)
    if buf: units.append("\n".join(buf))

    pieces = []
    for u in units:
        if count(u) <= budget or TABLE_LINE.match(u):
            pieces.append(u)
        else:
            pieces += [s for s in SENT_END.split(u) if s.strip()]
    out, cur = [], ""
    for p in pieces:
        cand = f"{cur}\n{p}" if cur else p
        if cur and count(cand) > budget:
            out.append(cur); cur = p
        else:
            cur = cand
    if cur: out.append(cur)
    return out


def chunk_page(page: Page, budget: int = 512, min_tokens: int = 60,
               count: Callable[[str], int] = approx_tokens) -> list[Chunk]:
    raw: list[tuple[list[str], str]] = []
    for hp, body in _blocks(page.text):
        text = body if body else ""
        if hp and not body:          # bare heading: carry it into the next section's path only
            continue
        for part in (_split_long(text, budget, count) if count(text) > budget else [text]):
            raw.append((hp, part))

    merged: list[tuple[list[str], str]] = []
    for hp, txt in raw:              # merge tiny neighbours so chunks aren't fragments
        if merged and (count(merged[-1][1]) < min_tokens or count(txt) < min_tokens) \
                and count(merged[-1][1] + txt) <= budget:
            php, ptxt = merged[-1]
            head = f"## {hp[-1]}\n" if hp and hp != php else ""
            merged[-1] = (php or hp, f"{ptxt}\n{head}{txt}".strip())
        else:
            merged.append((hp, txt))

    chunks = []
    for i, (hp, txt) in enumerate(merged):
        crumb = " > ".join([page.filename.rsplit(".", 1)[0], page.module, *hp][: 2 + len(hp)])
        chunks.append(Chunk(
            chunk_id=f"{page.doc_id}#p{page.page}#c{i}", doc_id=page.doc_id, page=page.page,
            language=page.language, heading_path=hp, breadcrumb=f"[{crumb} | p.{page.page}]",
            content=txt, parent_id=f"{page.doc_id}#p{page.page}"))
    return chunks


def chunk_corpus(pages: Iterable[Page], **kw) -> list[Chunk]:
    return [c for p in pages for c in chunk_page(p, **kw)]
