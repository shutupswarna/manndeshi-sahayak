"""Build the knowledge index: chunk -> sarvam-30b enrichment -> bge-m3 embeddings -> LanceDB.

    python -m scripts.build_index                 # full build (resumes enrichment from cache)
    python -m scripts.build_index --no-enrich     # skip the LLM step (structure + breadcrumb only)
    python -m scripts.build_index --limit 40      # quick trial on the first 40 chunks
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_corpus, load_pages  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.enrich import enrich_all, index_text  # noqa: E402
from app.index import ChunkRow, KnowledgeIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fake-embed", action="store_true", help="hash embeddings (tests only)")
    a = ap.parse_args()
    s = get_settings()
    t0 = time.time()

    pages = load_pages(s.corpus_path)
    embedder = None
    if not a.fake_embed:
        from app.embed import BGEM3
        embedder = BGEM3(s.embed_model)
        count = lambda t: len(embedder.tok.encode(t, add_special_tokens=False))  # noqa: E731
    else:
        from app.chunking import approx_tokens as count
        from app.embed import HashEmbedder
        embedder = HashEmbedder()
    chunks = chunk_corpus(pages, count=count)
    if a.limit:
        chunks = chunks[: a.limit]
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    parents = {f"{p.doc_id}#p{p.page}": p for p in pages}
    meta = {p.doc_id: (p.filename, p.module) for p in pages}
    enrich = {}
    if not a.no_enrich:
        from app.sarvam import SarvamClient
        client = SarvamClient(s.sarvam_api_key, s.sarvam_base_url)
        cache = s.index_dir / "enrichment.jsonl"
        s.index_dir.mkdir(parents=True, exist_ok=True)
        enrich = enrich_all(client, s.chunk_model, chunks, {k: v.text for k, v in parents.items()},
                            meta, cache, workers=a.workers)

    rows = []
    for c in chunks:
        e = enrich.get(c.chunk_id) or {}
        p = parents[c.parent_id]
        rows.append(ChunkRow(c.chunk_id, c.parent_id, c.doc_id, p.filename, p.module, c.page, c.language,
                             " > ".join(c.heading_path) or e.get("title_mr", ""), c.content, index_text(c, e), e))
    print("embedding…")
    dense, sparse = embedder.encode([r.index_text for r in rows], batch_size=8)
    used = {r.parent_id for r in rows}
    parent_rows = {k: {"doc_id": p.doc_id, "filename": p.filename, "module": p.module, "page": p.page,
                       "language": p.language, "text": p.text} for k, p in parents.items() if k in used}
    KnowledgeIndex.save(s.index_dir, rows, dense, sparse, parent_rows, {
        "built": dt.datetime.now(dt.timezone.utc).isoformat(), "chunks": len(rows), "pages": len(parent_rows),
        "documents": len({r.doc_id for r in rows}), "enriched": sum(1 for r in rows if r.enrichment),
        "chunk_model": None if a.no_enrich else s.chunk_model,
        "embed_model": "hash" if a.fake_embed else s.embed_model, "seconds": round(time.time() - t0, 1)})
    print(f"done in {time.time() - t0:.0f}s -> {s.index_dir}")


if __name__ == "__main__":
    main()
