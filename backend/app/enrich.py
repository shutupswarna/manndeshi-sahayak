"""LLM-assisted chunking: sarvam-30b reads every chunk and writes retrieval metadata.

Boundaries stay structural (chunking.py); the model adds what a Marathi reader
would know about the chunk: a Marathi + English title, an English summary,
bilingual keywords (incl. romanized spellings) and questions people would ask.
All of it is indexed next to the original text, so a Hindi, English or
romanized-Marathi question can still land on a Marathi passage.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .chunking import Chunk

PROMPT = """तुम्ही माणदेशी फाउंडेशन/माण देशी महिला सहकारी बँकेच्या आर्थिक साक्षरता साहित्याचे तज्ज्ञ आहात.
You are given one CHUNK from a document page, plus the full PAGE for context.
Document: {doc} | Module: {module} | Page: {page}

<page>
{parent}
</page>
<chunk>
{chunk}
</chunk>

Return ONLY a JSON object, no prose:
{{"title_mr": "<short Marathi heading for the chunk>",
  "title_en": "<short English heading>",
  "summary_en": "<1-2 English sentences: what this chunk says, keep exact numbers/amounts/days>",
  "keywords": ["<6-12 search terms: Marathi words, English equivalents, romanized Marathi, scheme names, numbers>"],
  "questions_mr": ["<2-3 questions a rural woman entrepreneur might ask in Marathi that this chunk answers>"]}}
Use only facts present in the chunk. If the chunk is a form or blank template, say so in the summary."""


def enrich_chunk(client, model: str, chunk: Chunk, parent_text: str, doc: str, module: str) -> dict:
    msg = PROMPT.format(doc=doc, module=module, page=chunk.page,
                        parent=parent_text[:6000], chunk=chunk.content[:4000])
    data = client.chat_json(model, [{"role": "user", "content": msg}], max_tokens=700, reasoning=None)
    return {
        "title_mr": str(data.get("title_mr", ""))[:200],
        "title_en": str(data.get("title_en", ""))[:200],
        "summary_en": str(data.get("summary_en", ""))[:800],
        "keywords": [str(k)[:60] for k in (data.get("keywords") or [])][:15],
        "questions_mr": [str(q)[:300] for q in (data.get("questions_mr") or [])][:3],
    }


def enrich_all(client, model: str, chunks: list[Chunk], parents: dict[str, str], meta: dict[str, tuple[str, str]],
               cache_path: Path, workers: int = 6, log=print) -> dict[str, dict]:
    """Resumable: results are appended to ``cache_path`` (jsonl) as they arrive."""
    done: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["chunk_id"]] = r["enrichment"]
    todo = [c for c in chunks if c.chunk_id not in done]
    log(f"enrichment: {len(done)} cached, {len(todo)} to do with {model}")
    if not todo:
        return done
    failures = 0
    with ThreadPoolExecutor(workers) as pool, cache_path.open("a", encoding="utf-8") as f:
        futs = {pool.submit(enrich_chunk, client, model, c, parents[c.parent_id], *meta[c.doc_id]): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                e = fut.result()
            except Exception as err:  # keep going; the chunk is still indexed without enrichment
                failures += 1
                log(f"  ! {c.chunk_id}: {err}")
                continue
            done[c.chunk_id] = e
            f.write(json.dumps({"chunk_id": c.chunk_id, "enrichment": e}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                log(f"  {i}/{len(todo)}")
    log(f"enrichment finished: {len(done)} ok, {failures} failed")
    return done


def index_text(chunk: Chunk, e: dict | None) -> str:
    """Text that gets embedded and keyword-indexed for a chunk."""
    if not e:
        return chunk.index_text
    return "\n".join(filter(None, [
        chunk.breadcrumb,
        f"{e.get('title_mr', '')} / {e.get('title_en', '')}",
        e.get("summary_en", ""),
        "Keywords: " + ", ".join(e.get("keywords", [])),
        "प्रश्न: " + " | ".join(e.get("questions_mr", [])),
        chunk.content,
    ]))
