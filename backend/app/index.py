"""The knowledge-base index: LanceDB on disk, three search legs in memory.

On disk (``data/index``):
  lancedb/            table ``chunks`` (text, metadata, enrichment, dense vector, sparse weights)
  parents.jsonl       page texts (what the LLM reads)
  manifest.json       build info

At startup everything is loaded into memory; the corpus is ~1k chunks, so
exact search with numpy is faster and simpler than an ANN index.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import sparse_score

# ---------------------------------------------------------------- tokenizing
_SPLIT = re.compile(r"[\s\.,;:!?।॥\(\)\[\]{}\"'“”‘’\-–—/\\|*#<>=_+&%@~`^]+")
# common Marathi/Hindi case suffixes and postpositions, longest first
_SUFFIXES = sorted("""च्या ची चा चे ची साठी करिता करीता मध्ये मधील मधे वर वरील ला ना ने नी नं ंना ांना ांनी ाला ाची ाचा ाचे ाच्या
ात ीत ेत त ून हून पासून पर्यंत पेक्षा शी ाशी ही ं ा ी े ो का की के को में से ने पर""".split(), key=len, reverse=True)


def tokenize(text: str) -> list[str]:
    """Script-safe tokenizer: never splits inside a Devanagari word (matras/virama kept)."""
    text = unicodedata.normalize("NFC", text.lower())
    out = []
    for tok in _SPLIT.split(text):
        if not tok or len(tok) < 2 and not tok.isdigit():
            continue
        out.append(tok)
        if re.search(r"[ऀ-ॿ]", tok):          # add a crude stem for inflected forms
            for s in _SUFFIXES:
                if tok.endswith(s) and len(tok) - len(s) >= 2:
                    out.append(tok[: -len(s)])
                    break
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


# ---------------------------------------------------------------- the index
@dataclass
class ChunkRow:
    chunk_id: str
    parent_id: str
    doc_id: str
    filename: str
    module: str
    page: int
    language: str
    heading: str
    content: str
    index_text: str
    enrichment: dict


class KnowledgeIndex:
    def __init__(self, rows: list[ChunkRow], dense: np.ndarray, sparse: list[dict[int, float]],
                 parents: dict[str, dict]):
        self.rows, self.dense, self.sparse, self.parents = rows, dense, sparse, parents
        self.bm25 = BM25([tokenize(r.index_text) for r in rows])
        self.by_id = {r.chunk_id: i for i, r in enumerate(rows)}

    # --- persistence ---------------------------------------------------------
    @staticmethod
    def save(path: Path, rows: list[ChunkRow], dense: np.ndarray, sparse: list[dict[int, float]],
             parents: dict[str, dict], manifest: dict) -> None:
        import lancedb
        import pyarrow as pa

        path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(path / "lancedb"))
        table = pa.table({
            "chunk_id": [r.chunk_id for r in rows], "parent_id": [r.parent_id for r in rows],
            "doc_id": [r.doc_id for r in rows], "filename": [r.filename for r in rows],
            "module": [r.module for r in rows], "page": [r.page for r in rows],
            "language": [r.language for r in rows], "heading": [r.heading for r in rows],
            "content": [r.content for r in rows], "index_text": [r.index_text for r in rows],
            "enrichment": [json.dumps(r.enrichment, ensure_ascii=False) for r in rows],
            "sparse": [json.dumps({str(k): round(v, 4) for k, v in s.items()}) for s in sparse],
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(dense.astype(np.float32).ravel()), dense.shape[1]),
        })
        db.create_table("chunks", table, mode="overwrite")
        with (path / "parents.jsonl").open("w", encoding="utf-8") as f:
            for pid, p in parents.items():
                f.write(json.dumps({"parent_id": pid, **p}, ensure_ascii=False) + "\n")
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeIndex":
        import lancedb

        df = lancedb.connect(str(path / "lancedb")).open_table("chunks").to_arrow().to_pylist()
        rows = [ChunkRow(d["chunk_id"], d["parent_id"], d["doc_id"], d["filename"], d["module"], d["page"],
                         d["language"], d["heading"], d["content"], d["index_text"], json.loads(d["enrichment"]))
                for d in df]
        dense = np.array([d["vector"] for d in df], dtype=np.float32)
        sparse = [{int(k): v for k, v in json.loads(d["sparse"]).items()} for d in df]
        parents = {}
        for line in (path / "parents.jsonl").read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            parents[p.pop("parent_id")] = p
        return cls(rows, dense, sparse, parents)

    # --- search legs (each returns chunk indices, best first) ------------------
    def dense_search(self, qv: np.ndarray, k: int) -> list[int]:
        return list(np.argsort(-(self.dense @ qv))[:k])

    def sparse_search(self, qs: dict[int, float], k: int) -> list[int]:
        s = np.array([sparse_score(qs, d) for d in self.sparse], dtype=np.float32)
        return [i for i in np.argsort(-s)[:k] if s[i] > 0]

    def bm25_search(self, text: str, k: int) -> list[int]:
        s = self.bm25.scores(tokenize(text))
        return [i for i in np.argsort(-s)[:k] if s[i] > 0]
