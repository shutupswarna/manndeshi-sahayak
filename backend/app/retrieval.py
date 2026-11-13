"""Hybrid retrieval: dense + sparse + BM25 on every query variant, fused with weighted RRF,
optionally reranked, then lifted to parent pages (small-to-big)."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embed import Embedder
from .index import KnowledgeIndex


@dataclass
class Source:
    n: int                 # [n] label used in the answer
    parent_id: str
    filename: str
    module: str
    page: int
    language: str
    text: str              # full page text given to the LLM
    matched: str           # the chunk that matched (shown as a preview)
    heading: str
    score: float


def rrf(legs: list[tuple[str, list[int]]], weights: dict[str, float], k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for leg, ids in legs:
        w = weights.get(leg, 1.0)
        for rank, i in enumerate(ids, 1):
            scores[i] = scores.get(i, 0.0) + w / (k + rank)
    return scores


class Retriever:
    def __init__(self, index: KnowledgeIndex, embedder: Embedder, settings: Settings, reranker=None):
        self.index, self.embedder, self.s, self.reranker = index, embedder, settings, reranker
        self.weights = dict(settings.leg_weights)

    def search(self, queries: list[str], top_parents: int | None = None) -> list[Source]:
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        if not queries:
            return []
        dense, sparse = self.embedder.encode(queries)
        legs = []
        for i, q in enumerate(queries):
            legs += [("dense", self.index.dense_search(dense[i], self.s.leg_limit)),
                     ("sparse", self.index.sparse_search(sparse[i], self.s.leg_limit)),
                     ("bm25", self.index.bm25_search(q, self.s.leg_limit))]
        fused = rrf(legs, self.weights, self.s.rrf_k)
        ranked = sorted(fused, key=fused.get, reverse=True)[:30]

        if self.reranker is not None and ranked:
            texts = [self.index.rows[i].index_text for i in ranked[:12]]
            rs = self.reranker(queries[0], texts)
            ranked = [i for _, i in sorted(zip(rs, ranked[:12]), key=lambda x: -x[0])] + ranked[12:]

        out, seen = [], {}
        for i in ranked:
            row = self.index.rows[i]
            if row.parent_id in seen:
                continue
            p = self.index.parents.get(row.parent_id, {})
            seen[row.parent_id] = True
            out.append(Source(n=len(out) + 1, parent_id=row.parent_id, filename=row.filename, module=row.module,
                              page=row.page, language=row.language, text=p.get("text", row.content),
                              matched=row.content, heading=row.heading, score=round(fused[i], 5)))
            if len(out) >= (top_parents or self.s.top_parents):
                break
        return out


def load_reranker(model_name: str):
    """Cross-encoder via sentence-transformers (FlagEmbedding's reranker breaks on transformers 5.x)."""
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, max_length=512)
    return lambda q, docs: list(ce.predict([(q, d) for d in docs]))
