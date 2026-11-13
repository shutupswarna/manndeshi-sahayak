"""Retrieval quality on the golden set: recall@k / MRR against page references.

    python -m scripts.eval_retrieval                  # dense+sparse+bm25 on raw question only
    python -m scripts.eval_retrieval --legs dense     # one leg at a time
    python -m scripts.eval_retrieval --english        # add an English query (uses the router LLM)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.embed import BGEM3  # noqa: E402
from app.index import KnowledgeIndex  # noqa: E402
from app.retrieval import Retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="dense,sparse,bm25")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--english", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    s = get_settings()
    idx = KnowledgeIndex.load(s.index_dir)
    emb = BGEM3(s.embed_model)
    weights = {k: v for k, v in s.leg_weights}
    legs = set(a.legs.split(","))
    r = Retriever(idx, emb, s)
    r.weights = {k: (weights[k] if k in legs else 0.0) for k in weights}

    router = None
    if a.english:
        from app.chat import ChatService
        from app.memory import MemoryStore
        from app.sarvam import SarvamClient
        router = ChatService(s, SarvamClient(s.sarvam_api_key), r, MemoryStore(Path("/tmp/eval_mem.sqlite3")))

    gold = [json.loads(l) for l in (Path(__file__).parent.parent / "tests" / "golden_eval.jsonl").open()]
    if a.limit:
        gold = gold[: a.limit]
    have = set(idx.parents)
    stats = defaultdict(lambda: [0, 0, 0.0])        # lang -> [n, hits, rr]
    skipped = 0
    t0 = time.time()
    for g in gold:
        want = {ref.replace("#", "#p") for ref in g["page_refs"]}
        if not want & have:                          # page dropped (image-only) -> can't be retrieved
            skipped += 1
            continue
        queries = [g["question"]]
        if router:
            queries.append(router.understand(g["question"], [], [], "auto")["english_query"])
        got = [src.parent_id for src in r.search(queries, top_parents=a.k)]
        rank = next((i + 1 for i, p in enumerate(got) if p in want), None)
        for key in (g["language"], "all"):
            st = stats[key]
            st[0] += 1
            st[1] += rank is not None
            st[2] += 1 / rank if rank else 0
    print(f"legs={sorted(legs)} english={a.english} k={a.k}  ({time.time() - t0:.0f}s, {skipped} skipped)")
    for key in ("all", "en", "mr"):
        n, h, rr = stats[key]
        if n:
            print(f"  {key:3s} n={n:3d}  recall@{a.k}={h / n:.3f}  mrr@{a.k}={rr / n:.3f}")


if __name__ == "__main__":
    main()
