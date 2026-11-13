"""bge-m3 dense + sparse encoder, implemented directly on transformers.

Mirrors FlagEmbedding's BGEM3 (CLS-pooled normalized dense vector; ReLU'd
per-token weights from ``sparse_linear.pt`` for lexical matching) without
depending on FlagEmbedding, whose classes break on recent transformers.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int
    def encode(self, texts: list[str], batch_size: int = 16) -> tuple[np.ndarray, list[dict[int, float]]]: ...


class BGEM3:
    dim = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, max_length: int = 1024):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.sparse = torch.nn.Linear(self.model.config.hidden_size, 1)
        state = torch.load(hf_hub_download(model_name, "sparse_linear.pt"), map_location="cpu")
        self.sparse.load_state_dict(state)
        self.sparse.to(self.device).eval()
        self.max_length = max_length
        self.skip = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}

    def encode(self, texts, batch_size: int = 16):
        torch = self.torch
        dense_out, sparse_out = [], []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                dense = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
                weights = torch.relu(self.sparse(hidden)).squeeze(-1)
                dense_out.append(dense.float().cpu().numpy())
                for ids, w in zip(batch["input_ids"].cpu().tolist(), weights.float().cpu().tolist()):
                    lex: dict[int, float] = defaultdict(float)
                    for t, v in zip(ids, w):
                        if t not in self.skip and v > 0:
                            lex[t] = max(lex[t], v)
                    sparse_out.append(dict(lex))
        return np.vstack(dense_out), sparse_out


class HashEmbedder:
    """Deterministic, dependency-free stand-in for tests (bag of character trigrams)."""
    dim = 256

    def encode(self, texts, batch_size: int = 16):
        dense, sparse = [], []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            lex: dict[int, float] = {}
            words = [w for w in t.lower().split() if w]
            for w in words:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                lex[h % 250000] = lex.get(h % 250000, 0) + 1.0
                for j in range(max(1, len(w) - 2)):
                    g = int(hashlib.md5(w[j:j + 3].encode()).hexdigest(), 16)
                    v[g % self.dim] += 1.0
            n = math.sqrt(float((v ** 2).sum())) or 1.0
            dense.append(v / n)
            sparse.append(lex)
        return np.vstack(dense), sparse


def sparse_score(q: dict[int, float], d: dict[int, float]) -> float:
    if len(q) > len(d):
        q, d = d, q
    return sum(w * d.get(t, 0.0) for t, w in q.items())
