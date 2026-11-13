"""Runtime settings, all overridable with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY", ""))
    sarvam_base_url: str = field(default_factory=lambda: _env("SARVAM_BASE_URL", "https://api.sarvam.ai"))

    # Models
    chunk_model: str = field(default_factory=lambda: _env("CHUNK_MODEL", "sarvam-105b-conversations"))      # index-time enrichment
    router_model: str = field(default_factory=lambda: _env("ROUTER_MODEL", "sarvam-105b-conversations"))    # query rewrite + memory
    answer_model: str = field(default_factory=lambda: _env("ANSWER_MODEL", "sarvam-105b-conversations"))   # final answer
    answer_reasoning: str = field(default_factory=lambda: _env("ANSWER_REASONING", "none"))   # low|medium|high|none
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "saaras:v3"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "bulbul:v3"))
    tts_speaker: str = field(default_factory=lambda: _env("TTS_SPEAKER", "priya"))
    translate_model: str = field(default_factory=lambda: _env("TRANSLATE_MODEL", "sarvam-translate:v1"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-m3"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", "0") == "1")

    # Retrieval
    leg_weights: tuple = (("dense", 1.0), ("sparse", 0.8), ("bm25", 0.8))
    rrf_k: int = 60
    leg_limit: int = 40
    top_parents: int = 4
    history_turns: int = 6
    max_facts: int = 25

    # Storage
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    state_dir: Path = field(default_factory=lambda: Path(_env("STATE_DIR", str(ROOT / "data" / "state"))))

    # Web
    allowed_origins: str = field(default_factory=lambda: _env("ALLOWED_ORIGINS", "*"))

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus" / "pages.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "memory.sqlite3"


def get_settings() -> Settings:
    return Settings()
