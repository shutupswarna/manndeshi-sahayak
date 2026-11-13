"""FastAPI app: chat, sessions, memory, speech. Run: uvicorn app.main:app --port 7860"""
from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .chat import LANGS, ChatService, as_dict
from .config import Settings, get_settings
from .memory import MemoryStore, valid_id
from .sarvam import SarvamError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sahayak")


class State:
    settings: Settings
    chat: ChatService | None = None
    memory: MemoryStore | None = None
    sarvam = None
    index_info: dict = {}
    error: str | None = None


state = State()


def build_services(settings: Settings, sarvam=None, embedder=None, index=None) -> None:
    """Wire everything. Tests pass fakes; production loads the real index and models."""
    from .index import KnowledgeIndex
    from .retrieval import Retriever, load_reranker

    state.settings = settings
    state.memory = MemoryStore(settings.db_path)
    if sarvam is None:
        from .sarvam import SarvamClient
        sarvam = SarvamClient(settings.sarvam_api_key, settings.sarvam_base_url)
    if index is None:
        index = KnowledgeIndex.load(settings.index_dir)
    if embedder is None:
        from .embed import BGEM3
        embedder = BGEM3(settings.embed_model)
    reranker = load_reranker(settings.rerank_model) if settings.use_reranker else None
    state.sarvam = sarvam
    state.chat = ChatService(settings, sarvam, Retriever(index, embedder, settings, reranker), state.memory)
    state.index_info = {"chunks": len(index.rows), "pages": len(index.parents),
                        "documents": len({r.doc_id for r in index.rows})}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.chat is None:          # not pre-wired by tests
        try:
            build_services(get_settings())
            log.info("ready: %s", state.index_info)
        except Exception as e:      # keep serving /api/health so the problem is visible
            state.error = f"{type(e).__name__}: {e}"
            log.exception("startup failed")
    yield


app = FastAPI(title="Manndeshi Sahayak API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in get_settings().allowed_origins.split(",")],
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

# ---------------------------------------------------------------- helpers
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, per_minute: int = 20):
    key = request.headers.get("x-user-id") or (request.client.host if request.client else "anon")
    q, now = _hits[key], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= per_minute:
        raise HTTPException(429, "Too many requests, please wait a minute.")
    q.append(now)


def ready() -> ChatService:
    if state.chat is None:
        raise HTTPException(503, f"Service not ready: {state.error or 'starting'}")
    return state.chat


def check_user(user_id: str) -> str:
    if not valid_id(user_id):
        raise HTTPException(400, "invalid user_id")
    return user_id


# ---------------------------------------------------------------- routes
@app.get("/")
def root():
    return {"name": "Manndeshi Sahayak API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    s = getattr(state, "settings", None)
    return {"ok": state.chat is not None, "error": state.error, "index": state.index_info,
            "models": {"chunking": s.chunk_model, "router": s.router_model, "answer": s.answer_model,
                       "reranker": s.use_reranker} if s else {}}


class ChatIn(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    language: str = "auto"          # auto | mr | hi | en


@app.post("/api/chat", dependencies=[Depends(rate_limit)])
def chat(body: ChatIn):
    svc = ready()
    check_user(body.user_id)
    pref = body.language if body.language in LANGS else "auto"
    try:
        return as_dict(svc.turn(body.user_id, body.session_id, body.message.strip(), pref))
    except SarvamError as e:
        log.error("chat failed: %s", e)
        raise HTTPException(502, "The language model is unavailable right now. Please try again.")


@app.get("/api/sessions")
def sessions(user_id: str):
    ready()
    return {"sessions": state.memory.sessions(check_user(user_id))}


@app.get("/api/sessions/{session_id}")
def session(session_id: str, user_id: str):
    ready()
    return {"messages": state.memory.transcript(session_id, check_user(user_id))}


@app.get("/api/memory")
def memory(user_id: str):
    ready()
    return {"facts": state.memory.facts(check_user(user_id))}


@app.delete("/api/memory/{fact_id}")
def forget_fact(fact_id: str, user_id: str):
    ready()
    if not state.memory.delete_fact(check_user(user_id), fact_id):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.delete("/api/memory")
def forget_all(user_id: str):
    ready()
    state.memory.forget_user(check_user(user_id))
    return {"ok": True}


# ---------------------------------------------------------------- speech
def _to_wav(data: bytes, filename: str) -> tuple[bytes, str]:
    """Browsers record webm/ogg; convert to 16 kHz mono WAV when ffmpeg is available."""
    if filename.lower().endswith(".wav") or not shutil.which("ffmpeg"):
        return data, filename
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / Path(filename).name, Path(d) / "out.wav"
        src.write_bytes(data)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000",
                            str(dst)], capture_output=True, timeout=60)
        if r.returncode != 0:
            raise HTTPException(400, "Could not read that audio.")
        return dst.read_bytes(), "audio.wav"


@app.post("/api/stt", dependencies=[Depends(rate_limit)])
async def stt(file: UploadFile = File(...), language: str = Form("auto")):
    ready()
    data = await file.read()
    if len(data) > 8_000_000:
        raise HTTPException(413, "Recording too long (max ~30 seconds).")
    wav, name = _to_wav(data, file.filename or "audio.webm")
    code = LANGS[language][2] if language in LANGS else None
    try:
        return state.sarvam.stt(wav, name, state.settings.stt_model, code)
    except SarvamError as e:
        log.error("stt failed: %s", e)
        raise HTTPException(502, "Speech recognition failed.")


class TTSIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = "mr"


def _speakable(text: str) -> str:
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)          # citations
    text = re.sub(r"[*_#`>|]+", " ", text)                     # markdown
    return re.sub(r"\s+", " ", text).strip()


@app.post("/api/tts", dependencies=[Depends(rate_limit)])
def tts(body: TTSIn):
    ready()
    lang = body.language if body.language in LANGS else "mr"
    try:
        audio = state.sarvam.tts(_speakable(body.text), LANGS[lang][2], state.settings.tts_model,
                                 state.settings.tts_speaker)
    except SarvamError as e:
        log.error("tts failed: %s", e)
        raise HTTPException(502, "Text to speech failed.")
    return Response(audio, media_type="audio/wav")
