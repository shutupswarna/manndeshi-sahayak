"""Thin Sarvam AI client (chat, translate, speech-to-text, text-to-speech).

Uses plain httpx so the app doesn't depend on SDK versions. Every method is
also implemented by ``FakeSarvam`` in tests.
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

import httpx


class SarvamError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model reply (tolerates ```json fences)."""
    if text is None:
        raise ValueError("empty model reply")
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for open_, close in (("{", "}"), ("[", "]")):
        s, e = t.find(open_), t.rfind(close)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in: {text[:200]!r}")


class RateLimiter:
    """Thread-safe sliding-window limiter (Sarvam 105B: 60 req/min on pay-as-you-go)."""
    def __init__(self, per_minute: int):
        import collections, threading
        self.per_minute, self.lock, self.calls = per_minute, threading.Lock(), collections.deque()

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                delay = 60 - (now - self.calls[0]) + 0.05
            time.sleep(delay)


class SarvamClient:
    def __init__(self, api_key: str, base_url: str = "https://api.sarvam.ai", timeout: float = 90.0,
                 llm_per_minute: int = 50):
        if not api_key:
            raise SarvamError("SARVAM_API_KEY is not set")
        self.base = base_url.rstrip("/")
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.llm_limiter = RateLimiter(llm_per_minute)
        # Docs show api-subscription-key; some guides show Bearer for chat. Sending both is harmless.
        self.http = httpx.Client(timeout=timeout, headers={
            "api-subscription-key": api_key, "Authorization": f"Bearer {api_key}"})

    def _post(self, path: str, retries: int = 5, **kw) -> httpx.Response:
        for attempt in range(retries):
            if path.startswith("/v1/chat"):
                self.llm_limiter.wait()
            r = self.http.post(f"{self.base}{path}", **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                ra = r.headers.get("retry-after")
                time.sleep(float(ra) if ra and ra.replace(".", "").isdigit() else min(30, 2 ** attempt * 2))
                continue
            if r.status_code >= 400:
                raise SarvamError(f"{path} {r.status_code}: {r.text[:300]}")
            return r
        raise SarvamError(f"{path} failed after {retries} attempts")

    # ---- chat -------------------------------------------------------------
    def chat(self, model: str, messages: list[dict], temperature: float = 0.2,
             max_tokens: int = 1200, reasoning: str | None = None) -> str:
        body: dict[str, Any] = {"model": model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens}
        # Reasoning is ON by default for Sarvam chat; send null to switch it off.
        body["reasoning_effort"] = None if reasoning in (None, "none", "") else reasoning
        data = self._post("/v1/chat/completions", json=body).json()
        u = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("prompt_tokens") or 0
        self.usage["completion_tokens"] += u.get("completion_tokens") or 0
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        if not content:  # all tokens went to thinking
            raise SarvamError("model returned no visible content (raise max_tokens or lower reasoning)")
        return content.strip()

    def chat_json(self, model: str, messages: list[dict], **kw) -> Any:
        return extract_json(self.chat(model, messages, temperature=0.0, **kw))

    # ---- translate --------------------------------------------------------
    def translate(self, text: str, source: str, target: str, model: str = "sarvam-translate:v1") -> str:
        if source == target or not text.strip():
            return text
        out = []
        for piece in _split_for_translate(text, 1900):
            r = self._post("/translate", json={"input": piece, "source_language_code": source,
                                               "target_language_code": target, "model": model})
            out.append(r.json()["translated_text"])
        return "\n".join(out)

    # ---- speech -----------------------------------------------------------
    def stt(self, audio: bytes, filename: str, model: str = "saaras:v3", language_code: str | None = None) -> dict:
        data = {"model": model, "mode": "transcribe"}
        if language_code:
            data["language_code"] = language_code
        r = self._post("/speech-to-text", data=data, files={"file": (filename, audio)})
        j = r.json()
        return {"transcript": j.get("transcript", ""), "language_code": j.get("language_code")}

    def tts(self, text: str, language_code: str, model: str = "bulbul:v3", speaker: str = "priya") -> bytes:
        chunks = []
        for piece in _split_for_translate(text, 1400):
            r = self._post("/text-to-speech", json={"text": piece, "target_language_code": language_code,
                                                     "model": model, "speaker": speaker})
            chunks += [base64.b64decode(a) for a in r.json().get("audios", [])]
        return _concat_wav(chunks) if len(chunks) > 1 else (chunks[0] if chunks else b"")


def _split_for_translate(text: str, limit: int) -> list[str]:
    parts, cur = [], ""
    for sent in re.split(r"(?<=[।॥\.\?\!\n])\s*", text):
        if len(cur) + len(sent) + 1 > limit and cur:
            parts.append(cur); cur = ""
        cur = f"{cur} {sent}".strip()
        while len(cur) > limit:
            parts.append(cur[:limit]); cur = cur[limit:]
    if cur:
        parts.append(cur)
    return parts or [""]


def _concat_wav(chunks: list[bytes]) -> bytes:
    """Join WAV files that share a format (Bulbul returns 16-bit PCM WAV)."""
    import io
    import wave
    out = io.BytesIO()
    with wave.open(io.BytesIO(chunks[0])) as first:
        params = first.getparams()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        for c in chunks:
            with wave.open(io.BytesIO(c)) as r:
                w.writeframes(r.readframes(r.getnframes()))
    return out.getvalue()
