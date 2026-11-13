"""Conversation + long-term memory in SQLite.

* sessions / messages: every turn of every conversation (short-term context)
* facts: durable things a user told us about themselves (long-term memory),
  scoped per user id, viewable and deletable from the website.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, language TEXT, sources TEXT, rewritten TEXT, created REAL);
CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, user_id TEXT NOT NULL, fact TEXT NOT NULL, created REAL);
CREATE INDEX IF NOT EXISTS ix_msg_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS ix_fact_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS ix_sess_user ON sessions(user_id, updated);
"""

# never store identifiers, even if a user types them
_SENSITIVE = re.compile(r"(\b\d{4}\s?\d{4}\s?\d{4}\b|\b\d{9,18}\b|\b[A-Z]{5}\d{4}[A-Z]\b|\b\d{10}\b|otp|पासवर्ड|password|pin\b)", re.I)
_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def valid_id(x: str | None) -> bool:
    return bool(x and _ID.match(x))


class MemoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # ---- sessions -------------------------------------------------------
    def ensure_session(self, session_id: str | None, user_id: str) -> str:
        sid = session_id if valid_id(session_id) else uuid.uuid4().hex
        now = time.time()
        with self.lock:
            row = self.db.execute("SELECT user_id FROM sessions WHERE id=?", (sid,)).fetchone()
            if row and row["user_id"] != user_id:      # someone else's session id: start a fresh one
                sid = uuid.uuid4().hex
                row = None
            if not row:
                self.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, user_id, None, now, now))
                self.db.commit()
        return sid

    def add_message(self, session_id: str, role: str, content: str, language: str | None = None,
                    sources: list | None = None, rewritten: str | None = None) -> None:
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO messages(session_id,role,content,language,sources,rewritten,created) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (session_id, role, content, language, json.dumps(sources or [], ensure_ascii=False),
                             rewritten, now))
            self.db.execute("UPDATE sessions SET updated=?, title=COALESCE(title, ?) WHERE id=?",
                            (now, content[:80] if role == "user" else None, session_id))
            self.db.commit()

    def history(self, session_id: str, turns: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                                   (session_id, turns * 2)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def transcript(self, session_id: str, user_id: str) -> list[dict]:
        with self.lock:
            owner = self.db.execute("SELECT user_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not owner or owner["user_id"] != user_id:
                return []
            rows = self.db.execute("SELECT role, content, language, sources, created FROM messages "
                                   "WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
        return [{"role": r["role"], "content": r["content"], "language": r["language"],
                 "sources": json.loads(r["sources"] or "[]"), "created": r["created"]} for r in rows]

    def sessions(self, user_id: str, limit: int = 30) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, title, updated FROM sessions WHERE user_id=? AND title IS NOT NULL "
                                   "ORDER BY updated DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ---- long-term facts --------------------------------------------------
    def facts(self, user_id: str, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, fact, created FROM facts WHERE user_id=? ORDER BY created DESC LIMIT ?",
                                   (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def add_facts(self, user_id: str, facts: list[str], cap: int = 25) -> list[str]:
        existing = {f["fact"].strip().lower() for f in self.facts(user_id, 200)}
        added = []
        for f in facts:
            f = " ".join(str(f).split())[:240]
            if len(f) < 4 or f.lower() in existing or _SENSITIVE.search(f):
                continue
            existing.add(f.lower())
            added.append(f)
        if added:
            now = time.time()
            with self.lock:
                self.db.executemany("INSERT INTO facts VALUES (?,?,?,?)",
                                    [(uuid.uuid4().hex, user_id, f, now + i * 1e-3) for i, f in enumerate(added)])
                # keep only the newest `cap`
                self.db.execute("DELETE FROM facts WHERE user_id=? AND id NOT IN "
                                "(SELECT id FROM facts WHERE user_id=? ORDER BY created DESC LIMIT ?)",
                                (user_id, user_id, cap))
                self.db.commit()
        return added

    def delete_fact(self, user_id: str, fact_id: str) -> bool:
        with self.lock:
            cur = self.db.execute("DELETE FROM facts WHERE user_id=? AND id=?", (user_id, fact_id))
            self.db.commit()
        return cur.rowcount > 0

    def forget_user(self, user_id: str) -> None:
        with self.lock:
            sids = [r["id"] for r in self.db.execute("SELECT id FROM sessions WHERE user_id=?", (user_id,))]
            self.db.executemany("DELETE FROM messages WHERE session_id=?", [(s,) for s in sids])
            self.db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            self.db.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
            self.db.commit()
