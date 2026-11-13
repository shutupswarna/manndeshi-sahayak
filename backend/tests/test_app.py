"""End-to-end tests with deterministic fakes (no network, no model downloads)."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_corpus, load_pages, approx_tokens  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embed import HashEmbedder  # noqa: E402
from app.enrich import index_text  # noqa: E402
from app.index import ChunkRow, KnowledgeIndex, tokenize  # noqa: E402
from app.memory import MemoryStore  # noqa: E402
from app.sarvam import SarvamError, extract_json  # noqa: E402


class FakeSarvam:
    """Router returns a rewrite that resolves 'त्याची' using history; answers cite [1]."""
    def __init__(self, fail_answer=False):
        self.calls, self.fail_answer = [], fail_answer

    def chat_json(self, model, messages, **kw):
        prompt = messages[0]["content"]
        self.calls.append(("router", model, prompt))
        msg = prompt.split("<<<")[1].split(">>>")[0]
        follow_up = "त्याची" in msg and "सिबील" in prompt
        return {"language": "en" if msg.isascii() else "mr",
                "standalone_question": "उद्योजिका कर्जासाठी सिबील स्कोर किती लागतो?" if follow_up else msg,
                "english_query": "Udyojika loan CIBIL score requirement",
                "needs_search": not msg.lower().startswith("hello"),
                "new_facts": ["runs a tailoring business in Satara"] if "शिवणकाम" in msg else
                             (["account number 123456789012"] if "खाते" in msg else [])}

    def chat(self, model, messages, **kw):
        self.calls.append(("answer", model, messages))
        if self.fail_answer:
            raise SarvamError("down")
        sys_prompt = messages[0]["content"]
        if "SOURCES" in sys_prompt:
            return "सिबील स्कोर किमान 700 असावा [1]." if "Marathi" in sys_prompt.split("Write the WHOLE")[1][:40] \
                else "The CIBIL score must be at least 700 [1]."
        return "नमस्कार! मी माणदेशी सहायक आहे."

    def translate(self, text, src, tgt, model=None):
        return f"[{tgt}] {text}"

    def stt(self, audio, filename, model=None, language_code=None):
        return {"transcript": "बचत गट म्हणजे काय?", "language_code": "mr-IN"}

    def tts(self, text, language_code, model=None, speaker=None):
        return b"RIFFfakewav"


def build_test_index(tmp: Path) -> KnowledgeIndex:
    pages = load_pages(ROOT / "data" / "corpus" / "pages.jsonl")
    pages = [p for p in pages if p.doc_id in ("sop_manndeshi_udyojika", "cgtmse_document")] + \
            [p for p in pages if p.language == "en"][:20]
    chunks = chunk_corpus(pages)
    parents = {f"{p.doc_id}#p{p.page}": p for p in pages}
    rows = [ChunkRow(c.chunk_id, c.parent_id, c.doc_id, parents[c.parent_id].filename, parents[c.parent_id].module,
                     c.page, c.language, " > ".join(c.heading_path), c.content, index_text(c, None), {})
            for c in chunks]
    emb = HashEmbedder()
    dense, sparse = emb.encode([r.index_text for r in rows])
    prow = {k: {"doc_id": p.doc_id, "filename": p.filename, "module": p.module, "page": p.page,
                "language": p.language, "text": p.text} for k, p in parents.items()}
    KnowledgeIndex.save(tmp / "index", rows, dense, sparse, prow, {"test": True})
    return KnowledgeIndex.load(tmp / "index")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.index = build_test_index(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def settings(self):
        os.environ["STATE_DIR"] = str(self.tmp / f"state_{self.id().split('.')[-1]}")
        os.environ["DATA_DIR"] = str(self.tmp)
        return Settings()


class TokenizerAndIndexTests(Base):
    def test_tokenizer_keeps_marathi_words_whole(self):
        toks = tokenize("कर्जाची परतफेड वेळेवर करा।")
        self.assertIn("कर्जाची", toks)
        self.assertIn("कर्ज", toks)          # crude stem added
        self.assertIn("परतफेड", toks)

    def test_bm25_finds_marathi_terms(self):
        hits = self.index.bm25_search("सिबील स्कोर", 5)
        self.assertTrue(any(self.index.rows[i].doc_id == "sop_manndeshi_udyojika" for i in hits))

    def test_index_round_trip(self):
        self.assertGreater(len(self.index.rows), 10)
        self.assertEqual(self.index.dense.shape[0], len(self.index.rows))

    def test_extract_json_handles_fences(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})


class ChatFlowTests(Base):
    def service(self, sarvam=None):
        from app.chat import ChatService
        from app.retrieval import Retriever
        s = self.settings()
        mem = MemoryStore(s.db_path)
        sv = sarvam or FakeSarvam()
        return ChatService(s, sv, Retriever(self.index, HashEmbedder(), s), mem), sv, mem

    def test_answer_has_sources_and_language(self):
        svc, sv, _ = self.service()
        r = svc.turn("user_00000001", None, "उद्योजिका कर्जासाठी सिबील स्कोर किती लागतो?")
        self.assertEqual(r.language, "mr")
        self.assertTrue(r.sources)
        self.assertIn("[1]", r.answer)
        self.assertEqual([c[1] for c in sv.calls], ["sarvam-105b-conversations", "sarvam-105b-conversations"])

    def test_follow_up_is_rewritten_using_history(self):
        svc, _, mem = self.service()
        r1 = svc.turn("user_00000002", None, "उद्योजिका कर्जासाठी सिबील स्कोर किती लागतो?")
        r2 = svc.turn("user_00000002", r1.session_id, "आणि त्याची फी किती?")
        self.assertEqual(r2.session_id, r1.session_id)
        self.assertIn("सिबील", r2.rewritten)
        self.assertEqual(len(mem.transcript(r1.session_id, "user_00000002")), 4)

    def test_long_term_memory_is_stored_and_reused(self):
        svc, sv, mem = self.service()
        svc.turn("user_00000003", None, "मी साताऱ्यात शिवणकाम करते, कर्ज कसे मिळेल?")
        self.assertEqual([f["fact"] for f in mem.facts("user_00000003")], ["runs a tailoring business in Satara"])
        svc.turn("user_00000003", None, "बचत कशी करावी?")          # new session, same user
        router_prompt = [c for c in sv.calls if c[0] == "router"][-1][2]
        self.assertIn("tailoring business", router_prompt)

    def test_sensitive_facts_are_not_stored(self):
        svc, _, mem = self.service()
        svc.turn("user_00000004", None, "माझे खाते क्रमांक सांगू का?")
        self.assertEqual(mem.facts("user_00000004"), [])

    def test_memory_is_isolated_between_users(self):
        svc, _, mem = self.service()
        r = svc.turn("user_00000005", None, "मी शिवणकाम करते")
        self.assertEqual(mem.transcript(r.session_id, "someone_else_1"), [])
        r2 = svc.turn("someone_else_1", r.session_id, "hello")    # cannot hijack another user's session
        self.assertNotEqual(r2.session_id, r.session_id)
        self.assertEqual(mem.facts("someone_else_1"), [])

    def test_english_detection(self):
        from app.chat import _looks_english
        self.assertTrue(_looks_english("How many days does the bank take to disburse the loan?"))
        self.assertFalse(_looks_english("Self help group ke kya fayde hain?"))
        self.assertFalse(_looks_english("बचत गट म्हणजे काय?"))

    def test_greeting_skips_search(self):
        svc, _, _ = self.service()
        r = svc.turn("user_00000006", None, "hello there")
        self.assertEqual(r.sources, [])

    def test_answer_failure_returns_polite_message(self):
        svc, _, _ = self.service(FakeSarvam(fail_answer=True))
        r = svc.turn("user_00000007", None, "बचत कशी करावी?")
        self.assertIn("माफ करा", r.answer)


class ApiTests(Base):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app import main
        main.state.chat = None
        main.build_services(self.settings(), sarvam=FakeSarvam(), embedder=HashEmbedder(), index=self.index)
        self.client = TestClient(main.app)

    def test_health_and_chat(self):
        self.assertTrue(self.client.get("/api/health").json()["ok"])
        r = self.client.post("/api/chat", json={"user_id": "web_user_01", "message": "What is the CIBIL score needed?",
                                                "language": "en"}).json()
        self.assertEqual(r["language"], "en")
        self.assertTrue(r["sources"])
        s = self.client.get("/api/sessions", params={"user_id": "web_user_01"}).json()["sessions"]
        self.assertEqual(len(s), 1)

    def test_memory_endpoints(self):
        self.client.post("/api/chat", json={"user_id": "web_user_02", "message": "मी शिवणकाम करते"})
        facts = self.client.get("/api/memory", params={"user_id": "web_user_02"}).json()["facts"]
        self.assertEqual(len(facts), 1)
        self.client.delete(f"/api/memory/{facts[0]['id']}", params={"user_id": "web_user_02"})
        self.assertEqual(self.client.get("/api/memory", params={"user_id": "web_user_02"}).json()["facts"], [])

    def test_bad_user_id_rejected(self):
        r = self.client.post("/api/chat", json={"user_id": "x", "message": "hi"})
        self.assertEqual(r.status_code, 400)

    def test_tts_and_stt(self):
        self.assertEqual(self.client.post("/api/tts", json={"text": "नमस्कार [1]", "language": "mr"}).content,
                         b"RIFFfakewav")
        r = self.client.post("/api/stt", files={"file": ("a.wav", b"RIFF....", "audio/wav")}, data={"language": "mr"})
        self.assertEqual(r.json()["language_code"], "mr-IN")


if __name__ == "__main__":
    unittest.main()
