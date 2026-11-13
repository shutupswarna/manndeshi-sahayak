# माणदेशी सहायक · Manndeshi Sahayak

A Marathi-first RAG chatbot for Mann Deshi's financial-literacy training material
and Mann Deshi Mahila Sahakari Bank's loan documents. You can ask in Marathi,
Hindi or English, by typing or speaking. Answers come back in the same language,
cite their sources, and the bot remembers useful context about each user.

- **Website:** GitHub Pages, served from `web/`
- **API:** Hugging Face Space (Docker), built from `backend/`

```
browser (GitHub Pages) ──► FastAPI on HF Space ──► Sarvam AI (LLM, speech, translation)
                               │
                               ├── LanceDB  : 938 chunks + bge-m3 vectors + LLM-written titles/keywords
                               └── SQLite   : conversations + per-user long-term memory
```

## How it works

| Step | What happens | Model |
|---|---|---|
| **Corpus** | 75 documents, 852 pages. The text comes from OCR because the PDFs' own Marathi text layers are broken. | Mistral OCR / Tesseract (reviewed) |
| **Chunking** | Each page becomes a parent, and each heading section becomes a chunk. Tables are kept whole and sentences are split at `।`. | structural |
| **LLM chunk enrichment** | For every chunk, the model writes a Marathi title, English title, English summary, bilingual keywords (including romanized forms) and the questions it answers. These are indexed with the chunk. | `sarvam-105b-conversations` |
| **Embeddings** | Dense vectors plus learned keyword weights, from one model. | `BAAI/bge-m3` |
| **Understanding a message** | One call rewrites follow-ups into full questions, detects the language (Marathi or Hindi), makes an English search query and pulls out facts worth remembering. | `sarvam-105b-conversations` |
| **Retrieval** | Dense, sparse and BM25 searches run on both the original and the English query. Results are fused (weighted RRF) and returned as their parent pages. A reranker is optional. | bge-m3, Devanagari-safe BM25 |
| **Answer** | The answer uses only the retrieved sources, cites them like [1], and is written in the user's language. | `sarvam-105b` (thinking low) |
| **Memory** | Each session keeps its history. Durable facts about each user are stored, shown in the UI and deletable. ID numbers are never stored. | SQLite |
| **Voice** | The mic button sends speech for transcription; the speaker button reads answers aloud. | Saaras v3 STT / Bulbul v3 TTS |

## Repository layout

```
backend/
  app/            chunking, enrichment, embeddings, index, retrieval, memory, chat, FastAPI app
  data/corpus/    pages.jsonl (OCR'd, cleaned) + README on how it was built
  data/index/     prebuilt LanceDB index + enrichment cache (committed, so the Space starts instantly)
  scripts/        build_index.py
  tests/          unit + API tests with fakes; golden_eval.jsonl (262 questions)
  Dockerfile      Hugging Face Space image
web/              static site: index.html, styles.css, app.js, config.js
.github/workflows pages.yml (website), deploy-space.yml (tests + backend to HF)
```

## Run locally

```bash
cd backend
pip install -r requirements.txt
echo "SARVAM_API_KEY=sk_..." > .env
set -a; . ./.env; set +a
uvicorn app.main:app --port 7860                 # API on http://localhost:7860
# in another terminal
cd web && python -m http.server 8000             # open http://localhost:8000
```

Rebuild the index after changing the corpus or chunking. The enrichment step resumes
from `data/index/enrichment.jsonl`.

```bash
python -m scripts.build_index            # about 25 min on the pay-as-you-go plan (60 LLM req/min)
python -m scripts.build_index --no-enrich   # structure-only index, no LLM calls
```

## Deploy (one-time setup)

1. **Hugging Face:** create an account and a **write** token at huggingface.co/settings/tokens.
2. **GitHub repo → Settings → Secrets and variables → Actions:**
   - Secrets: `HF_TOKEN`, `SARVAM_API_KEY`
   - Variables: `HF_SPACE` = `<hf-username>/manndeshi-sahayak`,
     `SAHAYAK_API_URL` = `https://<hf-username>-manndeshi-sahayak.hf.space`
3. **GitHub repo → Settings → Pages → Source: GitHub Actions.**
4. Push to `main`, or run both workflows from the Actions tab.

## Costs and limits

- **Indexing:** about ₹20–25 in Sarvam credits.
- **Per question:** well under ₹1 (Sarvam 105B is ₹15 per 1M input tokens and ₹60 per 1M output tokens).
- **Rate limit:** Sarvam's pay-as-you-go plan allows 60 requests per minute to 105B. The client throttles itself.
- **Space downtime:** a free HF Space sleeps after inactivity, so the first request then takes 1–2 minutes.
- **Memory resets:** SQLite memory lives in the container and resets when the Space restarts. Attach HF persistent storage and set `STATE_DIR=/data/state` to keep it.

## Tests

```bash
cd backend && python -m unittest discover -s tests -v
```
