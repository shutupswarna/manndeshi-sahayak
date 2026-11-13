# Nalanda-Lite corpus — clean page-level extract

Source: `SourceMaterial/Financial Literacy` (73 PDFs, 843 pages) from nalanda-lite-main.

## Files
- `pages.jsonl` — one row per page (843). Fields: doc_id, path, filename, module,
  language (detected from script), path_language (old folder-based label),
  page, source (mistral_ocr | none), is_junk, chars, devanagari_ratio, text.
- `docs/<doc_id>.md` — one clean Markdown file per document, junk pages removed,
  `<!-- page N -->` markers kept for citations.
- `build_corpus.py`, `extract_local.py` — scripts that produced this.

## How it was built
1. Text comes from the Mistral OCR output already cached in the repo's LanceDB index
   (chunks never cross a page, so pages were rebuilt exactly by joining chunks in order).
2. The PDFs' own text layer was also extracted and compared, then rejected for Marathi:
   it drops conjuncts (आर्थिक साक्षरता -> "आथक सारता") and DMBA pages use a legacy
   Devanagari font that extracts as Latin garbage.
3. 13 junk pages (number ladders, empty tables) are flagged `is_junk=true`.

## Stats (usable pages)
| language | pages | median chars | p90 chars | pages with OCR headings |
|---|---|---|---|---|
| Marathi | 519 | 835 | 1,793 | 397 |
| English | 311 | 411 | 1,939 | 223 |

About 65% of the corpus by characters is Marathi. 308 pages the old pipeline
labelled `en` (DMBA, Digital Literacy) are actually Marathi; `language` here is corrected.

## Known gaps
- OCR quality is Mistral OCR's; it has not been re-checked against a second OCR engine.
- 21 pages in the Marathi folder are mainly English (forms/tables) and are labelled `en`.
