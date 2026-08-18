# Website RAG Backend (Backend 2)

A website-specific AI RAG chatbot backend built with FastAPI, LangGraph, Qdrant, BGE embeddings,
and Gemini. This service handles **only** RAG functionality. It is one of two backend services in
the overall application.

---

## 1. Project purpose

This service answers user questions using a specific website's content plus supporting documents
(PDF, DOCX, TXT, Markdown, CSV). It recognizes greetings, classifies messages, retrieves relevant
knowledge with hybrid (dense + sparse) search, reranks candidates, falls back to live website
pages when the indexed knowledge base is insufficient, and generates grounded, validated answers.

**There is no conversational memory.** Every request is processed independently — no
`conversation_id`, no message history, no database of past turns. LangGraph maintains state only
for the duration of a single request.

---

## 2. Two-backend architecture

The overall application has two independent backend services:

- **Backend 1 — Express.js** (not part of this repo): authentication, authorization,
  user/account functionality, and all non-RAG application logic. Talks to the React frontend.
- **Backend 2 — FastAPI** (this repo): document/website ingestion, embeddings, Qdrant, hybrid
  retrieval, reranking, web fallback, LangGraph orchestration, and Gemini-based classification
  and generation. Nothing else.

This service does not implement authentication and never rebuilds or replaces Express.js
functionality.

---

## 3. Request flow vs. startup order

These are **not** the same thing.

**Request flow** (what happens per user message):

```
React -> Express.js -> [auth/authorization] -> FastAPI -> LangGraph
       -> retrieval / fallback / generation -> FastAPI response
       -> Express.js -> React -> User
```

**Startup order** (what you start, and in what sequence, during local development):

```
1. Qdrant (Cloud or local instance) must already be reachable
2. Start FastAPI      (this repo)
3. Start Express.js   (teammate's repo)
4. Start React        (teammate's repo)
```

Ports are all configurable via `.env` (`API_HOST` / `API_PORT`) — nothing is hard-coded.

---

## 4. Qdrant's role

Qdrant is the **only** vector database in this project. It stores, per chunk:

- the dense vector (BGE) and a sparse vector (BM25, for hybrid search)
- payload: `text`, `document_id`, `chunk_id`, `title`, `source_type`, `source_ref`, and (when
  applicable) `url`, `page`, `category`, `content_type`, `row_index`, `file_name`

Payload indexes are created once (`ingestion/vector_store/payload_index.py`) on the fields used
for filtering (`source_type`, `document_id`, `category`, `content_type`, `url`, `page`).

## 5. BGE's role

`BAAI/bge-small-en-v1.5` (configurable via `DENSE_EMBEDDING_MODEL`) is used for **both** chunk and
query embeddings, so they live in the same vector space. Queries get BGE's recommended instruction
prefix; documents do not.

---

## 6. Ingestion

```
file / website page -> loader -> cleaner -> preprocessor -> chunker
                     -> dense + sparse embeddings -> Qdrant upsert
```

Supported sources: PDF, DOCX, TXT, Markdown, CSV, and website pages (crawled within the configured
domain only). Chunking is recursive by default; semantic chunking is available and configurable
(`USE_SEMANTIC_CHUNKING=true`) but not assumed to be better.

Run ingestion with:

```bash
python scripts/ingest.py                        # files + website crawl
python scripts/ingest.py --no-website            # files only
python scripts/ingest.py --max-website-pages 100
```

Place source files under `data/raw/pdfs/` (PDFs) and `data/raw/documents/` (DOCX/TXT/MD/CSV)
before running ingestion.

---

## 7. Retrieval: hybrid search + reranking

For each knowledge query:

1. **Dense search** (`rag/retrieval/dense_search.py`) — BGE semantic similarity.
2. **Sparse search** (`rag/retrieval/sparse_search.py`) — BM25 lexical match (good for exact
   terms, product names, IDs).
3. **Hybrid fusion** (`rag/retrieval/hybrid_search.py`) — weighted Reciprocal Rank Fusion,
   weight configurable via `HYBRID_DENSE_WEIGHT`.
4. **Metadata filtering** (`rag/retrieval/filters.py`) — optional, by `source_type`,
   `document_id`, `category`, `content_type`, `url`.
5. **Reranking** (`rag/retrieval/reranker.py`) — a cross-encoder (`RERANKER_MODEL`) re-scores
   the top `TOP_K` candidates down to `RERANK_TOP_K`.

`rag/retrieval/retriever.py` is the single high-level entry point the graph calls, and it also
decides evidence sufficiency (`MIN_RERANK_SCORE`, `MIN_EVIDENCE_CHUNKS`).

**Important principle:** insufficient Qdrant evidence means the knowledge base is incomplete for
that question — not that the question is irrelevant. That distinction drives the web-fallback
routing in the chat graph, rather than treating a low-evidence result as out-of-scope.

---

## 8. Web fallback

When evidence is insufficient, `rag/retrieval/web_fallback.py` fetches a small number of live
pages **from the configured website only** (`WEBSITE_URL` / its domain), extracts clean text, and
scores passages against the query. It:

- never leaves the configured domain
- handles fetch failures, timeouts, and malformed HTML gracefully (returns fewer/no results, never
  raises to the caller)
- always returns the source URL for attribution
- is capped by `WEB_FALLBACK_MAX_PAGES` / `WEB_FALLBACK_TIMEOUT_SECONDS`
- only runs when needed, not on every request

---

## 9. LangGraph

- `graph/chat_graph.py` — the per-request chat workflow: classify → (direct response | retrieve →
  rerank → evidence check → generate-or-fallback → generate → validate → retry-or-safe-response).
- `graph/ingestion_graph.py` — the ingestion pipeline as an explicit, inspectable graph.
- `graph/state.py` — request-scoped state only. No persisted history.

LangGraph coordinates Qdrant, BGE, Gemini, and web fallback; it does not reimplement any of them.

---

## 10. Gemini

Two responsibilities, kept logically separate in `rag/generation/gemini.py`:

1. **Classification** — `GREETING` / `KNOWLEDGE_QUERY` / `OUT_OF_SCOPE`.
2. **Generation** — grounded answers from retrieved evidence, plus a validation pass checking the
   answer is actually supported by that evidence (not just similar in wording).

Prompts are centralized in `rag/prompts/`.

---

## 11. Configuration

All configuration lives in `config/settings.py`, sourced from `.env`. Copy the template:

```bash
cp .env.example .env
```

Then fill in at minimum: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (if using Qdrant Cloud),
`WEBSITE_URL`. See `.env.example` for every available setting and its default.

`WEBSITE_URL` is the single source of truth for the target site — used for ingestion, web
fallback, and domain restriction. It is never hard-coded elsewhere in the codebase.

---

## 12. Running the service

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit values
python scripts/ingest.py
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 13. API endpoints

### `GET /health`
Returns `{"status": "ok", "service": "rag-fastapi"}`.

### `POST /rag/chat`

Request:
```json
{ "message": "What services does the website provide?" }
```

Response:
```json
{
  "answer": "...",
  "sources": [
    { "title": "Pricing", "source_type": "web", "url": "https://example.com/pricing" }
  ]
}
```

No `conversation_id`, no history — each call is independent. `message` is validated as
non-empty, non-whitespace-only, and under `MAX_MESSAGE_LENGTH` characters (default 2000).
A message like `"1234"` may fail meaningful-content checks; `"What is product 1234?"` remains
valid — digits alone aren't rejected.

---

## 14. Express.js ↔ FastAPI integration contract

- Express.js owns authentication/authorization. It calls `POST /rag/chat` **after** it has
  already authenticated the user — FastAPI does not authenticate anyone itself.
- FastAPI does not trust a raw `user_id` or identity field placed on the request body by a
  client. If Express needs to pass trusted context, it should be done via a signed header (see
  below) rather than an unsigned field.
- Optional service-to-service signature verification: set `REQUIRE_SERVICE_SIGNATURE=true` and
  share `SERVICE_SHARED_SECRET` between the two services. Express computes
  `HMAC_SHA256(SERVICE_SHARED_SECRET, raw_json_body)` (hex digest) and sends it as the
  `X-Service-Signature` header. FastAPI rejects the request with 401 if the signature is missing
  or invalid. This is disabled by default for local development.
- CORS (`ALLOWED_ORIGINS`) should, in production, list only the Express.js service's own origin
  — the browser is not expected to call FastAPI directly.

---

## 15. Testing

```bash
pytest
```

Tests cover ingestion (loaders, cleaning, chunking), embeddings, Qdrant integration points,
retrieval (dense/sparse/hybrid/filtering/reranking), web fallback (including domain restriction),
classification, the LangGraph chat/ingestion graphs, generation, validation, and the FastAPI API
layer (including malformed-request handling). Tests that require live Qdrant/Gemini credentials
are marked and skip automatically when those credentials aren't configured — see
`tests/conftest.py`.

---

## 16. Project structure

See the authoritative structure enforced throughout this repo:

```
rag-fastapi/
├── ingestion/        # get/process knowledge: loaders, processors, chunking, embeddings, vector_store
├── rag/              # find knowledge (retrieval) + generate answers (generation, prompts, pipeline)
├── graph/             # LangGraph orchestration (chat_graph, ingestion_graph, nodes/)
├── api/               # FastAPI app, routes, schemas
├── config/            # settings.py — all configuration
├── scripts/           # ingest.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

No PostgreSQL, no conversation database, no second vector database — by design.
