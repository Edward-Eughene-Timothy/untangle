# Conflict Resolution Assistant (v2)

A **100% local, privacy-first** counseling and inner-conflict assistant. It listens, reflects feelings,
asks Socratic questions, and remembers you between sessions — using CBT, Internal Family Systems and
Nonviolent Communication as its grounding. Nothing you type ever leaves your machine.

> ⚠️ This is a reflective-thinking tool, **not** a therapist and not a crisis service. If you might harm
> yourself or are in danger, contact your local emergency number or a crisis line.

## How it works

```
                          FastAPI gateway  (/api/v1)
                                 │
     ┌───────────────────────────┴───────────────────────────┐
 PHASE 1 · one-time warm-up                        PHASE 2 · fully offline runtime
 crawl chosen counseling sources                   Ollama (gemma2:9b / qwen2.5:7b), local
 → clean → chunk (500 / 50 overlap)                SSE token stream, meta event in ms
 → FastEmbed bge-small-en-v1.5 (local)             emotion + distortion extraction (JSON)
 → on-disk Qdrant · SQLite flag initialized=true   SQLite memory + long-term profile summary
```

Each reply is built from four streams: the counselor persona, your long-term profile, the top-K framework
chunks retrieved from Qdrant, and the last N turns of the conversation.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with a model pulled: `ollama pull gemma2:9b` (or `qwen2.5:7b`)
- Internet **only** for the one-time warm-up (fetching pages + downloading the ~130 MB embedding model)

## Setup

```bash
git clone https://github.com/Edward-Eughene-Timothy/CONFLICT-RESOLUTION-ASSISTANT.git
cd CONFLICT-RESOLUTION-ASSISTANT
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # optional; defaults work
ollama serve                                           # if not already running
uvicorn app.main:app --reload
```

Interactive docs: <http://localhost:8000/docs>

### First run: warm-up

```bash
curl localhost:8000/api/v1/setup/topics                # pick from the catalog
curl -X POST localhost:8000/api/v1/setup/warmup \
     -H 'content-type: application/json' \
     -d '{"topics": ["cbt", "ifs", "nvc", "imposter_syndrome"]}'
curl localhost:8000/api/v1/setup/status                # poll until "state": "completed"
```

You can also pass `"extra_urls": ["https://..."]`. The crawler identifies itself, honours `robots.txt`,
and re-running a warm-up updates pages in place instead of duplicating them.

### Chat

```bash
curl -N -X POST localhost:8000/api/v1/stream_counsel \
     -H 'content-type: application/json' \
     -d '{"message": "I keep going back and forth on whether to leave my job."}'
```

Server-Sent Events, in order:

| event    | payload                                               | when                                   |
|----------|-------------------------------------------------------|----------------------------------------|
| `meta`   | `{session_id}`                                        | immediately                            |
| `safety` | `{message}`                                           | only if self-harm language is detected |
| `token`  | `{text}`                                              | repeated, as the model generates       |
| `done`   | `{session_id, ttft_ms, total_ms, emotion}`            | success                                |
| `error`  | `{message}`                                           | e.g. Ollama not running                |

Pass the returned `session_id` in later requests to continue the conversation. It is a `POST` (message in
the body), so browsers should read it with `fetch()` and a stream reader rather than `EventSource`.

## API

| Method & path                     | Purpose                                                        |
|-----------------------------------|----------------------------------------------------------------|
| `GET  /api/v1/setup/topics`       | Warm-up topic catalog                                          |
| `POST /api/v1/setup/warmup`       | Start crawling + indexing (202, runs in background)            |
| `GET  /api/v1/setup/status`       | Warm-up progress and `initialized` flag                        |
| `POST /api/v1/stream_counsel`     | Live counseling stream (SSE)                                   |
| `POST /api/v1/sessions`           | Create an empty session                                        |
| `GET  /api/v1/sessions`           | List sessions                                                  |
| `GET  /api/v1/sessions/{id}`      | Full history, with per-message emotion analysis                |
| `DELETE /api/v1/sessions/{id}`    | Delete a conversation                                          |
| `GET/PUT/DELETE /api/v1/profile`  | View, edit or erase the long-term "User Journey & Profile"     |
| `GET  /api/v1/health`             | Ollama / Qdrant / SQLite status                                |

## Configuration

All settings are environment variables (or `.env`); see `app/core/config.py` for the full list.

| Variable                | Default                    | Notes                                                        |
|-------------------------|----------------------------|--------------------------------------------------------------|
| `OLLAMA_BASE_URL`       | `http://localhost:11434`   |                                                              |
| `MODEL_NAME`            | `gemma2:9b`                | or `qwen2.5:7b`                                              |
| `QDRANT_STORAGE_PATH`   | `./qdrant_storage`         | on-disk vector DB                                            |
| `SQLITE_DB_PATH`        | `./memory.db`              | history, profile, setup flag                                 |
| `FASTEMBED_CACHE_PATH`  | `./fastembed_cache`        | kept outside the OS temp dir so offline runs survive reboots |
| `RECENT_TURNS`          | `8`                        | messages of history sent to the model                        |
| `SUMMARY_EVERY_N_TURNS` | `6`                        | how often the long-term profile is refreshed                 |
| `REQUIRE_WARMUP`        | `true`                     | set `false` to chat without retrieval material               |
| `CRISIS_RESOURCES_TEXT` | generic text               | **set this to your region's crisis resources**               |

## Verifying the acceptance criteria

1. **Zero external calls at runtime.** After warm-up, start the server with `HF_HUB_OFFLINE=1`, turn off
   Wi-Fi, and chat. Inference (Ollama), retrieval (Qdrant + cached FastEmbed) and memory (SQLite) are all local.
2. **Fast first token.** The `meta` event arrives within milliseconds and tokens stream unbuffered. The
   `done` event reports `ttft_ms`. Real time-to-first-token is dominated by your hardware: the model is
   kept resident (`OLLAMA_KEEP_ALIVE`) and preloaded on startup, but a 9B model on CPU will not hit 200 ms.
   Try `qwen2.5:3b`/`gemma2:2b` or a GPU if you need it.
3. **Persistence.** Restart the server and `GET /api/v1/sessions/{id}`; the history is still there and the
   next message sees it.
4. **Clean codebase.** `grep -ri -e streamlit -e faiss -e langchain .` returns only this section of the README.

## Tests

```bash
pytest
```

The suite fakes Ollama, the web and the embedding model, so it runs offline in about a second while
exercising the real FastAPI, SQLite and Qdrant code.

## Project layout

```
app/
  api/v1/          router + endpoints (setup, stream_counsel, sessions, health)
  core/            config, Qdrant client, SQLite, Ollama client
  schemas/         pydantic models
  services/        warm-up, emotion analysis, memory, RAG, counselor agent
  main.py          app factory, CORS, lifespan
tests/
```

## Migrating from v1

v1 was a Streamlit + FAISS + LangChain + CTransformers (Llama-2) app that embedded three PDF books. v2 removes
all of that. The book PDFs and the FAISS index were deleted from the working tree; they remain in git history.

## License

MIT — see `LICENSE`.
