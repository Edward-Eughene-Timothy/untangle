# Untangle

**A private, local-first AI companion for working through hard decisions, inner conflicts and difficult
conversations.**

Untangle listens, reflects what you're feeling, offers reframes and practical steps drawn from CBT,
Nonviolent Communication and Internal Family Systems, and remembers you between conversations. The language
model, the search over its reference material and your entire history all run **on your own computer**.
Nothing you write is sent anywhere.

> **Untangle is a reflective tool, not a therapist and not a crisis service.** It can't diagnose or treat
> anything. If you might harm yourself or you're in danger, contact your local emergency number or a crisis
> line, or reach out to someone you trust.

Originally built by a team for the **Intel oneAPI Hackathon (2024)** as the *Conflict Resolution
Assistant*, and rebuilt in 2026 with a modern architecture. See [History](#history-from-hackathon-prototype-to-untangle).

---

## Highlights

- **Fully local.** Inference through [Ollama](https://ollama.com); search through an on-disk
  [Qdrant](https://qdrant.tech) index; memory in SQLite. After a one-time setup it works with the network off.
- **A proper chat interface.** Streaming replies, markdown rendering, a conversation list, a Stop button and a
  dark/light theme, served by the app itself. No external scripts, fonts or CDNs.
- **Counsels, not just questions.** Every reply reflects the feeling, offers something useful and asks at most
  one question. Ask for help ("give me steps to cope") and it switches to concrete, numbered guidance.
- **Remembers you, on your terms.** A short private profile is built from your conversations. You can view,
  edit or erase it, and delete any conversation, from the UI.
- **Safety net.** Self-harm language is detected by a deterministic keyword check that doesn't depend on the
  model, and triggers a visible notice and a safer response mode.
- **Built for modest hardware.** Runs on a CPU-only laptop with a small model. Hidden "thinking" is switched
  off, and per-reply timing is shown so you can see where time goes.
- **Tested.** An automated pytest suite covers warm-up, streaming, persistence, safety and the UI server.

## How it works

```
                    Browser UI  (http://localhost:8000)
                               │  POST /api/v1/stream_counsel   (Server-Sent Events)
                               ▼
                       FastAPI application
     ┌───────────────────────┬──────────────────────┬─────────────────────────┐
     │ Retrieval             │ Counselor            │ Memory                  │
     │ FastEmbed vectors     │ persona + mode notes │ SQLite: conversations   │
     │ Qdrant (on disk)      │ Ollama (local LLM)   │ + long-term profile     │
     └───────────────────────┴──────────────────────┴─────────────────────────┘
```

**Phase 1: one-time setup (needs the internet).** You choose topics (CBT, IFS, NVC, decision paralysis,
imposter syndrome, and more). Untangle downloads reference pages for them, strips the HTML, splits the text into
~500-character passages, embeds them locally with `BAAI/bge-small-en-v1.5` and stores them in Qdrant. It honours
`robots.txt`, and re-running setup updates pages in place instead of duplicating them.

**Phase 2: everyday use (fully offline).** For each message:

1. Your message is saved. A bare greeting gets an instant reply with no model call.
2. Reference passages are retrieved and, optionally, a fast emotion/thinking-pattern pass runs alongside.
3. The prompt is assembled from the persona, a mode note (practical guidance if you asked for help; a nudge
   away from repeated phrases), your private profile, the emotional read, the passages, and recent turns.
4. Tokens stream to the browser as the model writes them.
5. The reply is saved. Every few exchanges the long-term profile is refreshed in the background.

## Quick start

### Requirements

- Python 3.11 or newer (developed and tested on Linux)
- [Ollama](https://ollama.com), with a model pulled
- Internet access for the first-run setup only

### Install and run

```bash
git clone https://github.com/Edward-Eughene-Timothy/untangle.git
cd untangle
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

ollama pull gemma4:e2b-it-qat                          # see "Choosing a model" below
cp .env.example .env                                   # then set MODEL_NAME to the model you pulled

uvicorn app.main:app --reload
```

Open **<http://localhost:8000>**. On the first visit you'll see a short setup screen: pick a few topics and press
**Download and index**. When it finishes, the chat opens. The small status pill at the top shows whether Ollama
and your model are reachable.

Stop the server with **Ctrl+C**. Your conversations and index stay in `memory.db` and `qdrant_storage/` in the
project folder.

### Choosing a model

Untangle works with any chat model Ollama can run. Set `MODEL_NAME` in `.env` to exactly what `ollama list`
shows.

| Model | Notes |
|---|---|
| `gemma4:e2b-it-qat` | The author's choice: good quality for its size. About 17 tokens/s on a CPU-only laptop. |
| `gemma3:1b-it-qat` | Faster (about 27 tokens/s in the same test) with simpler replies. |
| `gemma3:4b` | Larger and slower than the E2B; not benchmarked by the authors. |
| `gemma2:9b` | The largest of these, but far too slow without a GPU (about two minutes to the first word on the authors' CPU). |

Speeds are measured on the authors' hardware and will differ on yours. Run `python scripts/bench.py <model>` to
measure your own. Two more tips: keep `EMOTION_ANALYSIS_ENABLED=false` on slow machines (it saves an extra
model call per message), and lower `RAG_TOP_K` for a shorter prompt and a faster first word.

## Using Untangle

- **Enter** sends, **Shift+Enter** adds a new line, and pasting multi-line text works. While a reply is
  streaming, the send button becomes **Stop**.
- The sidebar lists past conversations. Open one to continue it, or delete it with **x**.
- **What I remember** shows the private summary the counselor sees at the start of every conversation. Edit it
  or erase it at any time.
- Under each reply, small grey text shows the time to the first word, how many tokens were read, and the
  writing speed.

## Configuration

Set values in `.env` or as environment variables. Relative data paths are resolved against the project folder,
so starting the server from another directory can't create a second, empty database.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `gemma2:9b` | Ollama model to chat with. **Set this to your model.** |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama listens |
| `SQLITE_DB_PATH` | `./memory.db` | Conversations, profile, setup flag |
| `QDRANT_STORAGE_PATH` | `./qdrant_storage` | On-disk vector index |
| `FASTEMBED_CACHE_PATH` | `./fastembed_cache` | Embedding model files (kept out of the OS temp dir so offline use survives reboots) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Must match each other |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Passage size in characters |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `4` / `0.3` | Passages per reply / minimum similarity |
| `RECENT_TURNS` | `8` | Earlier messages sent to the model |
| `SUMMARY_EVERY_N_TURNS` | `6` | How often the profile is refreshed |
| `EMOTION_ANALYSIS_ENABLED` | `true` | Extra model call per message that tags emotion and thinking patterns |
| `EMOTION_TIMEOUT_S` | `2.0` | Longest the reply waits for that analysis |
| `LLM_TEMPERATURE` / `LLM_NUM_CTX` | `0.6` / `4096` | Sampling temperature / context window |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model in memory |
| `PRELOAD_ON_STARTUP` | `true` | Load the models at startup so the first message is faster |
| `REQUIRE_WARMUP` | `true` | Refuse to chat until setup has completed once |
| `CRISIS_RESOURCES_TEXT` | generic text | **Replace with real resources for your region** |
| `CRAWL_USER_AGENT` / `CRAWL_TIMEOUT_S` / `CRAWL_CONCURRENCY` | see `config.py` | Setup crawler behaviour |
| `CORS_ORIGINS` | localhost dev ports | Only needed if you build a separate front end |

## API

Interactive docs are at <http://localhost:8000/docs>. Everything is under `/api/v1`.

| Method and path | Purpose |
|---|---|
| `GET /setup/topics` | Topic catalog: `cbt`, `ifs`, `nvc`, `decision_paralysis`, `imposter_syndrome`, `conflict_resolution`, `act`, `dbt`, `inner_conflict`, `attachment` |
| `POST /setup/warmup` | Start indexing `{"topics": [...], "extra_urls": [...]}` (returns 202) |
| `GET /setup/status` | Setup progress and the `initialized` flag |
| `POST /stream_counsel` | Send `{"message": "...", "session_id": "..."?}`; streams the reply as Server-Sent Events |
| `POST /sessions`, `GET /sessions` | Create or list conversations |
| `GET /sessions/{id}`, `DELETE /sessions/{id}` | Read or delete one conversation |
| `GET`, `PUT`, `DELETE /profile` | View, edit or erase the long-term profile |
| `GET /health` | Status of Ollama, the vector store and SQLite |

`stream_counsel` sends these events, in order: `meta` (session id, sent immediately), `safety` (only if
self-harm language is detected), `token` (repeated), then `done` (timing and the emotional read) or `error`.
It is a `POST`, so read it with `fetch()` and a stream reader; `EventSource` only supports `GET`.

```bash
curl -N -X POST localhost:8000/api/v1/stream_counsel \
     -H 'content-type: application/json' \
     -d '{"message": "I keep going back and forth on whether to leave my job."}'
```

## Privacy, data and safety

**Where your data lives.** Three places, all inside the project folder:
`memory.db` (conversations and your profile), `qdrant_storage/` (the reference passages, not your data) and
`fastembed_cache/` (the embedding model). Delete a conversation or your profile from the UI, or delete
`memory.db` to remove everything. Keep these files out of version control; the `.gitignore` already does.

**Network use.** Only the one-time setup touches the internet, to fetch reference pages and the embedding model
(plus `ollama pull`, which you run yourself). By default the pages come from Wikipedia, whose text is available
under CC BY-SA 4.0. The crawler identifies itself and respects `robots.txt`. After that, run with
`HF_HUB_OFFLINE=1` and the network off if you like.

**Safety.** Every message goes through a keyword check for self-harm language. It runs locally and doesn't rely
on the model. When it fires, the UI shows a notice and the model is told to respond with care, ask whether you're
safe, and point you to real help rather than problem-solving. Keyword matching can't catch everything, so
treat it as a safety net, not a guarantee. Set `CRISIS_RESOURCES_TEXT` to resources that fit where you live.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite fakes Ollama, the web and the embedding model, so it runs offline in seconds while exercising the real
FastAPI, SQLite and Qdrant code.

Helper scripts (the server must be running for the first):

| Script | Purpose |
|---|---|
| `python scripts/chat_cli.py` | Terminal chat client (`pip install -e ".[cli]"`) |
| `python scripts/bench.py [model]` | Measures prompt-reading speed, writing speed and hidden thinking for a model |

```
untangle/
├── app/
│   ├── api/v1/            router and endpoints (setup, stream_counsel, sessions, health)
│   ├── core/              settings, Qdrant client, SQLite, Ollama client
│   ├── schemas/           pydantic models
│   ├── services/          warm-up crawler, retrieval, emotion analysis, memory, counselor agent
│   ├── static/            the web UI (index.html, style.css, app.js)
│   └── main.py            app factory, CORS, lifespan
├── scripts/               chat_cli.py, bench.py
├── tests/
└── pyproject.toml
```

## Known limitations

- Small local models can be repetitive or shallow. Untangle steers them with concrete instructions, but a
  larger model gives noticeably richer replies if your hardware allows it.
- On a CPU, the first word can take several seconds. A GPU or a smaller model helps most.
- Reference material is general-purpose (Wikipedia by default), not clinical guidance. Add your own pages with
  `extra_urls` if you have better sources.
- Developed and tested on Linux. It should work elsewhere but hasn't been verified there.

## History: from hackathon prototype to Untangle

The first version was the **Conflict Resolution Assistant**, built for the Intel oneAPI Hackathon in 2024. It
proved the idea. This rebuild keeps the purpose and replaces almost everything underneath it.

| | 2024 prototype | Untangle |
|---|---|---|
| Interface | Streamlit text box | Streaming web chat with markdown, sessions and themes |
| Backend | Single script | Async FastAPI with a clean service layout |
| Language model | Llama-2 7B (GGML via CTransformers) | Any Ollama model; Gemma family recommended |
| Retrieval | LangChain + FAISS over a few bundled PDFs | Qdrant + FastEmbed over pages you choose at setup |
| Memory | None: each question answered in isolation | Persistent conversations and a long-term profile you control |
| Replies | Whole reply at once | Token-by-token streaming |
| Safety | None | Local self-harm detection and a safer response mode |
| Behaviour | One prompt, intent by keyword | Reflect, contribute, then at most one question; guidance mode on request |
| Testing | No automated tests | Automated test suite |

## Team

Untangle began as a team project for the Intel oneAPI Hackathon 2024:

- [**Edward-Eughene-Timothy**](https://github.com/Edward-Eughene-Timothy)
- [**ariya10**](https://github.com/ariya10)
- [**arputhan06-tech**](https://github.com/arputhan06-tech)

## Acknowledgements

Built with [FastAPI](https://fastapi.tiangolo.com), [Ollama](https://ollama.com),
[Qdrant](https://qdrant.tech), [FastEmbed](https://github.com/qdrant/fastembed) and
[Google's Gemma](https://ai.google.dev/gemma) models. Reference material is drawn from Wikipedia contributors
(CC BY-SA 4.0). The 2024 prototype was created for the Intel oneAPI Hackathon.

## License

MIT. See [LICENSE](LICENSE).
