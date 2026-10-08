# Northern Star — v0.3 Backend (Ingestion + Evidence Index + Evidence-Grounded Q&A)

**Northern Star** is a software intelligence platform.
v0.3 covers the pipeline through evidence-grounded Q&A:

```
GitHub URL
  → clone repository
  → inspect every file
  → detect languages and frameworks
  → structural metadata report (manifest)
  → chunk source/config/docs into evidence segments
  → store chunks in a SQLite FTS5 index
  → lexical retrieval with exact file:line provenance
  → grounded Q&A against a local Ollama model,
    answering ONLY from retrieved evidence with validated [E#] citations
```

Everything from ingestion through retrieval is deterministic and locally
runnable, with *provenance on every piece of retrieved evidence*. The LLM
(v0.3) sits *on top of* retrieval — never instead of it. Northern Star must
never be "an LLM reading a repository and inventing answers": the retrieval
layer supplies the evidence, the model cites it by `[E#]` label, and every
citation in an answer is validated against the evidence that was actually
provided.

---

## M3 at a glance

| Layer | What it does | Where |
|---|---|---|
| Source-file extraction | Picks indexable files (source/config/documentation/data) and skips binaries, build output, secrets | `indexing.py` (reuses M1 categories) |
| Chunking | Splits large files into sequential line chunks; small files stay whole | `chunking.py` |
| SQLite index | `repositories` → `files` → `chunks`, plus an FTS5 virtual table | `indexing.py` |
| Retrieval | FTS5 `MATCH` + BM25 ranking, filtered by repository | `retrieval.py` |
| Prompt builder | Grounding contract + `[E#]`-labelled evidence formatting + message assembly | `prompts.py` |
| LLM client | Thin httpx client for Ollama's `/api/chat`; typed errors, mockable transport | `llm.py` |
| Q&A orchestrator | retrieve → label evidence → prompt → LLM → parse → validate citations → answer | `qa.py` |
| API | `POST /repos/{o}/{r}/index`, `GET /repos/{o}/{r}/search`, `POST /repos/{o}/{r}/ask` | `routes.py` |

---

## Project structure

```
backend/
├── app/
│   ├── main.py               # FastAPI application
│   ├── config.py              # Settings (reads environment variables)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # HTTP endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic request/response schemas (incl. search + Q&A)
│   └── services/
│       ├── __init__.py
│       ├── github.py          # URL parsing + git fetch service
│       ├── detection.py       # Deterministic language/framework detection
│       ├── ingestion.py       # Orchestrates clone → analyse → persist manifest
│       ├── chunking.py        # M2: line-based evidence chunking
│       ├── indexing.py        # M2: SQLite evidence index (schema + build)
│       ├── retrieval.py       # M2: FTS5 search + BM25 ranking
│       ├── prompts.py         # M3: grounding rules + [E#] evidence formatting
│       ├── llm.py             # M3: Ollama /api/chat client (typed errors)
│       └── qa.py              # M3: Q&A orchestrator + citation validation
├── tests/
│   ├── conftest.py
│   ├── test_detection.py
│   ├── test_github.py
│   ├── test_ingestion.py
│   ├── test_api.py
│   ├── test_chunking.py       # M2
│   ├── test_indexing.py       # M2
│   ├── test_retrieval.py      # M2
│   ├── test_prompts.py        # M3
│   ├── test_llm.py            # M3
│   └── test_qa.py             # M3
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# From the NorthernStar root:
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

## Running the server

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok","service":"northern-star","version":"0.3.0"}
```

---

## SQLite evidence schema

The evidence database lives at `<storage>/northern_star.db` (single shared
database for all repositories — isolation is by `repo_id`).

```sql
CREATE TABLE repositories (
    id TEXT PRIMARY KEY,                -- "owner/repo"
    commit_hash TEXT,
    default_branch TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    path TEXT NOT NULL,                 -- relative to repo root (citation anchor)
    language TEXT,
    size_bytes INTEGER,
    is_source INTEGER NOT NULL DEFAULT 0,
    UNIQUE(repo_id, path)
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id TEXT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language TEXT,
    start_line INTEGER NOT NULL,        -- 1-indexed, inclusive
    end_line INTEGER NOT NULL,          -- 1-indexed, inclusive
    content TEXT NOT NULL
);

-- Lexical search index over chunk content (external-content FTS5).
-- INSERT/DELETE triggers keep it in sync automatically.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content, content='chunks', content_rowid='id', tokenize='unicode61'
);
```

FTS5 ships inside Python's built-in `sqlite3` — no native dependencies.

---

## Chunking strategy

`chunking.py` uses *deterministic* line-based chunking (no AST, no tree-sitter):

- **Small files (≤150 lines)** → a single chunk covering the whole file,
  with the verbatim file text as `content`.
- **Large files (>150 lines)** → sequential, non-overlapping chunks of
  ~100 lines each (final chunk may be shorter).
- Chunks never split a line; `start_line`/`end_line` are 1-indexed and inclusive.
- Blank files produce no chunks.
- Files are read lazily (one at a time); the indexer also skips files above
  a size guard (1 MB) so generated blobs never dominate the index.

Every chunk carries full provenance:

```
chunk_id · repo_id · file_id · file_path · language · start_line · end_line · content
```

---

## Indexing flow

Indexing runs automatically after every ingestion
(`POST /api/v1/repos` → clone → analyse → persist manifest → build index).
It is safe to run repeatedly — re-indexing deletes the repository's prior
rows inside the same transaction before re-inserting, so it never duplicates.

```text
ingestion
   ↓
index_repository(manifest, checkout_root, db_path)   # indexing.py
   ├─ read manifest file_inventory (skip BINARY/BUILD/OTHER)
   ├─ read each file's content from the checkout (≤1 MB, lazily)
   ├─ chunk_file(...)                                 # chunking.py
   └─ single transaction: DELETE old repo rows → INSERT repository/files/chunks
                                             (triggers keep FTS5 in sync)
```

To rebuild an index explicitly (e.g. after touching the checkout on disk),
call `POST /api/v1/repos/{owner}/{repo}/index`.

---

## API usage

### Ingest a repository (auto-indexes)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/repos \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/pallets/flask"}'
```

**HTTP 201** — returns the `RepositoryManifest` and builds the evidence index.

### Retrieve a cached manifest

```bash
curl http://127.0.0.1:8000/api/v1/repos/pallets/flask
```

**HTTP 200** — the previously ingested manifest, no re-clone.

### Search the evidence index

```bash
curl "http://127.0.0.1:8000/api/v1/repos/pallets/flask/search?q=routing&limit=5"
```

**HTTP 200** — structured, provenance-carrying results:

```json
{
  "query": "routing",
  "repo_id": "pallets/flask",
  "total": 2,
  "results": [
    {
      "file_path": "src/flask/app.py",
      "start_line": 501,
      "end_line": 600,
      "language": "Python",
      "score": 4.449,
      "content": "            # context processor for efficiency reasons...\n            request=request,\n            session=session,\n            ..."
    },
    {
      "file_path": "src/flask/debughelpers.py",
      "start_line": 1,
      "end_line": 100,
      "language": "Python",
      "score": 4.39,
      "content": "from __future__ import annotations\n\nimport typing as t\n\nfrom jinja2.loaders import BaseLoader\nfrom werkzeug.routing import RequestRedirect\n..."
    }
  ]
}
```

Scoring is SQLite's FTS5 BM25 ranking (`bm25()`) — `score` is negated so
*a higher value means more relevant*. Results are ordered best-first.

### (Re-)build the index

```bash
curl -X POST http://127.0.0.1:8000/api/v1/repos/pallets/flask/index
# → {"repo_id":"pallets/flask","files_indexed":215,"chunks_created":472}
```

Idempotent: calling it twice produces the same counts, never duplicates.

### Exact provenance, guaranteed

The hard rule of M2: **no unattributed text leaves the retrieval layer.**
Every query result is traceable to an exact repository, file, and line range:

```text
Query:   "authentication"
Result:  src/auth/middleware.py   lines 1–10   language=Python
         "def require_auth(request: Request) -> None:
              '''Reject requests that carry no authentication.'''
              auth = request.headers.get('Authorization')
              ..."

Every chunk carries: repo_id · file_path · start_line · end_line · language · content
```

These are the citation anchors the Q&A layer (M3) attaches to answers.

### Ask a question (evidence-grounded Q&A)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/repos/pallets/flask/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Flask handle routing?", "top_k": 5}'
```

**HTTP 200** — an answer grounded in retrieved evidence, with validated
`file:line` citations:

```json
{
  "question": "How does Flask handle routing?",
  "repo_id": "pallets/flask",
  "answer": "Flask maps URLs to view functions on the Flask object... "
            "the URL map is built in [E1] and routing resolution happens in [E2].",
  "citations": [
    {
      "id": "E1",
      "file_path": "src/flask/app.py",
      "start_line": 501,
      "end_line": 600,
      "language": "Python",
      "content": "..."
    },
    {
      "id": "E2",
      "file_path": "src/flask/wrappers.py",
      "start_line": 1,
      "end_line": 100,
      "language": "Python",
      "content": "..."
    }
  ],
  "confidence": "high",
  "evidence_sufficient": true,
  "confidence_source": "model",
  "evidence_grounding": "cited"
}
```

`top_k` (1–20, default from `QA_TOP_K`) controls how many evidence chunks
are retrieved and placed in the prompt — the evidence **budget**. The model
never sees the whole repository, only the top-ranked relevant chunks.

#### Why citations can be trusted

1. **Retrieval is evidence.** The `top_k` chunks come from the M2 index with
   exact `file_path:start-end` provenance — this is the *only* text the model
   is allowed to reason about.
2. **Grounding is forced in the prompt.** The system instruction forbids
   inventing files/functions/behavior, requires an `[E#]` citation for every
   factual claim, forbids citing evidence that doesn't support a claim, and
   requires an explicit "evidence insufficient" statement when appropriate.
3. **Validation is absolute.** Every `[E#]` the model emits is checked against
   the blocks that were actually supplied. Invalid IDs (e.g. `[E99]` when only
   `E1`–`E5` exist) are removed from both the answer text and the `citations`
   array. **The model cannot make evidence appear.**
4. **Citations must self-anchor (M3.1 AND rule).** For a citation to survive,
   *two independent signals must agree*: the model's `citations` array (checked
   against real block IDs) **and** the `[E#]` marker actually written into the
   answer text. A self-inconsistent answer — text `[E4]` with an array declaring
   `E1` — drops both, so no citation reaches the response that the answer does
   not genuinely reference. `evidence_grounding` is then derived *only* from
   these survivors (see [M3.1: what the response tells you](#m31-what-the-response-tells-you)).
5. **README claims stay claims.** README/documentation chunks are labelled
   `Language: Markdown` (or whatever the docs language is) and the prompt
   instructs the model to treat them as *claims*, not proof of implementation.

#### When evidence is insufficient

If retrieval returns nothing (or nothing relevant), the LLM is **never
called** — the endpoint returns immediately with
`evidence_sufficient: false`, `confidence: "low"`, and a canned statement.
So a question like *"Prove that this project supports 10,000 concurrent
users"* that no evidence backs cannot produce an invented citation: there is
no evidence to cite, and the answer says so.

#### Structured output with a fallback

The model is asked for strict JSON. If its output is unreliable (unparseable
JSON), the pipeline falls back to treating the whole text as the answer and
mining `[E#]` markers from it, then still validating every mined citation
against the supplied evidence.

#### M3.1: what the response tells you

Two extra fields separate **what the LLM claims** from **what Northern Star
can deterministically verify about the evidence**:

| Field | Values | Meaning |
|---|---|---|
| `confidence_source` | `model` \| `deterministic` | Where `confidence` / `evidence_sufficient` came from. `model` = the LLM reported them. `deterministic` = the empty-retrieval short-circuit (no LLM). **`confidence` is never proof that the evidence supports the answer** — use `evidence_grounding` for that. |
| `evidence_grounding` | `none` \| `cited` | Derived *only* from the surviving validated citations, never from the model's own confidence. `none` → zero evidence blocks are anchored in the answer (includes the empty-retrieval path); `cited` → one or more are. This is **citation-level** anchoring, *not* claim-level support verification (that is reserved for M4 — `partial` is never emitted here). |

Distinguishing case: the model can answer `confidence: high`,
`evidence_sufficient: true` while attaching **no** citation at all, or a
citation that doesn't match anything it actually cited inline. The response
then still reports `confidence_source: model` (so you know the confidence is a
model judgment, not a platform guarantee) and `evidence_grounding: none` (so
you know nothing in the answer is anchored in the supplied evidence).

### Errors

| Condition | Code | Detail |
|---|---|---|
| Missing/empty URL or `http://` | 400 | `Invalid repository URL: …` |
| `ssh://` or `git@…` | 400 | SSH not supported |
| Private or non-existent repo | 404 | `Repository not found or not accessible` |
| Clone/network failure | 502 | Upstream git error message |
| Timeout (>300s by default) | 504 | `Timed out while fetching …` |
| Empty search query | 422 | FastAPI query-param validation |
| Empty question | 422 | FastAPI body validation (`min_length=1`) |
| Ingested but never indexed | 404 | `Repository '…' is not indexed yet.` |
| Unknown owner/repo slug | 400 | `Invalid owner/repo slug.` |
| Ollama down or model not installed | 503 | `Ollama unavailable: …` |
| Ollama timed out | 504 | `Ollama timed out while generating the answer.` |
| Ollama returned unusable content | 502 | `Ollama returned …` |

---

## CLI (in-process)

Prefer the terminal over curl? The CLI drives the *same* services directly —
no HTTP server needs to be running. It uses the same storage, SQLite evidence
DB and local Ollama as the API, and adds **zero new dependencies** (stdlib
`argparse`).

```bash
cd backend
.venv/bin/python -m app.cli --help
```

Use the project's virtualenv interpreter (`.venv/bin/python`) — the system
Python won't have the app's dependencies.

### Commands

| Command | What it does |
|---|---|
| `ingest <https://github.com/owner/repo>` | Clone + analyse + index (same as `POST /repos/…/ingest`); `--ref` is recorded in the manifest metadata (the clone itself still uses the default branch) |
| `manifest <owner>/<repo>` | Show the cached metadata report for an ingested repo |
| `index <owner>/<repo>` | (Re-)build the FTS5 evidence index idempotently |
| `search <owner>/<repo> <query>` | Lexical FTS5 search with exact provenance. `--limit N` |
| `ask <owner>/<repo> "<question>"` | Evidence-grounded Q&A via local Ollama. `--top-k N`, `--model <ollama-model>` |
| `version` | Print the app version |

Every data command accepts `--json` to emit a machine-readable payload (same
shapes as the corresponding API responses: `RepositoryManifest`,
`IndexSummary`, `SearchResponse`, `AnswerResponse`).

### Examples

```bash
.venv/bin/python -m app.cli manifest pallets/flask
.venv/bin/python -m app.cli search pallets/flask "routing" --limit 5
.venv/bin/python -m app.cli ask pallets/flask "How does Flask handle routing?" --top-k 5
OLLAMA_MODEL=qwen2.5:3b .venv/bin/python -m app.cli ask pallets/flask "Prove this supports 10,000 concurrent users."
.venv/bin/python -m app.cli ask pallets/flask "…" --model qwen2.5:3b --json
```

`ask --model M` overrides `OLLAMA_MODEL` for a single run; all other env vars
from the [table below](#environment-variables) apply as usual. Exit codes are
`0` success, `1` service/runtime error (message on stderr), `2` usage error.

`ask` renders the M3.1 honesty fields under the answer, and labels each
citation's evidence kind from the manifest's deterministic file inventory
(`source` → `[code]`, documentation → `[documentation]`, other kinds labelled
by name; unknown paths get no label rather than a guess):

```text
$ OLLAMA_MODEL=qwen2.5:3b .venv/bin/python -m app.cli ask pallets/flask "How does Flask handle routing?" --top-k 5

Flask registers routes against the app's URL map ...
confidence: high (model-reported)
evidence_sufficient: True
evidence_grounding: cited
evidence_blocks: 1
citations:
  E1  src/flask/app.py:501-600 [code]
```

The `(model-reported)` suffix and the `evidence_grounding` line exist so a
`high` confidence never reads as "Northern Star proved this" — it is what the
LLM claims, and the grounding line says how much of the answer is actually
anchored in retrieved evidence. Empty retrieval renders
`confidence: low (deterministic)` and `evidence_grounding: none`.

---

## Determinism & isolation

- **Repository isolation is explicit.** Search (and therefore Q&A) is filtered
  by `repo_id`; a query against `owner/repo-a` can never return chunks from
  `owner/repo-b`. Covered by tests that index two repositories into one
  database and assert zero cross-repository leakage.
- **Re-indexing is idempotent.** Old rows for a repo are removed in the same
  transaction as the new inserts.
- **Deterministic chunking** means the same file always produces the same
  (start_line, end_line, content) tuple — stable citations across runs.
- **No LLM when there's no evidence.** Q&A short-circuits before calling
  Ollama when retrieval returns nothing, so the model can never invent an
  answer out of thin air.

---

## Running the tests

```bash
.venv/bin/python -m pytest backend/tests/ -v
```

**377 tests**, ~5 seconds (discovery tests run against a mocked httpx
transport; only the Q&A/verify/judge/challenge/improvement paths need Ollama,
and those tests use a stub LLM).
Coverage includes: chunk generation + line-number correctness + small/large
file behavior, SQLite schema, index creation + idempotent re-index,
source-file filtering (binaries/secrets excluded), FTS5 retrieval + BM25
ranking + repository isolation, exact provenance, API endpoints, empty
queries, nonexistent repositories, the prompt-builder grounding contract,
every Ollama failure path (connect/timeout/model-missing/empty/non-JSON),
citation validation + sanitization, the JSON/plain-text answer fallback, the
`/ask` endpoint (including mapping Ollama failures to HTTP codes), and the
in-process CLI (arg parsing, exit codes 0/1/2, `--json` output, offline ingest/
ask with a stubbed Ollama).

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NORTHERN_STAR_STORAGE` | `storage/repos` (relative to cwd) | Where cloned repos, manifests and the evidence DB are stored |
| `NORTHERN_STAR_CLONE_TIMEOUT_SECONDS` | `300` | Git timeout (clone + ls-remote) |
| `NORTHERN_STAR_MAX_FILES` | `50000` | Stop analysing after this many files |
| `NORTHERN_STAR_MAX_REPO_SIZE_BYTES` | `500000000` | Stop if cumulative file size exceeds this |
| `NORTHERN_STAR_BINARY_SNIFF_BYTES` | `1024` | Bytes read from unknown files to detect binary |
| `NORTHERN_STAR_MAX_CONFIG_SIZE_BYTES` | `524288` | Config files larger than this are skipped for framework detection |
| `NORTHERN_STAR_TREE_MAX_DEPTH` | `5` | Directory-tree nesting depth in the manifest |
| `NORTHERN_STAR_TREE_MAX_CHILDREN` | `200` | Max children shown per directory in the tree |
| `NORTHERN_STAR_CHUNK_LINES` | `100` | Target lines per chunk for large files |
| `NORTHERN_STAR_SEARCH_LIMIT` | `20` | Default max results per search |
| `NORTHERN_STAR_DB_FILENAME` | `northern_star.db` | Evidence database filename (inside `NORTHERN_STAR_STORAGE`) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server base URL for Q&A |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model used for evidence-grounded Q&A. Default is the fast CPU-friendly model; `qwen3:4b` (or any other) via `--model` / this env var |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Timeout for a single Ollama request (180s headroom for a cold 4B-model load on CPU-only hardware) |
| `OLLAMA_THINK` | `false` | qwen3-style hidden reasoning. OFF by default — on this hardware it roughly halves answer time (`qwen3:4b` ~128s → ~53s on a grounded 5-block prompt). Set `true` to re-enable on a model that reasons well. Ignored by non-thinking models (`qwen2.5:3b`) |
| `QA_TOP_K` | `5` | Evidence budget: chunks retrieved + placed in the prompt per question |
| `GITHUB_TOKEN` | *(unset)* | Optional GitHub personal access token for discovery. Raises API rate limits. Sent only as an `Authorization: Bearer` header — never logged, stored, or returned |
| `GITHUB_API_BASE_URL` | `https://api.github.com` | GitHub REST API base URL (overridable for tests) |
| `GITHUB_TIMEOUT_SECONDS` | `15` | Timeout for a single GitHub API request |
| `DISCOVERY_CACHE_TTL_SECONDS` | `300` | In-process TTL cache for discovery responses (search + trending) |

---

## GitHub discovery (M8.1)

Discovery finds **relevant repositories without ingesting or analyzing them**.
It returns curated GitHub metadata only — no cloning, no indexing, no M1–M7.1
analysis. Each result carries `full_name` / `html_url`, so a frontend can
later offer *"Analyze with Northern Star"* by sending that URL to the
existing M1 ingestion endpoint.

Uses GitHub's official REST API (`GET /search/repositories`) via a thin
httpx client (`services/discovery.py`). No HTML scraping, no SDK, no new
dependencies. Responses are cached in-process for 5 minutes (keyed by all
request parameters); the cache holds **discovery metadata only** and is never
treated as analysis evidence.

```bash
GET /api/v1/discover/search?q=AI%20coding%20agents
# → {"query": ..., "repositories": [...], "total_count": N,
#     "page": 1, "per_page": 10, "has_more": true}
#    params: page >= 1, per_page 1-30, language, sort (best-match|stars|forks|updated), order (asc|desc)

GET /api/v1/discover/trending?limit=10
# → {"repositories": [{"full_name": ..., "rank": 1, "trend_score": 0.99, "rank_change": null, ...}],
#     "total": 10, "limit": 10, "generated_at": "..."}
```

**Trending is Northern Star's API-derived discovery ranking — not an
official GitHub ranking.** It merges two curated searches (recently created
popular repos + recently pushed active repos), dedupes by `full_name`, and
orders by a deterministic trend score:

```text
S = log10(stars+1) / log10(max_stars+1)        (popularity)
F = log10(forks+1) / log10(max_forks+1)        (adoption)
R = max(0, 1 - days_since_push / 365)          (recency)
trend_score = round(0.5*S + 0.25*F + 0.25*R, 4)
```

`rank_change` is always `null` in M8.1: no snapshots are stored yet, so the
service does not pretend to know history (historical trend tracking belongs
to M8.2).

Rate limits are surfaced cleanly: HTTP 429 with a reset time when GitHub's
quota is exhausted (set `GITHUB_TOKEN` to raise limits), 504 on timeout,
502 on upstream failures, 503 when GitHub is unreachable. Transient 5xx gets
at most one retry. Tokens never appear in responses, errors, or logs.

CLI (note: `search` remains the M2 evidence-index search, so discovery uses
`discover`):

```bash
.venv/bin/python -m app.cli discover "AI coding agents" --json
.venv/bin/python -m app.cli discover "local AI" --language Python --sort stars
.venv/bin/python -m app.cli trending --limit 20 --json
```

---

## Historical trends (M8.2)

M8.1 discovery is live-only. M8.2 adds **stored snapshots** so the system can
answer "what is emerging?" with actual evidence instead of vibes. Snapshots
live in the existing SQLite evidence database (`discovery_snapshots` table —
created with `IF NOT EXISTS`, so existing M1–M8.1 data is untouched) and hold
**discovery metadata only**, never analysis evidence.

**Northern Star's trend rankings are its own deterministic analysis of GitHub
API data and are not an official GitHub ranking.** Historical quality improves
as snapshots accumulate: with zero or one snapshot, growth fields are honestly
`null` rather than fabricated.

```bash
POST /api/v1/discover/snapshots?limit=100   # explicit capture only — no scheduler
GET  /api/v1/discover/trends?window=7d&limit=20
GET  /api/v1/discover/repositories/{owner}/{repo}/history?window=30d
```

```bash
.venv/bin/python -m app.cli snapshot-trending --limit 100 --json
.venv/bin/python -m app.cli trends --window 7d --limit 20 --json
.venv/bin/python -m app.cli history openclaw/openclaw --window 30d --json
```

Windows are `24h` / `7d` / `30d`. The comparison matches the latest snapshot
against the stored snapshot closest to (latest − window) within a ±20%
tolerance; if none qualifies, the response carries `has_history: false` plus a
`history_reason` instead of a bogus comparison.

`rank_change = previous_rank − current_rank`: positive moved **UP**, negative
moved **DOWN**, zero unchanged.

**Popularity vs emergence.** M8.1 `trend_score` measures absolute popularity
(big repos always win). The M8.2 `emerging_score` rewards *growth* — absolute
size never enters the formula:

```text
star_rate = clamp(star_delta / max(prev_stars,1), 0, 5)
fork_rate = clamp(fork_delta / max(prev_forks,1), 0, 5)
sg, fg    = rates normalized by the set max (0 when the max is 0)
rank_imp  = (clamp(prev_rank − curr_rank, −10, 20) + 10) / 30
activity  = 1.0 if pushed ≤30d ago, linear to 0.0 at 365d, 0.0 if unknown
emerging_score = round(0.45·sg + 0.20·fg + 0.25·rank_imp + 0.10·activity, 4)
```

`null` whenever the repo has no previous snapshot in the window. Trend lists
rank history-backed rows by emerging score first, then history-less rows by
popularity — the `history_available` flag lets the frontend render the two
groups distinctly.

---

## Limitations (v0.3)

- **Lexical retrieval only.** FTS5 matches words, not meaning. Synonyms and
  paraphrases ("auth" ≠ "authentication") are missed; the Q&A evidence budget
  inherits this gap.
- **Chunk content, not filenames, is searchable.** A concept that appears
  only in a filename will not be found by FTS5, so it won't reach the LLM.
- **Answers are no better than retrieval.** The Q&A short-circuit guarantees
  no invented citations from *empty* retrieval, but retrieval can still miss
  relevant evidence; a question may then get a confident-but-incomplete
  answer. `evidence_sufficient` reflects what the **model** judged — use
  `evidence_grounding` (deterministic) and `confidence_source` (`"model"` vs
  `"deterministic"`) instead to know whether the platform independently
  anchors any claim.
- **Citation-level, not claim-level.** M3.1's `evidence_grounding` confirms
  that *a* citation survives, not that every claim in the answer is
  individually supported. Claim-level coverage scoring is deferred to M4.
- **CLI kind labels require an intact manifest.** `[code]` / `[documentation]`
  in `ask` output come from the manifest's `file_inventory`; a missing or
  corrupt manifest means citations render unlabelled (never guessed).
- **Local Ollama needed.** Q&A requires `ollama serve` running and the
  configured model pulled (`ollama pull <model>`); otherwise the `/ask`
  endpoint returns 503/404-model errors.
- **Large models are slow on CPU-only boxes.** The default `qwen2.5:3b`
  answers cleanly in ~8s, which is why it is the default. `qwen3:4b`
  (via `--model qwen3:4b` / `OLLAMA_MODEL`) takes ~53-128s on this machine even
  with `OLLAMA_THINK=false`, and its hidden reasoning leaks into the visible
  answer — so prefer the default model for interactive use, and avoid leaving
  two multi-GB models resident at once (they thrash the 14Gi RAM: load spikes,
  answers crawl, output degenerates).
- **One-shot Q&A, no conversation.** No chat history or follow-ups yet.
- **Shallow clone only** (`--depth 1`): no commit history.
- **No authentication**: private repositories are not supported.
- **Framework detection is config-file driven** (unchanged from v0.1).
- **Manifest is the indexer's inventory.** If the working tree changes after
  ingestion, re-indexing uses the persisted manifest; files added later
  require re-ingesting (or a manifest refresh).
- **Empty repositories** may surface as 404 (GitHub `ls-remote` behaviour).

---

## What's next: Milestone 4 (planned, not implemented)

M4 will move past *answering questions* toward *evaluating the project* using
the same evidence discipline:

1. **Fact-checking claims** — validate `readme_claims` against the evidence
   index (rank, then verify with the same citation machinery).
2. **Judging** — structured assessments (design, security, performance,
   maintainability) where every claim is backed by evidence and every gap is
   stated as "evidence insufficient".
3. **Optional semantic retrieval** — embeddings layered on FTS5 to close the
   synonym gap, reusing this milestone's retrieval queries as a re-ranker.

M3 deliberately stops at single-shot, evidence-grounded Q&A. No embeddings,
no agents, no multi-turn conversation, no judging.