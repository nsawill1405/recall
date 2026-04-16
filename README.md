# recall

Local-first memory for Python apps and AI agents. One file on disk, with a tiny API: `store()`, `search()`, `forget()`, `redact()`.

`recall-ai` (PyPI) installs with import path `recall`.

Use it for:

- AI memory (agent context, user preferences, conversation facts)
- General app memory (saved notes, profile facts, project metadata, app-level recall)

## Install

```bash
pip install recall-ai
```

Optional local sentence-transformer embeddings:

```bash
pip install "recall-ai[local]"
```

## 10-second quickstart

```python
from recall import Memory

mem = Memory()  # ~/.recall/default.db

mem.store("User prefers short responses")
mem.store("Project uses FastAPI", tags=["project"])
mem.store("Favorite coffee: flat white", tags=["profile"])

results = mem.search("what does the user prefer?", top_k=3)
for r in results:
    print(r.text, r.score)

mem.forget(tag="project")
```

## API

### Sync

```python
from recall import Memory

mem = Memory(path="./app.db", namespace="user_abc")
mem.store("Remember this", tags=["note"], ttl_days=7)
mem.redact(id="<memory-id>", remove="sensitive token")
results = mem.search("what should we remember?", top_k=5, tags=["note"])
deleted = mem.forget(id=results[0].id)
```

### Async

```python
from recall import AsyncMemory

mem = AsyncMemory()
await mem.store("Remember this")
await mem.redact(id="<memory-id>", remove="sensitive token")
results = await mem.search("what should we remember?")
await mem.forget(id=results[0].id)
```

## How Search Works

`search(query, top_k, tags)` works in four steps:

1. Embed the query with the active embedder (OpenAI/Anthropic/Cohere/local/custom).
2. Retrieve candidates from the current namespace, excluding expired rows:
   - If `sqlite-vec` is available, candidate retrieval is ANN via `vec0` cosine distance.
   - Otherwise, recall falls back to SQLite table candidates.
3. Apply optional tag filtering.
4. Compute cosine similarity in Python for candidates, convert to score in `[0, 1]`, sort descending, return top `k`.

Versioned memory behavior:

- Search only returns `is_current = 1` memory versions.
- Redacted/superseded historical versions are retained internally but hidden from normal search/list output.

## Provider behavior

Provider auto-detection order:

1. `OPENAI_API_KEY`
2. `ANTHROPIC_API_KEY`
3. `COHERE_API_KEY`
4. Local fallback

You can override explicitly:

```python
Memory(embedder="openai", model="text-embedding-3-small")
Memory(embedder="local", model="sentence-transformers/all-MiniLM-L6-v2")
```

Default local mode uses sentence-transformers when available. If not installed, it falls back to a deterministic hash embedder so the library stays usable offline.

## CLI

```bash
recall search "user preferences" --top-k 5
recall list --limit 20
recall forget --id <memory-id>
recall forget --tag project
recall redact --id <memory-id> --remove "secret"
recall stats
recall rebuild-index --embedder local
```

Common flags:

- `--path ./app.db`
- `--namespace user_abc`
- `--embedder openai|anthropic|cohere|local`
- `--model <provider-model-name>`

## Embedding migration and dimension mismatch

Embedding dimension is persisted in DB metadata. Opening a DB with a different embedding dimension fails fast.

Use:

```bash
recall rebuild-index --path ./app.db --namespace default --embedder local
```

This command updates metadata and recomputes every embedding in the namespace.

## Notes

- Storage is SQLite + `sqlite-vec` when available.
- If `sqlite-vec` is unavailable at runtime, recall falls back to exact cosine search over stored embeddings.
- This release is Python-only (`recall-js` is out of scope for v0.1).
