# recall

Local-first memory for Python apps and AI agents. One file on disk, with a tiny API: `store()`, `find()`, `search()`, `forget()`, `redact()`.

`recall` installs with import path `recall`.

Use it for:

- AI memory (agent context, user preferences, conversation facts)
- General app memory (saved notes, profile facts, project metadata, app-level recall)

## Install

```bash
pip install recall
```

Optional local sentence-transformer embeddings:

```bash
pip install "recall[local]"
```

## 10-second quickstart

```python
from recall import Memory

mem = Memory()  # ~/.recall/default.db

mem.store("User prefers short responses")
mem.store("Project uses FastAPI", tags=["project"])
mem.store("Favorite coffee: flat white", tags=["profile"])

best = mem.find("what does the user prefer?")
if best:
    print(best.text, best.score)

mem.forget(tag="project")
```

## API

### Sync

```python
from recall import Memory

mem = Memory(path="./app.db", namespace="user_abc")
mem.store("Remember this", tags=["note"], ttl_days=7)
mem.redact(id="<memory-id>", remove="sensitive token")
best = mem.find("what should we remember?", tags=["note"], min_score=0.45)
if best:
    deleted = mem.forget(id=best.id)
```

### Async

```python
from recall import AsyncMemory

mem = AsyncMemory()
await mem.store("Remember this")
await mem.redact(id="<memory-id>", remove="sensitive token")
best = await mem.find("what should we remember?", min_score=0.45)
if best:
    await mem.forget(id=best.id)
```

Use `find()` when you want one best result. Use `search()` when you need ranked lists.

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
The hash fallback is non-semantic and intended as a last-resort dev fallback, not production-quality retrieval.

## CLI

```bash
recall find "user preferences" --min-score 0.5
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
- `--reranker local` (optional lexical reranking for `find`/`search`)

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
