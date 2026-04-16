# recall

Local-first AI memory for Python. One file on disk, three methods: `store()`, `search()`, `forget()`.

`recall-ai` (PyPI) installs with import path `recall`.

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
results = mem.search("what should we remember?", top_k=5, tags=["note"])
deleted = mem.forget(id=results[0].id)
```

### Async

```python
from recall import AsyncMemory

mem = AsyncMemory()
await mem.store("Remember this")
results = await mem.search("what should we remember?")
await mem.forget(id=results[0].id)
```

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
