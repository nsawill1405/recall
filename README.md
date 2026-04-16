![recall](logo.png)

# recall

Local-first memory for Python apps and AI agents. One file on disk, with a compact API:
`store()`, `store_many()`, `find()`, `search()`, `edit()`, `redact()`, `forget()`,
`prune_expired()`, `export_jsonl()`, `import_jsonl()`.

`recall` installs with import path `recall`.

Use it for:

- AI memory (agent context, user preferences, conversation facts)
- General app memory (saved notes, profile facts, project metadata, app-level recall)

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/nsawill1405/recall.git
cd recall
pip install -e .
```

Optional local sentence-transformer embeddings:

```bash
pip install -e ".[local]"
```

## 10-second quickstart

```python
from recall import Memory, StoreItem

mem = Memory()  # ~/.recall/default.db

mem.store("User prefers short responses")
mem.store_many(
    [
        StoreItem(text="Project uses FastAPI", tags=["project"]),
        StoreItem(text="Favorite coffee: flat white", tags=["profile"]),
    ]
)

best = mem.find("what does the user prefer?")
if best:
    print(best.text, best.score)
    updated = mem.edit(best.id, text="User prefers concise responses")
    mem.redact(updated, "concise ")
mem.forget(tag="project")
```

## API

### Sync

```python
from recall import Memory, StoreItem

mem = Memory(path="./app.db", namespace="user_abc")
mem.store("Remember this", tags=["note"], ttl_days=7)
mem.store_many([StoreItem(text="Batch memory", tags=["batch"])])
mem.redact(id="<memory-id>", remove="sensitive token")
mem.edit(id="<memory-id>", text="Updated memory text", tags=["note", "updated"], ttl_days=14)
best = mem.find("what should we remember?", tags=["note"], min_score=0.45)
if best:
    deleted = mem.forget(id=best.id)
pruned = mem.prune_expired()
mem.export_jsonl("./backup.jsonl")
mem.import_jsonl("./backup.jsonl", conflict="skip")
```

### Async

```python
from recall import AsyncMemory, StoreItem

mem = AsyncMemory()
await mem.store("Remember this")
await mem.store_many([StoreItem(text="Batch memory")])
await mem.redact(id="<memory-id>", remove="sensitive token")
await mem.edit(id="<memory-id>", text="Updated memory text")
best = await mem.find("what should we remember?", min_score=0.45)
if best:
    await mem.forget(id=best.id)
await mem.prune_expired()
await mem.export_jsonl("./backup.jsonl")
await mem.import_jsonl("./backup.jsonl", conflict="skip")
```

Use `find()` when you want one best result. Use `search()` when you need ranked lists.

## API Reference

`Memory(...)` / `AsyncMemory(...)`
- `path`: custom SQLite path (default `~/.recall/default.db`)
- `namespace`: logical partition for multi-tenant memory
- `embedder`, `model`: explicit provider/model override
- `max_scan_rows`: optional fallback full-scan cap

Methods:
- `store(text, tags=None, ttl_days=None) -> str`
- `store_many(items: list[StoreItem]) -> list[str]`
- `find(query, tags=None, min_score=0.0, reranker=None) -> MemoryResult | None`
- `search(query, top_k=5, tags=None, reranker=None) -> list[MemoryResult]`
- `edit(id, text=None, tags=None, ttl_days=None) -> str`
- `redact(id, remove) -> str`
- `forget(id=None, tag=None) -> int`
- `prune_expired() -> int`
- `export_jsonl(path, namespace=None, include_history=False) -> int`
- `import_jsonl(path, namespace=None, conflict="skip|overwrite|new") -> int`

`StoreItem`:
- `StoreItem(text: str, tags: list[str] | None = None, ttl_days: int | None = None)`

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
recall edit --id <memory-id> --text "updated text" --tag note --ttl-days 7
recall redact --id <memory-id> --remove "secret"
recall prune
recall export --out memories.jsonl
recall import --in memories.jsonl --conflict skip
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
- This release is Python-only (`recall-js` is out of scope).
