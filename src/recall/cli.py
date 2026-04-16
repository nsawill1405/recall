from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .errors import DimensionMismatchError, RecallError
from .memory import Memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recall",
        description="Local-first memory for Python applications.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="SQLite DB path (default: ~/.recall/default.db)",
    )
    parser.add_argument("--namespace", default="default", help="Memory namespace")
    parser.add_argument("--embedder", default=None, help="openai | anthropic | cohere | local")
    parser.add_argument("--model", default=None, help="Embedding model override")

    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Search memories")
    search.add_argument("query", help="Search query text")
    search.add_argument("--top-k", type=int, default=5, help="Number of results")
    search.add_argument("--tag", action="append", default=None, help="Filter by tag (repeatable)")
    search.add_argument(
        "--reranker",
        default=None,
        choices=["local"],
        help="Optional reranker strategy",
    )

    find = subparsers.add_parser("find", help="Find the best memory match")
    find.add_argument("query", help="Query text")
    find.add_argument("--tag", action="append", default=None, help="Filter by tag (repeatable)")
    find.add_argument("--min-score", type=float, default=0.0, help="Minimum accepted score")
    find.add_argument(
        "--reranker",
        default=None,
        choices=["local"],
        help="Optional reranker strategy",
    )

    list_cmd = subparsers.add_parser("list", help="List latest memories")
    list_cmd.add_argument("--limit", type=int, default=20, help="Max memories")

    forget = subparsers.add_parser("forget", help="Delete memory by id or tag")
    forget.add_argument("--id", default=None, help="Memory ID")
    forget.add_argument("--tag", default=None, help="Delete memories by tag")

    redact = subparsers.add_parser("redact", help="Redact literal text from a memory")
    redact.add_argument("--id", required=True, help="Current memory ID")
    redact.add_argument("--remove", required=True, help="Literal text to remove")

    edit = subparsers.add_parser("edit", help="Create an edited current version of a memory")
    edit.add_argument("--id", required=True, help="Current memory ID")
    edit.add_argument("--text", default=None, help="Replacement text")
    edit.add_argument("--tag", action="append", default=None, help="Replacement tag (repeatable)")
    edit.add_argument("--ttl-days", type=int, default=None, help="Replacement TTL in days")
    edit.add_argument(
        "--clear-tags",
        action="store_true",
        help="Clear all tags",
    )
    edit.add_argument(
        "--clear-ttl",
        action="store_true",
        help="Remove existing TTL",
    )

    subparsers.add_parser("stats", help="Show memory stats")
    subparsers.add_parser("prune", help="Delete fully expired memory lineages")
    export_cmd = subparsers.add_parser("export", help="Export namespace memories as JSONL")
    export_cmd.add_argument("--out", required=True, help="Output JSONL path")
    export_cmd.add_argument(
        "--include-history",
        action="store_true",
        help="Include internal non-current lineage rows",
    )

    import_cmd = subparsers.add_parser("import", help="Import memories from JSONL")
    import_cmd.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    import_cmd.add_argument(
        "--conflict",
        choices=["skip", "overwrite", "new"],
        default="skip",
        help="Conflict strategy when IDs already exist",
    )

    subparsers.add_parser("rebuild-index", help="Rebuild embeddings and vector index")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    allow_mismatch = args.command == "rebuild-index"

    try:
        memory = Memory(
            path=args.path,
            namespace=args.namespace,
            embedder=args.embedder,
            model=args.model,
            _allow_dimension_mismatch=allow_mismatch,
        )

        try:
            if args.command == "search":
                return _run_search(memory, args.query, args.top_k, args.tag, args.reranker)
            if args.command == "find":
                return _run_find(memory, args.query, args.tag, args.min_score, args.reranker)
            if args.command == "list":
                return _run_list(memory, args.limit)
            if args.command == "forget":
                return _run_forget(memory, args.id, args.tag)
            if args.command == "redact":
                return _run_redact(memory, args.id, args.remove)
            if args.command == "edit":
                return _run_edit(
                    memory,
                    memory_id=args.id,
                    text=args.text,
                    tags=args.tag,
                    ttl_days=args.ttl_days,
                    clear_tags=args.clear_tags,
                    clear_ttl=args.clear_ttl,
                )
            if args.command == "stats":
                return _run_stats(memory)
            if args.command == "prune":
                return _run_prune(memory)
            if args.command == "export":
                return _run_export(memory, out_path=args.out, include_history=args.include_history)
            if args.command == "import":
                return _run_import(memory, in_path=args.in_path, conflict=args.conflict)
            if args.command == "rebuild-index":
                return _run_rebuild(memory)
            parser.error(f"unknown command: {args.command}")
        finally:
            memory.close()

    except DimensionMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (RecallError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _run_search(
    memory: Memory,
    query: str,
    top_k: int,
    tags: list[str] | None,
    reranker: str | None,
) -> int:
    results = memory.search(query, top_k=top_k, tags=tags, reranker=reranker)
    if not results:
        print("No memories found.")
        return 0

    for item in results:
        print(
            f"[{item.score:.3f}] {item.id}  {item.text} "
            f"(tags={item.tags}, created={_fmt(item.created_at)})"
        )
    return 0


def _run_find(
    memory: Memory,
    query: str,
    tags: list[str] | None,
    min_score: float,
    reranker: str | None,
) -> int:
    result = memory.find(query=query, tags=tags, min_score=min_score, reranker=reranker)
    if result is None:
        print("No memory found.")
        return 0
    print(
        f"[{result.score:.3f}] {result.id}  {result.text} "
        f"(tags={result.tags}, created={_fmt(result.created_at)})"
    )
    return 0


def _run_list(memory: Memory, limit: int) -> int:
    entries = memory.list(limit=limit)
    if not entries:
        print("No memories stored.")
        return 0

    for entry in entries:
        print(
            f"{entry['id']}  {entry['text']} "
            f"(tags={entry['tags']}, created={_fmt(entry['created_at'])}, "
            f"expires={_fmt(entry['expires_at'])})"
        )
    return 0


def _run_forget(memory: Memory, memory_id: str | None, tag: str | None) -> int:
    deleted = memory.forget(id=memory_id, tag=tag)
    print(f"Deleted {deleted} memories.")
    return 0


def _run_redact(memory: Memory, memory_id: str, remove: str) -> int:
    new_id, removed_count = memory._redact_internal(id=memory_id, remove=remove)
    print(f"Created redacted memory {new_id} (removed {removed_count} occurrence(s)).")
    return 0


def _run_edit(
    memory: Memory,
    memory_id: str,
    text: str | None,
    tags: list[str] | None,
    ttl_days: int | None,
    clear_tags: bool,
    clear_ttl: bool,
) -> int:
    if clear_tags and tags is not None:
        raise ValueError("cannot use --tag with --clear-tags")
    if clear_ttl and ttl_days is not None:
        raise ValueError("cannot use --ttl-days with --clear-ttl")

    effective_tags = [] if clear_tags else tags
    effective_ttl_days = 0 if clear_ttl else ttl_days
    if text is None and effective_tags is None and effective_ttl_days is None:
        raise ValueError("edit requires at least one change")

    new_id = memory.edit(
        id=memory_id,
        text=text,
        tags=effective_tags,
        ttl_days=effective_ttl_days,
    )
    print(f"Created edited memory {new_id}.")
    return 0


def _run_stats(memory: Memory) -> int:
    stats = memory.stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


def _run_prune(memory: Memory) -> int:
    deleted = memory.prune_expired()
    print(f"Pruned {deleted} memories.")
    return 0


def _run_export(memory: Memory, out_path: str, include_history: bool) -> int:
    exported = memory.export_jsonl(path=out_path, include_history=include_history)
    print(f"Exported {exported} memories to {out_path}.")
    return 0


def _run_import(memory: Memory, in_path: str, conflict: str) -> int:
    imported = memory.import_jsonl(path=in_path, conflict=conflict)
    print(f"Imported {imported} memories from {in_path}.")
    return 0


def _run_rebuild(memory: Memory) -> int:
    rebuilt = memory.rebuild_index()
    print(f"Rebuilt embeddings for {rebuilt} memories.")
    return 0


def _fmt(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
