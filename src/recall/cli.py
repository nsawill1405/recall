from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from .errors import DimensionMismatchError, RecallError
from .memory import Memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall", description="Local-first AI memory library")
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

    list_cmd = subparsers.add_parser("list", help="List latest memories")
    list_cmd.add_argument("--limit", type=int, default=20, help="Max memories")

    forget = subparsers.add_parser("forget", help="Delete memory by id or tag")
    forget.add_argument("--id", default=None, help="Memory ID")
    forget.add_argument("--tag", default=None, help="Delete memories by tag")

    subparsers.add_parser("stats", help="Show memory stats")

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
                return _run_search(memory, args.query, args.top_k, args.tag)
            if args.command == "list":
                return _run_list(memory, args.limit)
            if args.command == "forget":
                return _run_forget(memory, args.id, args.tag)
            if args.command == "stats":
                return _run_stats(memory)
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


def _run_search(memory: Memory, query: str, top_k: int, tags: list[str] | None) -> int:
    results = memory.search(query, top_k=top_k, tags=tags)
    if not results:
        print("No memories found.")
        return 0

    for item in results:
        print(
            f"[{item.score:.3f}] {item.id}  {item.text} "
            f"(tags={item.tags}, created={_fmt(item.created_at)})"
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


def _run_stats(memory: Memory) -> int:
    stats = memory.stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
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
