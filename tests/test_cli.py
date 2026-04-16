from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from recall import Memory


def _run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "recall.cli", *args],
        cwd=os.path.dirname(__file__) + "/..",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_search_list_forget_stats_rebuild(db_path: Path) -> None:
    mem = Memory(
        path=str(db_path),
        namespace="default",
        embedder="local",
        model="hash-embedding-v1",
    )
    try:
        memory_id = mem.store("CLI seeded memory", tags=["cli"])
        assert memory_id
    finally:
        mem.close()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("COHERE_API_KEY", None)

    base_args = [
        "--path",
        str(db_path),
        "--namespace",
        "default",
        "--embedder",
        "local",
        "--model",
        "hash-embedding-v1",
    ]

    list_run = _run_cli(*base_args, "list", "--limit", "5", env=env)
    assert list_run.returncode == 0
    assert "CLI seeded memory" in list_run.stdout

    search_run = _run_cli(*base_args, "search", "seeded", "--top-k", "3", env=env)
    assert search_run.returncode == 0
    assert "CLI seeded memory" in search_run.stdout

    stats_run = _run_cli(*base_args, "stats", env=env)
    assert stats_run.returncode == 0
    assert '"total": 1' in stats_run.stdout

    forget_run = _run_cli(*base_args, "forget", "--tag", "cli", env=env)
    assert forget_run.returncode == 0
    assert "Deleted 1 memories." in forget_run.stdout

    rebuild_run = _run_cli(*base_args, "rebuild-index", env=env)
    assert rebuild_run.returncode == 0
    assert "Rebuilt embeddings" in rebuild_run.stdout
