# Contributing to recall

Thanks for your interest in contributing.

## Development setup

1. Clone the repo.
2. Use Python 3.10-3.13.
3. Create a virtual environment and install dev deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run checks

```bash
ruff check .
mypy src tests
pytest
```

## Project principles

- Keep the public API small and stable (`store`, `search`, `forget`, `redact`).
- Prioritize local-first behavior and predictable data ownership.
- Preserve cross-provider behavior consistency where possible.

## Pull request guidelines

- Keep PRs focused and scoped to one problem.
- Add tests for behavior changes and bug fixes.
- Update docs/README when user-facing behavior changes.
- Keep compatibility with supported Python versions.

## Commit style

- Use clear, imperative commit messages.
- Include context in the PR description: problem, approach, test evidence.

## Reporting bugs

Please open an issue with:

- Environment details (OS, Python version, package version)
- Minimal reproduction
- Expected vs actual behavior
- Relevant logs/errors

## Feature requests

Open an issue describing:

- The use case
- Why current API is insufficient
- Proposed API shape and tradeoffs
