from __future__ import annotations

import json
import os
from urllib import error, request

from .base import BaseEmbedder


class AnthropicEmbedder(BaseEmbedder):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or "voyage-3-lite"
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic embeddings.")
        self.api_key: str = resolved_key
        self.dimension = 1024

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": [text]}).encode("utf-8")
        req = request.Request(
            "https://api.anthropic.com/v1/embeddings",
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Anthropic embedding request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Anthropic embedding request failed: {exc}") from exc

        vector: list[float] | None = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                    vector = first["embedding"]
            if vector is None and isinstance(body.get("embedding"), list):
                vector = body["embedding"]

        if vector is None:
            raise RuntimeError("Unexpected Anthropic embedding response shape.")
        return [float(v) for v in vector]
