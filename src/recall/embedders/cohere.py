from __future__ import annotations

import json
import os
from urllib import error, request

from .base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    name = "cohere"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or "embed-english-v3.0"
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise RuntimeError("COHERE_API_KEY is required for Cohere embeddings.")
        self.dimension = 1024

    def embed(self, text: str) -> list[float]:
        payload = json.dumps(
            {
                "model": self.model,
                "input_type": "search_document",
                "texts": [text],
            }
        ).encode("utf-8")
        req = request.Request(
            "https://api.cohere.com/v2/embed",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Cohere embedding request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cohere embedding request failed: {exc}") from exc

        vector: list[float] | None = None
        if isinstance(body, dict):
            embeddings = body.get("embeddings")
            if isinstance(embeddings, dict):
                float_vectors = embeddings.get("float")
                if isinstance(float_vectors, list) and float_vectors:
                    first = float_vectors[0]
                    if isinstance(first, list):
                        vector = first

        if vector is None:
            raise RuntimeError("Unexpected Cohere embedding response shape.")
        return [float(v) for v in vector]
