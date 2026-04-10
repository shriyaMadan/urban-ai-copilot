"""Embedding client utilities for RAG."""
from __future__ import annotations

from typing import List
from urllib.parse import quote_plus

import requests


class OpenAICompatibleEmbedder:
    """Minimal OpenAI-compatible embeddings client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.timeout = timeout

        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("Embedding configuration is incomplete.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using an OpenAI-compatible endpoint."""
        if not texts:
            return []

        if self._is_gemini_openai_base_url():
            return self._embed_texts_gemini_native(texts)

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        raw_items = payload.get("data", [])
        vectors: List[List[float]] = []
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                embedding = item.get("embedding", [])
                if isinstance(embedding, list) and embedding:
                    vectors.append([float(v) for v in embedding])

        if len(vectors) != len(texts):
            raise ValueError("Embedding provider returned unexpected vector count.")

        return vectors

    def _is_gemini_openai_base_url(self) -> bool:
        """Detect Gemini OpenAI-compatible base URLs."""
        normalized = self.base_url.lower()
        return (
            "generativelanguage.googleapis.com" in normalized
            and normalized.endswith("/openai")
        )

    def _embed_texts_gemini_native(self, texts: List[str]) -> List[List[float]]:
        """Embed via Gemini native endpoint (fallback for /openai embeddings 404)."""
        model_resource = (
            self.model if self.model.startswith("models/") else f"models/{self.model}"
        )
        api_root = self.base_url.rsplit("/openai", 1)[0]
        endpoint = (
            f"{api_root}/{model_resource}:batchEmbedContents"
            f"?key={quote_plus(self.api_key)}"
        )

        payload = {
            "requests": [
                {
                    "model": model_resource,
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
        }

        response = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        response_payload = response.json()

        raw_embeddings = response_payload.get("embeddings", [])
        vectors: List[List[float]] = []
        if isinstance(raw_embeddings, list):
            for item in raw_embeddings:
                if not isinstance(item, dict):
                    continue
                values = item.get("values", [])
                if isinstance(values, list) and values:
                    vectors.append([float(v) for v in values])

        if len(vectors) != len(texts):
            raise ValueError("Gemini embedding provider returned unexpected vector count.")

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []
