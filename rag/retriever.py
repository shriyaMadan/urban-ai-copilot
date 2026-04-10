"""Retriever orchestration for embedding + Qdrant search."""
from __future__ import annotations

from typing import List

from rag.embeddings import OpenAICompatibleEmbedder
from rag.schemas import RetrievedChunk
from rag.vector_store import QdrantVectorStore


class RAGRetriever:
    """Minimal retriever used by the copilot agent."""

    def __init__(
        self,
        embedder: OpenAICompatibleEmbedder,
        vector_store: QdrantVectorStore,
        default_top_k: int = 4,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """Retrieve top matching guidance chunks for a query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_vector = self.embedder.embed_query(cleaned_query)
        if not query_vector:
            return []

        limit = top_k if isinstance(top_k, int) and top_k > 0 else self.default_top_k
        return self.vector_store.search(query_vector=query_vector, limit=limit)
