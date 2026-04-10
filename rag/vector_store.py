"""Qdrant local vector store wrapper for RAG."""
from __future__ import annotations

import builtins
from threading import Lock
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag.chunker import TextChunk
from rag.schemas import RetrievedChunk
from utils.config import resolve_project_path


if not hasattr(builtins, "_urban_qdrant_client_cache_lock"):
    builtins._urban_qdrant_client_cache_lock = Lock()

if not hasattr(builtins, "_urban_qdrant_local_client_cache"):
    builtins._urban_qdrant_local_client_cache = {}

_CLIENT_CACHE_LOCK: Lock = builtins._urban_qdrant_client_cache_lock
_LOCAL_CLIENT_CACHE: dict[str, QdrantClient] = builtins._urban_qdrant_local_client_cache


class QdrantVectorStore:
    """Minimal wrapper for local persistent Qdrant operations."""

    def __init__(self, qdrant_path: str, collection_name: str) -> None:
        self.collection_name = collection_name
        normalized_path = resolve_project_path(qdrant_path)
        self.client = self._get_or_create_local_client(normalized_path)

    def _get_or_create_local_client(self, normalized_path: str) -> QdrantClient:
        """Reuse one local Qdrant client per path within a process.

        Local Qdrant storage uses file locks and does not allow multiple client
        instances for the same path in one process lifecycle. Streamlit reruns can
        reload modules, so the cache is stored on `builtins` to survive re-imports
        within the same Python process.
        """
        with _CLIENT_CACHE_LOCK:
            existing = _LOCAL_CLIENT_CACHE.get(normalized_path)
            if existing is not None:
                return existing

            client = QdrantClient(path=normalized_path)
            _LOCAL_CLIENT_CACHE[normalized_path] = client
            return client

    def recreate_collection(self, vector_size: int) -> None:
        """Recreate collection to keep ingestion deterministic for MVP."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

    def collection_exists(self) -> bool:
        """Return whether the configured collection already exists."""
        try:
            if hasattr(self.client, "collection_exists"):
                return bool(self.client.collection_exists(self.collection_name))  # type: ignore[attr-defined]

            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def count_points(self) -> int:
        """Return number of indexed points in the collection."""
        if not self.collection_exists():
            return 0

        try:
            response = self.client.count(
                collection_name=self.collection_name,
                exact=False,
            )
            return int(getattr(response, "count", 0) or 0)
        except Exception:
            return 0

    def upsert_chunks(self, chunks: List[TextChunk], vectors: List[List[float]]) -> None:
        """Insert chunk vectors and metadata into Qdrant."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors length mismatch.")

        points: List[qm.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                qm.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], limit: int = 4) -> List[RetrievedChunk]:
        """Search nearest chunks for the query vector."""
        if not query_vector:
            return []

        if hasattr(self.client, "search"):
            hits = self.client.search(  # type: ignore[attr-defined]
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            hits = getattr(response, "points", [])

        results: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            results.append(
                RetrievedChunk(
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "unknown")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    score=float(hit.score or 0.0),
                    metadata=metadata,
                )
            )

        return results
