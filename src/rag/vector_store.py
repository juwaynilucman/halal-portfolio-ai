"""ChromaDB vector store setup, indexing, and retrieval.

Manages the persistent ChromaDB collection that backs the RAG chain.
Documents are upserted (not duplicated) using content hashes as IDs,
so re-running the indexer on the same corpus is safe and idempotent.

Two collections are maintained:
  - "portfolio" — current allocation, screening results, redistribution reports
  - "knowledge" — static Islamic finance rulings, ETF fact sheets, scholarly opinions

Typical usage:
    store = VectorStore()
    store.index(chunks_with_embeddings, collection="knowledge")
    results = store.retrieve("What makes a stock halal?", top_k=5)
"""

import hashlib
import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from src.rag.embeddings import DocumentChunk, EmbeddingPipeline
from src.utils.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_PORTFOLIO = "portfolio"
COLLECTION_KNOWLEDGE = "knowledge"


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity search."""

    text: str
    source: str
    score: float          # cosine similarity (0–1, higher is more relevant)
    metadata: dict


class VectorStore:
    """Manages ChromaDB collections for the RAG pipeline.

    Args:
        persist_dir: Directory where ChromaDB stores its data files.
            Should be excluded from git (already in .gitignore).
        embedding_pipeline: EmbeddingPipeline instance used to embed
            query strings at retrieval time (must use the same model
            that was used at indexing time).
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_pipeline: EmbeddingPipeline | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, name: str) -> chromadb.Collection:
        """Return (or create) a named ChromaDB collection.

        Args:
            name: Collection name — use COLLECTION_PORTFOLIO or COLLECTION_KNOWLEDGE.

        Returns:
            ChromaDB Collection object.
        """
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def index(
        self,
        chunks_with_embeddings: list[tuple[DocumentChunk, list[float]]],
        collection: str = COLLECTION_KNOWLEDGE,
    ) -> int:
        """Upsert document chunks and their embeddings into ChromaDB.

        Uses a SHA-256 hash of the chunk text as the document ID so that
        re-indexing the same content is idempotent.

        Args:
            chunks_with_embeddings: Output from EmbeddingPipeline.embed().
            collection: Target collection name.

        Returns:
            Number of chunks upserted.

        TODO: Implement the upsert loop:
              col = self._get_collection(collection)
              ids = [_hash(chunk.text) for chunk, _ in chunks_with_embeddings]
              documents = [chunk.text for chunk, _ in chunks_with_embeddings]
              embeddings = [vec for _, vec in chunks_with_embeddings]
              metadatas = [chunk.metadata | {"source": chunk.source}
                           for chunk, _ in chunks_with_embeddings]
              col.upsert(ids=ids, documents=documents,
                         embeddings=embeddings, metadatas=metadatas)
        """
        logger.warning("index is a stub — no data written to ChromaDB")
        return 0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection: str = COLLECTION_KNOWLEDGE,
    ) -> list[RetrievedChunk]:
        """Retrieve the most semantically similar chunks for a query.

        Args:
            query: Natural language question from the user.
            top_k: Number of chunks to return (more = more context but
                increases LLM prompt size).
            collection: Collection to search — use COLLECTION_PORTFOLIO for
                questions about the user's current holdings.

        Returns:
            List of RetrievedChunk objects sorted by similarity descending.

        TODO: Implement retrieval:
              query_embedding = self.embedding_pipeline.embed(
                  [DocumentChunk(text=query, source="query", chunk_index=0, metadata={})]
              )[0][1]
              col = self._get_collection(collection)
              results = col.query(
                  query_embeddings=[query_embedding],
                  n_results=top_k,
                  include=["documents", "metadatas", "distances"],
              )
              Convert distances to similarity scores (cosine: score = 1 - distance).
        """
        logger.warning("retrieve is a stub — returning empty results")
        return []

    def retrieve_from_both(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve from both portfolio and knowledge collections and merge.

        Args:
            query: User's natural language question.
            top_k: Number of results to pull from each collection.

        Returns:
            Combined and re-ranked list of up to 2×top_k RetrievedChunk objects.
        """
        portfolio_results = self.retrieve(query, top_k, COLLECTION_PORTFOLIO)
        knowledge_results = self.retrieve(query, top_k, COLLECTION_KNOWLEDGE)
        combined = portfolio_results + knowledge_results
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:top_k]

    def clear_collection(self, collection: str) -> None:
        """Delete all documents from a collection (non-destructive to schema).

        Args:
            collection: Collection name to clear.
        """
        col = self._get_collection(collection)
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])
            logger.info("Cleared %d documents from collection '%s'", len(existing["ids"]), collection)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        """Return a SHA-256 hex digest of text for use as a document ID."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
