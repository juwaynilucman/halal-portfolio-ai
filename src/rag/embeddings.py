"""Document embedding pipeline for the portfolio RAG system.

Converts raw text documents (Islamic finance rulings, portfolio reports,
ETF fact sheets) into dense vector embeddings that can be stored in
ChromaDB and retrieved by semantic similarity.

Documents are split into overlapping chunks before embedding so that
long documents don't exceed the embedding model's token limit and so
that retrieved chunks retain enough surrounding context to be useful.

Typical usage:
    pipeline = EmbeddingPipeline()
    docs = pipeline.load_documents("docs/islamic_finance_guide.pdf")
    chunks = pipeline.split(docs)
    embeddings = pipeline.embed(chunks)
    # embeddings is a list of (chunk_text, vector) tuples ready for ChromaDB
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single text chunk with associated metadata."""

    text: str
    source: str          # file path or URL the chunk came from
    chunk_index: int     # position of this chunk within the source document
    metadata: dict       # arbitrary key-value pairs (category, date, etc.)


class EmbeddingPipeline:
    """Loads, splits, and embeds documents for the RAG vector store.

    Args:
        chunk_size: Maximum number of characters per chunk (default 800).
        chunk_overlap: Number of overlapping characters between adjacent
            chunks (default 100). Overlap preserves context at boundaries.
        embedding_model: Name of the sentence-transformers model to use for
            local embedding (default "all-MiniLM-L6-v2"). Ignored when
            use_anthropic_embeddings is True.
        use_anthropic_embeddings: If True, use the Anthropic embeddings API
            instead of a local sentence-transformers model.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_anthropic_embeddings: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.use_anthropic_embeddings = use_anthropic_embeddings

    def load_documents(self, path: str | Path) -> list[str]:
        """Load raw text from a file or directory of files.

        Supports .txt, .md, .pdf, and .json formats.

        Args:
            path: Path to a file or directory. Directories are scanned
                recursively for supported file types.

        Returns:
            List of raw document strings.

        TODO: Implement loaders for each format:
              - .txt / .md: open and read directly
              - .pdf: use pypdf or pdfminer.six
              - .json: load and stringify relevant fields
              Use LangChain's DirectoryLoader for recursive directory scans.
        """
        logger.warning("load_documents is a stub — returning empty list")
        return []

    def split(self, documents: list[str], source: str = "unknown") -> list[DocumentChunk]:
        """Split documents into overlapping chunks for embedding.

        Uses a sliding window of size chunk_size with chunk_overlap overlap.

        Args:
            documents: Raw document strings from load_documents().
            source: Label applied to all chunks' metadata (file name / URL).

        Returns:
            List of DocumentChunk objects ready for embedding.

        TODO: Implement the sliding window splitter.
              For each document:
                start = 0
                chunk_index = 0
                while start < len(doc):
                    end = min(start + chunk_size, len(doc))
                    yield DocumentChunk(text=doc[start:end], ...)
                    start += (chunk_size - chunk_overlap)
                    chunk_index += 1
              Prefer LangChain's RecursiveCharacterTextSplitter for
              smarter boundary detection (splits on paragraphs > sentences
              > words before resorting to character boundaries).
        """
        logger.warning("split is a stub — returning empty list")
        return []

    def embed(self, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, list[float]]]:
        """Generate vector embeddings for a list of document chunks.

        Args:
            chunks: DocumentChunk objects from split().

        Returns:
            List of (DocumentChunk, embedding_vector) tuples. The vector
            dimensionality depends on the model (384 for MiniLM-L6-v2).

        TODO: Implement embedding generation.
              Option A (local, free):
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.embedding_model)
                vectors = model.encode([c.text for c in chunks])
              Option B (Anthropic, paid but high-quality):
                Use anthropic client.embeddings.create() when
                self.use_anthropic_embeddings is True.
              Return list of (chunk, vector.tolist()) tuples.
        """
        logger.warning("embed is a stub — returning empty list")
        return []

    def load_portfolio_documents(
        self,
        portfolio_report: dict,
        screening_results: dict,
    ) -> list[DocumentChunk]:
        """Convert live portfolio data into embeddable text chunks.

        This lets the RAG chain answer questions about the user's current
        portfolio without requiring separate document uploads.

        Args:
            portfolio_report: The PortfolioPlan dict (serialised from builder).
            screening_results: Ticker -> ScreeningResult dict from the screener.

        Returns:
            List of DocumentChunk objects describing the current portfolio.

        TODO: Serialise portfolio_report into natural-language sentences:
              "AAPL is allocated $1,234.56 (Technology sector, 8.2% weight)"
              Include screening summaries:
              "MSFT has a 3% non-permissible income fraction — purification
               of $X.XX is due this quarter."
              Split into chunks and return.
        """
        logger.warning("load_portfolio_documents is a stub — returning empty list")
        return []
