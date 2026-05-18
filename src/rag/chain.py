"""LangChain RAG chain connecting ChromaDB retrieval with Claude.

Implements a retrieval-augmented generation pipeline:
  1. User asks a question (portfolio or Islamic finance).
  2. VectorStore retrieves the top-K most relevant chunks.
  3. Chunks are formatted into a context block.
  4. Claude receives a system prompt + context + user question and generates
     a grounded, citation-aware answer.

The system prompt instructs Claude to:
  - Only answer from the provided context (no hallucination).
  - Cite the source of each claim (e.g., "According to [docs/fiqh_guide.md]…").
  - Decline questions outside the scope of Islamic finance and halal investing.
  - Default to the more conservative scholarly opinion when rulings differ.

Typical usage:
    chain = RAGChain()
    answer = chain.ask("How much should I purify from MSFT dividends?")
    print(answer.text)
    print(answer.sources)
"""

import logging
from dataclasses import dataclass, field

import anthropic

from src.rag.vector_store import RetrievedChunk, VectorStore
from src.utils.config import ANTHROPIC_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI assistant for a Sharia-compliant portfolio management application.
Your role is to help Muslim investors understand their portfolio allocations,
purification obligations, and principles of Islamic finance (halal investing).

Guidelines:
- Answer ONLY from the context provided below. Do not add information not present in context.
- Cite the source of each factual claim using the format [source].
- When scholarly opinions differ, present the more conservative view as primary
  and note the alternative.
- If the context does not contain enough information to answer, say so clearly.
- Keep answers concise and practical. The user wants actionable guidance.
- Never give fatwa (religious rulings) — refer users to a qualified scholar
  for personal religious decisions.
"""


@dataclass
class RAGAnswer:
    """The answer returned by the RAG chain."""

    text: str
    sources: list[str] = field(default_factory=list)      # unique source files cited
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class RAGChain:
    """Orchestrates retrieval and generation for the portfolio Q&A feature.

    Args:
        vector_store: VectorStore instance to pull context from.
        model: Anthropic model ID to use for generation.
        top_k: Number of context chunks to retrieve per query.
        max_context_chars: Truncate context to this many characters to stay
            within token limits (approximate: 1 token ≈ 4 chars).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        model: str = LLM_MODEL,
        top_k: int = 6,
        max_context_chars: int = 12_000,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.model = model
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def ask(self, question: str, portfolio_context: bool = True) -> RAGAnswer:
        """Answer a natural language question using RAG.

        Args:
            question: The user's question in plain English.
            portfolio_context: If True, searches the portfolio collection in
                addition to the knowledge collection so questions about the
                user's specific holdings are grounded in their current data.

        Returns:
            RAGAnswer with the LLM-generated response and retrieved sources.
        """
        # Retrieve relevant context
        if portfolio_context:
            chunks = self.vector_store.retrieve_from_both(question, top_k=self.top_k)
        else:
            chunks = self.vector_store.retrieve(question, top_k=self.top_k)

        context_block = self._format_context(chunks)
        user_message = self._format_user_message(question, context_block)

        logger.debug("Sending RAG query to %s (%d chunks retrieved)", self.model, len(chunks))

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer_text = response.content[0].text if response.content else ""
        sources = list({chunk.source for chunk in chunks})

        return RAGAnswer(
            text=answer_text,
            sources=sources,
            retrieved_chunks=chunks,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def ask_stream(self, question: str, portfolio_context: bool = True):
        """Stream a RAG answer token-by-token (for real-time UI updates).

        Args:
            question: The user's question.
            portfolio_context: Whether to search the portfolio collection too.

        Yields:
            Text delta strings as they are generated by the model.

        TODO: Implement using self._client.messages.stream() context manager.
              Yield chunk.delta.text for each content_block_delta event.
              This enables the FastAPI endpoint to use StreamingResponse.
        """
        logger.warning("ask_stream is a stub — falling back to non-streaming ask()")
        answer = self.ask(question, portfolio_context)
        yield answer.text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build the context block injected into the LLM prompt.

        Each chunk is preceded by its source so Claude can cite it.

        Args:
            chunks: Retrieved chunks sorted by relevance.

        Returns:
            Formatted multi-line context string (truncated if too long).
        """
        parts = []
        total_chars = 0

        for chunk in chunks:
            block = f"[{chunk.source}]\n{chunk.text}\n"
            if total_chars + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)

        return "\n---\n".join(parts)

    def _format_user_message(self, question: str, context: str) -> str:
        """Combine context and question into the final user message.

        Args:
            question: The user's original question.
            context: Formatted context block from _format_context().

        Returns:
            Complete user message string for the API call.
        """
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above. "
            f"Cite sources using [source] notation."
        )
