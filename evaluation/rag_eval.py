"""RAG evaluation framework — retrieval accuracy, answer relevance, faithfulness.

Runs a set of test queries against the live RAG pipeline and reports three
metrics:

  Retrieval accuracy
    Given a query, do the top-K retrieved chunks contain the expected answer?
    Measured as recall@K: fraction of queries where at least one relevant
    chunk appears in the top-K results.

  Answer relevance
    Does the LLM's answer actually address the user's question?
    Scored with a lightweight LLM call: ask Claude to rate relevance 1–5.

  Faithfulness
    Is every claim in the answer grounded in the retrieved context?
    Scored with a lightweight LLM call: ask Claude to identify claims not
    supported by the context and compute a faithfulness ratio.

Usage:
    python -m evaluation.rag_eval --queries evaluation/test_queries.json
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from src.rag.chain import RAGChain
from src.rag.vector_store import VectorStore
from src.utils.config import ANTHROPIC_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Evaluation result for a single test query."""

    query: str
    expected_answer_snippet: str
    actual_answer: str
    retrieved_sources: list[str]
    retrieved_texts: list[str]
    retrieval_hit: bool        # True if any chunk contains expected snippet
    relevance_score: float     # 1–5 LLM-judged score, normalised to 0–1
    faithfulness_score: float  # 0–1 fraction of claims grounded in context


@dataclass
class EvaluationReport:
    """Aggregated metrics across all test queries."""

    results: list[QueryResult] = field(default_factory=list)
    recall_at_k: float = 0.0          # fraction of queries with a retrieval hit
    mean_relevance: float = 0.0       # average relevance score (0–1)
    mean_faithfulness: float = 0.0    # average faithfulness score (0–1)
    num_queries: int = 0


class RAGEvaluator:
    """Runs the RAG evaluation suite against a set of test queries.

    Args:
        rag_chain: The RAGChain instance to evaluate.
        judge_model: Anthropic model used to score relevance and faithfulness.
            A smaller model (Haiku) is fine here — it keeps eval costs low.
    """

    def __init__(
        self,
        rag_chain: RAGChain | None = None,
        judge_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.chain = rag_chain or RAGChain()
        self.judge_model = judge_model
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def run(self, test_queries_path: str | Path) -> EvaluationReport:
        """Run evaluation over all queries in the test file.

        Args:
            test_queries_path: Path to test_queries.json. Each entry must have:
                - "query": str
                - "expected_answer_snippet": str (substring that should appear
                  in a good answer or retrieved chunk)
                - "category": str (optional, for filtered reporting)

        Returns:
            EvaluationReport with per-query results and aggregate metrics.
        """
        queries = self._load_queries(test_queries_path)
        report = EvaluationReport(num_queries=len(queries))

        for item in queries:
            result = self._evaluate_single(item)
            report.results.append(result)
            logger.info(
                "Query: %s | hit=%s | relevance=%.2f | faithfulness=%.2f",
                item["query"][:60],
                result.retrieval_hit,
                result.relevance_score,
                result.faithfulness_score,
            )

        report.recall_at_k = self._recall_at_k(report.results)
        report.mean_relevance = statistics.mean(r.relevance_score for r in report.results) if report.results else 0.0
        report.mean_faithfulness = statistics.mean(r.faithfulness_score for r in report.results) if report.results else 0.0

        self._print_summary(report)
        return report

    def score_retrieval(self, query: str, expected_snippet: str, top_k: int = 5) -> bool:
        """Check whether retrieval for a query returns a relevant chunk.

        Args:
            query: Natural language question.
            expected_snippet: A substring that should appear in a relevant chunk.
            top_k: Number of chunks to retrieve.

        Returns:
            True if any retrieved chunk contains the expected snippet (case-insensitive).
        """
        chunks = self.chain.vector_store.retrieve(query, top_k=top_k)
        snippet_lower = expected_snippet.lower()
        return any(snippet_lower in chunk.text.lower() for chunk in chunks)

    def score_relevance(self, question: str, answer: str) -> float:
        """Ask the judge model to rate answer relevance on a 1–5 scale.

        Args:
            question: The original user question.
            answer: The LLM-generated answer to evaluate.

        Returns:
            Normalised relevance score (0–1). Returns 0.0 on judge failure.

        TODO: Implement the LLM judge call:
              Prompt: "Rate how well this answer addresses the question. "
                      "Scale: 1 (irrelevant) to 5 (fully answers the question). "
                      "Reply with just the number.\n"
                      f"Question: {question}\nAnswer: {answer}"
              Parse the response as int, normalise to (score - 1) / 4.
        """
        logger.warning("score_relevance is a stub — returning 0.0")
        return 0.0

    def score_faithfulness(self, answer: str, context_chunks: list[str]) -> float:
        """Check whether the answer is fully grounded in the retrieved context.

        Args:
            answer: The LLM-generated answer.
            context_chunks: The text of the retrieved chunks used to generate it.

        Returns:
            Faithfulness score (0–1). 1.0 = all claims are grounded in context.

        TODO: Implement the LLM judge call:
              Prompt: "Given the context below, identify any claims in the answer "
                      "that are NOT supported by the context. "
                      "Reply with: SUPPORTED_CLAIMS=N, UNSUPPORTED_CLAIMS=M\n"
                      f"Context:\n{context}\n\nAnswer:\n{answer}"
              Compute faithfulness = N / (N + M). Return 1.0 if M == 0.
        """
        logger.warning("score_faithfulness is a stub — returning 0.0")
        return 0.0

    def generate_report_json(self, report: EvaluationReport, output_path: str) -> None:
        """Write the evaluation report to a JSON file.

        Args:
            report: Output from run().
            output_path: File path to write (e.g. "evaluation/results.json").
        """
        data = {
            "num_queries": report.num_queries,
            "recall_at_k": report.recall_at_k,
            "mean_relevance": report.mean_relevance,
            "mean_faithfulness": report.mean_faithfulness,
            "results": [
                {
                    "query": r.query,
                    "expected": r.expected_answer_snippet,
                    "actual": r.actual_answer,
                    "retrieval_hit": r.retrieval_hit,
                    "relevance_score": r.relevance_score,
                    "faithfulness_score": r.faithfulness_score,
                    "sources": r.retrieved_sources,
                }
                for r in report.results
            ],
        }
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Evaluation report written to %s", output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_single(self, item: dict[str, Any]) -> QueryResult:
        """Run the full evaluation pipeline for one query.

        Args:
            item: Dict from test_queries.json with "query" and
                "expected_answer_snippet" keys.

        Returns:
            QueryResult with all three metric scores.
        """
        query = item["query"]
        expected = item["expected_answer_snippet"]

        answer_obj = self.chain.ask(query)

        retrieval_hit = expected.lower() in " ".join(
            c.text for c in answer_obj.retrieved_chunks
        ).lower()

        relevance = self.score_relevance(query, answer_obj.text)
        faithfulness = self.score_faithfulness(
            answer_obj.text,
            [c.text for c in answer_obj.retrieved_chunks],
        )

        return QueryResult(
            query=query,
            expected_answer_snippet=expected,
            actual_answer=answer_obj.text,
            retrieved_sources=answer_obj.sources,
            retrieved_texts=[c.text for c in answer_obj.retrieved_chunks],
            retrieval_hit=retrieval_hit,
            relevance_score=relevance,
            faithfulness_score=faithfulness,
        )

    @staticmethod
    def _load_queries(path: str | Path) -> list[dict[str, Any]]:
        """Load and validate the test queries JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("test_queries.json must be a JSON array")
        return data

    @staticmethod
    def _recall_at_k(results: list[QueryResult]) -> float:
        """Compute recall@K across all query results."""
        if not results:
            return 0.0
        hits = sum(1 for r in results if r.retrieval_hit)
        return hits / len(results)

    @staticmethod
    def _print_summary(report: EvaluationReport) -> None:
        """Print a formatted summary table to stdout."""
        print("\n" + "=" * 60)
        print("RAG Evaluation Summary")
        print("=" * 60)
        print(f"  Queries evaluated : {report.num_queries}")
        print(f"  Recall@K          : {report.recall_at_k:.1%}")
        print(f"  Mean relevance    : {report.mean_relevance:.3f} / 1.0")
        print(f"  Mean faithfulness : {report.mean_faithfulness:.3f} / 1.0")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument(
        "--queries",
        default="evaluation/test_queries.json",
        help="Path to test_queries.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        help="Path to write the evaluation report JSON",
    )
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    report = evaluator.run(args.queries)
    evaluator.generate_report_json(report, args.output)
