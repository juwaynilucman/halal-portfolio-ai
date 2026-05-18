"""FastAPI application — REST endpoints for portfolio operations and RAG chat.

Endpoints
---------
GET  /health                    — liveness probe
GET  /etfs                      — list tracked ETFs and their configs
POST /portfolio/build           — build a new allocation plan
GET  /portfolio/current         — retrieve the most recent plan from the DB
POST /portfolio/redistribute    — run quarterly redistribution
POST /portfolio/purification    — calculate purification for current positions
POST /chat                      — ask a question via the RAG chain
POST /chat/stream               — streaming version of /chat

All endpoints return JSON. Errors are returned as {"detail": "..."} with an
appropriate HTTP status code.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.config import ETFS, PORTFOLIO

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Halal Portfolio AI",
    description="Sharia-compliant stock portfolio builder powered by RAG",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class BuildPortfolioRequest(BaseModel):
    available_cash: float = Field(..., gt=0, description="Total investable amount in USD")
    num_companies: int = Field(30, gt=0, le=200, description="Target number of stocks")
    etf_tickers: list[str] = Field(
        default=["SCHD", "VYMI", "IVW"],
        description="ETFs to source holdings from",
    )
    use_zoya: bool = Field(False, description="Enable Zoya API for Sharia screening")


class PurificationRequest(BaseModel):
    positions: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of position dicts. Each must contain: ticker, old_fraction, "
            "delta_fraction, old_price, new_price, dividend_rate, purification_pct."
        ),
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    include_portfolio_context: bool = Field(
        True,
        description="Search the portfolio collection in addition to the knowledge base",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 OK when the service is up."""
    return {"status": "ok"}


@app.get("/etfs")
async def list_etfs() -> dict[str, Any]:
    """Return metadata for all tracked ETFs."""
    return {
        ticker: {
            "name": cfg.name,
            "strategy": cfg.strategy,
            "target_companies": cfg.target_companies,
        }
        for ticker, cfg in ETFS.items()
    }


@app.post("/portfolio/build")
async def build_portfolio(request: BuildPortfolioRequest) -> dict[str, Any]:
    """Build a new Sharia-compliant allocation plan.

    Steps:
      1. Scrape ETF holdings for each requested ticker.
      2. Fetch stock info (prices, sectors) via yfinance.
      3. Screen each holding for Sharia compliance.
      4. Build the portfolio allocation plan (Method 1).
      5. Persist the plan to PostgreSQL.
      6. Index the plan in ChromaDB for RAG queries.

    Returns the full allocation plan as JSON.

    TODO: Wire up ETFScraper, StockDataFetcher, ShariaScreener,
          PortfolioBuilder, and VectorStore calls here.
    """
    raise HTTPException(status_code=501, detail="Portfolio build endpoint not yet implemented")


@app.get("/portfolio/current")
async def get_current_portfolio() -> dict[str, Any]:
    """Return the most recently built portfolio plan from the database.

    TODO: Query PostgreSQL for the latest PortfolioPlan record and return it.
    """
    raise HTTPException(status_code=501, detail="Portfolio retrieval not yet implemented")


@app.post("/portfolio/redistribute")
async def redistribute_portfolio(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Trigger quarterly redistribution.

    Runs as a background task because fetching fresh ETF data and re-screening
    all stocks can take several minutes.

    TODO: Launch RedistributionEngine.run() in the background task, persist
          the RedistributionReport, update the ChromaDB portfolio collection.
    """
    raise HTTPException(status_code=501, detail="Redistribution endpoint not yet implemented")


@app.post("/portfolio/purification")
async def calculate_purification(request: PurificationRequest) -> dict[str, Any]:
    """Calculate purification amounts for a list of positions.

    Args:
        request: Contains a list of position dicts with purification inputs.

    Returns:
        Per-stock purification breakdown and total charity amount due.

    TODO: Instantiate PurificationCalculator and call calculate_bulk().
    """
    raise HTTPException(status_code=501, detail="Purification endpoint not yet implemented")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer a portfolio or Islamic finance question using RAG.

    Args:
        request: Contains the user's question and retrieval preferences.

    Returns:
        LLM-generated answer with source citations and token usage.

    TODO: Instantiate RAGChain and call ask().
    """
    raise HTTPException(status_code=501, detail="Chat endpoint not yet implemented")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a RAG answer token-by-token for real-time UI display.

    Args:
        request: Same as /chat.

    Returns:
        StreamingResponse with text/event-stream content type.

    TODO: Instantiate RAGChain and call ask_stream(), yielding SSE events.
    """
    raise HTTPException(status_code=501, detail="Streaming chat not yet implemented")
