"""Central configuration for ETF tickers, portfolio parameters, and runtime settings.

Values here are defaults; anything with a matching env var is overridden at
runtime via python-dotenv so the repo stays secret-free.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# ETF definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ETFConfig:
    """Static metadata for a tracked ETF."""

    ticker: str
    name: str
    strategy: str          # "dividend" | "growth"
    target_companies: int  # how many halal holdings to pick from this ETF


ETFS: dict[str, ETFConfig] = {
    "SCHD": ETFConfig(
        ticker="SCHD",
        name="Schwab US Dividend Equity ETF",
        strategy="dividend",
        target_companies=int(os.getenv("NUM_DIVIDEND_COMPANIES", "30")),
    ),
    "VYMI": ETFConfig(
        ticker="VYMI",
        name="Vanguard International High Dividend Yield ETF",
        strategy="dividend",
        target_companies=int(os.getenv("NUM_DIVIDEND_COMPANIES", "30")),
    ),
    "IVW": ETFConfig(
        ticker="IVW",
        name="iShares S&P 500 Growth ETF",
        strategy="growth",
        target_companies=int(os.getenv("NUM_GROWTH_COMPANIES", "20")),
    ),
}


# ---------------------------------------------------------------------------
# Portfolio parameters
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    """Tuneable knobs for portfolio construction and rebalancing."""

    # Fraction of the total portfolio to hold as cash buffer (0–1).
    cash_buffer_pct: float = 0.02

    # Minimum sector weight (%) before a sector is dropped from allocation.
    min_sector_weight_pct: float = 1.0

    # How often redistribution runs (in calendar days).
    redistribution_period_days: int = 90

    # Default purification percentage applied to mixed-income stocks (0–1).
    # Override per-stock when Zoya returns a specific non-permissible revenue %.
    default_purification_pct: float = 0.05


PORTFOLIO = PortfolioConfig()


# ---------------------------------------------------------------------------
# External service credentials (read from env)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/halal_portfolio")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# LLM model used by the RAG chain.
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
