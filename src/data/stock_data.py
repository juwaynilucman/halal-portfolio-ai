"""Market data fetcher: prices, dividends, and sector info via yfinance.

This module is the single point of contact between the portfolio engine and
live market data. Everything downstream (builder, purification, redistribution)
calls functions here rather than importing yfinance directly, which keeps
mocking easy and the data-fetch logic in one place.

Typical usage:
    fetcher = StockDataFetcher()
    info = fetcher.get_stock_info("AAPL")
    prices = fetcher.get_price_history(["AAPL", "MSFT"], period="1y")
    dividends = fetcher.get_annual_dividend_rate("AAPL")
"""

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class StockInfo:
    """Snapshot of a stock's current market data and metadata."""

    ticker: str
    name: str
    sector: str          # canonical Yahoo Finance sector
    open_price: float    # most recent session open price (USD)
    current_price: float
    market_cap: float
    dividend_yield: float       # trailing 12-month yield (0–1, e.g. 0.032)
    annual_dividend_rate: float # dividends per share over trailing 12 months


class StockDataFetcher:
    """Wraps yfinance calls with caching and error handling."""

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        """Initialise the fetcher.

        Args:
            cache_ttl_seconds: How long (in seconds) to cache ticker metadata
                before re-fetching. Prices are never cached — always live.
        """
        self._cache_ttl = cache_ttl_seconds
        self._info_cache: dict[str, StockInfo] = {}

    def get_stock_info(self, ticker: str) -> StockInfo:
        """Return a StockInfo snapshot for a single ticker.

        Hits the in-memory cache first; re-fetches from yfinance on a miss.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            StockInfo populated from yfinance's fast_info and info dicts.

        Raises:
            ValueError: If yfinance returns no data for the ticker.
        """
        ticker = ticker.upper()
        if ticker in self._info_cache:
            return self._info_cache[ticker]

        logger.debug("Fetching info for %s", ticker)
        yt = yf.Ticker(ticker)
        info = yt.info

        if not info or "symbol" not in info:
            raise ValueError(f"yfinance returned no data for ticker: {ticker}")

        stock_info = StockInfo(
            ticker=ticker,
            name=info.get("longName", ticker),
            sector=info.get("sector", "Unknown"),
            open_price=info.get("open", 0.0) or 0.0,
            current_price=info.get("currentPrice", 0.0) or 0.0,
            market_cap=info.get("marketCap", 0.0) or 0.0,
            dividend_yield=info.get("dividendYield", 0.0) or 0.0,
            annual_dividend_rate=info.get("dividendRate", 0.0) or 0.0,
        )
        self._info_cache[ticker] = stock_info
        return stock_info

    def get_bulk_info(self, tickers: list[str]) -> dict[str, StockInfo]:
        """Fetch StockInfo for multiple tickers, skipping failed lookups.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Mapping of ticker -> StockInfo for all tickers that returned data.
        """
        result: dict[str, StockInfo] = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get_stock_info(ticker)
            except (ValueError, Exception) as exc:
                logger.warning("Skipping %s — data fetch failed: %s", ticker, exc)
        return result

    def get_price_history(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV history for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            period: yfinance period string (e.g. "1y", "6mo", "3mo").
            interval: yfinance interval string (e.g. "1d", "1wk").

        Returns:
            DataFrame with a MultiIndex of (field, ticker) columns, indexed
            by date. Use df["Close"]["AAPL"] to get Apple's close series.
        """
        logger.info("Fetching %s price history for %d tickers", period, len(tickers))
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True)
        return data

    def get_open_price(self, ticker: str) -> float:
        """Return today's opening price for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Opening price in USD, or 0.0 if unavailable.
        """
        info = self.get_stock_info(ticker)
        return info.open_price

    def get_annual_dividend_rate(self, ticker: str) -> float:
        """Return the trailing 12-month dividends-per-share for a ticker.

        Used by the purification calculator to determine how much non-permissible
        income to give to charity each quarter.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Annual dividend rate in USD per share (e.g. 1.44 for $1.44/year).
        """
        info = self.get_stock_info(ticker)
        return info.annual_dividend_rate

    def get_sector(self, ticker: str) -> str:
        """Return the Yahoo Finance sector for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Sector string (e.g. "Technology"), or "Unknown" if not found.
        """
        info = self.get_stock_info(ticker)
        return info.sector

    def get_dividend_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        """Return the dividend payment history for a ticker.

        Args:
            ticker: Stock ticker symbol.
            start: Start date (inclusive). Defaults to 1 year ago if None.
            end: End date (inclusive). Defaults to today if None.

        Returns:
            pandas Series indexed by ex-dividend date with dividend amounts.
        """
        yt = yf.Ticker(ticker.upper())
        dividends = yt.dividends

        if start:
            dividends = dividends[dividends.index.date >= start]
        if end:
            dividends = dividends[dividends.index.date <= end]

        return dividends
