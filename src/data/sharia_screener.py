"""Sharia compliance screener for individual stocks.

Classifies stocks as halal, haram, or mixed based on their business
activities and financial ratios. The screening methodology follows the
AAOIFI (Accounting and Auditing Organization for Islamic Financial
Institutions) standard:

  1. Qualitative screen — exclude companies whose primary business involves
     alcohol, tobacco, weapons, adult content, pork, conventional banking,
     or gambling.
  2. Financial ratios screen — exclude companies where:
       - Interest-bearing debt / total assets > 33%
       - Non-permissible income / total revenue > 5%
       - Accounts receivable / total assets > 45% (some scholars: 33%)

The module is designed for pluggable backends:
  - Rule-based stub (default, offline)
  - Zoya API integration (TODO — requires API key and rate-limit handling)
  - Manual override list (CSV/JSON that the user maintains)

Typical usage:
    screener = ShariaScreener()
    result = screener.screen("AAPL")
    # ScreeningResult(ticker="AAPL", status="halal", purification_pct=0.0, ...)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Sharia compliance classification for a stock."""

    HALAL = "halal"        # passes all screens — full investment permitted
    HARAM = "haram"        # fails a qualitative screen — excluded entirely
    MIXED = "mixed"        # passes qualitative but has some non-permissible revenue;
                           # investment allowed with purification of that income share
    UNKNOWN = "unknown"    # screener could not determine status (data unavailable)


@dataclass
class ScreeningResult:
    """Output of the Sharia screener for a single stock."""

    ticker: str
    status: ComplianceStatus
    # Fraction of revenue that is non-permissible (0–1). Used to calculate
    # the purification amount the investor must donate to charity.
    purification_pct: float = 0.0
    # Ratio checks — stored for transparency / audit trail.
    debt_to_assets: float = 0.0
    non_permissible_income_pct: float = 0.0
    accounts_receivable_to_assets: float = 0.0
    # Human-readable explanation of why the stock was classified as it was.
    reason: str = ""
    # Which backend produced this result.
    source: str = "rule_based"
    # Tickers in the known-haram list that triggered an immediate exclusion.
    triggered_rules: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Known haram businesses — maintained manually as a fast-reject list.
# The Zoya integration (TODO) will supersede this for most tickers.
# ---------------------------------------------------------------------------

_HARAM_SECTORS: set[str] = {
    "Financial Services",  # conventional banking / insurance with riba
}

# Tickers confirmed haram by major scholars — hard exclusions.
_HARAM_TICKERS: set[str] = {
    # Conventional banks
    "JPM", "BAC", "WFC", "C", "GS", "MS",
    # Alcohol
    "BUD", "TAP", "STZ",
    # Tobacco
    "MO", "PM", "BTI",
    # Casinos / gambling
    "MGM", "WYNN", "LVS",
    # Adult entertainment — major publicly-traded operators
    # (add as identified)
}


class ShariaScreener:
    """Classifies stocks by Sharia compliance status.

    The screener can be extended by registering additional backend callables.
    By default it uses rule-based heuristics only (offline, no API key needed).
    """

    def __init__(self, use_zoya: bool = False, manual_overrides_path: str | None = None) -> None:
        """Initialise the screener.

        Args:
            use_zoya: If True, attempt to call the Zoya API for each ticker
                after the fast-reject pass. Requires ZOYA_API_KEY in env.
            manual_overrides_path: Path to a JSON file mapping ticker ->
                {"status": "halal"|"haram"|"mixed", "purification_pct": 0.0}.
                Overrides take precedence over all other sources.
        """
        self.use_zoya = use_zoya
        self._overrides: dict[str, dict] = {}

        if manual_overrides_path:
            self._load_overrides(manual_overrides_path)

    def screen(self, ticker: str) -> ScreeningResult:
        """Screen a single ticker for Sharia compliance.

        Evaluation order:
          1. Manual override (highest priority)
          2. Known-haram ticker list
          3. Known-haram sector list
          4. Zoya API (if enabled)
          5. Rule-based financial ratio checks

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").

        Returns:
            ScreeningResult with the compliance verdict and supporting data.
        """
        ticker = ticker.upper()
        logger.debug("Screening %s", ticker)

        # 1. Manual override
        if ticker in self._overrides:
            override = self._overrides[ticker]
            return ScreeningResult(
                ticker=ticker,
                status=ComplianceStatus(override["status"]),
                purification_pct=override.get("purification_pct", 0.0),
                reason="Manual override",
                source="manual_override",
            )

        # 2. Known-haram ticker fast-reject
        if ticker in _HARAM_TICKERS:
            return ScreeningResult(
                ticker=ticker,
                status=ComplianceStatus.HARAM,
                reason="Ticker is on the confirmed-haram exclusion list",
                triggered_rules=[ticker],
                source="rule_based",
            )

        # 3. Sector fast-reject
        # NOTE: sector info must be injected by caller or fetched here.
        # TODO: inject sector from StockDataFetcher to avoid duplicate API calls.

        # 4. Zoya API
        if self.use_zoya:
            result = self._screen_via_zoya(ticker)
            if result.status != ComplianceStatus.UNKNOWN:
                return result

        # 5. Rule-based financial ratio screening (stub)
        return self._screen_rule_based(ticker)

    def screen_bulk(self, tickers: list[str]) -> dict[str, ScreeningResult]:
        """Screen multiple tickers, skipping those that error out.

        Args:
            tickers: List of ticker symbols to screen.

        Returns:
            Mapping of ticker -> ScreeningResult.
        """
        results: dict[str, ScreeningResult] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.screen(ticker)
            except Exception as exc:
                logger.warning("Screening failed for %s: %s", ticker, exc)
                results[ticker] = ScreeningResult(
                    ticker=ticker,
                    status=ComplianceStatus.UNKNOWN,
                    reason=f"Screening error: {exc}",
                )
        return results

    def filter_halal(self, tickers: list[str]) -> list[str]:
        """Return only the tickers that pass the Sharia screen.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Subset of tickers classified as halal or mixed (investable).
        """
        results = self.screen_bulk(tickers)
        return [
            t for t, r in results.items()
            if r.status in (ComplianceStatus.HALAL, ComplianceStatus.MIXED)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _screen_via_zoya(self, ticker: str) -> ScreeningResult:
        """Call the Zoya API to get a compliance verdict.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            ScreeningResult from Zoya, or UNKNOWN status if the call fails.

        TODO: Implement Zoya API integration.
              - GET https://api.zoya.finance/graphql with ticker query
              - Parse response: { compliance: "COMPLIANT"|"NON_COMPLIANT"|"QUESTIONABLE",
                                  purificationPercentage: float }
              - Map COMPLIANT -> HALAL, NON_COMPLIANT -> HARAM, QUESTIONABLE -> MIXED
              - Handle rate limits with exponential back-off (Zoya free tier: 100 req/day)
              - Cache results to avoid re-fetching the same ticker within a session
        """
        logger.warning("Zoya integration not yet implemented — falling back to rule-based")
        return ScreeningResult(
            ticker=ticker,
            status=ComplianceStatus.UNKNOWN,
            reason="Zoya API not implemented",
            source="zoya",
        )

    def _screen_rule_based(self, ticker: str) -> ScreeningResult:
        """Apply AAOIFI financial ratio screens to a ticker.

        Fetches financial statements from yfinance (balance sheet + income
        statement) and evaluates the three AAOIFI ratio thresholds.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            ScreeningResult with ratio values populated.

        TODO: Implement ratio calculations.
              Steps:
                1. Fetch balance sheet: yf.Ticker(ticker).balance_sheet
                2. Fetch income statement: yf.Ticker(ticker).income_stmt
                3. Compute:
                     debt_ratio = interest_bearing_debt / total_assets
                     npi_pct    = non_permissible_income / total_revenue
                     ar_ratio   = accounts_receivable / total_assets
                4. Classify:
                     if debt_ratio > 0.33 or ar_ratio > 0.45 -> HARAM
                     elif npi_pct > 0.05 -> MIXED (purification_pct = npi_pct)
                     else -> HALAL
        """
        logger.warning(
            "Rule-based ratio screener is a stub — classifying %s as UNKNOWN", ticker
        )
        return ScreeningResult(
            ticker=ticker,
            status=ComplianceStatus.UNKNOWN,
            reason="Rule-based ratio screener not yet implemented",
            source="rule_based",
        )

    def _load_overrides(self, path: str) -> None:
        """Load manual override entries from a JSON file.

        Expected format:
            {
              "AAPL": {"status": "halal", "purification_pct": 0.0},
              "BRK-B": {"status": "haram", "purification_pct": 0.0}
            }

        Args:
            path: Filesystem path to the JSON override file.
        """
        import json

        try:
            with open(path, "r", encoding="utf-8") as fh:
                self._overrides = json.load(fh)
            logger.info("Loaded %d manual overrides from %s", len(self._overrides), path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Failed to load override file %s: %s", path, exc)
