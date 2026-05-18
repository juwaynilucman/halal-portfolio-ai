"""ETF holdings scraper for SCHD, VYMI, and IVW.

Each ETF publishes its full holdings list on a public webpage. This module
fetches and parses those pages with BeautifulSoup, returning a standardised
list of holdings that the rest of the pipeline can consume without caring
about the source ETF's HTML layout.

Typical usage:
    scraper = ETFScraper()
    holdings = scraper.get_holdings("SCHD")
    # [{"ticker": "PEP", "name": "PepsiCo Inc", "weight": 4.21, "sector": "Consumer Defensive"}, ...]
"""

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from src.utils.sectors import to_yahoo

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    """A single stock position within an ETF."""

    ticker: str
    name: str
    weight_pct: float    # percentage weight in the ETF (e.g. 4.21 for 4.21%)
    sector: str          # canonical Yahoo Finance sector name


# Source URLs — update if fund providers restructure their sites.
_ETF_URLS: dict[str, str] = {
    "SCHD": "https://www.schwab.com/research/etfs/quotes/holdings/schd",
    "VYMI": "https://investor.vanguard.com/investment-products/etfs/profile/vymi#portfolio-composition",
    "IVW": "https://www.ishares.com/us/products/239728/ishares-sp-500-growth-etf",
}

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class ETFScraper:
    """Scrapes ETF holdings pages and returns normalised Holding objects."""

    def __init__(self, timeout: int = 30) -> None:
        """Initialise the scraper.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def get_holdings(self, ticker: str) -> list[Holding]:
        """Return all holdings for a given ETF ticker.

        Dispatches to the correct private parser based on the ticker.

        Args:
            ticker: ETF ticker symbol — one of "SCHD", "VYMI", "IVW".

        Returns:
            List of Holding objects sorted by weight descending.

        Raises:
            ValueError: If the ticker is not supported.
            requests.HTTPError: If the holdings page returns a non-2xx status.
        """
        ticker = ticker.upper()
        parsers = {
            "SCHD": self._parse_schd,
            "VYMI": self._parse_vymi,
            "IVW": self._parse_ivw,
        }
        if ticker not in parsers:
            raise ValueError(f"Unsupported ETF ticker: {ticker}. Choose from {list(parsers)}")

        logger.info("Fetching holdings for %s", ticker)
        url = _ETF_URLS[ticker]
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        holdings = parsers[ticker](soup)
        holdings.sort(key=lambda h: h.weight_pct, reverse=True)
        logger.info("Retrieved %d holdings for %s", len(holdings), ticker)
        return holdings

    def get_all_holdings(self) -> dict[str, list[Holding]]:
        """Fetch holdings for all tracked ETFs.

        Returns:
            Mapping of ticker -> list of Holding objects.
        """
        return {ticker: self.get_holdings(ticker) for ticker in _ETF_URLS}

    # ------------------------------------------------------------------
    # Private parsers — one per ETF because each site has its own layout
    # ------------------------------------------------------------------

    def _parse_schd(self, soup: BeautifulSoup) -> list[Holding]:
        """Parse the Schwab SCHD holdings table.

        Args:
            soup: Parsed HTML of the SCHD holdings page.

        Returns:
            List of Holding objects extracted from the table.

        TODO: Implement once Schwab's exact table structure is confirmed.
              The table ID or class selector may need updating if Schwab
              redesigns their holdings page.
        """
        holdings: list[Holding] = []

        # TODO: Locate the holdings <table> by its CSS class or ID.
        #       Iterate <tr> rows, skip the header, extract:
        #         col 0 -> ticker
        #         col 1 -> company name
        #         col 2 -> weight (strip "%" and cast to float)
        #         col 4 -> sector (run through to_yahoo())
        #       Append a Holding for each row.

        logger.warning("SCHD parser is a stub — returning empty list")
        return holdings

    def _parse_vymi(self, soup: BeautifulSoup) -> list[Holding]:
        """Parse the Vanguard VYMI holdings table.

        Args:
            soup: Parsed HTML of the VYMI portfolio composition page.

        Returns:
            List of Holding objects.

        TODO: Vanguard renders holdings data via JavaScript. A Selenium or
              Playwright fetch may be required instead of plain requests.
              Alternatively, Vanguard exposes a CSV download link — use that
              as the primary source and fall back to HTML scraping.
        """
        holdings: list[Holding] = []

        # TODO: Check for a JSON endpoint in the page's <script> tags
        #       (Vanguard sometimes embeds holdings JSON in the HTML).
        #       If not found, launch a headless browser to wait for the
        #       React/Angular component to render, then parse the table.

        logger.warning("VYMI parser is a stub — returning empty list")
        return holdings

    def _parse_ivw(self, soup: BeautifulSoup) -> list[Holding]:
        """Parse the iShares IVW holdings table.

        Args:
            soup: Parsed HTML of the iShares IVW product page.

        Returns:
            List of Holding objects.

        TODO: iShares provides a CSV export at a stable URL pattern:
              https://www.ishares.com/us/products/239728/...holdings.csv
              Prefer the CSV route over HTML parsing for reliability.
              The CSV columns are: Ticker, Name, Asset Class, Weight(%), Price,
              Shares, Market Value, Notional Value, Sector, ISIN, CUSIP.
        """
        holdings: list[Holding] = []

        # TODO: Fetch the CSV export URL, parse with csv.DictReader, and
        #       filter rows where "Asset Class" == "Equity".
        #       Map the "Sector" column through to_yahoo().

        logger.warning("IVW parser is a stub — returning empty list")
        return holdings
