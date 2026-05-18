"""Purification (Zakat al-Tazkiyah) calculator for mixed-income stocks.

When a halal-classified stock earns a fraction of its revenue from
non-permissible activities (e.g., interest income from a treasury
desk), the investor must donate that proportional amount to charity.
This is called purification (tazkiyah).

The module is intentionally pluggable: the default implementation uses
the formulas below, but the caller can inject a custom_purification_fn
to swap in a different scholarly opinion without touching the rest of
the codebase.

Formulas
--------
Dividend purification (quarterly):
    dividend_purification = (dividend_rate / 4) × old_fraction
                            × purification_pct × 0.7

    The 0.7 factor accounts for tax drag on dividends
    (approximate net-of-tax adjustment). Adjust per jurisdiction.

Growth purification (on redistribution, only when delta < 0 and price rose):
    growth_purification = (new_price - old_price) × delta_fraction
                          × purification_pct

    This captures the purification owed on the capital-gain portion
    attributable to the non-permissible income fraction.

Typical usage:
    calc = PurificationCalculator()
    result = calc.calculate(
        ticker="MSFT",
        old_fraction=10.5,
        delta_fraction=-2.0,
        old_price=380.0,
        new_price=410.0,
        dividend_rate=3.0,
        purification_pct=0.05,
    )
    print(result.total_purification)  # amount to donate to charity
"""

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

# Type alias for a custom purification function.
# Receives the same kwargs as PurificationCalculator.calculate() and must
# return a float (total charity amount in USD).
PurificationFn = Callable[..., float]


@dataclass
class PurificationResult:
    """Breakdown of the purification calculation for a single stock position."""

    ticker: str
    dividend_purification: float   # charity amount from dividend income
    growth_purification: float     # charity amount from capital appreciation
    total_purification: float      # sum of the above
    # Inputs used — stored for audit trail / display in the dashboard.
    old_fraction: float
    delta_fraction: float
    old_price: float
    new_price: float
    dividend_rate: float
    purification_pct: float


class PurificationCalculator:
    """Computes purification amounts for positions with non-permissible income.

    Args:
        tax_drag_factor: Multiplier applied to dividend purification to
            account for withholding tax (default 0.7 — roughly 30% tax).
            Set to 1.0 to disable the tax adjustment.
        custom_purification_fn: Optional callable that fully replaces the
            built-in formula. When provided, the default formulas are
            bypassed entirely and this function is called instead.
    """

    def __init__(
        self,
        tax_drag_factor: float = 0.7,
        custom_purification_fn: PurificationFn | None = None,
    ) -> None:
        self.tax_drag_factor = tax_drag_factor
        self._custom_fn = custom_purification_fn

    def calculate(
        self,
        ticker: str,
        old_fraction: float,
        delta_fraction: float,
        old_price: float,
        new_price: float,
        dividend_rate: float,
        purification_pct: float,
    ) -> PurificationResult:
        """Calculate the purification amount owed for a single position.

        Args:
            ticker: Stock ticker symbol (for logging / audit trail).
            old_fraction: Number of fractional shares held at start of period.
            delta_fraction: Change in fractional shares this period
                (new_fraction - old_fraction). Negative means a partial sale.
            old_price: Share price at the start of the period (USD).
            new_price: Share price at calculation time (USD).
            dividend_rate: Annual dividends per share paid by the stock (USD).
            purification_pct: Fraction of revenue that is non-permissible (0–1).
                Comes from the Sharia screener (ScreeningResult.purification_pct).

        Returns:
            PurificationResult with itemised and total charity amounts.
        """
        if self._custom_fn is not None:
            logger.info("Using custom purification function for %s", ticker)
            total = self._custom_fn(
                ticker=ticker,
                old_fraction=old_fraction,
                delta_fraction=delta_fraction,
                old_price=old_price,
                new_price=new_price,
                dividend_rate=dividend_rate,
                purification_pct=purification_pct,
            )
            return PurificationResult(
                ticker=ticker,
                dividend_purification=0.0,
                growth_purification=0.0,
                total_purification=total,
                old_fraction=old_fraction,
                delta_fraction=delta_fraction,
                old_price=old_price,
                new_price=new_price,
                dividend_rate=dividend_rate,
                purification_pct=purification_pct,
            )

        div_purification = self._dividend_purification(
            dividend_rate=dividend_rate,
            old_fraction=old_fraction,
            purification_pct=purification_pct,
        )
        growth_purification = self._growth_purification(
            old_price=old_price,
            new_price=new_price,
            delta_fraction=delta_fraction,
            purification_pct=purification_pct,
        )

        return PurificationResult(
            ticker=ticker,
            dividend_purification=div_purification,
            growth_purification=growth_purification,
            total_purification=div_purification + growth_purification,
            old_fraction=old_fraction,
            delta_fraction=delta_fraction,
            old_price=old_price,
            new_price=new_price,
            dividend_rate=dividend_rate,
            purification_pct=purification_pct,
        )

    def calculate_bulk(
        self,
        positions: list[dict],
    ) -> list[PurificationResult]:
        """Calculate purification for multiple positions at once.

        Args:
            positions: List of dicts, each containing the keyword arguments
                accepted by calculate() (ticker, old_fraction, etc.).

        Returns:
            List of PurificationResult objects in the same order as positions.
        """
        return [self.calculate(**pos) for pos in positions]

    def total_charity_due(self, results: list[PurificationResult]) -> float:
        """Sum total purification across all positions.

        Args:
            results: Output from calculate_bulk().

        Returns:
            Total USD amount to donate to charity this period.
        """
        return sum(r.total_purification for r in results)

    # ------------------------------------------------------------------
    # Private formula implementations
    # ------------------------------------------------------------------

    def _dividend_purification(
        self,
        dividend_rate: float,
        old_fraction: float,
        purification_pct: float,
    ) -> float:
        """Compute the quarterly dividend purification amount.

        Formula:
            dividend_purification = (dividend_rate / 4)
                                    × old_fraction
                                    × purification_pct
                                    × tax_drag_factor

        Args:
            dividend_rate: Annual dividends per share (USD).
            old_fraction: Fractional shares held at period start.
            purification_pct: Non-permissible revenue fraction (0–1).

        Returns:
            USD amount to donate from dividend income.
        """
        quarterly_dividend = dividend_rate / 4
        purification = quarterly_dividend * old_fraction * purification_pct * self.tax_drag_factor
        logger.debug(
            "Dividend purification: (%.4f / 4) × %.4f × %.4f × %.2f = %.6f",
            dividend_rate,
            old_fraction,
            purification_pct,
            self.tax_drag_factor,
            purification,
        )
        return purification

    def _growth_purification(
        self,
        old_price: float,
        new_price: float,
        delta_fraction: float,
        purification_pct: float,
    ) -> float:
        """Compute purification on capital gains owed during redistribution.

        Growth purification only applies when:
          - delta_fraction < 0 (shares were reduced / sold)
          - new_price > old_price (the position appreciated)

        Formula:
            growth_purification = (new_price - old_price)
                                  × delta_fraction
                                  × purification_pct

        Note: delta_fraction is negative here, so the result is negative
        before the purification_pct multiplier — this is intentional because
        the formula produces a positive donation amount when price rose.
        The absolute value is returned.

        Args:
            old_price: Share price at period start (USD).
            new_price: Share price now (USD).
            delta_fraction: Change in fractional shares (new - old).
            purification_pct: Non-permissible revenue fraction (0–1).

        Returns:
            USD amount to donate from capital appreciation, or 0.0 if
            conditions are not met.
        """
        if delta_fraction >= 0 or new_price <= old_price:
            return 0.0

        purification = abs((new_price - old_price) * delta_fraction * purification_pct)
        logger.debug(
            "Growth purification: (%.2f - %.2f) × %.4f × %.4f = %.6f",
            new_price,
            old_price,
            delta_fraction,
            purification_pct,
            purification,
        )
        return purification
