"""Quarterly portfolio redistribution (rebalancing) engine.

Every quarter the portfolio is rebuilt from scratch using the current ETF
holdings and fresh Sharia screening results. Some stocks will:
  - Gain or lose weight in the ETF (weight drift)
  - Change compliance status (newly halal or newly haram)
  - Be added or removed from the ETF's top holdings

This module computes the delta between the old portfolio state and the new
allocation plan, then outputs actionable buy/sell instructions.

Key formula:
    delta_fraction   = new_fraction - old_fraction
    adjusted_amount  = (new_open_price - old_open_price) × delta_fraction

  - delta_fraction > 0 → buy more shares
  - delta_fraction < 0 → sell shares (also triggers growth purification check)
  - delta_fraction == 0 → no trade required

Typical usage:
    engine = RedistributionEngine(purification_calc)
    report = engine.run(old_portfolio, new_plan, stock_info_map)
    for action in report.actions:
        print(action.ticker, action.action_type, action.dollar_amount)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from src.portfolio.builder import PortfolioPlan, StockAllocation
from src.portfolio.purification import PurificationCalculator, PurificationResult

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Type of trade action required during redistribution."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"     # stock was halal but is now haram — full sale required
    ENTER = "enter"   # new stock added to portfolio this quarter


@dataclass
class RedistributionAction:
    """A single buy/sell instruction produced by the redistribution engine."""

    ticker: str
    action_type: ActionType
    old_fraction: float     # fractional shares before rebalance
    new_fraction: float     # target fractional shares after rebalance
    delta_fraction: float   # new_fraction - old_fraction
    old_open_price: float
    new_open_price: float
    adjusted_amount: float  # (new_open_price - old_open_price) × delta_fraction (USD)
    purification: PurificationResult | None = None


@dataclass
class RedistributionReport:
    """Full quarterly redistribution output."""

    actions: list[RedistributionAction] = field(default_factory=list)
    purification_results: list[PurificationResult] = field(default_factory=list)
    total_charity_due: float = 0.0
    newly_excluded: list[str] = field(default_factory=list)  # tickers removed (now haram)
    newly_added: list[str] = field(default_factory=list)     # tickers added (new halal picks)
    total_buys_usd: float = 0.0
    total_sells_usd: float = 0.0


class RedistributionEngine:
    """Computes the quarterly rebalancing delta and purification amounts."""

    def __init__(self, purification_calculator: PurificationCalculator) -> None:
        """Initialise the engine.

        Args:
            purification_calculator: Instance of PurificationCalculator used to
                compute charity amounts for any mixed-income stocks that are
                partially sold during rebalancing.
        """
        self.purification_calc = purification_calculator

    def run(
        self,
        old_allocations: list[StockAllocation],
        new_plan: PortfolioPlan,
        purification_pcts: dict[str, float],
        dividend_rates: dict[str, float],
    ) -> RedistributionReport:
        """Execute the quarterly redistribution computation.

        Compares the previous allocation to the new plan, computes delta
        fractions and adjusted amounts, and triggers purification calculations
        for any stocks with non-permissible income.

        Args:
            old_allocations: The StockAllocation list from last quarter's plan.
            new_plan: The freshly built PortfolioPlan for this quarter.
            purification_pcts: Mapping of ticker -> purification_pct from the
                Sharia screener (0.0 for fully halal stocks).
            dividend_rates: Mapping of ticker -> annual dividend rate (USD/share)
                from StockDataFetcher.

        Returns:
            RedistributionReport with all actions and purification amounts.
        """
        report = RedistributionReport()

        old_map: dict[str, StockAllocation] = {a.ticker: a for a in old_allocations}
        new_map: dict[str, StockAllocation] = {a.ticker: a for a in new_plan.allocations}

        all_tickers = set(old_map) | set(new_map)

        for ticker in all_tickers:
            old = old_map.get(ticker)
            new = new_map.get(ticker)

            action = self._compute_action(
                ticker=ticker,
                old=old,
                new=new,
                purification_pct=purification_pcts.get(ticker, 0.0),
                dividend_rate=dividend_rates.get(ticker, 0.0),
            )
            report.actions.append(action)

            if action.action_type == ActionType.ENTER:
                report.newly_added.append(ticker)
            elif action.action_type == ActionType.EXIT:
                report.newly_excluded.append(ticker)

            if action.purification:
                report.purification_results.append(action.purification)

            if action.adjusted_amount > 0:
                report.total_buys_usd += action.adjusted_amount
            elif action.adjusted_amount < 0:
                report.total_sells_usd += abs(action.adjusted_amount)

        report.total_charity_due = self.purification_calc.total_charity_due(
            report.purification_results
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_action(
        self,
        ticker: str,
        old: StockAllocation | None,
        new: StockAllocation | None,
        purification_pct: float,
        dividend_rate: float,
    ) -> RedistributionAction:
        """Determine the trade action for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            old: Previous quarter's allocation (None if stock is new).
            new: This quarter's allocation (None if stock was removed).
            purification_pct: Non-permissible revenue fraction from screener.
            dividend_rate: Annual dividends per share (USD).

        Returns:
            RedistributionAction with all fields populated.
        """
        old_fraction = old.stock_fraction if old else 0.0
        new_fraction = new.stock_fraction if new else 0.0
        old_price = old.open_price if old else 0.0
        new_price = new.open_price if new else 0.0

        delta_fraction = new_fraction - old_fraction

        # Formula: adjusted_amount = (new_open_price - old_open_price) × delta_fraction
        adjusted_amount = (new_price - old_price) * delta_fraction

        # Determine action type
        if old is None:
            action_type = ActionType.ENTER
        elif new is None:
            action_type = ActionType.EXIT
        elif abs(delta_fraction) < 1e-8:
            action_type = ActionType.HOLD
        elif delta_fraction > 0:
            action_type = ActionType.BUY
        else:
            action_type = ActionType.SELL

        # Purification — only for mixed-income stocks with non-zero pct
        purification: PurificationResult | None = None
        if purification_pct > 0 and old_fraction > 0:
            purification = self.purification_calc.calculate(
                ticker=ticker,
                old_fraction=old_fraction,
                delta_fraction=delta_fraction,
                old_price=old_price,
                new_price=new_price,
                dividend_rate=dividend_rate,
                purification_pct=purification_pct,
            )

        return RedistributionAction(
            ticker=ticker,
            action_type=action_type,
            old_fraction=old_fraction,
            new_fraction=new_fraction,
            delta_fraction=delta_fraction,
            old_open_price=old_price,
            new_open_price=new_price,
            adjusted_amount=adjusted_amount,
            purification=purification,
        )

    def summarise(self, report: RedistributionReport) -> str:
        """Return a human-readable text summary of the redistribution report.

        Args:
            report: Output from run().

        Returns:
            Multi-line string suitable for logging or dashboard display.
        """
        lines = [
            f"Quarterly Redistribution Summary",
            f"  Stocks: {len(report.actions)} total",
            f"  New additions: {len(report.newly_added)} ({', '.join(report.newly_added) or 'none'})",
            f"  Removed (haram/dropped): {len(report.newly_excluded)} ({', '.join(report.newly_excluded) or 'none'})",
            f"  Total buys: ${report.total_buys_usd:,.2f}",
            f"  Total sells: ${report.total_sells_usd:,.2f}",
            f"  Purification due: ${report.total_charity_due:.4f}",
        ]
        return "\n".join(lines)
