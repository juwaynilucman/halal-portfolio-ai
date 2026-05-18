"""Portfolio construction engine — Method 1: sector-weighted allocation.

This module takes a filtered list of halal stocks (already screened) and
produces a money allocation plan: how many fractional shares to buy of each
stock given the investor's available cash.

Algorithm (Method 1):
  1. Group stocks by canonical Yahoo Finance sector.
  2. Compute each sector's weight from the ETF holdings data.
  3. If an entire sector is haram (or empty after screening), redistribute
     its weight equally across the remaining halal sectors.
  4. For each sector, compute how many companies to include:
       companies_in_sector = (sector_weight / 100) × num_desired_companies
  5. Compute per-stock allocation:
       stock_units       = sector_weight / num_companies_in_sector
       sector_avg_price  = mean(open_price for each stock in sector)
       stock_cost        = sector_avg_price × stock_units
       ratio             = stock_cost / total_portfolio_cost
       money_allocated   = available_cash × ratio
       stock_fraction    = money_allocated / open_price

Typical usage:
    builder = PortfolioBuilder(available_cash=10_000.0, num_companies=30)
    plan = builder.build(halal_holdings, stock_info_map)
    for allocation in plan.allocations:
        print(allocation.ticker, allocation.money_allocated, allocation.stock_fraction)
"""

import logging
from dataclasses import dataclass, field

from src.data.etf_scraper import Holding
from src.data.stock_data import StockInfo
from src.utils.sectors import CANONICAL_SECTORS

logger = logging.getLogger(__name__)


@dataclass
class StockAllocation:
    """Money allocation for a single stock in the portfolio."""

    ticker: str
    name: str
    sector: str
    sector_weight_pct: float    # effective weight of this stock's sector after redistribution
    stock_units: float          # proportional unit count (sector_weight / num_companies)
    open_price: float           # price used for all calculations this session
    sector_avg_price: float
    stock_cost: float           # sector_avg_price × stock_units
    ratio: float                # stock_cost / total_portfolio_cost
    money_allocated: float      # available_cash × ratio
    stock_fraction: float       # money_allocated / open_price (fractional shares to buy)


@dataclass
class PortfolioPlan:
    """Complete allocation plan produced by the builder."""

    available_cash: float
    allocations: list[StockAllocation] = field(default_factory=list)
    total_portfolio_cost: float = 0.0   # sum of all stock_costs
    excluded_sectors: list[str] = field(default_factory=list)  # sectors dropped (all haram)
    redistributed_weight_pct: float = 0.0  # total weight redistributed from excluded sectors


class PortfolioBuilder:
    """Constructs a Sharia-compliant portfolio allocation from halal holdings."""

    def __init__(self, available_cash: float, num_companies: int) -> None:
        """Initialise the builder.

        Args:
            available_cash: Total investable amount in USD.
            num_companies: Target number of stocks across the entire portfolio.
                The per-sector count is derived from this proportionally.
        """
        self.available_cash = available_cash
        self.num_companies = num_companies

    def build(
        self,
        halal_holdings: list[Holding],
        stock_info_map: dict[str, StockInfo],
    ) -> PortfolioPlan:
        """Build the allocation plan from pre-filtered halal holdings.

        Args:
            halal_holdings: Holdings that have already passed the Sharia screen.
                Each Holding carries the ETF-reported sector weight_pct.
            stock_info_map: Mapping of ticker -> StockInfo with current prices.
                Tickers absent from this map are silently skipped.

        Returns:
            PortfolioPlan with per-stock allocations and summary metadata.
        """
        plan = PortfolioPlan(available_cash=self.available_cash)

        # Step 1 — group by sector
        sectors = self._group_by_sector(halal_holdings, stock_info_map)

        # Step 2 — compute raw sector weights
        sector_weights = self._compute_sector_weights(halal_holdings, sectors)

        # Step 3 — redistribute weight from empty/all-haram sectors
        sector_weights, excluded = self._redistribute_haram_sectors(sector_weights, sectors)
        plan.excluded_sectors = excluded
        plan.redistributed_weight_pct = sum(
            w for s, w in sector_weights.items() if s in excluded
        )

        # Step 4 — compute companies per sector
        companies_per_sector = self._companies_per_sector(sector_weights)

        # Step 5 — compute per-stock allocation
        allocations, total_cost = self._allocate(
            sectors, sector_weights, companies_per_sector, stock_info_map
        )
        plan.allocations = allocations
        plan.total_portfolio_cost = total_cost

        # Step 6 — finalize money allocation and fractional shares
        self._finalize_allocation(plan)

        logger.info(
            "Portfolio plan: %d stocks, total cost $%.2f, cash $%.2f",
            len(plan.allocations),
            plan.total_portfolio_cost,
            self.available_cash,
        )
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_by_sector(
        self,
        holdings: list[Holding],
        stock_info_map: dict[str, StockInfo],
    ) -> dict[str, list[Holding]]:
        """Group holdings by canonical sector name.

        Holdings whose tickers are missing from stock_info_map are dropped
        because we cannot compute prices for them.

        Args:
            holdings: Pre-filtered halal holdings list.
            stock_info_map: Ticker -> StockInfo for price data.

        Returns:
            Mapping of sector -> list of Holding objects in that sector.
        """
        grouped: dict[str, list[Holding]] = {}

        # TODO: Iterate holdings, skip tickers not in stock_info_map,
        #       use holding.sector (already canonical) as the key.
        #       grouped.setdefault(holding.sector, []).append(holding)

        logger.warning("_group_by_sector is a stub — returning empty dict")
        return grouped

    def _compute_sector_weights(
        self,
        holdings: list[Holding],
        sectors: dict[str, list[Holding]],
    ) -> dict[str, float]:
        """Aggregate ETF weight percentages per sector.

        The weight of a sector is the sum of the individual holding weights
        within that sector (as reported by the ETF).

        Args:
            holdings: Full halal holdings list.
            sectors: Grouped holdings by sector (from _group_by_sector).

        Returns:
            Mapping of sector -> total weight percentage (should sum to ~100).
        """
        weights: dict[str, float] = {}

        # TODO: For each sector in sectors, sum holding.weight_pct for all
        #       holdings in that sector.

        logger.warning("_compute_sector_weights is a stub — returning empty dict")
        return weights

    def _redistribute_haram_sectors(
        self,
        sector_weights: dict[str, float],
        sectors: dict[str, list[Holding]],
    ) -> tuple[dict[str, float], list[str]]:
        """Remove sectors that are empty or fully haram and spread their weight.

        When an entire sector has no halal stocks, its ETF weight is
        distributed equally across all remaining investable sectors.

        Formula per surviving sector:
            extra_weight = total_excluded_weight / num_surviving_sectors
            new_weight   = original_weight + extra_weight

        Args:
            sector_weights: Raw per-sector weights before redistribution.
            sectors: Grouped holdings (sectors with 0 stocks are excluded).

        Returns:
            Tuple of (updated_weights, list_of_excluded_sector_names).
        """
        excluded: list[str] = []
        updated = dict(sector_weights)

        # TODO: Identify sectors where len(sectors.get(s, [])) == 0.
        #       Sum their weights as total_excluded_weight.
        #       Remove them from updated.
        #       Add total_excluded_weight / len(remaining) to each remaining sector.

        logger.warning("_redistribute_haram_sectors is a stub — returning unchanged weights")
        return updated, excluded

    def _companies_per_sector(self, sector_weights: dict[str, float]) -> dict[str, int]:
        """Compute the integer number of companies to include per sector.

        Formula:
            companies_in_sector = round((sector_weight / 100) × num_desired_companies)
            Minimum of 1 company per sector to avoid zero-allocation.

        Args:
            sector_weights: Effective per-sector weight percentages after redistribution.

        Returns:
            Mapping of sector -> integer company count.
        """
        counts: dict[str, int] = {}

        # TODO: Apply the formula above, enforce minimum of 1, ensure the
        #       total doesn't exceed self.num_companies (round-trip correction).

        logger.warning("_companies_per_sector is a stub — returning empty dict")
        return counts

    def _allocate(
        self,
        sectors: dict[str, list[Holding]],
        sector_weights: dict[str, float],
        companies_per_sector: dict[str, int],
        stock_info_map: dict[str, StockInfo],
    ) -> tuple[list[StockAllocation], float]:
        """Compute stock_cost for each stock and the total portfolio cost.

        Per-stock formula:
            stock_units      = sector_weight / num_companies_in_sector
            sector_avg_price = mean(open_price for all stocks in sector)
            stock_cost       = sector_avg_price × stock_units

        Args:
            sectors: Grouped halal holdings by sector.
            sector_weights: Effective sector weights after redistribution.
            companies_per_sector: Number of stocks to pick per sector.
            stock_info_map: Ticker -> StockInfo for open prices.

        Returns:
            Tuple of (list of StockAllocation with cost fields filled,
                      total_portfolio_cost).

        TODO: Implement the selection of top-N holdings per sector
              (e.g., by ETF weight descending) and run the formula above
              for each selected stock.
        """
        allocations: list[StockAllocation] = []
        total_cost = 0.0

        logger.warning("_allocate is a stub — returning empty allocations")
        return allocations, total_cost

    def _finalize_allocation(self, plan: PortfolioPlan) -> None:
        """Fill in ratio, money_allocated, and stock_fraction for each allocation.

        Mutates plan.allocations in place.

        Formula:
            ratio            = stock_cost / total_portfolio_cost
            money_allocated  = available_cash × ratio
            stock_fraction   = money_allocated / open_price

        Args:
            plan: PortfolioPlan with total_portfolio_cost already set.
        """
        if plan.total_portfolio_cost == 0:
            logger.warning("total_portfolio_cost is 0 — skipping finalization")
            return

        # TODO: Iterate plan.allocations, compute ratio, money_allocated,
        #       stock_fraction for each StockAllocation.
        logger.warning("_finalize_allocation is a stub — no values updated")
