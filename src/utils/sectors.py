"""Sector name mapping between ETF data sources and Yahoo Finance.

SCHD, VYMI, and IVW use slightly different sector labels than Yahoo Finance.
This module provides bidirectional lookup so the screener and portfolio builder
always work with a single canonical name.
"""

# Canonical name is the Yahoo Finance label (right-hand side).
SCHD_TO_YAHOO: dict[str, str] = {
    "Materials": "Basic Materials",
    "Industrials": "Industrials",
    "Consumer Staples": "Consumer Defensive",
    "Energy": "Energy",
    "Consumer Discretionary": "Consumer Cyclical",
    "Health Care": "Healthcare",
    "Information Technology": "Technology",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
    "Financials": "Financial Services",
}

YAHOO_TO_SCHD: dict[str, str] = {v: k for k, v in SCHD_TO_YAHOO.items()}

# All canonical Yahoo Finance sector names used across the project.
CANONICAL_SECTORS: list[str] = list(SCHD_TO_YAHOO.values())

# Sectors that are almost always haram — used as a fast-reject hint for the
# screener before it fires a full API call.
TYPICALLY_HARAM_SECTORS: set[str] = {
    "Financial Services",  # interest-based banking/insurance
}


def to_yahoo(etf_sector: str) -> str:
    """Convert an ETF sector label to the canonical Yahoo Finance name.

    Args:
        etf_sector: Sector label as it appears in SCHD / VYMI / IVW holdings.

    Returns:
        Canonical Yahoo Finance sector string, or the original value if no
        mapping exists (so unknown sectors pass through rather than crashing).
    """
    return SCHD_TO_YAHOO.get(etf_sector, etf_sector)


def to_etf(yahoo_sector: str) -> str:
    """Convert a Yahoo Finance sector name back to the ETF label.

    Args:
        yahoo_sector: Canonical Yahoo Finance sector string.

    Returns:
        ETF-facing sector label, or the original value if no mapping exists.
    """
    return YAHOO_TO_SCHD.get(yahoo_sector, yahoo_sector)


def is_typically_haram(yahoo_sector: str) -> bool:
    """Return True if the sector is flagged as commonly non-compliant.

    This is a fast heuristic only — the full Sharia screener should still be
    run per-stock even when this returns True.

    Args:
        yahoo_sector: Canonical Yahoo Finance sector string.
    """
    return yahoo_sector in TYPICALLY_HARAM_SECTORS
