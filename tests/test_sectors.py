"""Unit tests for sector name mapping utilities."""

import pytest

from src.utils.sectors import (
    CANONICAL_SECTORS,
    SCHD_TO_YAHOO,
    is_typically_haram,
    to_etf,
    to_yahoo,
)


class TestSectorMapping:
    def test_schd_to_yahoo_known_mapping(self):
        assert to_yahoo("Information Technology") == "Technology"
        assert to_yahoo("Consumer Staples") == "Consumer Defensive"
        assert to_yahoo("Health Care") == "Healthcare"
        assert to_yahoo("Financials") == "Financial Services"

    def test_unknown_sector_passes_through(self):
        assert to_yahoo("Unknown Sector XYZ") == "Unknown Sector XYZ"

    def test_yahoo_to_etf_round_trip(self):
        for etf_name, yahoo_name in SCHD_TO_YAHOO.items():
            assert to_etf(yahoo_name) == etf_name

    def test_canonical_sectors_complete(self):
        assert len(CANONICAL_SECTORS) == len(SCHD_TO_YAHOO)
        assert "Technology" in CANONICAL_SECTORS
        assert "Financial Services" in CANONICAL_SECTORS


class TestHaramSectorFlag:
    def test_financial_services_is_haram(self):
        assert is_typically_haram("Financial Services") is True

    def test_technology_is_not_haram(self):
        assert is_typically_haram("Technology") is False

    def test_energy_is_not_haram(self):
        assert is_typically_haram("Energy") is False
