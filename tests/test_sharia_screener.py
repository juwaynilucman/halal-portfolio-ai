"""Unit tests for the Sharia screener."""

import pytest

from src.data.sharia_screener import ComplianceStatus, ShariaScreener


class TestKnownHaramTickers:
    def test_conventional_bank_is_haram(self):
        screener = ShariaScreener()
        result = screener.screen("JPM")
        assert result.status == ComplianceStatus.HARAM

    def test_alcohol_ticker_is_haram(self):
        screener = ShariaScreener()
        result = screener.screen("MO")
        assert result.status == ComplianceStatus.HARAM

    def test_casino_ticker_is_haram(self):
        screener = ShariaScreener()
        result = screener.screen("MGM")
        assert result.status == ComplianceStatus.HARAM


class TestManualOverrides:
    def test_override_halal_supersedes_haram_list(self, tmp_path):
        import json

        override_file = tmp_path / "overrides.json"
        override_file.write_text(
            json.dumps({"JPM": {"status": "halal", "purification_pct": 0.0}}),
            encoding="utf-8",
        )
        screener = ShariaScreener(manual_overrides_path=str(override_file))
        result = screener.screen("JPM")
        assert result.status == ComplianceStatus.HALAL
        assert result.source == "manual_override"

    def test_override_mixed_sets_purification_pct(self, tmp_path):
        import json

        override_file = tmp_path / "overrides.json"
        override_file.write_text(
            json.dumps({"AAPL": {"status": "mixed", "purification_pct": 0.03}}),
            encoding="utf-8",
        )
        screener = ShariaScreener(manual_overrides_path=str(override_file))
        result = screener.screen("AAPL")
        assert result.status == ComplianceStatus.MIXED
        assert result.purification_pct == pytest.approx(0.03)


class TestFilterHalal:
    def test_filter_removes_haram_tickers(self):
        screener = ShariaScreener()
        tickers = ["JPM", "BAC", "AAPL"]  # first two haram, AAPL unknown (stub)
        halal = screener.filter_halal(tickers)
        assert "JPM" not in halal
        assert "BAC" not in halal

    def test_bulk_screen_skips_errors(self):
        screener = ShariaScreener()
        # "$$INVALID" is not a real ticker but should not crash the bulk call
        results = screener.screen_bulk(["JPM", "$$INVALID"])
        assert "JPM" in results
        assert "$$INVALID" in results
