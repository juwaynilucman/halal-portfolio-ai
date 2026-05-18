"""Unit tests for the purification calculator."""

import pytest

from src.portfolio.purification import PurificationCalculator, PurificationResult


class TestDividendPurification:
    def test_basic_dividend_purification(self):
        calc = PurificationCalculator(tax_drag_factor=0.7)
        result = calc.calculate(
            ticker="TEST",
            old_fraction=10.0,
            delta_fraction=0.0,
            old_price=100.0,
            new_price=100.0,
            dividend_rate=4.0,        # $4/year
            purification_pct=0.05,    # 5% non-permissible
        )
        # (4.0 / 4) × 10.0 × 0.05 × 0.7 = 0.035
        assert result.dividend_purification == pytest.approx(0.035, rel=1e-6)
        assert result.growth_purification == 0.0

    def test_no_purification_for_zero_pct(self):
        calc = PurificationCalculator()
        result = calc.calculate(
            ticker="CLEAN",
            old_fraction=50.0,
            delta_fraction=0.0,
            old_price=200.0,
            new_price=210.0,
            dividend_rate=2.0,
            purification_pct=0.0,
        )
        assert result.total_purification == 0.0


class TestGrowthPurification:
    def test_growth_purification_on_sell_with_gain(self):
        calc = PurificationCalculator()
        result = calc.calculate(
            ticker="TEST",
            old_fraction=10.0,
            delta_fraction=-3.0,     # sold 3 shares
            old_price=100.0,
            new_price=130.0,         # price rose
            dividend_rate=0.0,
            purification_pct=0.05,
        )
        # |((130 - 100) × -3) × 0.05| = |-90 × 0.05| = 4.5 — wait, formula is:
        # (new - old) × delta × purification_pct = 30 × -3 × 0.05 = -4.5, abs = 4.5
        assert result.growth_purification == pytest.approx(4.5, rel=1e-6)

    def test_no_growth_purification_when_delta_positive(self):
        calc = PurificationCalculator()
        result = calc.calculate(
            ticker="TEST",
            old_fraction=5.0,
            delta_fraction=2.0,      # bought more — no growth purification
            old_price=100.0,
            new_price=120.0,
            dividend_rate=0.0,
            purification_pct=0.05,
        )
        assert result.growth_purification == 0.0

    def test_no_growth_purification_when_price_fell(self):
        calc = PurificationCalculator()
        result = calc.calculate(
            ticker="TEST",
            old_fraction=10.0,
            delta_fraction=-2.0,
            old_price=150.0,
            new_price=100.0,         # price fell — no growth purification
            dividend_rate=0.0,
            purification_pct=0.05,
        )
        assert result.growth_purification == 0.0


class TestCustomPurificationFn:
    def test_custom_function_is_called(self):
        def flat_fee(**kwargs) -> float:
            return 99.99

        calc = PurificationCalculator(custom_purification_fn=flat_fee)
        result = calc.calculate(
            ticker="X",
            old_fraction=1.0,
            delta_fraction=0.0,
            old_price=10.0,
            new_price=10.0,
            dividend_rate=1.0,
            purification_pct=0.1,
        )
        assert result.total_purification == pytest.approx(99.99)
