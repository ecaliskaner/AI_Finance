from __future__ import annotations

import pytest

from ai_finance.backtest.costs import CostModel


def test_default_round_trip_is_slightly_pessimistic():
    """The plan assumes 20bp; the default charges 22bp so results cannot flatter."""
    costs = CostModel()
    assert costs.round_trip_cost == pytest.approx(0.0022)
    assert costs.round_trip_cost > 0.0020


def test_cost_decomposition():
    costs = CostModel(fee_rate=0.001, half_spread_bps=1.0, slippage_bps=2.0)
    assert costs.edge == pytest.approx(0.0003)
    assert costs.per_side_cost == pytest.approx(0.0013)
    assert costs.round_trip_cost == pytest.approx(0.0026)


def test_buying_pays_up_and_selling_receives_less():
    costs = CostModel(half_spread_bps=5.0, slippage_bps=5.0)  # 10bp edge
    assert costs.fill_price(100.0, +1.0) == pytest.approx(100.10)
    assert costs.fill_price(100.0, -1.0) == pytest.approx(99.90)
    assert costs.fill_price(100.0, 0.0) == 100.0


def test_fee_is_charged_on_absolute_notional():
    costs = CostModel(fee_rate=0.001)
    assert costs.fee(1000.0) == pytest.approx(1.0)
    assert costs.fee(-1000.0) == pytest.approx(1.0)


def test_free_model_charges_nothing():
    costs = CostModel.free()
    assert costs.round_trip_cost == 0.0
    assert costs.fill_price(100.0, 1.0) == 100.0
    assert costs.fee(1_000_000.0) == 0.0


def test_bnb_discount_is_worth_5bp_a_round_trip():
    standard = CostModel(fee_rate=0.001)
    discounted = CostModel(fee_rate=0.00075)
    saving = standard.round_trip_cost - discounted.round_trip_cost
    assert saving == pytest.approx(0.0005)


def test_negative_components_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        CostModel(fee_rate=-0.001)
    with pytest.raises(ValueError, match="cannot be negative"):
        CostModel(slippage_bps=-1.0)


def test_describe_reports_the_round_trip():
    assert "22.0bp round trip" in CostModel().describe()
