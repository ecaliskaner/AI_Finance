from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.risk.engine import RiskEngine, RiskLimits
from ai_finance.strategy.base import Signal

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def sig(weight, confidence=1.0):
    return Signal(symbol="BTCUSDT", target_weight=weight, confidence=confidence, reason="test")


class TestLimitsValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_position_weight": 1.5},
            {"max_position_weight": -0.1},
            {"max_total_exposure": -1.0},
            {"max_daily_loss": 0.0},
            {"max_drawdown": -0.1},
            {"min_order_notional": -1.0},
            {"max_orders_per_hour": 0},
        ],
    )
    def test_nonsense_limits_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RiskLimits(**kwargs)

    def test_defaults_match_the_risk_document(self):
        limits = RiskLimits()
        assert limits.max_position_weight == 0.25
        assert limits.max_drawdown == 0.15
        assert limits.max_daily_loss == 0.03
        assert limits.min_order_notional == 20.0
        assert limits.max_orders_per_hour == 20
        assert limits.allow_short is False

    def test_unconstrained_never_binds(self):
        engine = RiskEngine(limits=RiskLimits.unconstrained())
        assert engine.apply(sig(1.0), T0).target_weight == 1.0
        assert engine.apply(sig(-1.0), T0).target_weight == -1.0


class TestPositionCap:
    def test_caps_at_max_position_weight(self):
        engine = RiskEngine(limits=RiskLimits(max_position_weight=0.25))
        decision = engine.apply(sig(1.0), T0)

        assert decision.target_weight == 0.25
        assert decision.was_adjusted
        assert "capped at 0.25" in decision.adjustments[0]

    def test_leaves_a_smaller_request_alone(self):
        engine = RiskEngine(limits=RiskLimits(max_position_weight=0.25))
        decision = engine.apply(sig(0.1), T0)
        assert decision.target_weight == 0.1
        assert not decision.was_adjusted

    def test_total_exposure_can_bind_before_position_weight(self):
        limits = RiskLimits(max_position_weight=0.9, max_total_exposure=0.3)
        assert RiskEngine(limits=limits).apply(sig(0.9), T0).target_weight == pytest.approx(0.3)

    def test_shorts_blocked_by_default(self):
        decision = RiskEngine().apply(sig(-1.0), T0)
        assert decision.target_weight == 0.0
        assert "shorting disabled" in decision.adjustments[0]

    def test_shorts_allowed_when_enabled_and_still_capped(self):
        limits = RiskLimits(max_position_weight=0.25).allowing_short()
        assert RiskEngine(limits=limits).apply(sig(-1.0), T0).target_weight == pytest.approx(-0.25)

    def test_confidence_scales_size(self):
        limits = RiskLimits(max_position_weight=1.0)
        decision = RiskEngine(limits=limits).apply(sig(1.0, confidence=0.4), T0)

        assert decision.target_weight == pytest.approx(0.4)
        assert "confidence 0.40" in decision.adjustments[0]

    def test_confidence_scaling_can_be_disabled(self):
        limits = RiskLimits(max_position_weight=1.0, scale_by_confidence=False)
        assert RiskEngine(limits=limits).apply(sig(1.0, confidence=0.4), T0).target_weight == 1.0


class TestOrderPermission:
    def test_rejects_orders_too_small_to_be_worth_the_fee(self):
        engine = RiskEngine(limits=RiskLimits(min_order_notional=20.0))
        permitted, why = engine.permits_order(19.99, T0)

        assert not permitted
        assert "below minimum" in why
        assert "fees would dominate" in why

    def test_allows_an_order_at_the_minimum(self):
        engine = RiskEngine(limits=RiskLimits(min_order_notional=20.0))
        assert engine.permits_order(20.0, T0)[0]

    def test_rate_limit_blocks_a_runaway_loop(self):
        engine = RiskEngine(limits=RiskLimits(max_orders_per_hour=3, min_order_notional=0.0))
        for i in range(3):
            assert engine.permits_order(100.0, T0 + pd.Timedelta(minutes=i))[0]
            engine.record_order(T0 + pd.Timedelta(minutes=i))

        permitted, why = engine.permits_order(100.0, T0 + pd.Timedelta(minutes=4))
        assert not permitted
        assert "rate limit" in why

    def test_rate_limit_window_rolls_forward(self):
        engine = RiskEngine(limits=RiskLimits(max_orders_per_hour=2, min_order_notional=0.0))
        engine.record_order(T0)
        engine.record_order(T0 + pd.Timedelta(minutes=1))
        assert not engine.permits_order(100.0, T0 + pd.Timedelta(minutes=2))[0]

        # An hour later the early orders have aged out.
        assert engine.permits_order(100.0, T0 + pd.Timedelta(hours=1, minutes=2))[0]


class TestHalts:
    def test_daily_loss_halt_forces_flat(self):
        engine = RiskEngine(limits=RiskLimits(max_daily_loss=0.03, max_drawdown=0.9))
        engine.observe(10_000.0, T0)
        engine.observe(9_650.0, T0 + pd.Timedelta(hours=1))  # -3.5%

        assert engine.is_halted(T0 + pd.Timedelta(hours=1))
        assert engine.apply(sig(1.0), T0 + pd.Timedelta(hours=1)).target_weight == 0.0
        assert "daily loss" in engine.halt_reason

    def test_daily_loss_halt_expires_after_24h(self):
        engine = RiskEngine(limits=RiskLimits(max_daily_loss=0.03, max_drawdown=0.9))
        engine.observe(10_000.0, T0)
        engine.observe(9_600.0, T0 + pd.Timedelta(hours=1))

        assert engine.is_halted(T0 + pd.Timedelta(hours=23))
        assert not engine.is_halted(T0 + pd.Timedelta(hours=25))
        assert not engine.halted_permanently

    def test_drawdown_halt_is_permanent(self):
        engine = RiskEngine(limits=RiskLimits(max_drawdown=0.15, max_daily_loss=0.9))
        engine.observe(10_000.0, T0)
        engine.observe(8_400.0, T0 + pd.Timedelta(days=1))  # -16% from peak

        assert engine.halted_permanently
        assert "manual restart required" in engine.halt_reason
        # Not even a full recovery re-enables it.
        engine.observe(20_000.0, T0 + pd.Timedelta(days=30))
        assert engine.is_halted(T0 + pd.Timedelta(days=365))
        assert engine.apply(sig(1.0), T0 + pd.Timedelta(days=365)).target_weight == 0.0

    def test_drawdown_measured_from_peak_not_from_start(self):
        engine = RiskEngine(limits=RiskLimits(max_drawdown=0.15, max_daily_loss=0.9))
        engine.observe(10_000.0, T0)
        engine.observe(20_000.0, T0 + pd.Timedelta(days=1))
        engine.observe(18_000.0, T0 + pd.Timedelta(days=2))  # -10% from peak, still up on start

        assert not engine.halted_permanently

        engine.observe(16_900.0, T0 + pd.Timedelta(days=3))  # -15.5% from peak
        assert engine.halted_permanently

    def test_daily_anchor_resets_each_utc_day(self):
        engine = RiskEngine(limits=RiskLimits(max_daily_loss=0.03, max_drawdown=0.9))
        engine.observe(10_000.0, T0)
        engine.observe(9_800.0, T0 + pd.Timedelta(hours=12))  # -2%, under the limit
        assert not engine.is_halted(T0 + pd.Timedelta(hours=12))

        # New day: the anchor moves to 9,800 so a further 2% is not cumulative.
        engine.observe(9_800.0, T0 + pd.Timedelta(days=1))
        engine.observe(9_610.0, T0 + pd.Timedelta(days=1, hours=1))
        assert not engine.is_halted(T0 + pd.Timedelta(days=1, hours=1))

    def test_a_quiet_account_is_never_halted(self):
        engine = RiskEngine()
        for day in range(10):
            engine.observe(10_000.0, T0 + pd.Timedelta(days=day))
        assert not engine.is_halted(T0 + pd.Timedelta(days=10))
        assert engine.halt_reason == ""
