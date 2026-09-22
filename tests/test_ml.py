from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.ml import (
    ModelStrategy,
    PredictionPolicy,
    linear_coefficients,
    make_model,
)
from tests.test_engine import synth

OPEN = RiskLimits.unconstrained()


class TestPredictionPolicy:
    def test_threshold_is_a_multiple_of_the_round_trip(self):
        policy = PredictionPolicy(CostModel(), safety_factor=1.5)
        assert policy.threshold == pytest.approx(0.0022 * 1.5)

    def test_a_prediction_below_the_threshold_is_not_traded(self):
        """Being right about a 5bp move is worthless when a round trip costs 22."""
        policy = PredictionPolicy(CostModel())
        assert policy.weight(0.0005) == 0.0
        assert "needed to clear costs" in policy.explain(0.0005)

    def test_a_prediction_just_above_the_threshold_trades_small(self):
        policy = PredictionPolicy(CostModel(), full_size_return=0.02)
        weight = policy.weight(0.004)
        assert 0 < weight < 0.25

    def test_a_large_prediction_reaches_max_weight(self):
        policy = PredictionPolicy(CostModel(), full_size_return=0.02, max_weight=1.0)
        assert policy.weight(0.05) == pytest.approx(1.0)

    def test_raising_costs_makes_the_policy_pickier(self):
        cheap = PredictionPolicy(CostModel(fee_rate=0.001))
        pricey = PredictionPolicy(CostModel(fee_rate=0.005))

        assert pricey.threshold > cheap.threshold
        assert cheap.weight(0.005) > 0
        assert pricey.weight(0.005) == 0.0

    def test_shorts_are_refused_by_default(self):
        policy = PredictionPolicy(CostModel())
        assert policy.weight(-0.05) == 0.0
        assert "shorting is disabled" in policy.explain(-0.05)

    def test_shorts_are_sized_when_enabled(self):
        policy = PredictionPolicy(CostModel(), allow_short=True, full_size_return=0.02)
        assert policy.weight(-0.05) == pytest.approx(-1.0)

    def test_a_missing_prediction_means_flat(self):
        policy = PredictionPolicy(CostModel())
        assert policy.weight(float("nan")) == 0.0
        assert "no prediction" in policy.explain(float("nan"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"safety_factor": 0.0},
            {"full_size_return": 0.0},
            {"max_weight": 0.0},
            {"max_weight": 2.0},
        ],
    )
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            PredictionPolicy(CostModel(), **kwargs)


class TestModelStrategy:
    def test_acts_on_a_confident_prediction(self):
        bars = synth(200)
        predictions = pd.Series(0.05, index=pd.DatetimeIndex(bars["close_time"]))

        result = run_backtest(
            bars,
            ModelStrategy(predictions, PredictionPolicy(CostModel())),
            symbol="S",
            costs=CostModel(),
            limits=OPEN,
        )

        assert result.n_fills >= 1
        assert result.weight_curve.max() > 0.5

    def test_stays_flat_when_no_prediction_clears_the_threshold(self):
        bars = synth(200)
        predictions = pd.Series(0.0001, index=pd.DatetimeIndex(bars["close_time"]))

        result = run_backtest(
            bars,
            ModelStrategy(predictions, PredictionPolicy(CostModel())),
            symbol="S",
            costs=CostModel(),
            limits=OPEN,
        )

        assert result.n_fills == 0
        assert result.total_costs == 0.0

    def test_a_bar_without_a_prediction_leaves_the_position_alone(self):
        """A skipped model run must not silently flatten the book."""
        bars = synth(200)
        times = pd.DatetimeIndex(bars["close_time"])
        # Predict confidently for the first half only.
        predictions = pd.Series(0.05, index=times[:100])

        result = run_backtest(
            bars,
            ModelStrategy(predictions, PredictionPolicy(CostModel())),
            symbol="S",
            costs=CostModel(),
            limits=OPEN,
        )

        assert result.weight_curve.iloc[-1] > 0.5

    def test_matches_predictions_by_timestamp_not_position(self):
        """Walk-forward slices bars, so positional lookup would misalign."""
        bars = synth(200)
        times = pd.DatetimeIndex(bars["close_time"])
        predictions = pd.Series(0.05, index=times[50:])

        result = run_backtest(
            bars.iloc[40:].reset_index(drop=True),
            ModelStrategy(predictions, PredictionPolicy(CostModel())),
            symbol="S",
            costs=CostModel(),
            limits=OPEN,
        )

        # Entry happens after the tenth bar of this slice, not the first.
        assert (result.weight_curve.iloc[:10].abs() < 1e-12).all()
        assert result.weight_curve.iloc[-1] > 0.5


class TestModels:
    def test_ridge_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4))
        y = x[:, 0] * 0.01 + rng.normal(scale=0.001, size=200)

        model = make_model("ridge")
        model.fit(x, y)

        assert model.predict(x[:5]).shape == (5,)
        assert np.corrcoef(model.predict(x), y)[0, 1] > 0.9

    def test_gbm_fits_and_predicts(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 4))
        y = np.sign(x[:, 0]) * 0.01

        model = make_model("gbm", max_iter=30)
        model.fit(x, y)

        assert model.predict(x[:5]).shape == (5,)

    def test_ridge_standardises_its_inputs(self):
        """Without scaling, a feature measured in thousands dominates the penalty."""
        model = make_model("ridge")
        assert "standardscaler" in model.named_steps

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="unknown model"):
            make_model("deep-neural-net")

    def test_coefficients_are_reported_for_a_linear_model(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = x[:, 1] * 0.05
        model = make_model("ridge")
        model.fit(x, y)

        coefficients = linear_coefficients(model, ["a", "b", "c"])

        assert coefficients is not None
        assert coefficients.index[0] == "b"  # largest magnitude first

    def test_no_coefficients_for_a_tree_model(self):
        model = make_model("gbm", max_iter=10)
        model.fit(np.random.default_rng(0).normal(size=(100, 3)), np.zeros(100))
        assert linear_coefficients(model, ["a", "b", "c"]) is None
