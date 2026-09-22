from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.data.schema import normalize_bars
from ai_finance.data.sources import SyntheticSource
from ai_finance.features.pipeline import PointInTimeError
from ai_finance.research.ml_walkforward import accuracy_z_score, run_ml_walk_forward
from ai_finance.research.registry import Registry
from ai_finance.risk.engine import RiskLimits

FREE = CostModel.free()
OPEN = RiskLimits.unconstrained()


def bars_4h(n=3000, seed=1):
    return SyntheticSource(epoch_ms=0, seed=seed).fetch_chunk("S", "4h", 0, limit=n)


def mean_reverting_bars(n=3000, phi=-0.35, seed=0):
    """Bars whose next return is genuinely predictable from the last one.

    A positive control. Without it, "the model found nothing" is ambiguous
    between "there was nothing to find" and "the pipeline cannot find anything".
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = phi * returns[i - 1] + rng.normal(scale=0.01)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]])
    # Volume must vary: a constant column makes volume_z undefined for every
    # row, which drops the entire dataset.
    volume = np.exp(rng.normal(0.0, 0.3, n)) * 10.0
    step = 4 * 3600 * 1000
    return normalize_bars(
        pd.DataFrame(
            {
                "open_time": [i * step for i in range(n)],
                "open": open_,
                "high": np.maximum(open_, close) * 1.001,
                "low": np.minimum(open_, close) * 0.999,
                "close": close,
                "volume": volume,
                "trades": (volume * 5).astype(int),
                "close_time": [(i + 1) * step - 1 for i in range(n)],
            }
        )
    )


def run(bars, **kwargs):
    options = {
        "horizon": 1,
        "model_kind": "ridge",
        "train_size": 600,
        "test_size": 300,
        "embargo": 5,
        "symbol": "S",
        "interval": "4h",
        "costs": FREE,
        "limits": OPEN,
    }
    options.update(kwargs)
    return run_ml_walk_forward(bars, **options)


class TestPositiveControl:
    """Prove the pipeline can find a signal that is really there."""

    def test_detects_genuine_mean_reversion(self):
        result = run(mean_reverting_bars())

        assert result.accuracy > 0.55, "a real AR(1) signal should be detectable"
        assert result.accuracy_z > 3.0
        assert result.ic > 0.1

    def test_finds_nothing_in_a_random_walk(self):
        """The same pipeline, on data with no signal, must come up empty."""
        result = run(bars_4h(3000))

        assert abs(result.accuracy - 0.5) < 0.03
        assert abs(result.ic) < 0.1

    def test_the_learned_coefficient_points_the_right_way(self):
        """Mean reversion means yesterday's gain predicts tomorrow's loss."""
        result = run(mean_reverting_bars())

        assert result.coefficients is not None
        assert result.coefficients["ret_1"] < 0


class TestMechanics:
    def test_runs_end_to_end(self):
        result = run(bars_4h(2000))

        assert result.n_folds >= 1
        assert len(result.predictions) == len(result.actuals)
        assert len(result.oos.equity_curve) > 0
        assert "walk-forward" in result.oos.strategy_name

    def test_folds_chain_from_the_running_equity(self):
        result = run(bars_4h(3000))

        for previous, current in zip(result.folds, result.folds[1:], strict=False):
            assert current.backtest.initial_equity == pytest.approx(previous.backtest.final_equity)

    def test_predictions_are_indexed_by_close_time(self):
        bars = bars_4h(2000)
        result = run(bars)

        assert result.predictions.index.isin(pd.DatetimeIndex(bars["close_time"])).all()

    def test_gbm_also_runs(self):
        result = run(bars_4h(2000), model_kind="gbm", model_params={"max_iter": 30})
        assert result.n_folds >= 1
        assert result.coefficients is None  # trees have no coefficients

    def test_records_each_fold_to_the_registry(self, tmp_path):
        registry = Registry(tmp_path / "log.jsonl")
        result = run(bars_4h(3000), registry=registry)

        assert registry.count("test") == result.n_folds

    def test_report_covers_signal_quality_and_money(self):
        text = run(bars_4h(2000)).to_text()

        assert "SIGNAL QUALITY" in text
        assert "accuracy" in text
        assert "info coef" in text
        assert "WHAT IT EARNED" in text
        assert "benchmark" in text

    def test_report_flags_a_suspiciously_high_ic(self):
        text = run(mean_reverting_bars(phi=-0.9)).to_text()
        result = run(mean_reverting_bars(phi=-0.9))
        if abs(result.ic) >= 0.15:
            assert "SUSPICIOUSLY HIGH" in text

    def test_too_little_data_is_an_explicit_error(self):
        with pytest.raises(ValueError, match="too short"):
            run(bars_4h(300), train_size=600, test_size=300)

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="unknown model"):
            run(bars_4h(2000), model_kind="magic")


class TestLeakDefences:
    def test_point_in_time_check_runs_by_default(self, monkeypatch):
        calls = []
        real = run_ml_walk_forward.__globals__["assert_point_in_time"]

        def spy(bars, ppy, **kwargs):
            calls.append(1)
            return real(bars, ppy, **kwargs)

        monkeypatch.setitem(run_ml_walk_forward.__globals__, "assert_point_in_time", spy)
        run(bars_4h(2000))
        assert calls, "the leak check must run unless explicitly skipped"

    def test_a_leaky_feature_aborts_the_run(self, monkeypatch):
        def leaky(bars, ppy, **kwargs):
            raise PointInTimeError("feature 'tomorrow' saw the future")

        monkeypatch.setitem(run_ml_walk_forward.__globals__, "assert_point_in_time", leaky)

        with pytest.raises(PointInTimeError, match="saw the future"):
            run(bars_4h(2000))

    def test_the_check_can_be_skipped_explicitly(self, monkeypatch):
        calls = []
        monkeypatch.setitem(
            run_ml_walk_forward.__globals__,
            "assert_point_in_time",
            lambda *a, **k: calls.append(1),
        )
        run(bars_4h(2000), verify_point_in_time=False)
        assert not calls

    def test_early_folds_are_unaffected_by_later_data(self):
        """A fold's model must depend only on bars that closed before it."""
        bars = mean_reverting_bars(3000)
        clean = run(bars, max_splits=1)

        tampered = bars.copy()
        tail = tampered.index[2000:]
        for column in ("open", "high", "low", "close"):
            tampered.loc[tail, column] *= 5.0

        corrupted = run(tampered, max_splits=1)

        pd.testing.assert_series_equal(clean.predictions, corrupted.predictions)


class TestAccuracyZScore:
    def test_a_coin_flip_scores_zero(self):
        assert accuracy_z_score(0.5, 1000) == 0.0

    def test_the_same_edge_is_more_convincing_with_more_data(self):
        assert accuracy_z_score(0.52, 10_000) > accuracy_z_score(0.52, 100)

    def test_a_small_sample_cannot_establish_a_small_edge(self):
        """52% on 300 samples is noise; the z-score says so."""
        assert accuracy_z_score(0.52, 300) < 2.0

    def test_a_large_sample_can(self):
        assert accuracy_z_score(0.52, 30_000) > 3.0

    def test_matches_a_hand_computation(self):
        # (0.55 - 0.5) / sqrt(0.25 / 400) = 0.05 / 0.025 = 2.0
        assert accuracy_z_score(0.55, 400) == pytest.approx(2.0)

    def test_nan_inputs_and_tiny_samples(self):
        assert np.isnan(accuracy_z_score(0.5, 1))
        assert np.isnan(accuracy_z_score(float("nan"), 1000))
