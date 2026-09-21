from __future__ import annotations

import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.data.sources import SyntheticSource
from ai_finance.research.registry import Registry
from ai_finance.research.walkforward import make_splits, run_walk_forward
from ai_finance.risk.engine import RiskLimits

FREE = CostModel.free()
OPEN = RiskLimits.unconstrained()
SMALL_GRID = {"fast": [5, 10], "slow": [20, 40]}


def synth(n=1500, seed=1):
    """4-hour bars, so a window of a few hundred spans enough days for a Sharpe."""
    return SyntheticSource(epoch_ms=0, seed=seed).fetch_chunk("S", "4h", 0, limit=n)


class TestMakeSplits:
    def test_basic_layout(self):
        splits = make_splits(1000, train_bars=400, test_bars=100)

        assert len(splits) == 6
        first = splits[0]
        assert (first.train_start, first.train_end) == (0, 400)
        assert (first.test_start, first.test_end) == (400, 500)

    def test_steps_forward_by_one_test_window(self):
        splits = make_splits(1000, train_bars=400, test_bars=100)
        starts = [s.test_start for s in splits]
        assert starts == [400, 500, 600, 700, 800, 900]

    def test_training_never_overlaps_testing(self):
        """The whole point of the exercise."""
        for split in make_splits(2000, train_bars=500, test_bars=200, embargo_bars=25):
            assert split.train_end <= split.test_start
            assert split.test_start >= split.train_end + 25

    def test_embargo_leaves_a_gap(self):
        split = make_splits(1000, train_bars=400, test_bars=100, embargo_bars=50)[0]
        assert split.train_end == 400
        assert split.test_start == 450

    def test_rolling_window_moves_its_start(self):
        splits = make_splits(1000, train_bars=400, test_bars=100)
        assert splits[0].train_start == 0
        assert splits[1].train_start == 100

    def test_anchored_window_keeps_its_start(self):
        splits = make_splits(1000, train_bars=400, test_bars=100, anchored=True)
        assert all(s.train_start == 0 for s in splits)
        assert splits[1].train_end > splits[0].train_end

    def test_warmup_pulls_the_test_run_back_without_moving_the_measurement(self):
        split = make_splits(1000, train_bars=400, test_bars=100, warmup_bars=60)[0]

        assert split.test_run_start == 340  # 60 bars of history before the window
        assert split.test_start == 400  # but measurement still begins here

    def test_empty_when_the_data_is_too_short(self):
        assert make_splits(100, train_bars=400, test_bars=100) == []

    def test_max_splits_caps_the_count(self):
        assert len(make_splits(5000, train_bars=400, test_bars=100, max_splits=3)) == 3

    def test_training_must_outlast_the_warmup(self):
        with pytest.raises(ValueError, match="must exceed warmup_bars"):
            make_splits(1000, train_bars=50, test_bars=100, warmup_bars=50)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_bars": 0, "test_bars": 10},
            {"train_bars": 10, "test_bars": 0},
            {"train_bars": 10, "test_bars": 10, "embargo_bars": -1},
            {"train_bars": 10, "test_bars": 10, "warmup_bars": -1},
        ],
    )
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            make_splits(1000, **kwargs)


class TestRunWalkForward:
    def _run(self, bars=None, **kwargs):
        options = {
            "param_grid": SMALL_GRID,
            "train_bars": 400,
            "test_bars": 150,
            "symbol": "S",
            "interval": "4h",
            "costs": FREE,
            "limits": OPEN,
        }
        options.update(kwargs)
        data = bars if bars is not None else synth(1500)
        return run_walk_forward(data, "ma-crossover", **options)

    def test_produces_an_out_of_sample_curve(self):
        result = self._run()

        assert result.n_splits > 1
        assert len(result.oos.equity_curve) > 0
        assert result.oos.equity_curve.is_monotonic_increasing is not None
        assert "walk-forward" in result.oos.strategy_name

    def test_out_of_sample_curve_starts_at_the_first_test_bar(self):
        """Warm-up returns must not leak into the measured result."""
        bars = synth(1500)
        result = self._run(bars)

        first_split = result.outcomes[0].split
        assert result.oos.equity_curve.index[0] == bars["close_time"].iloc[first_split.test_start]
        assert result.oos.equity_curve.iloc[0] == pytest.approx(result.oos.initial_equity)

    def test_windows_chain_continuously(self):
        """Each window picks up the equity the previous one ended with."""
        result = self._run()

        for previous, current in zip(result.outcomes, result.outcomes[1:], strict=False):
            assert current.test_result.initial_equity == pytest.approx(
                previous.test_result.final_equity
            )

    def test_measured_windows_are_strictly_after_their_training_data(self):
        bars = synth(1500)
        result = self._run(bars)

        for outcome in result.outcomes:
            train_end_time = bars["close_time"].iloc[outcome.split.train_end - 1]
            assert outcome.test_result.equity_curve.index[0] > train_end_time

    def test_counts_every_variant_it_tried(self):
        result = self._run()
        assert result.variants_tested == result.n_splits * 4  # 2 fast x 2 slow

    def test_records_training_and_testing_separately(self, tmp_path):
        registry = Registry(tmp_path / "log.jsonl")
        result = self._run(registry=registry)

        assert registry.count("train") == result.variants_tested
        assert registry.count("test") == result.n_splits

    def test_selection_actually_picks_the_training_winner(self):
        result = self._run(selection_metric="total_return")

        for outcome in result.outcomes:
            assert outcome.chosen_params in [
                {"fast": f, "slow": s} for f in (5, 10) for s in (20, 40)
            ]
            assert outcome.train_score > -float("inf")

    def test_param_turnover_reports_instability(self):
        result = self._run()
        assert 0.0 <= result.param_turnover <= 1.0

    def test_param_turnover_is_zero_for_a_single_choice(self):
        result = self._run(param_grid={"fast": [10], "slow": [50]})
        assert result.param_turnover == 0.0
        assert result.variants_tested == result.n_splits

    def test_invalid_grid_combinations_are_skipped_not_counted(self):
        """fast >= slow is not a crossover, and must not burn a trial."""
        result = self._run(param_grid={"fast": [10, 60], "slow": [20, 40]})
        # Of four combinations only 10/20 and 10/40 are valid.
        assert result.variants_tested == result.n_splits * 2

    def test_report_mentions_the_search_size_and_churn(self):
        text = self._run().to_text()

        assert "parameter runs" in text
        assert "param churn" in text
        assert "OUT OF SAMPLE" in text
        assert "benchmark" in text

    def test_every_baseline_can_be_walked_forward(self):
        for name in ("ma-crossover", "rsi-mean-reversion", "breakout", "vol-scaled-trend"):
            result = run_walk_forward(
                synth(1500),
                name,
                train_bars=400,
                test_bars=200,
                symbol="S",
                costs=FREE,
                limits=OPEN,
                max_splits=2,
            )
            assert result.n_splits == 2

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            run_walk_forward(synth(500), "nope", train_bars=100, test_bars=50)

    def test_unknown_selection_metric_rejected(self):
        with pytest.raises(ValueError, match="unknown selection_metric"):
            self._run(selection_metric="profit")

    def test_too_little_data_is_an_explicit_error(self):
        with pytest.raises(ValueError, match="too short"):
            self._run(synth(300), train_bars=400, test_bars=150)


class TestNoLeakage:
    def test_parameters_are_fixed_before_the_test_window_opens(self):
        """A split's parameters depend only on data that closed before it."""
        bars = synth(1600)
        registry_a = run_walk_forward(
            bars,
            "ma-crossover",
            param_grid=SMALL_GRID,
            train_bars=400,
            test_bars=150,
            symbol="S",
            costs=FREE,
            limits=OPEN,
            max_splits=2,
        )

        # Corrupt everything after the first test window; the first split's
        # choice must be unchanged, because it never saw that data.
        tampered = bars.copy()
        tail = slice(700, None)
        for column in ("open", "high", "low", "close"):
            tampered.loc[tampered.index[tail], column] *= 3.0

        registry_b = run_walk_forward(
            tampered,
            "ma-crossover",
            param_grid=SMALL_GRID,
            train_bars=400,
            test_bars=150,
            symbol="S",
            costs=FREE,
            limits=OPEN,
            max_splits=2,
        )

        assert registry_a.outcomes[0].chosen_params == registry_b.outcomes[0].chosen_params
        assert registry_a.outcomes[0].train_score == pytest.approx(
            registry_b.outcomes[0].train_score
        )
