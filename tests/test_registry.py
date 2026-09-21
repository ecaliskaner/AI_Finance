from __future__ import annotations

import json

import numpy as np
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.research.registry import Registry, expected_max_sharpe
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import BuyAndHold
from tests.test_engine import synth


@pytest.fixture
def registry(tmp_path):
    return Registry(tmp_path / "experiments.jsonl")


def a_result():
    return run_backtest(
        synth(500),
        BuyAndHold(),
        symbol="S",
        costs=CostModel(),
        limits=RiskLimits.unconstrained(),
    )


class TestRecording:
    def test_empty_registry_reads_as_empty(self, registry):
        assert registry.count() == 0
        assert registry.all().empty
        assert "No experiments recorded" in registry.report()

    def test_round_trip(self, registry):
        registry.record_result(a_result(), params={"fast": 10}, interval="4h", kind="train")

        frame = registry.all()
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["strategy"] == "buy-and-hold"
        assert row["params"] == {"fast": 10}
        assert row["interval"] == "4h"
        assert row["kind"] == "train"
        assert row["costs"]["fee_rate"] == 0.001

    def test_appends_rather_than_overwrites(self, registry):
        for i in range(3):
            registry.record_result(a_result(), params={"fast": i}, interval="4h")

        assert registry.count() == 3
        assert [row["fast"] for row in registry.all()["params"]] == [0, 1, 2]

    def test_counts_by_kind(self, registry):
        registry.record_result(a_result(), params={}, interval="4h", kind="train")
        registry.record_result(a_result(), params={}, interval="4h", kind="train")
        registry.record_result(a_result(), params={}, interval="4h", kind="test")

        assert registry.count() == 3
        assert registry.count("train") == 2
        assert registry.count("test") == 1

    def test_non_finite_metrics_survive_the_json_round_trip(self, registry):
        """A Sharpe of nan must not corrupt the log."""
        result = run_backtest(
            synth(30),
            BuyAndHold(),
            symbol="S",
            costs=CostModel.free(),
            limits=RiskLimits.unconstrained(),
        )
        assert np.isnan(result.sharpe)

        registry.record_result(result, params={}, interval="4h")

        lines = registry.path.read_text().splitlines()
        assert json.loads(lines[0])["sharpe"] is None

    def test_creates_the_parent_directory(self, tmp_path):
        nested = Registry(tmp_path / "deep" / "nested" / "log.jsonl")
        nested.record_result(a_result(), params={}, interval="4h")
        assert nested.path.exists()

    def test_report_summarises_by_strategy(self, registry):
        registry.record_result(a_result(), params={}, interval="4h", kind="train")
        text = registry.report()

        assert "experiments   1" in text
        assert "buy-and-hold" in text

    def test_report_warns_about_multiple_testing(self, registry):
        for i in range(50):
            registry.record_result(a_result(), params={"i": i}, interval="4h", kind="train")

        text = registry.report(n_observations=1000)

        assert "MULTIPLE TESTING" in text
        assert "50 search runs" in text


class TestExpectedMaxSharpe:
    def test_a_single_trial_has_no_selection_bias(self):
        assert expected_max_sharpe(1, 1000) == 0.0

    def test_grows_with_the_number_of_trials(self):
        values = [expected_max_sharpe(n, 1000) for n in (5, 20, 100, 500)]
        assert values == sorted(values)
        assert all(v > 0 for v in values)

    def test_shrinks_with_more_observations(self):
        """More data makes a lucky result harder to come by."""
        assert expected_max_sharpe(100, 5000) < expected_max_sharpe(100, 500)

    def test_matches_a_hand_computation(self):
        """200 trials on 1,000 daily observations is about 1.7."""
        assert expected_max_sharpe(200, 1000) == pytest.approx(1.67, abs=0.05)

    def test_scales_with_the_square_root_of_annualisation(self):
        daily = expected_max_sharpe(100, 1000, periods_per_year=365)
        quarterly = expected_max_sharpe(100, 1000, periods_per_year=365 / 4)
        assert daily / quarterly == pytest.approx(2.0, rel=1e-6)

    def test_the_headline_warning_is_real(self):
        """With enough tries, a worthless strategy clears a Sharpe of 1."""
        assert expected_max_sharpe(1000, 1000) > 1.0

    @pytest.mark.parametrize("args", [(0, 1000), (-1, 1000), (10, 1), (10, 0)])
    def test_rejects_nonsense_inputs(self, args):
        with pytest.raises(ValueError):
            expected_max_sharpe(*args)


class TestJsonValidity:
    """The log must be readable by something other than Python."""

    def test_written_lines_are_strict_json(self, registry):
        result = run_backtest(
            synth(30),
            BuyAndHold(),
            symbol="S",
            costs=CostModel.free(),
            limits=RiskLimits.unconstrained(),
        )
        registry.record_result(result, params={"x": float("nan")}, interval="4h")

        for line in registry.path.read_text().splitlines():
            # parse_constant fires on NaN/Infinity, which are not valid JSON.
            json.loads(line, parse_constant=_reject)

    def test_infinities_are_encoded_as_null_too(self, registry):
        registry.record_result(
            a_result(), params={"ratio": float("inf")}, interval="4h", note="inf test"
        )
        row = json.loads(registry.path.read_text().splitlines()[-1])
        assert row["params"]["ratio"] is None


def _reject(constant):
    raise AssertionError(f"invalid JSON literal in the log: {constant}")
