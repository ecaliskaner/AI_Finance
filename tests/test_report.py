from __future__ import annotations

import re

import numpy as np
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.report import _downsample, _nice_ticks, write_report
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import BuyAndHold, RandomStrategy
from tests.test_engine import synth

OPEN = RiskLimits.unconstrained()


@pytest.fixture
def result():
    return run_backtest(
        synth(2000),
        RandomStrategy(seed=3, every_n_bars=50),
        symbol="BTCUSDT",
        costs=CostModel(),
        limits=OPEN,
    )


class TestOutput:
    def test_writes_a_file(self, result, tmp_path):
        path = write_report(result, tmp_path / "out.html")
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_creates_missing_directories(self, result, tmp_path):
        path = write_report(result, tmp_path / "deep" / "nested" / "out.html")
        assert path.exists()

    def test_is_self_contained(self, result, tmp_path):
        """It must render on a VPS with no outbound network."""
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")

        assert "<style>" in html and "<script>" in html
        for pattern in ("src=", "href=", "@import", "cdn.", "https://", "http://"):
            assert pattern not in html, f"external reference {pattern!r} found"

    def test_includes_both_series_and_a_legend(self, result, tmp_path):
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")

        assert 'class="series-1"' in html
        assert 'class="series-2"' in html
        assert "Buy and hold" in html
        assert "Strategy" in html

    def test_reports_benchmark_and_costs(self, result, tmp_path):
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")

        assert "Benchmark" in html
        assert "Cost drag" in html
        assert "Exchange fees" in html
        assert "Spread and slippage" in html

    def test_cost_drag_is_unsigned(self, result, tmp_path):
        """A cost is not a gain; '+2.4%' would read like one."""
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")
        tile = re.search(r"Cost drag</div><div class=\"tile-value\">([^<]+)", html)
        assert tile is not None
        assert not tile.group(1).startswith("+")

    def test_notes_are_rendered(self, result, tmp_path):
        html = write_report(
            result, tmp_path / "out.html", notes=["prices are synthetic"]
        ).read_text(encoding="utf-8")

        assert "Read this before believing any of it" in html
        assert "prices are synthetic" in html

    def test_no_notes_section_when_there_are_none(self, result, tmp_path):
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")
        assert "Read this before believing" not in html

    def test_titles_are_escaped(self, result, tmp_path):
        html = write_report(
            result, tmp_path / "out.html", title="<script>alert(1)</script>"
        ).read_text(encoding="utf-8")

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_declares_both_colour_schemes(self, result, tmp_path):
        html = write_report(result, tmp_path / "out.html").read_text(encoding="utf-8")
        assert "prefers-color-scheme: dark" in html
        assert "--series-1" in html

    def test_handles_a_strategy_that_never_traded(self, tmp_path):
        flat = run_backtest(
            synth(200), BuyAndHold(weight=0.0), symbol="S", costs=CostModel(), limits=OPEN
        )
        html = write_report(flat, tmp_path / "out.html").read_text(encoding="utf-8")
        assert "n/a" in html  # undefined win rate rather than a crash

    def test_handles_a_two_bar_backtest(self, tmp_path):
        tiny = run_backtest(synth(2), BuyAndHold(), symbol="S", costs=CostModel(), limits=OPEN)
        assert write_report(tiny, tmp_path / "out.html").exists()


class TestChartHelpers:
    def test_downsample_keeps_the_endpoints(self, result):
        thinned = _downsample(result.equity_curve, limit=50)

        assert len(thinned) <= 50
        assert thinned.index[0] == result.equity_curve.index[0]
        assert thinned.index[-1] == result.equity_curve.index[-1]

    def test_downsample_leaves_short_series_alone(self, result):
        short = result.equity_curve.iloc[:10]
        assert _downsample(short, limit=50) is short

    def test_ticks_span_the_data(self):
        ticks = _nice_ticks(3.2, 17.8)
        assert ticks[0] <= 3.2
        assert ticks[-1] >= 17.8

    def test_ticks_are_evenly_spaced_and_round(self):
        ticks = _nice_ticks(0.0, 100.0)
        gaps = np.diff(ticks)
        assert np.allclose(gaps, gaps[0])
        assert all(abs(t - round(t)) < 1e-9 for t in ticks)

    def test_ticks_handle_a_degenerate_range(self):
        assert _nice_ticks(5.0, 5.0) == [0.0, 1.0]
        assert _nice_ticks(float("nan"), 1.0) == [0.0, 1.0]

    def test_ticks_work_for_small_negative_ranges(self):
        ticks = _nice_ticks(-0.77, 0.0)
        assert ticks[0] <= -0.77
        assert ticks[-1] >= 0.0
