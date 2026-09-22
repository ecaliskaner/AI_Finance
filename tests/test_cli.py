from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from ai_finance.cli import main
from ai_finance.data.store import load_bars, write_bars
from tests.conftest import bars_frame


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A runner with AIFIN_DATA_DIR pointed at a temp directory."""
    monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path))
    return CliRunner()


def _bars_root(tmp_path):
    return tmp_path / "bars"


class TestFetchCommand:
    def test_synthetic_fetch_populates_the_store(self, cli, tmp_path):
        result = cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "SYNTH",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-01 02:00",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "wrote 121 bars" in result.output
        assert len(load_bars("SYNTH", root=_bars_root(tmp_path))) == 121

    def test_is_deterministic_for_a_given_seed(self, cli, tmp_path):
        args = [
            "fetch",
            "--source",
            "synthetic",
            "--symbol",
            "S",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-01 00:30",
            "--seed",
            "42",
        ]
        cli.invoke(main, args)
        first = load_bars("S", root=_bars_root(tmp_path))

        cli.invoke(main, [*args, "--no-resume"])
        second = load_bars("S", root=_bars_root(tmp_path))

        pd.testing.assert_frame_equal(first, second)

    def test_rerun_resumes_and_reports_it(self, cli, tmp_path):
        args = ["fetch", "--source", "synthetic", "--symbol", "S", "--start", "2024-01-01"]
        cli.invoke(main, [*args, "--end", "2024-01-01 01:00"])

        result = cli.invoke(main, [*args, "--end", "2024-01-01 02:00"])

        assert result.exit_code == 0, result.output
        assert "resumed after" in result.output
        assert len(load_bars("S", root=_bars_root(tmp_path))) == 121

    def test_multiple_symbols(self, cli, tmp_path):
        result = cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "AAA",
                "--symbol",
                "BBB",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-01 00:10",
            ],
        )

        assert result.exit_code == 0, result.output
        assert len(load_bars("AAA", root=_bars_root(tmp_path))) == 11
        assert len(load_bars("BBB", root=_bars_root(tmp_path))) == 11

    def test_warns_when_storing_a_non_base_interval(self, cli):
        result = cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "S",
                "--interval",
                "1h",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
            ],
        )
        assert "Convention is to store only 1m" in result.output

    def test_rejects_an_unknown_interval(self, cli):
        result = cli.invoke(main, ["fetch", "--interval", "7m"])
        assert result.exit_code != 0
        assert "7m" in result.output


class TestQualityCommand:
    def test_clean_data_exits_zero(self, cli, tmp_path):
        write_bars(bars_frame("2024-01-01", 500), "BTCUSDT", root=_bars_root(tmp_path))

        result = cli.invoke(main, ["quality", "--symbol", "BTCUSDT"])

        assert result.exit_code == 0, result.output
        assert "ERRORS     none" in result.output
        assert "All checked series are clean" in result.output

    def test_gappy_data_exits_nonzero(self, cli, tmp_path):
        gappy = pd.concat(
            [bars_frame("2024-01-01 00:00", 10), bars_frame("2024-01-01 01:00", 10)],
            ignore_index=True,
        )
        write_bars(gappy, "BTCUSDT", root=_bars_root(tmp_path))

        result = cli.invoke(main, ["quality", "--symbol", "BTCUSDT"])

        assert result.exit_code == 1
        assert "missing bars" in result.output
        assert "data quality errors" in result.output

    def test_empty_store_is_reported_not_crashed(self, cli):
        result = cli.invoke(main, ["quality", "--symbol", "NOTHING"])
        assert result.exit_code == 0
        assert "No bars stored" in result.output


class TestInfoAndShow:
    def test_info_on_empty_store(self, cli):
        result = cli.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "Store is empty" in result.output

    def test_info_lists_stored_symbols(self, cli, tmp_path):
        write_bars(bars_frame("2024-01-01", 42), "BTCUSDT", root=_bars_root(tmp_path))

        result = cli.invoke(main, ["info"])

        assert "BTCUSDT" in result.output
        assert "42" in result.output

    def test_show_prints_the_tail(self, cli, tmp_path):
        write_bars(bars_frame("2024-01-01", 300), "BTCUSDT", root=_bars_root(tmp_path))

        result = cli.invoke(main, ["show", "--symbol", "BTCUSDT", "--tail", "3"])

        assert result.exit_code == 0, result.output
        assert "300 bars" in result.output
        assert result.output.count("2024-01-01") >= 3

    def test_show_resamples(self, cli, tmp_path):
        write_bars(bars_frame("2024-01-01", 1440), "BTCUSDT", root=_bars_root(tmp_path))

        result = cli.invoke(main, ["show", "--symbol", "BTCUSDT", "--interval", "4h"])

        assert "6 bars" in result.output

    def test_show_on_missing_symbol(self, cli):
        result = cli.invoke(main, ["show", "--symbol", "NOPE"])
        assert result.exit_code == 0
        assert "No bars" in result.output


class TestBacktestCommand:
    @pytest.fixture
    def stored(self, cli, tmp_path):
        """A day of synthetic 1-minute bars in the store."""
        cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "SYNTH",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-31 23:59",
            ],
        )
        return tmp_path

    def test_runs_and_reports_benchmark_and_costs(self, cli, stored):
        result = cli.invoke(main, ["backtest", "--symbol", "SYNTH", "--interval", "4h"])

        assert result.exit_code == 0, result.output
        assert "Backtest: buy-and-hold on SYNTH" in result.output
        assert "benchmark" in result.output
        assert "total cost" in result.output

    def test_each_reference_strategy_runs(self, cli, stored):
        for name in ("buy-and-hold", "always-flat", "random"):
            result = cli.invoke(
                main, ["backtest", "--symbol", "SYNTH", "--interval", "4h", "--strategy", name]
            )
            assert result.exit_code == 0, f"{name}: {result.output}"
            assert name in result.output

    def test_always_flat_returns_exactly_zero(self, cli, stored):
        result = cli.invoke(
            main,
            ["backtest", "--symbol", "SYNTH", "--interval", "4h", "--strategy", "always-flat"],
        )
        assert "return       +0.00%" in result.output
        assert "total cost   0.00" in result.output

    def test_cost_sensitivity_flags_change_the_result(self, cli, stored):
        """PLAN section 5: re-run the winner at 30bp and see if the edge survives."""
        base = cli.invoke(
            main,
            [
                "backtest",
                "--symbol",
                "SYNTH",
                "--interval",
                "1h",
                "--strategy",
                "random",
                "--unconstrained",
                "--seed",
                "3",
            ],
        )
        pricey = cli.invoke(
            main,
            [
                "backtest",
                "--symbol",
                "SYNTH",
                "--interval",
                "1h",
                "--strategy",
                "random",
                "--unconstrained",
                "--seed",
                "3",
                "--fee",
                "0.005",
            ],
        )

        assert base.exit_code == pricey.exit_code == 0
        # Default is 2 x (10bp fee + 1bp edge) = 22bp; a 0.5% fee makes it 102bp.
        assert "22.0bp round trip" in base.output
        assert "102.0bp round trip" in pricey.output

    def test_unconstrained_flag_warns_loudly(self, cli, stored):
        result = cli.invoke(
            main, ["backtest", "--symbol", "SYNTH", "--interval", "4h", "--unconstrained"]
        )
        assert "measures the engine, not a strategy you could run" in result.output

    def test_default_limits_cap_the_position(self, cli, stored):
        result = cli.invoke(main, ["backtest", "--symbol", "SYNTH", "--interval", "4h"])
        assert "capped at 0.25" in result.output

    def test_liquidate_flag_completes_the_trade(self, cli, stored):
        held = cli.invoke(main, ["backtest", "--symbol", "SYNTH", "--interval", "4h"])
        closed = cli.invoke(
            main, ["backtest", "--symbol", "SYNTH", "--interval", "4h", "--liquidate"]
        )

        assert "trades       0 " in held.output
        assert "trades       1 " in closed.output

    def test_warns_when_the_data_has_quality_errors(self, cli, tmp_path):
        gappy = pd.concat(
            [bars_frame("2024-01-01 00:00", 500), bars_frame("2024-01-02 00:00", 500)],
            ignore_index=True,
        )
        write_bars(gappy, "GAPPY", root=tmp_path / "bars")

        result = cli.invoke(main, ["backtest", "--symbol", "GAPPY", "--interval", "1h"])

        assert "data quality errors" in result.output
        assert "data you have not vouched for" in result.output

    def test_missing_symbol_gives_a_useful_error(self, cli):
        result = cli.invoke(main, ["backtest", "--symbol", "NOPE"])
        assert result.exit_code != 0
        assert "Run 'aifin fetch' first" in result.output

    def test_rejects_an_unknown_strategy(self, cli, stored):
        result = cli.invoke(main, ["backtest", "--symbol", "SYNTH", "--strategy", "magic"])
        assert result.exit_code != 0
        assert "magic" in result.output


class TestWalkForwardCommand:
    @pytest.fixture
    def stored_4h(self, cli, tmp_path):
        """Enough 1-minute history that 4h bars cover a useful span."""
        cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "SYNTH",
                "--start",
                "2024-01-01",
                "--end",
                "2024-08-31 23:59",
            ],
        )
        return tmp_path

    def test_runs_and_prints_a_verdict(self, cli, stored_4h):
        result = cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--strategy",
                "ma-crossover",
                "--train-bars",
                "300",
                "--test-bars",
                "100",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "OUT OF SAMPLE" in result.output
        assert "VERDICT" in result.output
        assert "Noise floor" in result.output

    def test_reports_param_churn(self, cli, stored_4h):
        result = cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--strategy",
                "ma-crossover",
                "--train-bars",
                "300",
                "--test-bars",
                "100",
            ],
        )
        assert "param churn" in result.output
        assert "churn" in result.output

    def test_all_runs_every_baseline(self, cli, stored_4h):
        result = cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--train-bars",
                "300",
                "--test-bars",
                "150",
            ],
        )

        assert result.exit_code == 0, result.output
        for name in ("ma-crossover", "rsi-mean-reversion", "breakout", "vol-scaled-trend"):
            assert name in result.output

    def test_writes_to_the_registry(self, cli, stored_4h, tmp_path):
        cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--strategy",
                "breakout",
                "--train-bars",
                "300",
                "--test-bars",
                "150",
            ],
        )
        assert (tmp_path / "experiments.jsonl").exists()

    def test_no_registry_flag_writes_nothing(self, cli, stored_4h, tmp_path):
        cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--strategy",
                "breakout",
                "--train-bars",
                "300",
                "--test-bars",
                "150",
                "--no-registry",
            ],
        )
        assert not (tmp_path / "experiments.jsonl").exists()

    def test_unknown_strategy_is_rejected(self, cli, stored_4h):
        result = cli.invoke(main, ["walkforward", "--symbol", "SYNTH", "--strategy", "magic"])
        assert result.exit_code != 0
        assert "unknown strategy" in result.output

    def test_missing_data_gives_a_useful_error(self, cli):
        result = cli.invoke(main, ["walkforward", "--symbol", "NOPE"])
        assert result.exit_code != 0
        assert "Run 'aifin fetch' first" in result.output

    def test_too_little_data_is_explained(self, cli, stored_4h):
        result = cli.invoke(
            main,
            [
                "walkforward",
                "--symbol",
                "SYNTH",
                "--interval",
                "1d",
                "--train-bars",
                "5000",
                "--test-bars",
                "1000",
            ],
        )
        assert result.exit_code != 0
        assert "too short" in result.output


class TestVerdictJudgement:
    """The gate must not celebrate a losing strategy."""

    def test_a_negative_sharpe_never_passes(self):
        from ai_finance.cli import _judge

        class Losing:
            sharpe = -0.5
            benchmark_sharpe = -0.9

        verdict, reason = _judge(Losing(), floor=0.0)
        assert verdict == "fail"
        assert "negative Sharpe" in reason

    def test_beating_a_worse_benchmark_is_not_enough(self):
        from ai_finance.cli import _judge

        class Mediocre:
            sharpe = 0.3
            benchmark_sharpe = 0.1

        verdict, reason = _judge(Mediocre(), floor=1.5)
        assert verdict == "fail"
        assert "noise floor" in reason

    def test_losing_to_the_benchmark_fails(self):
        from ai_finance.cli import _judge

        class Trailing:
            sharpe = 0.5
            benchmark_sharpe = 1.2

        verdict, reason = _judge(Trailing(), floor=0.0)
        assert verdict == "fail"
        assert "did not beat buy-and-hold" in reason

    def test_undefined_sharpe_fails(self):
        from ai_finance.cli import _judge

        class Undefined:
            sharpe = float("nan")
            benchmark_sharpe = 0.5

        verdict, reason = _judge(Undefined(), floor=0.0)
        assert verdict == "fail"
        assert "no Sharpe" in reason

    def test_clearing_all_three_hurdles_passes(self):
        from ai_finance.cli import _judge

        class Good:
            sharpe = 2.1
            benchmark_sharpe = 0.8

        assert _judge(Good(), floor=1.5) == ("PASS", "")


class TestTrainCommand:
    @pytest.fixture
    def stored_year(self, cli, tmp_path):
        cli.invoke(
            main,
            [
                "fetch",
                "--source",
                "synthetic",
                "--symbol",
                "SYNTH",
                "--start",
                "2023-01-01",
                "--end",
                "2023-12-31 23:59",
            ],
        )
        return tmp_path

    def test_runs_and_prints_signal_quality_and_a_verdict(self, cli, stored_year):
        result = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--model",
                "ridge",
                "--train-size",
                "600",
                "--test-size",
                "300",
                "--horizon",
                "6",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "SIGNAL QUALITY" in result.output
        assert "accuracy" in result.output
        assert "info coef" in result.output
        assert "VERDICT" in result.output

    def test_reports_the_cost_threshold(self, cli, stored_year):
        result = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--train-size",
                "600",
                "--test-size",
                "300",
            ],
        )
        assert "predicted move to act" in result.output
        assert "round trip" in result.output

    def test_safety_factor_changes_the_threshold(self, cli, stored_year):
        cheap = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--train-size",
                "600",
                "--test-size",
                "300",
                "--safety-factor",
                "1.0",
            ],
        )
        strict = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--train-size",
                "600",
                "--test-size",
                "300",
                "--safety-factor",
                "3.0",
            ],
        )

        assert "0.220%" in cheap.output
        assert "0.660%" in strict.output

    def test_gbm_runs_too(self, cli, stored_year):
        result = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--model",
                "gbm",
                "--train-size",
                "600",
                "--test-size",
                "300",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "gbm" in result.output

    def test_a_leak_aborts_with_a_clear_message(self, cli, stored_year, monkeypatch):
        from ai_finance.features.pipeline import PointInTimeError
        from ai_finance.research import ml_walkforward

        def leaky(*args, **kwargs):
            raise PointInTimeError("feature 'tomorrow' saw the future")

        monkeypatch.setattr(ml_walkforward, "assert_point_in_time", leaky)

        result = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "4h",
                "--train-size",
                "600",
                "--test-size",
                "300",
            ],
        )

        assert result.exit_code != 0
        assert "LEAK DETECTED" in result.output
        assert "results discarded" in result.output

    def test_unknown_model_rejected(self, cli, stored_year):
        result = cli.invoke(main, ["train", "--symbol", "SYNTH", "--model", "transformer"])
        assert result.exit_code != 0
        assert "unknown model" in result.output

    def test_missing_data_gives_a_useful_error(self, cli):
        result = cli.invoke(main, ["train", "--symbol", "NOPE"])
        assert result.exit_code != 0
        assert "Run 'aifin fetch' first" in result.output

    def test_too_little_data_is_explained(self, cli, stored_year):
        result = cli.invoke(
            main,
            [
                "train",
                "--symbol",
                "SYNTH",
                "--interval",
                "1d",
                "--train-size",
                "5000",
                "--test-size",
                "1000",
            ],
        )
        assert result.exit_code != 0
        assert "too short" in result.output or "no usable rows" in result.output


class TestRunCommands:
    @pytest.fixture
    def paper(self, cli):
        """Options that drive the runner from a synthetic live feed."""
        return [
            "run",
            "--symbol",
            "PAPER",
            "--interval",
            "1h",
            "--strategy",
            "ma-crossover",
            "--source",
            "synthetic",
            "--equity",
            "5000",
            "--max-position",
            "1.0",
        ]

    def test_live_mode_is_refused_at_the_cli(self, cli, paper):
        result = cli.invoke(main, [*paper, "--mode", "live"])

        assert result.exit_code != 0
        assert "live trading is not implemented" in result.output
        assert "--mode paper" in result.output

    def test_a_run_reports_what_it_did(self, cli, paper):
        result = cli.invoke(main, paper)

        assert result.exit_code == 0, result.output
        assert "[paper]" in result.output
        assert "equity" in result.output

    def test_status_before_any_run(self, cli):
        result = cli.invoke(main, ["status", "--symbol", "PAPER"])
        assert "Run 'aifin run' first" in result.output

    def test_status_after_a_run(self, cli, paper):
        cli.invoke(main, paper)
        result = cli.invoke(main, ["status", "--symbol", "PAPER"])

        assert result.exit_code == 0, result.output
        assert "ma-crossover" in result.output
        assert "cash" in result.output

    def test_health_is_stale_before_any_run(self, cli):
        result = cli.invoke(main, ["health", "--symbol", "PAPER", "--interval", "1h"])
        assert result.exit_code == 1, "a monitoring cron needs a non-zero exit"
        assert "NEVER RUN" in result.output

    def test_health_is_ok_after_a_run(self, cli, paper):
        cli.invoke(main, paper)
        result = cli.invoke(main, ["health", "--symbol", "PAPER", "--interval", "1h"])

        assert result.exit_code == 0
        assert "OK" in result.output

    def test_halt_and_resume(self, cli, paper):
        cli.invoke(main, paper)

        halted = cli.invoke(main, ["halt", "--reason", "testing"])
        assert "Kill switch engaged" in halted.output
        assert "KILL SWITCH testing" in cli.invoke(main, ["status", "--symbol", "PAPER"]).output

        resumed = cli.invoke(main, ["resume"])
        assert "released" in resumed.output
        assert "KILL SWITCH" not in cli.invoke(main, ["status", "--symbol", "PAPER"]).output

    def test_resume_when_not_halted(self, cli):
        result = cli.invoke(main, ["resume"])
        assert "was not engaged" in result.output

    def test_rerunning_does_not_double_trade(self, cli, paper):
        cli.invoke(main, paper)
        second = cli.invoke(main, paper)
        assert "already-done" in second.output
