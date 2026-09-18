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
