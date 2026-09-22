from __future__ import annotations

import pytest
from click.testing import CliRunner

from ai_finance.cli import main

#: A short period with small windows, so the demo runs in seconds rather than a
#: minute. The stages exercised are identical; only the amount of data differs.
FAST = [
    "--quick",
    "--start",
    "2023-01-01",
    "--end",
    "2023-04-30 23:59",
    "--train-bars",
    "300",
    "--test-bars",
    "120",
]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path))
    return CliRunner()


class TestDemoCommand:
    def test_runs_every_stage(self, cli):
        result = cli.invoke(main, ["demo", *FAST])

        assert result.exit_code == 0, result.output
        for stage in (
            "Fetch and store bars",
            "Check the data before trusting it",
            "What trading frequency costs",
            "Walk the classic baselines forward",
            "Fit a model",
            "What this did and did not show",
        ):
            assert stage in result.output

    def test_says_plainly_what_it_does_not_show(self, cli):
        """The demo must not be mistakable for evidence about real markets."""
        result = cli.invoke(main, ["demo", *FAST])

        assert "random walk" in result.output
        assert "NOT shown" in result.output
        assert "aifin fetch --symbol BTCUSDT" in result.output

    def test_shows_the_cost_of_frequency(self, cli):
        result = cli.invoke(main, ["demo", *FAST])
        assert "cost drag" in result.output
        assert "1m" in result.output and "1d" in result.output

    def test_reports_the_noise_floor(self, cli):
        result = cli.invoke(main, ["demo", *FAST])
        assert "by luck alone" in result.output

    def test_writes_a_report_when_asked(self, cli, tmp_path):
        target = tmp_path / "demo.html"
        result = cli.invoke(main, ["demo", *FAST, "--report", str(target)])

        assert result.exit_code == 0, result.output
        assert target.exists()
        assert "random walk" in target.read_text(encoding="utf-8")

    def test_is_deterministic(self, tmp_path, monkeypatch):
        """Same seed, same numbers — in separate stores.

        Re-running against the *same* store correctly reports "already up to
        date" instead of re-fetching, so the two runs need their own directories
        for the fetch line to match too.
        """
        outputs = []
        for name in ("run-a", "run-b"):
            monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path / name))
            result = CliRunner().invoke(main, ["demo", *FAST, "--seed", "3"])
            assert result.exit_code == 0, result.output
            # Drop the timing line, which is wall-clock and will never match.
            outputs.append(
                [line for line in result.output.splitlines() if "Demo finished" not in line]
            )

        assert outputs[0] == outputs[1]

    def test_rerunning_against_the_same_store_refetches_nothing(self, cli):
        cli.invoke(main, ["demo", *FAST])
        second = cli.invoke(main, ["demo", *FAST])
        assert "already up to date" in second.output
