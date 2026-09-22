from __future__ import annotations

import pandas as pd

from ai_finance.ops.monitor import check_health, heartbeat_path, write_heartbeat

NOW = pd.Timestamp("2024-06-01 12:00", tz="UTC")
FOUR_HOURS = 4 * 3600.0


class TestHeartbeat:
    def test_written_and_read_back(self, tmp_path):
        write_heartbeat("BTCUSDT", "paper", action="traded", root=tmp_path, now=NOW)

        health = check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=NOW)

        assert health.status == "OK"
        assert health.last_run == NOW
        assert health.action == "traded"

    def test_path_separates_symbol_and_mode(self, tmp_path):
        a = heartbeat_path("BTCUSDT", "paper", tmp_path)
        b = heartbeat_path("ETHUSDT", "paper", tmp_path)
        assert a != b


class TestStaleness:
    def test_never_run(self, tmp_path):
        health = check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=NOW)

        assert health.status == "NEVER RUN"
        assert health.is_stale
        assert "has the scheduled job ever fired" in health.to_text()

    def test_a_recent_run_is_healthy(self, tmp_path):
        write_heartbeat("BTCUSDT", "paper", action="held", root=tmp_path, now=NOW)
        later = NOW + pd.Timedelta(hours=3)

        assert not check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=later).is_stale

    def test_one_skipped_run_is_tolerated(self, tmp_path):
        """Two cadence periods, so a single miss does not cry wolf."""
        write_heartbeat("BTCUSDT", "paper", action="held", root=tmp_path, now=NOW)
        later = NOW + pd.Timedelta(hours=7)

        assert not check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=later).is_stale

    def test_a_silently_dead_job_is_caught(self, tmp_path):
        """The dangerous failure: no crash, no log, just nothing happening."""
        write_heartbeat("BTCUSDT", "paper", action="held", root=tmp_path, now=NOW)
        later = NOW + pd.Timedelta(hours=12)

        health = check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=later)

        assert health.is_stale
        assert health.status == "STALE"
        assert "The job is not running" in health.to_text()

    def test_an_unreadable_heartbeat_counts_as_stale(self, tmp_path):
        path = heartbeat_path("BTCUSDT", "paper", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")

        health = check_health("BTCUSDT", "paper", FOUR_HOURS, root=tmp_path, now=NOW)

        assert health.is_stale
        assert health.note == "heartbeat unreadable"

    def test_a_faster_cadence_has_a_tighter_window(self, tmp_path):
        write_heartbeat("BTCUSDT", "paper", action="held", root=tmp_path, now=NOW)
        later = NOW + pd.Timedelta(hours=3)

        hourly = check_health("BTCUSDT", "paper", 3600.0, root=tmp_path, now=later)
        daily = check_health("BTCUSDT", "paper", 86400.0, root=tmp_path, now=later)

        assert hourly.is_stale
        assert not daily.is_stale
