from __future__ import annotations

import pandas as pd
import pytest

import ai_finance.data.fetch as fetch_module
from ai_finance.data.fetch import fetch_history
from ai_finance.data.quality import check_bars
from ai_finance.data.schema import empty_bars, normalize_bars
from ai_finance.data.sources import BinanceSource, RateLimiter, SyntheticSource
from ai_finance.data.store import load_bars, write_bars
from tests.conftest import FakeTransport, bars_frame, klines, ok

MINUTE = 60_000
START = "2024-01-01"
START_MS = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)


class ReplaySource:
    """Serves a fixed frame in pages, recording the cursors it was asked for."""

    name = "replay"

    def __init__(self, bars: pd.DataFrame) -> None:
        self._bars = bars
        self.cursors: list[int] = []

    def fetch_chunk(self, symbol, interval, start_ms, end_ms=None, limit=1000):
        self.cursors.append(start_ms)
        start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        window = self._bars.loc[self._bars["open_time"] >= start]
        if end_ms is not None:
            window = window.loc[window["open_time"] <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
        return normalize_bars(window.head(limit))


class StuckSource:
    """Always returns the same single bar. A buggy source must not hang the job."""

    name = "stuck"

    def __init__(self) -> None:
        self.calls = 0

    def fetch_chunk(self, symbol, interval, start_ms, end_ms=None, limit=1000):
        self.calls += 1
        return bars_frame(START, 1)


class TestPagination:
    def test_walks_the_cursor_across_pages(self, store_root):
        source = ReplaySource(bars_frame(START, 250))

        result = fetch_history(source, "BTCUSDT", START, limit=100, root=store_root)

        assert result.requests == 4  # 100, 100, 50, then empty
        assert result.bars_written == 250
        assert len(load_bars("BTCUSDT", root=store_root)) == 250

    def test_cursor_advances_by_exactly_one_bar(self, store_root):
        source = ReplaySource(bars_frame(START, 30))
        fetch_history(source, "BTCUSDT", START, limit=10, root=store_root)

        assert source.cursors == [
            START_MS,
            START_MS + 10 * MINUTE,
            START_MS + 20 * MINUTE,
            START_MS + 30 * MINUTE,
        ]

    def test_stops_on_empty_response(self, store_root):
        result = fetch_history(ReplaySource(empty_bars()), "BTCUSDT", START, root=store_root)

        assert result.requests == 1
        assert result.bars_written == 0
        assert "already up to date" in result.summary()

    def test_respects_end(self, store_root):
        source = ReplaySource(bars_frame(START, 500))

        fetch_history(source, "BTCUSDT", START, end="2024-01-01 00:59", limit=100, root=store_root)

        stored = load_bars("BTCUSDT", root=store_root)
        assert len(stored) == 60
        assert stored["open_time"].iloc[-1] == pd.Timestamp("2024-01-01 00:59", tz="UTC")

    def test_result_reports_the_written_range(self, store_root):
        result = fetch_history(
            ReplaySource(bars_frame(START, 120)), "BTCUSDT", START, limit=50, root=store_root
        )

        assert result.first == pd.Timestamp("2024-01-01 00:00", tz="UTC")
        assert result.last == pd.Timestamp("2024-01-01 01:59", tz="UTC")
        assert "wrote 120 bars" in result.summary()

    def test_progress_callback_sees_every_page(self, store_root):
        seen: list[tuple[int, int]] = []
        fetch_history(
            ReplaySource(bars_frame(START, 25)),
            "BTCUSDT",
            START,
            limit=10,
            root=store_root,
            on_progress=lambda n, cursor, count: seen.append((n, count)),
        )
        assert seen == [(1, 10), (2, 10), (3, 5)]


class TestResume:
    def test_resumes_after_the_newest_stored_bar(self, store_root):
        write_bars(bars_frame(START, 100), "BTCUSDT", root=store_root)
        source = ReplaySource(bars_frame(START, 150))

        result = fetch_history(source, "BTCUSDT", START, limit=100, root=store_root)

        assert result.resumed_from == pd.Timestamp("2024-01-01 01:39", tz="UTC")
        assert source.cursors[0] == START_MS + 100 * MINUTE
        assert result.bars_written == 50
        assert len(load_bars("BTCUSDT", root=store_root)) == 150

    def test_second_run_is_a_no_op(self, store_root):
        source = ReplaySource(bars_frame(START, 60))
        fetch_history(source, "BTCUSDT", START, root=store_root)

        again = fetch_history(source, "BTCUSDT", START, root=store_root)

        assert again.bars_written == 0
        assert len(load_bars("BTCUSDT", root=store_root)) == 60

    def test_no_resume_refetches_from_the_start(self, store_root):
        write_bars(bars_frame(START, 100), "BTCUSDT", root=store_root)
        source = ReplaySource(bars_frame(START, 100))

        result = fetch_history(source, "BTCUSDT", START, resume=False, root=store_root)

        assert result.resumed_from is None
        assert source.cursors[0] == START_MS
        assert result.bars_written == 100
        assert len(load_bars("BTCUSDT", root=store_root)) == 100

    def test_resume_ignores_a_store_that_predates_the_request(self, store_root):
        write_bars(bars_frame("2023-06-01", 10), "BTCUSDT", root=store_root)
        source = ReplaySource(bars_frame(START, 10))

        fetch_history(source, "BTCUSDT", START, root=store_root)

        assert source.cursors[0] == START_MS

    def test_interrupted_run_recovers_on_the_next_attempt(self, store_root):
        """The property a cron job depends on."""
        full = bars_frame(START, 300)
        partial = fetch_history(
            ReplaySource(full), "BTCUSDT", START, limit=50, max_requests=3, root=store_root
        )
        assert partial.bars_written == 150

        finished = fetch_history(ReplaySource(full), "BTCUSDT", START, limit=50, root=store_root)

        assert finished.bars_written == 150
        stored = load_bars("BTCUSDT", root=store_root)
        assert len(stored) == 300
        assert check_bars(stored, "BTCUSDT", "1m").is_clean()


class TestSafety:
    def test_a_source_that_will_not_advance_does_not_hang(self, store_root):
        source = StuckSource()

        result = fetch_history(source, "BTCUSDT", START, root=store_root)

        # Detected on the second call, when the cursor fails to move past the
        # bar already seen. Terminates instead of spinning to max_requests.
        assert source.calls == 2
        assert result.bars_written == 2
        # Idempotent writes mean the repeated bar is stored once.
        assert len(load_bars("BTCUSDT", root=store_root)) == 1

    def test_max_requests_is_honoured(self, store_root):
        source = ReplaySource(bars_frame(START, 10_000))

        result = fetch_history(source, "BTCUSDT", START, limit=100, max_requests=5, root=store_root)

        assert result.requests == 5
        assert result.bars_written == 500

    def test_unknown_interval_rejected_before_any_request(self, store_root):
        source = ReplaySource(bars_frame(START, 10))
        with pytest.raises(ValueError, match="unknown interval"):
            fetch_history(source, "BTCUSDT", START, interval="7m", root=store_root)
        assert source.cursors == []


class TestEndToEnd:
    def test_binance_source_through_to_a_clean_store(self, store_root):
        """Two pages of API-shaped rows, then empty. The real pipeline, faked HTTP."""
        transport = FakeTransport(
            [
                ok(klines(START_MS, 100)),
                ok(klines(START_MS + 100 * MINUTE, 60)),
                ok([]),
            ]
        )
        slept: list[float] = []
        source = BinanceSource(
            transport,
            rate_limiter=RateLimiter(60_000, clock=lambda: 0.0, sleep=slept.append),
            now_ms=lambda: START_MS + 10_000 * MINUTE,
            sleep=slept.append,
        )

        result = fetch_history(source, "BTCUSDT", START, limit=100, root=store_root)

        assert result.requests == 3
        assert result.bars_written == 160
        assert transport.exhausted

        stored = load_bars("BTCUSDT", root=store_root)
        assert len(stored) == 160
        assert check_bars(stored, "BTCUSDT", "1m").is_clean()

    def test_synthetic_source_fills_a_day_and_resamples_to_4h(self, store_root):
        source = SyntheticSource(epoch_ms=START_MS, seed=1)

        fetch_history(source, "SYNTH", START, end="2024-01-01 23:59", limit=500, root=store_root)

        minutes = load_bars("SYNTH", root=store_root)
        assert len(minutes) == 1440
        assert check_bars(minutes, "SYNTH", "1m").is_clean()

        four_hourly = load_bars("SYNTH", interval="4h", root=store_root)
        assert len(four_hourly) == 6
        assert four_hourly["volume"].sum() == pytest.approx(minutes["volume"].sum())
        assert four_hourly["high"].max() == pytest.approx(minutes["high"].max())


class ExplodingSource:
    """Serves a few pages, then raises. Models a dropped connection mid-backfill."""

    name = "exploding"

    def __init__(self, bars: pd.DataFrame, fail_after: int) -> None:
        self._bars = bars
        self._fail_after = fail_after
        self.calls = 0

    def fetch_chunk(self, symbol, interval, start_ms, end_ms=None, limit=1000):
        self.calls += 1
        if self.calls > self._fail_after:
            raise ConnectionError("network went away")
        start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        return normalize_bars(self._bars.loc[self._bars["open_time"] >= start].head(limit))


class TestBuffering:
    def test_batches_writes_instead_of_one_per_chunk(self, store_root, monkeypatch):
        writes: list[int] = []
        real_write = fetch_module.write_bars

        def counting_write(bars, *args, **kwargs):
            writes.append(len(bars))
            return real_write(bars, *args, **kwargs)

        monkeypatch.setattr(fetch_module, "write_bars", counting_write)

        fetch_history(
            ReplaySource(bars_frame(START, 1000)),
            "BTCUSDT",
            START,
            limit=100,
            flush_bars=500,
            root=store_root,
        )

        # Two full buffers plus a final partial flush, not ten chunk writes.
        assert writes == [500, 500]
        assert len(load_bars("BTCUSDT", root=store_root)) == 1000

    def test_final_partial_buffer_is_flushed(self, store_root):
        fetch_history(
            ReplaySource(bars_frame(START, 130)),
            "BTCUSDT",
            START,
            limit=50,
            flush_bars=1_000_000,
            root=store_root,
        )
        assert len(load_bars("BTCUSDT", root=store_root)) == 130

    def test_buffered_bars_survive_an_exception(self, store_root):
        """An interrupted backfill must keep what it already downloaded."""
        source = ExplodingSource(bars_frame(START, 500), fail_after=3)

        with pytest.raises(ConnectionError):
            fetch_history(source, "BTCUSDT", START, limit=50, flush_bars=1_000_000, root=store_root)

        stored = load_bars("BTCUSDT", root=store_root)
        assert len(stored) == 150
        assert check_bars(stored, "BTCUSDT", "1m").is_clean()

    def test_next_run_resumes_after_a_crash(self, store_root):
        full = bars_frame(START, 400)
        with pytest.raises(ConnectionError):
            fetch_history(
                ExplodingSource(full, fail_after=2),
                "BTCUSDT",
                START,
                limit=50,
                root=store_root,
            )

        result = fetch_history(ReplaySource(full), "BTCUSDT", START, limit=50, root=store_root)

        assert result.resumed_from == pd.Timestamp("2024-01-01 01:39", tz="UTC")
        assert len(load_bars("BTCUSDT", root=store_root)) == 400

    def test_rejects_nonsense_flush_size(self, store_root):
        with pytest.raises(ValueError, match="flush_bars must be positive"):
            fetch_history(
                ReplaySource(bars_frame(START, 10)),
                "BTCUSDT",
                START,
                flush_bars=0,
                root=store_root,
            )
