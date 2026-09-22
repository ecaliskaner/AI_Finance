from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.data.schema import assert_schema
from ai_finance.data.sources import (
    BinanceError,
    BinanceSource,
    HttpResponse,
    RateLimiter,
    SyntheticSource,
)
from tests.conftest import FakeTransport, klines, ok

START_MS = 1_700_000_000_000  # 2023-11-14 22:13:20 UTC, on a minute boundary
MINUTE = 60_000


def _source(transport, *, now_ms=None, **kwargs):
    slept: list[float] = []
    source = BinanceSource(
        transport,
        rate_limiter=RateLimiter(60_000, clock=lambda: 0.0, sleep=slept.append),
        now_ms=(lambda: now_ms) if now_ms is not None else (lambda: START_MS + 10_000 * MINUTE),
        sleep=slept.append,
        **kwargs,
    )
    return source, slept


class TestBinanceSource:
    def test_parses_klines_into_canonical_bars(self):
        transport = FakeTransport([ok(klines(START_MS, 3))])
        source, _ = _source(transport)

        bars = source.fetch_chunk("btcusdt", "1m", START_MS)

        assert_schema(bars)
        assert len(bars) == 3
        assert bars["open_time"].iloc[0] == pd.Timestamp(START_MS, unit="ms", tz="UTC")
        assert bars["open"].iloc[0] == pytest.approx(100.0)
        assert bars["trades"].iloc[0] == 42
        # Symbol is upper-cased for the API.
        assert transport.calls[0][1]["symbol"] == "BTCUSDT"

    def test_drops_the_still_forming_bar(self):
        """The API returns the current partial bar; storing it is look-ahead bias."""
        rows = klines(START_MS, 3)
        # "Now" lands inside the third bar, so only the first two have closed.
        now = START_MS + 2 * MINUTE + 30_000
        source, _ = _source(FakeTransport([ok(rows)]), now_ms=now)

        bars = source.fetch_chunk("BTCUSDT", "1m", START_MS)

        assert len(bars) == 2
        assert bars["close_time"].iloc[-1] < pd.Timestamp(now, unit="ms", tz="UTC")

    def test_keeps_unclosed_bar_when_explicitly_asked(self):
        rows = klines(START_MS, 3)
        now = START_MS + 2 * MINUTE + 30_000
        source, _ = _source(FakeTransport([ok(rows)]), now_ms=now, drop_unclosed=False)

        assert len(source.fetch_chunk("BTCUSDT", "1m", START_MS)) == 3

    def test_empty_response_yields_empty_frame(self):
        source, _ = _source(FakeTransport([ok([])]))
        bars = source.fetch_chunk("BTCUSDT", "1m", START_MS)
        assert bars.empty
        assert_schema(bars)

    def test_error_payload_with_status_200_is_raised(self):
        """Binance reports a bad symbol as a dict body under a 200."""
        body = {"code": -1121, "msg": "Invalid symbol."}
        source, _ = _source(FakeTransport([ok(body)]))

        with pytest.raises(BinanceError, match="Invalid symbol"):
            source.fetch_chunk("NOPE", "1m", START_MS)

    def test_retries_on_429_then_succeeds(self):
        transport = FakeTransport(
            [
                HttpResponse(status=429, body=None, headers={"Retry-After": "3"}),
                ok(klines(START_MS, 2)),
            ]
        )
        source, slept = _source(transport)

        bars = source.fetch_chunk("BTCUSDT", "1m", START_MS)

        assert len(bars) == 2
        assert 3.0 in slept, "Retry-After header should be honoured"

    def test_retries_on_server_error(self):
        transport = FakeTransport([HttpResponse(status=503, body=None), ok(klines(START_MS, 1))])
        source, _ = _source(transport)
        assert len(source.fetch_chunk("BTCUSDT", "1m", START_MS)) == 1

    def test_418_backs_off_much_harder_than_429(self):
        transport = FakeTransport([HttpResponse(status=418, body=None), ok(klines(START_MS, 1))])
        source, slept = _source(transport)
        source.fetch_chunk("BTCUSDT", "1m", START_MS)
        assert max(slept) >= 60.0

    def test_client_error_is_not_retried(self):
        transport = FakeTransport([HttpResponse(status=400, body={"msg": "bad"})])
        source, _ = _source(transport)

        with pytest.raises(BinanceError, match="400"):
            source.fetch_chunk("BTCUSDT", "1m", START_MS)
        assert transport.exhausted, "a 400 must not be retried"

    def test_gives_up_after_max_retries(self):
        transport = FakeTransport([HttpResponse(status=503, body=None)] * 3)
        source, _ = _source(transport, max_retries=3)

        with pytest.raises(BinanceError, match="giving up"):
            source.fetch_chunk("BTCUSDT", "1m", START_MS)

    def test_rejects_unknown_interval_before_spending_a_request(self):
        transport = FakeTransport([])
        source, _ = _source(transport)

        with pytest.raises(ValueError, match="unknown interval"):
            source.fetch_chunk("BTCUSDT", "7m", START_MS)
        assert not transport.calls

    def test_rejects_oversized_limit(self):
        source, _ = _source(FakeTransport([]))
        with pytest.raises(ValueError, match="limit must be"):
            source.fetch_chunk("BTCUSDT", "1m", START_MS, limit=5000)


class TestRateLimiter:
    def test_sleeps_to_maintain_minimum_gap(self):
        slept: list[float] = []
        now = [0.0]
        limiter = RateLimiter(60, clock=lambda: now[0], sleep=slept.append)

        limiter.acquire()  # first call is free
        limiter.acquire()  # needs a full second

        assert slept == [pytest.approx(1.0)]

    def test_no_sleep_when_enough_time_has_passed(self):
        slept: list[float] = []
        now = [0.0]
        limiter = RateLimiter(60, clock=lambda: now[0], sleep=slept.append)
        limiter.acquire()
        now[0] = 5.0
        limiter.acquire()
        assert slept == []

    def test_rejects_nonsense_rate(self):
        with pytest.raises(ValueError):
            RateLimiter(0)


class TestSyntheticSource:
    def test_produces_canonical_bars_on_the_grid(self):
        source = SyntheticSource(epoch_ms=START_MS, seed=1)
        bars = source.fetch_chunk("SYNTH", "1m", START_MS, limit=100)

        assert_schema(bars)
        assert len(bars) == 100
        deltas = bars["open_time"].diff().dropna().unique()
        assert list(deltas) == [pd.Timedelta(minutes=1)]

    def test_ohlc_invariants_hold(self):
        bars = SyntheticSource(epoch_ms=START_MS, seed=2).fetch_chunk(
            "SYNTH", "1m", START_MS, limit=5000
        )
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
        assert (bars[["open", "high", "low", "close"]] > 0).all().all()

    def test_is_deterministic_across_identical_requests(self):
        a = SyntheticSource(epoch_ms=START_MS, seed=7).fetch_chunk("S", "1m", START_MS, limit=500)
        b = SyntheticSource(epoch_ms=START_MS, seed=7).fetch_chunk("S", "1m", START_MS, limit=500)
        pd.testing.assert_frame_equal(a, b)

    def test_path_is_identical_however_it_is_chunked(self):
        """Re-fetching a range must not change prices, or nothing is reproducible."""
        whole = SyntheticSource(epoch_ms=START_MS, seed=3).fetch_chunk(
            "S", "1m", START_MS, limit=300
        )

        paged = SyntheticSource(epoch_ms=START_MS, seed=3)
        first = paged.fetch_chunk("S", "1m", START_MS, limit=100)
        second = paged.fetch_chunk("S", "1m", START_MS + 100 * MINUTE, limit=200)
        # The same instance asked again for page one.
        first_again = paged.fetch_chunk("S", "1m", START_MS, limit=100)

        pd.testing.assert_frame_equal(whole, pd.concat([first, second], ignore_index=True))
        pd.testing.assert_frame_equal(first, first_again)

    def test_a_bar_depends_only_on_its_index_not_on_the_request_size(self):
        """Regression: block-size-dependent draws broke cross-process replay.

        The generator used to grow by doubling and draw each column as one
        contiguous slice, so the random values landing on a given bar depended
        on how far the caller had asked for. Two processes paginating
        differently — exactly what the scheduled runner does across wake-ups —
        produced different prices for the same timestamp, and the stored series
        gained a discontinuity wherever the two met.
        """
        target = START_MS + 5_000 * MINUTE
        frames = [
            SyntheticSource(epoch_ms=START_MS, seed=11).fetch_chunk("S", "1m", target, limit=limit)
            for limit in (1, 7, 100, 1000)
        ]

        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frames[0], frame.head(1))

    def test_resuming_a_fetch_joins_without_a_price_jump(self):
        """The join between two fetch sessions must not be a discontinuity."""
        first = SyntheticSource(epoch_ms=START_MS, seed=5).fetch_chunk(
            "S", "1m", START_MS, limit=2_000
        )
        # A second process resumes where the first stopped.
        resumed = SyntheticSource(epoch_ms=START_MS, seed=5).fetch_chunk(
            "S", "1m", START_MS + 2_000 * MINUTE, limit=50
        )

        joined = pd.concat([first, resumed], ignore_index=True)
        returns = joined["close"].pct_change().dropna().abs()
        # 1-minute moves at 50% annual vol are ~7bp; 2% would be 30 sigma.
        assert returns.max() < 0.02
        assert resumed["open"].iloc[0] == pytest.approx(first["close"].iloc[-1])

    def test_different_seeds_give_different_paths(self):
        a = SyntheticSource(epoch_ms=START_MS, seed=1).fetch_chunk("S", "1m", START_MS, limit=200)
        b = SyntheticSource(epoch_ms=START_MS, seed=2).fetch_chunk("S", "1m", START_MS, limit=200)
        assert not a["close"].equals(b["close"])

    def test_respects_end_ms(self):
        source = SyntheticSource(epoch_ms=START_MS, seed=1)
        bars = source.fetch_chunk("S", "1m", START_MS, END := START_MS + 9 * MINUTE, limit=1000)
        assert len(bars) == 10
        assert bars["open_time"].iloc[-1] == pd.Timestamp(END, unit="ms", tz="UTC")

    def test_gap_probability_drops_bars(self):
        source = SyntheticSource(epoch_ms=START_MS, seed=5, gap_probability=0.2)
        bars = source.fetch_chunk("S", "1m", START_MS, limit=1000)
        assert 0 < len(bars) < 1000

    def test_realised_volatility_is_close_to_requested(self):
        """A fixture that lies about its volatility would invalidate cost analysis."""
        import numpy as np

        source = SyntheticSource(epoch_ms=0, seed=11, annual_vol=0.5)
        bars = source.fetch_chunk("S", "1m", 0, limit=200_000)
        per_minute = np.diff(np.log(bars["close"].to_numpy())).std()
        annualised = per_minute * np.sqrt(365 * 24 * 60)
        assert 0.45 < annualised < 0.55

    def test_rejects_start_before_epoch(self):
        source = SyntheticSource(epoch_ms=START_MS)
        with pytest.raises(ValueError, match="precedes epoch"):
            source.fetch_chunk("S", "1m", START_MS - MINUTE)

    def test_rejects_mixed_intervals_on_one_instance(self):
        source = SyntheticSource(epoch_ms=START_MS)
        source.fetch_chunk("S", "1m", START_MS, limit=10)
        with pytest.raises(ValueError, match="separate instance"):
            source.fetch_chunk("S", "1h", START_MS, limit=10)
