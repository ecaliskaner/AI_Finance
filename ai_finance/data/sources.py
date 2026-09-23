"""Bar sources: where OHLCV data comes from.

Two implementations, both satisfying :class:`BarSource`:

- :class:`BinanceSource` — the real thing, over HTTPS.
- :class:`SyntheticSource` — a deterministic geometric-Brownian-motion generator.

The synthetic source exists so the whole pipeline (store, quality report, CLI,
and later the backtest engine) can be exercised and unit-tested without network
access, and so tests can construct data with *known* defects — a gap here, a bad
print there — and assert that the quality layer finds them.

HTTP goes through an injectable :data:`Transport`, so no test ever touches the
network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ai_finance.config import interval_ms
from ai_finance.data.schema import empty_bars, normalize_bars

log = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"

#: Binance's hard cap on klines per request.
MAX_LIMIT = 1000

MS_PER_YEAR = 365 * 24 * 60 * 60 * 1000

#: Bars generated per block by :class:`SyntheticSource`.
#:
#: Fixed, and that is the point. An adaptive block size makes the random draws
#: landing on a given bar depend on how far the caller happened to ask for, so
#: two processes that paginate differently generate different prices for the
#: same timestamp. With a fixed size, block N always consumes the same slice of
#: the random stream, and a bar's value depends only on its index.
SYNTHETIC_BLOCK_BARS = 1 << 16


class BarSource(Protocol):
    """Anything that can produce a chunk of bars."""

    name: str

    def fetch_chunk(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
        limit: int = MAX_LIMIT,
    ) -> pd.DataFrame:
        """Return up to ``limit`` bars with ``open_time >= start_ms``, ascending.

        An empty frame means "no more data", which is how pagination terminates.
        """
        ...


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HttpResponse:
    """Just enough of an HTTP response for our purposes."""

    status: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)


#: ``(url, params) -> HttpResponse``. Injected so tests can fake it.
Transport = Callable[[str, Mapping[str, Any]], HttpResponse]


def requests_transport(timeout: float = 20.0) -> Transport:
    """A :data:`Transport` backed by a pooled ``requests`` session."""
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = "ai-finance/0.1 (research)"

    def _transport(url: str, params: Mapping[str, Any]) -> HttpResponse:
        response = session.get(url, params=dict(params), timeout=timeout)
        try:
            body = response.json()
        except ValueError:
            body = None
        return HttpResponse(status=response.status_code, body=body, headers=response.headers)

    return _transport


class BinanceError(RuntimeError):
    """A Binance API error we should not retry (bad symbol, bad interval, ...)."""


class RateLimiter:
    """Minimum-interval limiter.

    Binance's real budget is weight-based and generous; this stays well inside
    it rather than trying to model it exactly. Being throttled costs minutes;
    being IP-banned costs hours.
    """

    def __init__(
        self,
        requests_per_minute: int = 600,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._min_gap = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def acquire(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self._min_gap - (now - self._last)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last = now


# --------------------------------------------------------------------------- #
# Binance
# --------------------------------------------------------------------------- #

# Index of each field in a Binance kline row, which is a positional array.
_K_OPEN_TIME = 0
_K_OPEN = 1
_K_HIGH = 2
_K_LOW = 3
_K_CLOSE = 4
_K_VOLUME = 5
_K_CLOSE_TIME = 6
_K_TRADES = 8


class BinanceSource:
    """Fetches spot klines from Binance.

    Args:
        transport: HTTP transport. Defaults to a real ``requests`` session.
        base_url: API root. Override to point at a testnet or a mirror.
        rate_limiter: request pacing. Defaults to a conservative limiter.
        max_retries: attempts per request before giving up on a retryable error.
        drop_unclosed: drop the final, still-forming bar. **Keep this on.** The
            API happily returns the current partial bar, whose close price will
            change; storing it is a direct route to a backtest that trades on
            information it could not have had.
        now_ms: clock used to decide which bars have closed. Injectable for tests.
    """

    name = "binance"

    def __init__(
        self,
        transport: Transport | None = None,
        *,
        base_url: str = BINANCE_BASE_URL,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        drop_unclosed: bool = True,
        now_ms: Callable[[], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transport = transport if transport is not None else requests_transport()
        self._base_url = base_url.rstrip("/")
        self._limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self._max_retries = max_retries
        self._drop_unclosed = drop_unclosed
        self._now_ms = now_ms if now_ms is not None else (lambda: int(time.time() * 1000))
        self._sleep = sleep

    def fetch_chunk(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
        limit: int = MAX_LIMIT,
    ) -> pd.DataFrame:
        if limit < 1 or limit > MAX_LIMIT:
            raise ValueError(f"limit must be in 1..{MAX_LIMIT}, got {limit}")
        interval_ms(interval)  # validate early, before spending a request

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": int(start_ms),
            "limit": int(limit),
        }
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        rows = self._get(f"{self._base_url}/api/v3/klines", params)
        if not rows:
            return empty_bars()

        frame = pd.DataFrame(
            {
                "open_time": [r[_K_OPEN_TIME] for r in rows],
                "open": [r[_K_OPEN] for r in rows],
                "high": [r[_K_HIGH] for r in rows],
                "low": [r[_K_LOW] for r in rows],
                "close": [r[_K_CLOSE] for r in rows],
                "volume": [r[_K_VOLUME] for r in rows],
                "trades": [r[_K_TRADES] for r in rows],
                "close_time": [r[_K_CLOSE_TIME] for r in rows],
            }
        )
        bars = normalize_bars(frame)

        if self._drop_unclosed and not bars.empty:
            cutoff = pd.Timestamp(self._now_ms(), unit="ms", tz="UTC")
            bars = bars.loc[bars["close_time"] < cutoff].reset_index(drop=True)

        return bars

    def _get(self, url: str, params: Mapping[str, Any]) -> list[list[Any]]:
        """GET with pacing, retries and honest error classification."""
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            self._limiter.acquire()
            response = self._transport(url, params)

            if response.status == 200:
                body = response.body
                # A 200 can still carry an error payload, as a dict not a list.
                if isinstance(body, dict):
                    raise BinanceError(
                        f"binance error {body.get('code')}: {body.get('msg')} (params={dict(params)})"
                    )
                if body is None:
                    last_error = BinanceError("empty body with status 200")
                    self._backoff(attempt)
                    continue
                return list(body)

            if response.status in (429, 418):
                # 429 = rate limited, 418 = IP banned for ignoring 429s.
                wait = self._retry_after(response, attempt)
                log.warning(
                    "binance throttled (status=%s), sleeping %.1fs (attempt %d/%d)",
                    response.status,
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                last_error = BinanceError(f"throttled with status {response.status}")
                self._sleep(wait)
                continue

            if response.status >= 500:
                last_error = BinanceError(f"server error {response.status}")
                self._backoff(attempt)
                continue

            # Any other 4xx is our fault and will not fix itself.
            raise BinanceError(
                f"binance returned {response.status} for params={dict(params)}: {response.body!r}"
            )

        raise BinanceError(f"giving up after {self._max_retries} attempts") from last_error

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(2.0**attempt, 30.0))

    @staticmethod
    def _retry_after(response: HttpResponse, attempt: int) -> float:
        header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if header:
            try:
                return max(float(header), 1.0)
            except ValueError:
                pass
        # 418 means we have already been rude; back off much harder.
        base = 60.0 if response.status == 418 else 2.0
        return min(base * (2.0**attempt), 300.0)


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #


class SyntheticSource:
    """Deterministic geometric Brownian motion, shaped like exchange data.

    Same interface as :class:`BinanceSource`, so every downstream component can
    be tested offline. The path is reproducible for a given ``seed``: fetching
    the same range twice returns identical bars.

    Args:
        start_price: price at ``epoch_ms``.
        annual_vol: annualised volatility. 0.5 is roughly BTC-like.
        annual_drift: annualised drift. Default 0 — an honest fixture should not
            hand a strategy free money.
        epoch_ms: index origin for the path. Requests before this are rejected.
        gap_probability: per-bar chance of dropping a bar, simulating an exchange
            outage. Used by tests to prove the quality report catches gaps.
        now_ms: when set, bars are never generated past this moment, so the
            source behaves like a live feed rather than an oracle with unlimited
            future. Needed to exercise the scheduled runner without an exchange.
        seed: RNG seed.
    """

    name = "synthetic"

    def __init__(
        self,
        *,
        start_price: float = 30_000.0,
        annual_vol: float = 0.5,
        annual_drift: float = 0.0,
        epoch_ms: int = 0,
        gap_probability: float = 0.0,
        now_ms: Callable[[], int] | None = None,
        seed: int = 0,
    ) -> None:
        if start_price <= 0:
            raise ValueError("start_price must be positive")
        if not 0.0 <= gap_probability < 1.0:
            raise ValueError("gap_probability must be in [0, 1)")
        self._start_price = start_price
        self._annual_vol = annual_vol
        self._annual_drift = annual_drift
        self._epoch_ms = epoch_ms
        self._gap_probability = gap_probability
        self._now_ms = now_ms
        self._rng = np.random.default_rng(seed)
        # Column 0: return shock. 1: high wick. 2: low wick. 3: volume. 4: gap roll.
        self._draws = np.empty((0, 5), dtype=np.float64)
        self._log_price = np.empty(0, dtype=np.float64)
        self._interval: str | None = None

    def fetch_chunk(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
        limit: int = MAX_LIMIT,
    ) -> pd.DataFrame:
        step = interval_ms(interval)
        if self._interval is None:
            self._interval = interval
        elif self._interval != interval:
            raise ValueError(
                f"this SyntheticSource generates {self._interval} bars; asked for {interval}. "
                "Use a separate instance per interval."
            )
        if start_ms < self._epoch_ms:
            raise ValueError(f"start_ms {start_ms} precedes epoch_ms {self._epoch_ms}")
        if limit < 1:
            raise ValueError("limit must be positive")

        first = (start_ms - self._epoch_ms + step - 1) // step  # ceil to the grid
        count = limit
        horizon = end_ms
        if self._now_ms is not None:
            # A live feed cannot serve a bar that has not closed yet.
            latest_closed = self._now_ms() - step
            horizon = latest_closed if horizon is None else min(horizon, latest_closed)
        if horizon is not None:
            last_allowed = (horizon - self._epoch_ms) // step
            count = min(count, last_allowed - first + 1)
        if count <= 0:
            return empty_bars()

        self._extend_to(first + count, step)

        idx = np.arange(first, first + count)
        open_time = self._epoch_ms + idx * step
        close = np.exp(self._log_price[idx + 1])
        open_ = np.exp(self._log_price[idx])
        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        # Wicks scale with the bar's own volatility so they stay plausible.
        scale = self._annual_vol * np.sqrt(step / MS_PER_YEAR)
        high = body_hi * (1.0 + np.abs(self._draws[idx, 1]) * scale)
        low = body_lo * (1.0 - np.abs(self._draws[idx, 2]) * scale)
        volume = np.exp(self._draws[idx, 3]) * 10.0
        trades = np.maximum(1, (volume * 5.0).astype(np.int64))

        frame = pd.DataFrame(
            {
                "open_time": open_time,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "trades": trades,
                "close_time": open_time + step - 1,
            }
        )

        if self._gap_probability > 0.0:
            keep = self._draws[idx, 4] >= self._gap_probability
            frame = frame.loc[keep]

        return normalize_bars(frame)

    def _extend_to(self, n_bars: int, step: int) -> None:
        """Ensure the path covers ``n_bars`` bars, generating whole blocks.

        Blocks are a fixed size and are always generated in order, so block N
        consumes the same slice of the random stream no matter how the caller
        paginated to get there. That is what makes a bar's value a function of
        its index alone, and therefore reproducible across processes.
        """
        while self._draws.shape[0] < n_bars:
            self._generate_block(step)

    def _generate_block(self, step: int) -> None:
        size = SYNTHETIC_BLOCK_BARS
        block = np.empty((size, 5), dtype=np.float64)
        block[:, 0] = self._rng.standard_normal(size)
        block[:, 1] = self._rng.standard_normal(size)
        block[:, 2] = self._rng.standard_normal(size)
        block[:, 3] = self._rng.standard_normal(size)
        block[:, 4] = self._rng.random(size)

        dt = step / MS_PER_YEAR
        sigma = self._annual_vol
        drift = (self._annual_drift - 0.5 * sigma**2) * dt
        increments = drift + sigma * np.sqrt(dt) * block[:, 0]

        if self._draws.size == 0:
            anchor = np.log(self._start_price)
            self._log_price = np.concatenate([[anchor], anchor + np.cumsum(increments)])
            self._draws = block
        else:
            self._log_price = np.concatenate(
                [self._log_price, self._log_price[-1] + np.cumsum(increments)]
            )
            self._draws = np.vstack([self._draws, block])
