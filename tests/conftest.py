"""Shared test helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest

from ai_finance.config import interval_ms
from ai_finance.data.schema import normalize_bars
from ai_finance.data.sources import HttpResponse

MINUTE_MS = 60_000


def kline(open_time_ms: int, price: float, step_ms: int = MINUTE_MS) -> list[Any]:
    """One Binance kline row, in the exact positional layout the API returns."""
    return [
        open_time_ms,
        f"{price:.8f}",
        f"{price * 1.001:.8f}",
        f"{price * 0.999:.8f}",
        f"{price * 1.0005:.8f}",
        "12.34000000",
        open_time_ms + step_ms - 1,
        "370000.00000000",
        42,
        "6.00000000",
        "180000.00000000",
        "0",
    ]


def klines(start_ms: int, count: int, start_price: float = 100.0, step_ms: int = MINUTE_MS):
    """``count`` consecutive klines, price drifting gently upward."""
    return [kline(start_ms + i * step_ms, start_price + i, step_ms) for i in range(count)]


def bars_frame(start: str, count: int, interval: str = "1m", start_price: float = 100.0):
    """A canonical bars DataFrame, for tests that don't need a source."""
    step = interval_ms(interval)
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    return normalize_bars(
        pd.DataFrame(
            {
                "open_time": [start_ms + i * step for i in range(count)],
                "open": [start_price + i for i in range(count)],
                "high": [start_price + i + 1.0 for i in range(count)],
                "low": [start_price + i - 1.0 for i in range(count)],
                "close": [start_price + i + 0.5 for i in range(count)],
                "volume": [10.0 + i for i in range(count)],
                "trades": [5 + i for i in range(count)],
                "close_time": [start_ms + (i + 1) * step - 1 for i in range(count)],
            }
        )
    )


class FakeTransport:
    """Returns queued responses in order and records every request.

    Exhausting the queue is an error rather than a silent empty response: a test
    that makes more requests than it expected should fail loudly.
    """

    def __init__(self, responses: Sequence[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, params: Mapping[str, Any]) -> HttpResponse:
        self.calls.append((url, dict(params)))
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {url} {dict(params)}")
        return self._responses.pop(0)

    @property
    def exhausted(self) -> bool:
        return not self._responses


def ok(body: Any, **headers: str) -> HttpResponse:
    return HttpResponse(status=200, body=body, headers=headers)


@pytest.fixture
def store_root(tmp_path):
    """An isolated bar store root."""
    root = tmp_path / "bars"
    root.mkdir()
    return root


@pytest.fixture
def no_sleep():
    """A sleep that records durations instead of taking time."""
    slept: list[float] = []
    return slept.append, slept
