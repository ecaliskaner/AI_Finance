from __future__ import annotations

import json

import pandas as pd
import pytest

from ai_finance.ops.state import (
    STATE_VERSION,
    RunState,
    StateError,
    StateStore,
    default_state_path,
)


def a_state(**overrides):
    base = {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "mode": "paper",
        "strategy": "ma-crossover",
        "initial_equity": 10_000.0,
        "cash": 10_000.0,
    }
    return RunState(**{**base, **overrides})


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.json")


class TestRoundTrip:
    def test_save_and_load(self, store):
        store.save(a_state(units=0.5, cash=1234.5))
        loaded = store.load()

        assert loaded is not None
        assert loaded.units == 0.5
        assert loaded.cash == 1234.5
        assert loaded.version == STATE_VERSION

    def test_missing_file_loads_as_none(self, store):
        assert store.load() is None
        assert not store.exists()

    def test_equity_and_weight(self):
        state = a_state(cash=500.0, units=5.0)
        assert state.equity(100.0) == 1000.0
        assert state.weight(100.0) == pytest.approx(0.5)

    def test_default_path_separates_symbol_and_mode(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path))
        assert default_state_path("btcusdt", "paper") != default_state_path("ETHUSDT", "paper")
        assert "BTCUSDT-paper" in str(default_state_path("btcusdt", "paper"))


class TestCrashSafety:
    def test_write_is_atomic_leaving_no_temp_file(self, store):
        store.save(a_state())
        leftovers = list(store.path.parent.glob("*.tmp"))
        assert leftovers == []

    def test_previous_version_is_kept_as_a_backup(self, store):
        store.save(a_state(units=1.0))
        store.save(a_state(units=2.0))

        assert store.backup_path.exists()
        assert json.loads(store.backup_path.read_text())["units"] == 1.0
        assert store.load().units == 2.0

    def test_a_truncated_file_falls_back_to_the_backup(self, store):
        """A half-written state file must never read back as a position."""
        store.save(a_state(units=1.0))
        store.save(a_state(units=2.0))

        store.path.write_text('{"symbol": "BTC', encoding="utf-8")  # killed mid-write

        recovered = store.load()
        assert recovered is not None
        assert recovered.units == 1.0

    def test_both_files_unreadable_gives_none_not_a_lie(self, store):
        store.save(a_state(units=1.0))
        store.path.write_text("{", encoding="utf-8")
        store.backup_path.write_text("{", encoding="utf-8")
        assert store.load() is None

    def test_a_future_state_version_is_refused(self, store):
        store.save(a_state())
        payload = json.loads(store.path.read_text())
        payload["version"] = STATE_VERSION + 99
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        store.backup_path.unlink(missing_ok=True)

        with pytest.raises(StateError, match="state version"):
            store.load()


class TestInitialise:
    def test_creates_state_on_the_first_run(self, store):
        state = store.initialise(a_state())

        assert store.exists()
        assert state.started_at is not None
        assert state.peak_equity == 10_000.0

    def test_returns_existing_state_untouched(self, store):
        store.initialise(a_state())
        store.save(a_state(units=3.0, cash=1.0, started_at="2024-01-01T00:00:00+00:00"))

        state = store.initialise(a_state())

        assert state.units == 3.0
        assert state.started_at == "2024-01-01T00:00:00+00:00"

    @pytest.mark.parametrize(
        "field,value", [("symbol", "ETHUSDT"), ("interval", "1d"), ("mode", "backtest")]
    )
    def test_refuses_to_overwrite_state_for_a_different_run(self, store, field, value):
        """A mistyped command must not silently discard an open position."""
        store.initialise(a_state())

        with pytest.raises(StateError, match=field):
            store.initialise(a_state(**{field: value}))


class TestIdempotency:
    def test_a_fresh_state_has_acted_on_nothing(self):
        assert not a_state().has_acted_on(pd.Timestamp("2024-01-01", tz="UTC"))

    def test_the_same_bar_counts_as_already_handled(self):
        bar = pd.Timestamp("2024-01-01 03:59:59.999", tz="UTC")
        state = a_state(last_bar_time=bar.isoformat())
        assert state.has_acted_on(bar)

    def test_an_older_bar_counts_as_handled_too(self):
        """Out-of-order data must not re-open a decision already taken."""
        state = a_state(last_bar_time="2024-01-02T00:00:00+00:00")
        assert state.has_acted_on(pd.Timestamp("2024-01-01", tz="UTC"))

    def test_a_newer_bar_is_not_handled(self):
        state = a_state(last_bar_time="2024-01-01T00:00:00+00:00")
        assert not state.has_acted_on(pd.Timestamp("2024-01-02", tz="UTC"))
