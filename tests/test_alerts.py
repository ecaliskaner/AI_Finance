from __future__ import annotations

import pytest

from ai_finance.ops.alerts import (
    NullNotifier,
    TelegramNotifier,
    engage_halt,
    halt_file,
    halt_reason,
    is_halted,
    notifier_from_environment,
    release_halt,
)


class TestKillSwitch:
    def test_absent_by_default(self, tmp_path):
        assert not is_halted(tmp_path)
        assert halt_reason(tmp_path) == ""

    def test_engage_and_release(self, tmp_path):
        engage_halt("testing", tmp_path)

        assert is_halted(tmp_path)
        assert halt_reason(tmp_path) == "testing"
        assert release_halt(tmp_path) is True
        assert not is_halted(tmp_path)

    def test_release_when_not_engaged(self, tmp_path):
        assert release_halt(tmp_path) is False

    def test_engaging_twice_is_harmless(self, tmp_path):
        engage_halt("first", tmp_path)
        engage_halt("second", tmp_path)
        assert halt_reason(tmp_path) == "second"

    def test_needs_no_network_or_credentials(self, tmp_path):
        """The point of a file: it works when everything else is down."""
        engage_halt("exchange outage", tmp_path)
        assert halt_file(tmp_path).exists()
        assert halt_file(tmp_path).read_text().strip() == "exchange outage"

    def test_creates_the_directory_if_needed(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        engage_halt("x", nested)
        assert is_halted(nested)


class TestNotifiers:
    def test_null_notifier_always_succeeds(self):
        assert NullNotifier().send("hello") is True

    def test_telegram_posts_the_message(self):
        sent = []

        def transport(url, payload):
            sent.append((url, payload))
            return True

        assert TelegramNotifier("tok", "42", transport=transport).send("hi") is True
        url, payload = sent[0]
        assert "bottok/sendMessage" in url
        assert payload == {"chat_id": "42", "text": "hi"}

    def test_a_failing_notifier_never_raises(self):
        """Alerting must not be able to take the trading system down."""

        def broken(url, payload):
            raise ConnectionError("telegram is down")

        assert TelegramNotifier("tok", "42", transport=broken).send("hi") is False

    def test_falls_back_to_logging_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert notifier_from_environment().name == "log"

    def test_uses_telegram_when_configured(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        assert notifier_from_environment().name == "telegram"

    def test_half_configured_falls_back(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        assert notifier_from_environment().name == "log"


class TestSecrets:
    def test_the_token_is_not_in_the_repr(self):
        """A token in a traceback ends up in a log someone else can read."""
        notifier = TelegramNotifier("super-secret-token", "42")
        assert notifier.token == "super-secret-token"
        # The dataclass-free class has no auto repr that dumps fields.
        assert "super-secret-token" not in repr(notifier)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path))
