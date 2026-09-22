"""Notifications and the kill switch.

Two independent ways to stop the system, because the one that depends on a
third-party API is the one that will be unavailable when it matters:

1. **A file on disk.** ``aifin halt`` creates ``<data_dir>/HALT``; the runner
   refuses to open or hold a position while it exists. No network, no
   credentials, no API that can rate-limit you. This is the one that always
   works.
2. **A Telegram command.** Convenient from a phone, and useless when Telegram is
   down or the token has expired. It is the nice-to-have, not the guarantee.

Notifications degrade quietly. A missing bot token must never stop the runner
from trading correctly or, more importantly, from halting correctly — an alerting
failure that took the trading system down with it would be the alerting system
causing the outage it exists to report.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from ai_finance.config import data_dir

log = logging.getLogger(__name__)

TELEGRAM_TOKEN_VAR = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_VAR = "TELEGRAM_CHAT_ID"


def halt_file(root: Path | None = None) -> Path:
    """Path to the kill-switch file. Its existence means "stop trading"."""
    return (root if root is not None else data_dir()) / "HALT"


def is_halted(root: Path | None = None) -> bool:
    return halt_file(root).exists()


def engage_halt(reason: str, root: Path | None = None) -> Path:
    """Create the kill-switch file. Idempotent."""
    path = halt_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reason + "\n", encoding="utf-8")
    log.warning("kill switch engaged: %s", reason)
    return path


def release_halt(root: Path | None = None) -> bool:
    """Remove the kill-switch file. Returns whether it was there."""
    path = halt_file(root)
    if path.exists():
        path.unlink()
        log.warning("kill switch released")
        return True
    return False


def halt_reason(root: Path | None = None) -> str:
    path = halt_file(root)
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


class Notifier(Protocol):
    """Anything that can deliver a message."""

    name: str

    def send(self, message: str) -> bool:
        """Deliver ``message``. Returns whether it got through."""
        ...


class NullNotifier:
    """Logs instead of sending. The default when nothing is configured."""

    name = "log"

    def send(self, message: str) -> bool:
        log.info("notification: %s", message.replace("\n", " | "))
        return True


class TelegramNotifier:
    """Sends via the Telegram bot API.

    Failures are logged and swallowed. The runner must not be brought down by
    its own alerting.
    """

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: str,
        *,
        timeout: float = 10.0,
        transport=None,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self._transport = transport

    def send(self, message: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            if self._transport is not None:
                return bool(self._transport(url, payload))
            import requests

            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code != 200:
                log.warning("telegram returned %s: %s", response.status_code, response.text[:200])
                return False
            return True
        except Exception as exc:
            log.warning("telegram notification failed: %s", exc)
            return False


def notifier_from_environment() -> Notifier:
    """Build a Telegram notifier if it is configured, otherwise log locally."""
    token = os.environ.get(TELEGRAM_TOKEN_VAR)
    chat_id = os.environ.get(TELEGRAM_CHAT_VAR)
    if token and chat_id:
        return TelegramNotifier(token, chat_id)
    log.debug(
        "%s / %s not set; notifications will be logged rather than sent",
        TELEGRAM_TOKEN_VAR,
        TELEGRAM_CHAT_VAR,
    )
    return NullNotifier()
