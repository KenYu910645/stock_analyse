"""Monitor-list parsing and subscription-limit validation."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_MAX_SUBSCRIPTIONS = 200


def load_monitor_symbols(path: str | Path) -> list[str]:
    """Load 4-digit stock symbols from a monitor list.

    Lines may contain comments or descriptions after the code. Only lines whose
    first non-space token is a 4-digit code are used. Duplicates are removed
    while preserving the first occurrence.
    """

    monitor_path = Path(path)
    symbols: list[str] = []
    seen: set[str] = set()

    for line in monitor_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*(\d{4})\b", line)
        if not match:
            continue

        symbol = match.group(1)
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)

    return symbols


def validate_subscription_limit(
    symbols: list[str],
    channels: list[str],
    max_subscriptions: int = DEFAULT_MAX_SUBSCRIPTIONS,
) -> None:
    subscription_count = len(symbols) * len(channels)
    if subscription_count > max_subscriptions:
        raise ValueError(
            f"Monitor list requires {subscription_count} subscriptions "
            f"({len(symbols)} symbols x {len(channels)} channels), "
            f"which exceeds the one-connection limit of {max_subscriptions}."
        )
