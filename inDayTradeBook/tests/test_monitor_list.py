from __future__ import annotations

from pathlib import Path

import pytest

from realtime.monitor_list import load_monitor_symbols, validate_subscription_limit


def test_load_monitor_symbols_ignores_comments_descriptions_and_duplicates(tmp_path: Path) -> None:
    monitor = tmp_path / "monitor_list.txt"
    monitor.write_text(
        "\n".join(
            [
                "// comment",
                "",
                "2330 台積電 semiconductor",
                "2308 Delta",
                "2330 duplicate should be ignored",
                "not-a-symbol",
                " 2454 MediaTek",
            ]
        ),
        encoding="utf-8",
    )

    assert load_monitor_symbols(monitor) == ["2330", "2308", "2454"]


def test_validate_subscription_limit_accepts_current_one_connection_shape() -> None:
    symbols = [f"{index:04d}" for index in range(1000, 1059)]

    validate_subscription_limit(symbols, ["trades", "books"], max_subscriptions=200)


def test_validate_subscription_limit_rejects_more_than_one_connection() -> None:
    symbols = [f"{index:04d}" for index in range(1000, 1101)]

    with pytest.raises(ValueError, match="exceeds"):
        validate_subscription_limit(symbols, ["trades", "books"], max_subscriptions=200)


def test_current_monitor_list_has_59_symbols() -> None:
    monitor_path = Path(__file__).resolve().parents[1] / "monitor_list.txt"

    symbols = load_monitor_symbols(monitor_path)

    assert len(symbols) == 59
    assert symbols[:3] == ["2330", "2303", "2454"]
    assert len(symbols) == len(set(symbols))
