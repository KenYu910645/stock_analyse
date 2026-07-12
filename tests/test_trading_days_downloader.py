from datetime import date

import pandas as pd

from downloader import trading_days


def test_default_start_date_uses_recent_overlap_from_existing_calendar(tmp_path):
    output = tmp_path / "trading_days.csv"
    pd.DataFrame(
        {
            "date": [
                "2026-06-18",
                "2026-06-22",
                "2026-06-24",
            ]
        }
    ).to_csv(output, index=False, encoding="utf-8-sig")

    assert trading_days.default_start_date(output) == date(2026, 5, 24)


def test_default_start_date_uses_online_start_when_output_missing(tmp_path):
    output = tmp_path / "missing_trading_days.csv"

    assert trading_days.default_start_date(output) == trading_days.TWSE_ONLINE_START_DATE
