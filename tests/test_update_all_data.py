from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from column_schema import read_csv_canonical
from downloader import update_all_data as updater


def test_append_or_refresh_rows_refreshes_only_fetched_at(tmp_path):
    path = tmp_path / "1101_test.csv"
    pd.DataFrame(
        [
            {
                "FetchedAt": "old",
                "Date": "2026-06-01",
                "Code": "1101",
                "Value": "10",
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    same = pd.DataFrame(
        [
            {
                "FetchedAt": "new",
                "Date": "2026-06-01",
                "Code": "1101",
                "Value": "10",
            }
        ]
    )
    result = updater.append_or_refresh_rows(
        str(path),
        same,
        ["Date", "Code"],
        refresh_fetched_at=True,
    )

    stored = read_csv_canonical(path, dtype=str).fillna("")
    assert result["appended"] == 0
    assert result["refreshed"] == 1
    assert len(stored) == 1
    assert stored.loc[0, "FetchedAt"] == "new"
    assert stored.loc[0, "Value"] == "10"

    changed = same.copy()
    changed.loc[0, "FetchedAt"] = "newer"
    changed.loc[0, "Value"] = "11"
    result = updater.append_or_refresh_rows(
        str(path),
        changed,
        ["Date", "Code"],
        refresh_fetched_at=True,
    )

    stored = read_csv_canonical(path, dtype=str).fillna("")
    assert result["appended"] == 1
    assert result["refreshed"] == 0
    assert len(stored) == 2
    assert stored.loc[1, "Value"] == "11"


def test_append_new_by_keys_normalizes_chinese_date_key_and_deduplicates_batch(tmp_path):
    data_date = "\u8cc7\u6599\u65e5\u671f"
    code = "\u8b49\u5238\u4ee3\u865f"
    level = "\u6301\u80a1\u5206\u7d1a"
    path = tmp_path / "1101_test.csv"
    pd.DataFrame(
        [
            {
                data_date: "2026-06-12",
                code: "1101",
                level: "1",
                "\u4eba\u6578": "10",
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    incoming = pd.DataFrame(
        [
            {data_date: "20260612", code: "1101", level: "1", "\u4eba\u6578": "20"},
            {data_date: "20260619", code: "1101", level: "1", "\u4eba\u6578": "30"},
            {data_date: "2026-06-19", code: "1101", level: "1", "\u4eba\u6578": "40"},
        ]
    )

    written = updater.append_new_by_keys(
        str(path),
        incoming,
        [data_date, code, level],
    )

    stored = read_csv_canonical(path, dtype=str).fillna("")
    assert written == 1
    assert len(stored) == 2
    assert stored["\u4eba\u6578"].tolist() == ["10", "40"]


def test_append_or_fill_blank_rows_fills_existing_blank_without_duplicate(tmp_path):
    path = tmp_path / "1101_test.csv"
    pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Code": "1101",
                "Close": "",
                "DividendYield": "3.26",
                "PEratio": "",
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    incoming = pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Code": "1101",
                "Close": "33.75",
                "DividendYield": "3.26",
                "PEratio": "",
            }
        ]
    )

    result = updater.append_or_fill_blank_rows(
        str(path),
        incoming,
        ["Date", "Code"],
        fill_columns=["Close"],
    )

    stored = updater.read_csv_canonical(path, dtype=str).fillna("")
    assert result == {"appended": 0, "filled": 1}
    assert len(stored) == 1
    assert stored.loc[0, "Close"] == "33.75"


def test_add_price_close_to_valuation_uses_unadjusted_price_close(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    price_dir = data_dir / "price"
    price_dir.mkdir(parents=True)
    price_path = price_dir / "1101_test.csv"
    pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Capacity": "1",
                "Turnover": "1",
                "Open": "33",
                "High": "34",
                "Low": "32",
                "Close": "33.75",
                "Change": "0.25",
                "Transaction": "1",
                "open_adj": "30",
                "close_adj": "30.75",
                "high_adj": "31",
                "low_adj": "29",
                "AdjFactor": "0.91",
            }
        ]
    ).to_csv(price_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))

    valuation = pd.DataFrame(
        [
            {
                "Date": "2026-06-17",
                "Code": "1101",
                "Name": "台泥",
                "Close": "",
                "DividendYield": "3.26",
                "DividendYear": "",
                "PEratio": "",
                "PBratio": "0.78",
                "FiscalYearQuarter": "",
            }
        ]
    )

    out = updater.add_price_close_to_valuation(valuation)

    assert out.loc[0, "Close"] == "33.75"


def test_missing_trading_dates_after_uses_calendar(monkeypatch):
    monkeypatch.setattr(
        updater,
        "load_trading_days",
        lambda: [
            date(2026, 6, 10),
            date(2026, 6, 11),
            date(2026, 6, 12),
            date(2026, 6, 15),
            date(2026, 6, 16),
        ],
    )

    assert updater.missing_trading_dates_after("2026-06-11", date(2026, 6, 16)) == [
        date(2026, 6, 12),
        date(2026, 6, 15),
        date(2026, 6, 16),
    ]


def test_refresh_trading_days_fails_without_usable_cached_calendar(tmp_path, monkeypatch):
    calendar_path = tmp_path / "trading_days.csv"
    monkeypatch.setattr(updater, "TRADING_DAYS_PATH", str(calendar_path))
    monkeypatch.setattr(
        updater.trading_days,
        "download_trading_days",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(RuntimeError, match="no usable cached calendar"):
        updater.refresh_trading_days(date(2026, 6, 16))


def test_refresh_trading_days_reports_degraded_cached_calendar(tmp_path, monkeypatch):
    calendar_path = tmp_path / "trading_days.csv"
    pd.DataFrame([{"date": "2026-06-15"}]).to_csv(calendar_path, index=False)
    calls = []
    monkeypatch.setattr(updater, "TRADING_DAYS_PATH", str(calendar_path))
    monkeypatch.setattr(
        updater.trading_days,
        "download_trading_days",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        updater,
        "status",
        lambda dataset, action, rows=0, path=None, note="": calls.append(
            (dataset, action, rows, note)
        ),
    )

    assert updater.refresh_trading_days(date(2026, 6, 16)) is False
    assert calls == [
        (
            "trading_days",
            "warning",
            1,
            "refresh_failed=offline; using cached calendar through 2026-06-15",
        )
    ]


def test_latest_trading_day_requires_eligible_calendar_date(monkeypatch):
    monkeypatch.setattr(updater, "load_trading_days", lambda: [])

    with pytest.raises(RuntimeError, match="No canonical trading day"):
        updater.latest_trading_day_on_or_before(date(2026, 6, 16))


def test_main_validates_calendar_before_metadata_or_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        updater,
        "parse_args",
        lambda: SimpleNamespace(date=None, skip_repairs=False),
    )
    monkeypatch.setattr(
        updater,
        "run_preflight_tasks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("calendar unavailable")),
    )
    monkeypatch.setattr(
        updater,
        "scan_data_registry",
        lambda: calls.append("registry"),
    )
    monkeypatch.setattr(
        updater,
        "load_listed_common_stocks",
        lambda: calls.append("metadata"),
    )

    with pytest.raises(RuntimeError, match="calendar unavailable"):
        updater.main()

    assert calls == []


def test_openapi_snapshot_rejects_unregistered_aggregate_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(updater, "fetch_json", lambda url: [{"Code": "1101"}])
    monkeypatch.setattr(
        updater,
        "filter_listed_openapi_df",
        lambda frame, listed_codes: frame,
    )
    monkeypatch.setattr(
        updater,
        "update_openapi_per_stock",
        lambda dataset_name, output_dir, frame: False,
    )

    with pytest.raises(ValueError, match="Unsupported OpenAPI snapshot dataset"):
        updater.update_openapi_snapshot(
            "unregistered_dataset",
            "/example",
            "unsupported",
            {"1101"},
        )

    assert list(tmp_path.rglob("*.csv")) == []


def test_market_closed_daily_updates_skip_price_only(monkeypatch):
    calls = []

    monkeypatch.setattr(
        updater,
        "status",
        lambda dataset, action, rows=0, path=None, note="": calls.append(
            ("status", dataset, action, note)
        ),
    )
    monkeypatch.setattr(
        updater,
        "update_price",
        lambda query_date, listed_codes: calls.append(("price", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_institutional",
        lambda query_date, listed_codes: calls.append(("institutional", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_margin",
        lambda query_date: calls.append(("margin", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_shareholding",
        lambda: calls.append(("shareholding",)),
    )
    monkeypatch.setattr(
        updater,
        "update_dividend",
        lambda query_date: calls.append(("dividend", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_valuation",
        lambda listed_codes: calls.append(("valuation",)),
    )
    monkeypatch.setattr(
        updater,
        "update_day_trading",
        lambda query_date, listed_codes: calls.append(("day_trading", query_date)),
    )

    updater.run_daily_updates(
        date(2026, 6, 18),
        {"1101"},
        market_closed=True,
    )

    assert not any(call[0] == "price" for call in calls)
    assert ("status", "price_by_stock", "skipped", "market_closed; price update skipped") in calls
    assert ("shareholding",) in calls
    assert ("valuation",) in calls
    assert any(call[0] == "day_trading" for call in calls)


def test_daily_task_runner_continues_after_independent_failure(monkeypatch):
    calls = []

    monkeypatch.setattr(
        updater,
        "status",
        lambda dataset, action, rows=0, path=None, note="": calls.append(
            ("status", dataset, action, note)
        ),
    )
    monkeypatch.setattr(
        updater,
        "update_price",
        lambda query_date, listed_codes: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        updater,
        "update_institutional",
        lambda query_date, listed_codes: calls.append(("institutional", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_margin",
        lambda query_date: calls.append(("margin", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_shareholding",
        lambda: calls.append(("shareholding",)),
    )
    monkeypatch.setattr(
        updater,
        "update_dividend",
        lambda query_date: calls.append(("dividend", query_date)),
    )
    monkeypatch.setattr(
        updater,
        "update_valuation",
        lambda listed_codes: calls.append(("valuation",)),
    )
    monkeypatch.setattr(
        updater,
        "update_day_trading",
        lambda query_date, listed_codes: calls.append(("day_trading", query_date)),
    )

    updater.run_daily_updates(date(2026, 6, 18), listed_codes={"1101"})

    assert ("status", "price_by_stock", "failed", "boom") in calls
    assert ("institutional", date(2026, 6, 18)) in calls
    assert ("shareholding",) in calls
    assert ("day_trading", date(2026, 6, 18)) in calls


def test_repair_incomplete_price_tails_deletes_trailing_blank_adjusted_rows(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    price_dir = data_dir / "price"
    price_dir.mkdir(parents=True)
    path = price_dir / "1101_test.csv"
    pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Capacity": "1",
                "Turnover": "1",
                "Open": "10",
                "High": "10",
                "Low": "10",
                "Close": "10",
                "Change": "0",
                "Transaction": "1",
                "open_adj": "10",
                "close_adj": "10",
                "high_adj": "10",
                "low_adj": "10",
                "AdjFactor": "1",
            },
            {
                "Date": "2026-06-17",
                "Capacity": "1",
                "Turnover": "1",
                "Open": "11",
                "High": "11",
                "Low": "11",
                "Close": "11",
                "Change": "1",
                "Transaction": "1",
                "open_adj": "",
                "close_adj": "",
                "high_adj": "",
                "low_adj": "",
                "AdjFactor": "",
            },
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))

    repaired = updater.repair_incomplete_price_tails()

    stored = read_csv_canonical(path, dtype=str).fillna("")
    assert repaired == {"1101": str(path)}
    assert stored["Date"].tolist() == ["2026-06-16"]


def test_repair_margin_order_tails_deletes_inversion_suffix(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    margin_dir = data_dir / "margin"
    margin_dir.mkdir(parents=True)
    path = margin_dir / "1101_test.csv"
    pd.DataFrame(
        [
            {"Date": "20260615", "Code": "1101"},
            {"Date": "20260617", "Code": "1101"},
            {"Date": "20260616", "Code": "1101"},
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        updater,
        "load_trading_days",
        lambda: [date(2026, 6, 15), date(2026, 6, 16), date(2026, 6, 17)],
    )

    repaired = updater.repair_margin_order_tails(target_date=date(2026, 6, 17))

    stored = read_csv_canonical(path, dtype=str).fillna("")
    assert repaired == {"1101": str(path)}
    assert stored["Date"].tolist() == ["2026-06-15"]


def test_report_periods_to_check_skips_current_files(tmp_path):
    path = tmp_path / "1101_test.csv"
    pd.DataFrame([{"Year": "2026", "Quarter": "1"}]).to_csv(path, index=False)

    assert updater.report_periods_to_check(str(path), latest_available=(2026, 1)) == []
    assert updater.report_periods_to_check(str(path), latest_available=(2026, 2)) == [
        (2026, 2)
    ]


def test_refresh_adjusted_price_columns_preserves_raw_columns(tmp_path, monkeypatch):
    path = tmp_path / "1101_test.csv"
    raw = pd.DataFrame(
        [
            {
                "Date": "2026-06-15",
                "Capacity": "100",
                "Turnover": "1000",
                "Open": "10",
                "High": "11",
                "Low": "9",
                "Close": "10",
                "Change": "0",
                "Transaction": "5",
                "open_adj": "",
                "close_adj": "",
                "high_adj": "",
                "low_adj": "",
                "AdjFactor": "",
            }
        ]
    )
    raw.to_csv(path, index=False, encoding="utf-8-sig")
    metadata = pd.DataFrame({"Code": ["1101"], "Type": ["股票"]}).set_index("Code", drop=False)
    monkeypatch.setattr(
        updater.price_adjustments,
        "load_adjustment_events",
        lambda code, metadata: pd.DataFrame(columns=["ex_date", "adjustment_ratio"]),
    )

    updater.refresh_adjusted_price_columns(str(path), "1101", metadata, {})

    stored = read_csv_canonical(path, dtype=str).fillna("")
    for column in updater.PRICE_COLUMNS:
        assert stored.loc[0, column] == raw.loc[0, column]
    assert stored.loc[0, "AdjFactor"] == "1.0"
    assert stored.loc[0, "close_adj"] == "10.0"


def test_update_day_trading_writes_flat_output(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(updater, "load_listed_common_stock_names", lambda: {"1101": "台泥"})
    monkeypatch.setattr(
        updater,
        "fetch_day_trading_rows",
        lambda query_date, listed_codes: pd.DataFrame(
            [
                {
                    "Date": "2026-06-16",
                    "Code": "1101",
                    "Name": "台泥",
                    "SuspensionNote": "",
                    "DayTradingVolume": "1",
                    "DayTradingBuyAmount": "2",
                    "DayTradingSellAmount": "3",
                }
            ]
        ),
    )

    updater.update_day_trading(date(2026, 6, 16), {"1101"})

    output_path = data_dir / "day_trading" / "1101_台泥.csv"
    assert output_path.exists()
    stored = read_csv_canonical(output_path, dtype=str).fillna("")
    for column in updater.DAY_TRADING_FEATURE_COLUMNS:
        assert column in stored.columns
    for column in updater.DAY_TRADING_DEPRECATED_COLUMNS:
        assert column not in stored.columns
    assert not (data_dir / "day_trading" / "by_stock").exists()


def test_add_day_trading_features_from_price_context():
    raw = pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Code": "1101",
                "Name": "台泥",
                "SuspensionNote": "",
                "DayTradingVolume": "1000",
                "DayTradingBuyAmount": "20000",
                "DayTradingSellAmount": "21000",
            }
        ]
    )
    price = pd.DataFrame(
        [
            {
                "Date": "2026-06-16",
                "Capacity": 10000,
                "Turnover": 200000,
                "Open": 20,
                "High": 21,
                "Low": 19,
                "Close": 20.5,
            }
        ]
    )

    out = updater.add_day_trading_features(raw, price)

    assert out.loc[0, "DayTradingVolumeRatio"] == 0.1
    assert out.loc[0, "DayTradingTurnover"] == 20500
    assert out.loc[0, "DayTradingTurnoverRatio"] == 0.1025
    assert out.loc[0, "DayTradingAvgBuyPrice"] == 20
    assert out.loc[0, "DayTradingAvgSellPrice"] == 21
    assert out.loc[0, "DayTradingAvgSpreadRate"] == 0.05
    assert out.loc[0, "DayTradingAmountImbalanceRatio"] == 0.005


def test_update_margin_fetches_in_memory_without_temp_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        updater,
        "load_trading_days",
        lambda: [date(2026, 6, 15), date(2026, 6, 16)],
    )
    monkeypatch.setattr(updater, "load_listed_common_stock_names", lambda: {"1101": "台泥"})
    monkeypatch.setattr(updater.margin_trading, "fetch_payload", lambda session, query_date: {"stat": "OK"})
    monkeypatch.setattr(
        updater.margin_trading,
        "parse_payload_rows",
        lambda payload, query_date: [
            {
                "Date": "20260616",
                "Code": "1101",
                "Name": "台泥",
                "MarginPurchase": "1",
                "MarginSale": "2",
                "MarginCashRepayment": "0",
                "MarginPreviousBalance": "3",
                "MarginCurrentBalance": "4",
                "MarginNextDayLimit": "5",
                "ShortPurchase": "6",
                "ShortSale": "7",
                "ShortStockRepayment": "0",
                "ShortPreviousBalance": "8",
                "ShortCurrentBalance": "9",
                "ShortNextDayLimit": "10",
                "Offsetting": "11",
                "Note": "",
            }
        ],
    )

    updater.update_margin(date(2026, 6, 16))

    assert (data_dir / "margin" / "1101_台泥.csv").exists()
    assert not (data_dir / "margin" / "raw").exists()
    assert not list((data_dir / "margin").glob("*manifest*"))


def test_update_twse_broker_syncs_date_folder_and_builds_views(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    metadata_path = data_dir / "metadata.csv"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("Code,Name,Type,Market\n1101,test,股票,上市\n", encoding="utf-8-sig")
    monkeypatch.setattr(updater, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(updater, "STOCK_METADATA_PATH", str(metadata_path))

    build_calls = []
    status_calls = []

    def fake_download(args):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = args.output_dir / "1101_bsr_twse_20260630_120000.csv"
        raw_path.write_bytes(b"raw")
        return {
            "summary": {
                "date": "2026-06-30",
                "selected_stocks": 2,
                "status_counts": {"success": 1, "no_data": 1},
                "output_dir": str(args.output_dir),
                "csv_log": "",
                "metadata_update": {"metadata_has_broker": 1},
            },
            "records": [],
            "exit_code": 0,
        }

    def fake_build(by_stock_dir, by_broker_dir, summary_json):
        by_stock_dir = Path(by_stock_dir)
        by_broker_dir = Path(by_broker_dir)
        source_files = len(list(by_stock_dir.glob("*.csv")))
        by_broker_dir.mkdir(parents=True, exist_ok=True)
        if source_files:
            (by_broker_dir / "BrokerA.csv").write_text("Date\n", encoding="utf-8-sig")
        build_calls.append(
            {
                "by_stock_dir": by_stock_dir,
                "by_broker_dir": by_broker_dir,
                "summary_json": Path(summary_json),
            }
        )
        return {
            "stats": {
                "source_files": source_files,
                "parsed_files": source_files,
                "skipped_files": 0,
                "output_files": 1 if source_files else 0,
                "records": source_files * 2,
            }
        }

    monkeypatch.setattr(updater, "download_twse_broker_batch", fake_download)
    monkeypatch.setattr(updater, "build_twse_broker_outputs", fake_build)
    monkeypatch.setattr(
        updater,
        "status",
        lambda dataset, action, rows=0, path=None, note="": status_calls.append(
            (dataset, action, rows, path, note)
        ),
    )

    updater.update_twse_broker(
        date(2026, 6, 30),
        max_stocks=2,
        max_attempts=2,
        throttle_min=0,
        throttle_max=0,
    )

    twse_dir = data_dir / "broker" / "twse"
    cumulative_by_stock = twse_dir / "by_stock"
    cumulative_by_broker = twse_dir / "by_broker"
    date_dir = twse_dir / "by_date" / "2026-06-30"
    date_by_stock = date_dir / "by_stock"
    date_by_broker = date_dir / "by_broker"
    raw_name = "1101_bsr_twse_20260630_120000.csv"

    assert (cumulative_by_stock / raw_name).exists()
    assert (date_by_stock / raw_name).exists()
    assert build_calls[0]["by_stock_dir"] == cumulative_by_stock
    assert build_calls[0]["by_broker_dir"] == cumulative_by_broker
    assert build_calls[1]["by_stock_dir"] == date_by_stock
    assert build_calls[1]["by_broker_dir"] == date_by_broker
    assert status_calls[-1][0] == "broker_twse"
    assert status_calls[-1][1] == "updated"
    assert status_calls[-1][2] == 2
    assert status_calls[-1][3] == str(date_dir)
    assert "date_raw=1" in status_calls[-1][4]


def test_shareholding_refresh_does_not_skip_partial_latest_snapshot(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    shareholding_dir = data_dir / "shareholding"
    shareholding_dir.mkdir(parents=True)
    (shareholding_dir / "1101_台泥.csv").write_text(
        "資料日期,證券代號,證券名稱,持股分級,持股/單位數分級\n"
        "2026-07-10,1101,台泥,1,1-999\n",
        encoding="utf-8-sig",
    )
    response_text = (
        "資料日期,證券代號,證券名稱,持股分級,持股/單位數分級\n"
        "20260710,1101,台泥,1,1-999\n"
        "20260710,1102,亞泥,1,1-999\n"
    )

    class FakeResponse:
        text = response_text
        encoding = None

        @staticmethod
        def raise_for_status():
            return None

    class FakeSession:
        @staticmethod
        def get(url, timeout):
            return FakeResponse()

    monkeypatch.setattr(updater, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(
        updater,
        "load_listed_common_stock_names",
        lambda: {"1101": "台泥", "1102": "亞泥"},
    )
    monkeypatch.setattr(updater.tdcc_shareholding, "make_session", lambda: FakeSession())

    updater.update_shareholding()

    assert (shareholding_dir / "1102_亞泥.csv").exists()


def test_report_zero_rows_with_filing_evidence_is_failure(tmp_path, monkeypatch):
    status_calls = []
    output_path = tmp_path / "data" / "report" / "1101_台泥.csv"
    listed = pd.DataFrame([{"Code": "1101", "Name": "台泥"}])

    monkeypatch.setattr(updater, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(updater, "latest_financial_period_by_code", lambda: {"1101": (2026, 1)})
    monkeypatch.setattr(updater.report, "STATEMENTS", ["income"])
    monkeypatch.setattr(updater.report, "create_session", lambda: object())
    monkeypatch.setattr(updater.report, "get_output_path", lambda code, name: str(output_path))
    monkeypatch.setattr(
        updater.report,
        "fetch_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Parsed zero report rows.")),
    )
    monkeypatch.setattr(updater.report, "sleep_between_requests", lambda: None)
    monkeypatch.setattr(
        updater,
        "status",
        lambda dataset, action, rows=0, path=None, note="": status_calls.append(
            (dataset, action, rows, path, note)
        ),
    )

    updater.update_report_latest_periods(listed)

    assert any(dataset == "report" and action == "failed" for dataset, action, *_ in status_calls)
    assert not any(dataset == "report" and action == "no_source_data" for dataset, action, *_ in status_calls)
