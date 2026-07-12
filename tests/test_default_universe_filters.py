from pathlib import Path

import pandas as pd

import backtesting
import stock_analyse


def write_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.DataFrame(
        [
            {"Code": "1101", "Name": "台泥", "Type": "股票", "Market": "上市"},
            {"Code": "0050", "Name": "元大台灣50", "Type": "ETF", "Market": "上市"},
            {"Code": "ABCD", "Name": "測試指數", "Type": "INDEX", "Market": "INDEX"},
        ]
    )
    catalog.to_csv(path, index=False, encoding="utf-8-sig")
    return catalog


def create_price_files(price_dir: Path) -> None:
    price_dir.mkdir(parents=True)
    for name in (
        "1101_台泥.csv",
        "0050_元大台灣50.csv",
        "9999_非目錄.csv",
        "twse_price_2026-07-09.csv",
    ):
        (price_dir / name).write_text("Date,Close\n2026-07-09,1\n", encoding="utf-8")


def test_backtesting_all_uses_metadata_listed_common_stock_universe(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata_path = data_dir / "metadata.csv"
    write_catalog(metadata_path)
    create_price_files(data_dir / "price")
    (data_dir / "trading_days.csv").write_text("date\n2026-07-09\n", encoding="utf-8")

    monkeypatch.setattr(backtesting, "DATA_DIR", data_dir)
    monkeypatch.setattr(backtesting, "STOCK_METADATA_PATH", metadata_path)

    assert backtesting.get_all_cached_stock_codes() == ["1101"]


def test_stock_analysis_uses_metadata_listed_common_stock_universe(tmp_path, monkeypatch):
    price_dir = tmp_path / "price"
    create_price_files(price_dir)
    metadata = write_catalog(tmp_path / "metadata.csv")
    monkeypatch.setattr(stock_analyse, "PRICE_DIR", price_dir)

    latest = stock_analyse.get_latest_csv_by_stock(metadata)

    assert list(latest) == ["1101"]
    assert latest["1101"].name == "1101_台泥.csv"
