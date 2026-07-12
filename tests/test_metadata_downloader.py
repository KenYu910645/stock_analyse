from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical
from downloader import metadata


def catalog_row(code: str, name: str) -> dict[str, str]:
    return {
        "Code": code,
        "Name": name,
        "Type": "股票",
        "Market": "上市",
        "Group": "水泥工業",
        "ISIN": f"TW{code}",
        "Start": "2000-01-01",
        "CFI": "ESVUFR",
        "Board": "一般",
        "CompanyName": name,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_refresh_availability_uses_flat_catalog_keyed_files(tmp_path):
    data_root = tmp_path / "data"
    catalog = pd.DataFrame([catalog_row("1101", "台泥"), catalog_row("1102", "亞泥")])
    write_csv(
        data_root / "price" / "1101_台泥.csv",
        [
            {
                "Date": "2026-07-09",
                "Close": "40",
                "open_adj": "39",
                "close_adj": "40",
                "high_adj": "41",
                "low_adj": "38",
                "AdjFactor": "1",
            }
        ],
    )
    write_csv(
        data_root / "price" / "1102_亞泥.csv",
        [
            {
                "Date": "2026-07-09",
                "Close": "30",
                "open_adj": "",
                "close_adj": "",
                "high_adj": "",
                "low_adj": "",
                "AdjFactor": "",
            }
        ],
    )
    write_csv(data_root / "price" / "twse_price_2026-07-09.csv", [{"Code": "1101"}])
    write_csv(data_root / "institutional" / "1101_台泥.csv", [{"Date": "2026-07-09"}])
    write_csv(data_root / "yield_pe_pb" / "1101_台泥.csv", [{"Date": "2026-07-09"}])
    write_csv(data_root / "report" / "1102_亞泥.csv", [{"Year": "2026"}])
    write_csv(
        data_root / "broker" / "twse" / "by_stock" / "1102_亞泥_2026-07-09.csv",
        [{"序號": "1"}],
    )
    margin_path = data_root / "margin" / "1101_台泥.csv"
    margin_path.parent.mkdir(parents=True)
    margin_path.write_text("Date,Code\n", encoding="utf-8")

    refreshed = metadata.refresh_availability(catalog, data_root).set_index("Code")

    assert refreshed.at["1101", "has_price"] == 1
    assert refreshed.at["1101", "has_adj_price"] == 1
    assert refreshed.at["1101", "has_institutional"] == 1
    assert refreshed.at["1101", "has_margin"] == 0
    assert refreshed.at["1101", "available_dataset_count"] == 4
    assert refreshed.at["1102", "has_price"] == 1
    assert refreshed.at["1102", "has_adj_price"] == 0
    assert refreshed.at["1102", "has_report"] == 1
    assert refreshed.at["1102", "has_broker"] == 1
    assert refreshed.at["1102", "available_dataset_count"] == 3


def test_write_catalog_round_trips_storage_headers(tmp_path):
    catalog = pd.DataFrame([catalog_row("1101", "台泥")])
    for column in metadata.AVAILABILITY_COLUMNS:
        catalog[column] = 0
    catalog["available_dataset_count"] = 0
    output_path = tmp_path / "metadata.csv"

    metadata.write_catalog(catalog, output_path)

    stored_header = output_path.read_text(encoding="utf-8-sig").splitlines()[0]
    round_tripped = read_csv_canonical(output_path, dtype={"Code": str})
    assert "類型" in stored_header
    assert round_tripped.loc[0, "Code"] == "1101"
    assert round_tripped.loc[0, "Type"] == "股票"
