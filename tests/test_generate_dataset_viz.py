from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from viz import generate_dataset_viz as dataset_viz
from viz.generate_dataset_viz import ex_right_events_by_date


def test_ex_right_event_labels_cash_capital_increase_from_rate():
    df = pd.DataFrame(
        [
            {
                "ex_date": "2026-06-01",
                "right_or_dividend": "\u6b0a",
                "previous_close": "28.85",
                "ex_reference_price": "26.69",
                "cash_capital_increase_rate": "0.258085",
            }
        ]
    )

    events = ex_right_events_by_date(df)

    assert events["2026-06-01"]["label"] == "\u73fe\u91d1\u589e\u8cc7"


def test_ex_right_event_labels_cash_capital_increase_from_reference_base():
    df = pd.DataFrame(
        [
            {
                "ex_date": "2013-09-26",
                "right_or_dividend": "\u6b0a",
                "previous_close": "28.85",
                "ex_reference_price": "26.69",
                "opening_auction_base": "28.85",
                "deducted_dividend_reference_price": "28.85",
            }
        ]
    )

    events = ex_right_events_by_date(df)

    assert events["2013-09-26"]["label"] == "\u73fe\u91d1\u589e\u8cc7"


def test_ex_right_event_keeps_regular_ex_right_dividend_label():
    df = pd.DataFrame(
        [
            {
                "ex_date": "2013-08-27",
                "right_or_dividend": "\u6b0a",
                "previous_close": "38.3",
                "ex_reference_price": "27.35",
                "opening_auction_base": "27.35",
                "deducted_dividend_reference_price": "27.35",
            }
        ]
    )

    events = ex_right_events_by_date(df)

    assert events["2013-08-27"]["label"] == "\u9664\u6b0a\u606f"


def test_csv_files_for_dataset_uses_catalog_and_flat_layout(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    price_dir = data_root / "price"
    nested_dir = price_dir / "legacy"
    nested_dir.mkdir(parents=True)
    pd.DataFrame([{"Code": "1101"}]).to_csv(data_root / "metadata.csv", index=False)
    for path in (
        price_dir / "1101_台泥.csv",
        price_dir / "0050_元大台灣50.csv",
        price_dir / "twse_price_2026-07-09.csv",
        nested_dir / "1101_duplicate.csv",
    ):
        path.write_text("Date,Close\n2026-07-09,1\n", encoding="utf-8")

    monkeypatch.setattr(dataset_viz, "DATA_ROOT", data_root)

    assert dataset_viz.csv_files_for_dataset("price") == [price_dir / "1101_台泥.csv"]


def test_main_returns_failure_when_any_renderer_fails(tmp_path, monkeypatch):
    failed = dataset_viz.VizResult(Path("source.csv"), Path("output.html"), "failed", "boom")
    monkeypatch.setattr(
        dataset_viz,
        "parse_args",
        lambda: SimpleNamespace(datasets="price", limit=None, force=False),
    )
    monkeypatch.setattr(dataset_viz, "generate_dataset", lambda *args, **kwargs: [failed])
    monkeypatch.setattr(dataset_viz, "write_manifest", lambda results: tmp_path / "manifest.csv")

    assert dataset_viz.main() == 1
