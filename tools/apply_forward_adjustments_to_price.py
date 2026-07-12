"""Add forward-adjusted OHLC columns to consolidated price CSVs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical, to_csv_storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICE_DIR = PROJECT_ROOT / "data" / "price"
DIVIDEND_DIR = PROJECT_ROOT / "data" / "dividend" / "ex_right_dividend"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
TRADING_DAYS_PATH = PROJECT_ROOT / "data" / "trading_days.csv"
LOG_DIR = PROJECT_ROOT / "logs"

PRICE_COLUMNS = [
    "Date",
    "Capacity",
    "Turnover",
    "Open",
    "High",
    "Low",
    "Close",
    "Change",
    "Transaction",
]
ADJUSTED_COLUMNS = ["open_adj", "close_adj", "high_adj", "low_adj", "AdjFactor"]


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(value or "")).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip(" .") or "Unknown"


def load_metadata() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    if "Code" not in metadata.columns or "Name" not in metadata.columns:
        raise ValueError(f"{METADATA_PATH} must include Code and Name columns.")
    metadata["Code"] = metadata["Code"].astype(str).str.strip()
    return metadata.drop_duplicates("Code").set_index("Code", drop=False)


def load_previous_trading_day_map() -> dict[pd.Timestamp, pd.Timestamp]:
    trading_days = pd.read_csv(TRADING_DAYS_PATH, dtype=str)
    if "date" not in trading_days.columns:
        raise ValueError(f"{TRADING_DAYS_PATH} must include a date column.")
    dates = (
        pd.to_datetime(trading_days["date"], errors="coerce")
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    return {current: previous for previous, current in zip(dates, dates[1:])}


def code_from_price_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def dividend_path_for_code(code: str, metadata: pd.DataFrame) -> Path | None:
    if code not in metadata.index:
        return None
    name = str(metadata.at[code, "Name"]).strip()
    path = DIVIDEND_DIR / f"{code}_{safe_filename_part(name)}.csv"
    return path if path.exists() else None


def load_adjustment_events(code: str, metadata: pd.DataFrame) -> pd.DataFrame:
    dividend_path = dividend_path_for_code(code, metadata)
    if dividend_path is None:
        return pd.DataFrame(columns=["ex_date", "adjustment_ratio"])

    events = read_csv_canonical(
        dividend_path,
        dtype={"stock_id": str},
        usecols=["ex_date", "previous_close", "ex_reference_price"],
    )
    required = {"ex_date", "previous_close", "ex_reference_price"}
    if not required.issubset(events.columns):
        return pd.DataFrame(columns=["ex_date", "adjustment_ratio"])

    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce")
    events["previous_close"] = pd.to_numeric(events["previous_close"], errors="coerce")
    events["ex_reference_price"] = pd.to_numeric(events["ex_reference_price"], errors="coerce")
    events = events.dropna(subset=["ex_date", "previous_close", "ex_reference_price"])
    events = events[events["previous_close"].gt(0) & events["ex_reference_price"].gt(0)]
    if events.empty:
        return pd.DataFrame(columns=["ex_date", "adjustment_ratio"])

    events = events.sort_values("ex_date").drop_duplicates("ex_date", keep="last")
    events["adjustment_ratio"] = events["ex_reference_price"] / events["previous_close"]
    return events[["ex_date", "adjustment_ratio"]]


def infer_price_reference_events(
    df: pd.DataFrame,
    previous_trading_day_by_date: dict[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """Infer adjustment ratios from TWSE daily reference-price changes."""
    working = df[["Date", "Close", "Change"]].copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Close"] = pd.to_numeric(working["Close"], errors="coerce")
    working["Change"] = pd.to_numeric(working["Change"], errors="coerce")
    working = working.dropna(subset=["Date", "Close", "Change"]).sort_values("Date")
    working["previous_close"] = working["Close"].shift(1)
    working["previous_row_date"] = working["Date"].shift(1)
    working["expected_previous_trading_day"] = working["Date"].map(previous_trading_day_by_date)
    working = working[working["previous_row_date"].eq(working["expected_previous_trading_day"])]
    working["reference_price"] = working["Close"] - working["Change"]
    working = working[
        working["previous_close"].gt(0)
        & working["reference_price"].gt(0)
    ].copy()
    working["adjustment_ratio"] = working["reference_price"] / working["previous_close"]

    # Normal price-limit rounding and ordinary reference-price rounding create
    # tiny differences.  Corporate-action ratios are materially away from 1.
    working = working[(working["adjustment_ratio"] - 1).abs().gt(0.005)]
    if working.empty:
        return pd.DataFrame(columns=["ex_date", "adjustment_ratio"])

    return working.rename(columns={"Date": "ex_date"})[["ex_date", "adjustment_ratio"]]


def merge_adjustment_events(dividend_events: pd.DataFrame, inferred_events: pd.DataFrame) -> pd.DataFrame:
    """Merge dividend-file and price-inferred adjustment events by ex-date."""
    frames = []
    if not dividend_events.empty:
        frames.append(dividend_events.assign(source_priority=2))
    if not inferred_events.empty:
        frames.append(inferred_events.assign(source_priority=1))
    if not frames:
        return pd.DataFrame(columns=["ex_date", "adjustment_ratio"])

    events = pd.concat(frames, ignore_index=True)
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce")
    events["adjustment_ratio"] = pd.to_numeric(events["adjustment_ratio"], errors="coerce")
    events = events.dropna(subset=["ex_date", "adjustment_ratio"])
    events = events[events["adjustment_ratio"].gt(0)]
    return (
        events.sort_values(["ex_date", "source_priority"])
        .drop_duplicates("ex_date", keep="last")
        .sort_values("ex_date")
        [["ex_date", "adjustment_ratio"]]
    )


def add_adjusted_columns(
    price_df: pd.DataFrame,
    events: pd.DataFrame,
    previous_trading_day_by_date: dict[pd.Timestamp, pd.Timestamp],
    allow_price_inferred_events: bool,
) -> pd.DataFrame:
    missing = [column for column in PRICE_COLUMNS if column not in price_df.columns]
    if missing:
        raise ValueError(f"Price CSV is missing required columns: {missing}")

    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")

    for column in ["Open", "High", "Low", "Close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    inferred_events = (
        infer_price_reference_events(df, previous_trading_day_by_date)
        if allow_price_inferred_events
        else pd.DataFrame(columns=["ex_date", "adjustment_ratio"])
    )
    merged_events = merge_adjustment_events(events, inferred_events)
    df["AdjFactor"] = 1.0
    if not merged_events.empty:
        for _, event in merged_events.sort_values("ex_date", ascending=False).iterrows():
            df.loc[df["Date"] < event["ex_date"], "AdjFactor"] *= float(event["adjustment_ratio"])

    df["open_adj"] = (df["Open"] * df["AdjFactor"]).round(4)
    df["high_adj"] = (df["High"] * df["AdjFactor"]).round(4)
    df["low_adj"] = (df["Low"] * df["AdjFactor"]).round(4)
    df["close_adj"] = (df["Close"] * df["AdjFactor"]).round(4)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    output_columns = PRICE_COLUMNS + ADJUSTED_COLUMNS
    for column in output_columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[output_columns], merged_events, inferred_events


def main() -> None:
    metadata = load_metadata()
    previous_trading_day_by_date = load_previous_trading_day_map()
    price_paths = sorted(
        path for path in PRICE_DIR.glob("*.csv")
        if not path.name.startswith("twse_price_")
    )
    LOG_DIR.mkdir(exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "price_dir": str(PRICE_DIR),
        "files": {},
    }

    for index, price_path in enumerate(price_paths, start=1):
        code = code_from_price_path(price_path)
        print(f"[{index}/{len(price_paths)}] adjusting {price_path.name}")
        price_df = read_csv_canonical(price_path)
        events = load_adjustment_events(code, metadata)
        instrument_type = str(metadata.at[code, "Type"]).strip() if code in metadata.index else ""
        allow_price_inferred_events = instrument_type.upper() == "ETF"
        adjusted, merged_events, inferred_events = add_adjusted_columns(
            price_df,
            events,
            previous_trading_day_by_date,
            allow_price_inferred_events,
        )
        to_csv_storage(adjusted, price_path, index=False, encoding="utf-8-sig")
        manifest["files"][price_path.name] = {
            "code": code,
            "rows": len(adjusted),
            "events_used": len(merged_events),
            "dividend_events_used": len(events),
            "price_inferred_events_found": len(inferred_events),
            "price_inferred_events_enabled": allow_price_inferred_events,
            "date_min": adjusted["Date"].min() if not adjusted.empty else "",
            "date_max": adjusted["Date"].max() if not adjusted.empty else "",
            "min_adj_factor": float(adjusted["AdjFactor"].min()) if not adjusted.empty else None,
            "max_adj_factor": float(adjusted["AdjFactor"].max()) if not adjusted.empty else None,
        }

    manifest_path = LOG_DIR / f"price_forward_adjustment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
