"""Data loading and universe construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from column_schema import read_csv_canonical


NORMALIZED_COLUMNS = [
    "date",
    "stock_id",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "turnover",
]

COMMON_STOCK_TYPE = "\u80a1\u7968"
TWSE_MARKET = "\u4e0a\u5e02"


def stock_id_from_path(csv_path: Path) -> str:
    """Extract stock id from filenames like 2330_台積電.csv."""
    return csv_path.stem.split("_", 1)[0]


def load_price_csv(csv_path: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    """Load one adjusted price CSV into the normalized schema."""
    column_map = {
        data_config["date_col"]: "date",
        data_config["open_col"]: "adj_open",
        data_config["high_col"]: "adj_high",
        data_config["low_col"]: "adj_low",
        data_config["close_col"]: "adj_close",
        data_config["volume_col"]: "volume",
        data_config["turnover_col"]: "turnover",
    }
    df = read_csv_canonical(csv_path)
    missing = [source for source in column_map if source not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    df = df[list(column_map)].rename(columns=column_map)
    df["stock_id"] = stock_id_from_path(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["adj_open", "adj_high", "adj_low", "adj_close", "volume", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df[NORMALIZED_COLUMNS]


def load_price_data(config: dict[str, Any], stock_limit: int | None = None) -> pd.DataFrame:
    """Load all adjusted price CSV files from the configured directory."""
    price_dir = Path(config["data"]["price_dir"])
    csv_paths = sorted(
        csv_path for csv_path in price_dir.glob("*.csv")
        if not csv_path.name.startswith("twse_price_")
    )
    allow_unfiltered = bool(config["data"].get("allow_unfiltered_universe", False))
    allowed_codes = None
    if not allow_unfiltered:
        allowed_codes = load_twse_common_stock_codes(config["data"].get("metadata_path"))
    if allowed_codes is not None:
        csv_paths = [csv_path for csv_path in csv_paths if stock_id_from_path(csv_path) in allowed_codes]
    if stock_limit is not None:
        csv_paths = csv_paths[:stock_limit]
    if not csv_paths:
        raise FileNotFoundError(f"No price CSV files found in {price_dir}.")

    frames = []
    skipped = []
    for csv_path in csv_paths:
        try:
            frames.append(load_price_csv(csv_path, config["data"]))
        except Exception as exc:
            skipped.append(f"{csv_path.name}: {exc}")

    if not frames:
        raise ValueError("No usable price CSV files were loaded.")
    if skipped:
        print(f"Skipped {len(skipped)} price files with schema/read errors.")

    df = pd.concat(frames, ignore_index=True)
    duplicate_count = int(df.duplicated(["date", "stock_id"], keep=False).sum())
    if duplicate_count:
        print(f"Removed {duplicate_count} duplicate date/stock rows, keeping the last row per date.")
        df = df.drop_duplicates(["date", "stock_id"], keep="last")

    return (
        df.dropna(subset=["date", "stock_id", "adj_close"])
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )


def load_twse_common_stock_codes(metadata_path: str | None) -> set[str]:
    """Return TWSE listed common-stock codes, failing closed on bad metadata."""
    if not metadata_path:
        raise ValueError(
            "A metadata_path is required for the default TWSE listed common-stock universe. "
            "Set data.allow_unfiltered_universe=true only for an intentional broad load."
        )

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata catalog does not exist: {path}. "
            "Set data.allow_unfiltered_universe=true only for an intentional broad load."
        )

    try:
        metadata_df = read_csv_canonical(path, dtype={"Code": str})
    except Exception as exc:
        raise ValueError(f"Could not read metadata catalog {path}: {exc}") from exc
    required = {"Code", "Type", "Market"}
    missing = required - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata catalog {path} missing required columns: {sorted(missing)}")

    filtered = metadata_df[
        metadata_df["Type"].eq(COMMON_STOCK_TYPE)
        & metadata_df["Market"].eq(TWSE_MARKET)
    ]
    codes = set(filtered["Code"].astype(str).str.strip()) - {""}
    if not codes:
        raise ValueError(f"Metadata catalog {path} contains no TWSE listed common stocks.")
    return codes


def filter_date_range(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Apply optional inclusive date filters."""
    filtered = df
    if start:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start)]
    if end:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end)]
    return filtered.copy()


def build_universe(price_df: pd.DataFrame, universe_config: dict[str, Any]) -> pd.DataFrame:
    """Return price data annotated with tradable-universe flags."""
    df = price_df.sort_values(["stock_id", "date"]).copy()
    min_price = float(universe_config.get("min_price", 0))
    min_turnover = universe_config.get("min_avg_turnover_20d")
    min_history_days = int(universe_config.get("min_history_days", 0))

    grouped = df.groupby("stock_id", sort=False)
    df["history_days"] = grouped.cumcount() + 1
    df["avg_turnover_20d"] = (
        grouped["turnover"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["in_universe"] = (
        df["adj_close"].gt(min_price)
        & df["volume"].gt(0)
        & df["history_days"].ge(min_history_days)
    )
    if min_turnover is not None:
        df["in_universe"] &= df["avg_turnover_20d"].ge(float(min_turnover))
    return df
