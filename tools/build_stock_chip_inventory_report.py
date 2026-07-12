"""Build a per-stock flow-implied chip inventory and cost report."""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from column_schema import read_csv_canonical
from build_institutional_participation_report import compute_metrics


DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
METADATA_PATH = DATA_DIR / "metadata.csv"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_flow_inventory"
STOCK_PAGES_ROOT = DATA_VIZ_ROOT / "stocks"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_flow_inventory"
SHARE_EPSILON = 1e-3
RESIDUAL_OUTPUT_EPSILON = 1000.0


@dataclass(frozen=True)
class ParticipantSpec:
    key: str
    label: str
    color: str


@dataclass(frozen=True)
class InventoryMethodSpec:
    key: str
    label: str
    description: str


PARTICIPANTS = [
    ParticipantSpec("foreign", "外資", "#2563eb"),
    ParticipantSpec("trust", "投信", "#d97706"),
    ParticipantSpec("dealer", "自營商", "#7c3aed"),
    ParticipantSpec("other", "其他市場參與者", "#64748b"),
]

INVENTORY_METHODS = [
    InventoryMethodSpec("fifo", "FIFO", "先賣出最早建立的區間內籌碼，偏老籌碼被賣掉的假設。"),
    InventoryMethodSpec("lifo", "FILO/LIFO", "先賣出最近建立的區間內籌碼，偏短線加碼先被沖銷、核心籌碼留下的假設。"),
    InventoryMethodSpec("average", "平均成本法", "賣出時按當時開放庫存平均成本沖銷，並等比例減少既有批次。"),
]
DEFAULT_METHOD_KEY = "fifo"
METHOD_COLORS = {
    "fifo": "#2563eb",
    "lifo": "#d97706",
    "average": "#7c3aed",
}
MIN_COST_BIN_COUNT = 20
TARGET_COST_BIN_COUNT = 28
MAX_COST_BIN_COUNT = 80
MIN_COST_BIN_WIDTH = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flow-implied chip inventory and cost distribution report for one stock."
    )
    parser.add_argument("--all", action="store_true", help="Build the report for every listed TWSE common stock.")
    parser.add_argument("--code", default="2330", help="Stock code, default: 2330.")
    parser.add_argument("--years", type=float, default=5.0, help="Trailing year window, default: 5.")
    parser.add_argument("--start-date", default="", help="Optional explicit start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Optional explicit end date, YYYY-MM-DD.")
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.0,
        help="Optional cost-distribution bin width in TWD. Default 0 uses automatic price-spread bins.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N listed-stock limit for testing.")
    parser.add_argument("--skip-stock-pages", action="store_true", help="Only write all-market CSV and index outputs.")
    return parser.parse_args()


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def stock_name_from_path(path: Path, fallback: str = "") -> str:
    if "_" in path.stem:
        return path.stem.split("_", 1)[1]
    return fallback or code_from_path(path)


def path_by_code(directory: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(directory.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "")).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("._ ") or "stock"


def listed_common_metadata(code: str) -> dict[str, str]:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    row = metadata[
        metadata["Code"].astype(str).eq(code)
        & metadata["類型"].eq("股票")
        & metadata["市場"].eq("上市")
    ]
    if row.empty:
        raise ValueError(f"{code} is not a listed TWSE common stock in data/metadata.csv")
    return row.iloc[0].fillna("").to_dict()


def listed_common_metadata_frame(limit: int | None = None) -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    listed = metadata[metadata["類型"].eq("股票") & metadata["市場"].eq("上市")].copy()
    listed = listed.sort_values("Code").reset_index(drop=True)
    if limit:
        listed = listed.head(limit).copy()
    return listed.fillna("")


def window_slug(years: float) -> str:
    if float(years).is_integer():
        return f"{int(years)}y"
    return f"{years:g}y".replace(".", "p")


def window_label(years: float) -> str:
    labels = {1: "近一年", 3: "近三年", 5: "近五年", 10: "近十年"}
    if float(years).is_integer():
        value = int(years)
        return labels.get(value, f"近{value}年")
    return f"近{years:g}年"


def number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def fmt_number(value: Any, digits: int = 0) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    return f"{number:,.{digits}f}"


def fmt_price(value: Any) -> str:
    return fmt_number(value, 2)


def fmt_lots(shares: Any, digits: int = 0) -> str:
    number = finite_float(shares)
    if number is None:
        return "-"
    return f"{number / 1000:,.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    return f"{number * 100:,.{digits}f}%"


def nice_cost_bin_width(target_width: float, max_width: float) -> float:
    if not math.isfinite(target_width) or target_width <= 0:
        return MIN_COST_BIN_WIDTH
    max_width = max(max_width, MIN_COST_BIN_WIDTH)
    exponent = math.floor(math.log10(max(max_width, MIN_COST_BIN_WIDTH)))
    candidates: list[float] = []
    for power in range(exponent - 6, exponent + 3):
        scale = 10**power
        for step in (1, 2, 2.5, 5, 10):
            width = step * scale
            if MIN_COST_BIN_WIDTH <= width <= max_width * (1 + 1e-9):
                candidates.append(width)
    if not candidates:
        return max(MIN_COST_BIN_WIDTH, max_width)
    return min(candidates, key=lambda width: (abs(math.log(width / target_width)), -width))


def format_bucket_price(value: float, width: float) -> str:
    if width < 1:
        digits = 2
    elif width < 10:
        digits = 1
    else:
        digits = 0
    return fmt_number(value, digits)


def cost_bin_plan(prices: pd.Series, current_close: float, requested_width: float) -> tuple[float, float, float, int]:
    finite_prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    finite_prices = finite_prices[finite_prices.map(math.isfinite)]
    close = finite_float(current_close)
    price_values = finite_prices.tolist()
    if close is not None:
        price_values.append(close)
    if not price_values:
        return 0.0, 1.0, 1.0 / MIN_COST_BIN_COUNT, MIN_COST_BIN_COUNT

    price_min = min(price_values)
    price_max = max(price_values)
    if math.isclose(price_min, price_max):
        center = price_min
        spread = max(abs(center) * 0.05, 1.0)
        price_min = center - spread / 2
        price_max = center + spread / 2
    spread = max(price_max - price_min, MIN_COST_BIN_WIDTH * MIN_COST_BIN_COUNT)
    max_width_for_min_bins = spread / MIN_COST_BIN_COUNT
    if requested_width > 0:
        width = min(requested_width, max_width_for_min_bins)
        width = max(width, MIN_COST_BIN_WIDTH)
    else:
        width = nice_cost_bin_width(spread / TARGET_COST_BIN_COUNT, max_width_for_min_bins)

    lower = math.floor(price_min / width) * width
    upper = math.ceil(price_max / width) * width
    bin_count = max(1, int(round((upper - lower) / width)))
    while bin_count < MIN_COST_BIN_COUNT:
        width = max(width / 2, MIN_COST_BIN_WIDTH)
        lower = math.floor(price_min / width) * width
        upper = math.ceil(price_max / width) * width
        bin_count = max(1, int(round((upper - lower) / width)))
        if math.isclose(width, MIN_COST_BIN_WIDTH) and bin_count >= MIN_COST_BIN_COUNT:
            break
    while bin_count > MAX_COST_BIN_COUNT and requested_width <= 0:
        width = nice_cost_bin_width(width * 1.35, spread)
        lower = math.floor(price_min / width) * width
        upper = math.ceil(price_max / width) * width
        next_count = max(1, int(round((upper - lower) / width)))
        if next_count >= bin_count:
            break
        bin_count = next_count
    upper = lower + width * bin_count
    return lower, upper, width, bin_count


def fmt_money(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    sign = "-" if number < 0 else ""
    absolute = abs(number)
    if absolute >= 100_000_000:
        return f"{sign}{absolute / 100_000_000:,.2f} 億"
    if absolute >= 10_000:
        return f"{sign}{absolute / 10_000:,.1f} 萬"
    return f"{number:,.0f}"


def prepare_metrics(
    price_path: Path,
    institutional_path: Path,
    start_date: str,
    end_date: str,
    years: float,
) -> pd.DataFrame:
    price = read_csv_canonical(price_path, dtype={"Code": str})
    institutional = read_csv_canonical(institutional_path, dtype={"Code": str})

    institutional_dates = set(pd.to_datetime(institutional["Date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna())
    if not institutional_dates:
        return pd.DataFrame()

    price_dates = pd.to_datetime(price["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    price_for_metrics = price[price_dates.isin(institutional_dates)].copy()
    if price_for_metrics.empty:
        return pd.DataFrame()

    price_extra = price_for_metrics[["Date", "Turnover"]].copy()
    price_extra["Date"] = pd.to_datetime(price_extra["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    price_extra["Turnover"] = pd.to_numeric(price_extra["Turnover"], errors="coerce")
    price_extra = price_extra.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")

    metrics = compute_metrics(price_for_metrics, institutional)
    if metrics.empty:
        return metrics
    metrics = metrics[metrics["Date"].isin(institutional_dates)].copy()
    metrics = metrics.merge(price_extra, on="Date", how="left")
    metrics["DateTs"] = pd.to_datetime(metrics["Date"], errors="coerce")
    metrics = metrics.dropna(subset=["DateTs"]).sort_values("DateTs").reset_index(drop=True)

    for column in ["Capacity", "Close", "Turnover"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    average_price = metrics["Turnover"] / metrics["Capacity"]
    metrics["EntryPrice"] = average_price.where(average_price.gt(0), metrics["Close"])
    metrics["EntryPrice"] = metrics["EntryPrice"].where(metrics["EntryPrice"].gt(0), metrics["Close"])

    metrics["foreign_net"] = number_series(metrics, "foreign_buy") - number_series(metrics, "foreign_sell")
    metrics["trust_net"] = number_series(metrics, "trust_buy") - number_series(metrics, "trust_sell")
    metrics["dealer_net"] = number_series(metrics, "dealer_buy") - number_series(metrics, "dealer_sell")
    metrics["other_net"] = -(metrics["foreign_net"] + metrics["trust_net"] + metrics["dealer_net"])

    last_date = pd.to_datetime(end_date, errors="coerce") if end_date else metrics["DateTs"].max()
    if pd.isna(last_date):
        return pd.DataFrame()
    report_start_target = pd.to_datetime(start_date, errors="coerce") if start_date else last_date - pd.DateOffset(years=years)
    if pd.isna(report_start_target):
        raise ValueError(f"Invalid start date: {start_date}")

    metrics = metrics[metrics["DateTs"].le(last_date)].copy()
    if metrics.empty:
        return pd.DataFrame()
    metrics["InReportWindow"] = metrics["DateTs"].between(report_start_target, last_date)
    if not metrics["InReportWindow"].any():
        return pd.DataFrame()
    actual_report_start = metrics.loc[metrics["InReportWindow"], "DateTs"].min()
    metrics["WarmupStartDate"] = metrics["DateTs"].min().strftime("%Y-%m-%d")
    metrics["ReportStartDate"] = actual_report_start.strftime("%Y-%m-%d")
    metrics["ReportEndDate"] = last_date.strftime("%Y-%m-%d")
    metrics["Date"] = metrics["DateTs"].dt.strftime("%Y-%m-%d")
    return metrics.reset_index(drop=True)


def lot_shares(lots: list[dict[str, float | str]]) -> float:
    return sum(float(lot["Shares"]) for lot in lots)


def lot_cost(lots: list[dict[str, float | str]]) -> float:
    return sum(float(lot["Shares"]) * float(lot["EntryPrice"]) for lot in lots)


def infer_minimum_initial_inventory(metrics: pd.DataFrame) -> dict[str, float]:
    required: dict[str, float] = {}
    for participant in PARTICIPANTS:
        cumulative = metrics[f"{participant.key}_net"].astype(float).cumsum()
        min_cumulative = float(cumulative.min()) if not cumulative.empty else 0.0
        required[participant.key] = max(-min_cumulative, 0.0)
    return required


def consume_sell(
    lots: list[dict[str, float | str]],
    sell_shares: float,
    sell_price: float,
    method_key: str,
) -> tuple[float, float, float]:
    if sell_shares <= SHARE_EPSILON:
        return 0.0, 0.0, 0.0

    realized_pnl = 0.0
    realized_amount = 0.0
    remaining_sell = sell_shares

    if method_key == "average":
        open_shares = lot_shares(lots)
        if open_shares <= SHARE_EPSILON:
            return sell_shares, 0.0, 0.0
        consumed = min(remaining_sell, open_shares)
        average_cost = lot_cost(lots) / open_shares
        realized_pnl += consumed * (sell_price - average_cost)
        realized_amount += consumed * sell_price
        scale = max(open_shares - consumed, 0.0) / open_shares
        for lot in lots:
            lot["Shares"] = float(lot["Shares"]) * scale
        lots[:] = [lot for lot in lots if float(lot["Shares"]) > SHARE_EPSILON]
        remaining = max(remaining_sell - consumed, 0.0)
        return (0.0 if remaining <= SHARE_EPSILON else remaining), realized_pnl, realized_amount

    while remaining_sell > 0 and lots:
        lot_index = -1 if method_key == "lifo" else 0
        open_lot = lots[lot_index]
        open_lot_shares = float(open_lot["Shares"])
        consumed = min(open_lot_shares, remaining_sell)
        realized_pnl += consumed * (sell_price - float(open_lot["EntryPrice"]))
        realized_amount += consumed * sell_price
        open_lot_shares -= consumed
        remaining_sell -= consumed
        if open_lot_shares <= SHARE_EPSILON:
            lots.pop(lot_index)
        else:
            open_lot["Shares"] = open_lot_shares
    return (0.0 if remaining_sell <= SHARE_EPSILON else remaining_sell), realized_pnl, realized_amount


def estimate_inventory_for_method(
    code: str,
    name: str,
    metrics: pd.DataFrame,
    bin_width: float,
    method: InventoryMethodSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warmup_start_date = str(metrics["WarmupStartDate"].iloc[0])
    report_start_date = str(metrics["ReportStartDate"].iloc[0])
    latest_date = str(metrics["ReportEndDate"].iloc[0])
    current_close = float(metrics["Close"].iloc[-1])
    initial_cost_price = float(metrics["EntryPrice"].iloc[0]) if finite_float(metrics["EntryPrice"].iloc[0]) else float(metrics["Close"].iloc[0])
    initial_inventory = infer_minimum_initial_inventory(metrics)

    lots: dict[str, list[dict[str, float | str]]] = {participant.key: [] for participant in PARTICIPANTS}
    cumulative = {participant.key: 0.0 for participant in PARTICIPANTS}
    report_cumulative = {participant.key: 0.0 for participant in PARTICIPANTS}
    positive_flow = {participant.key: 0.0 for participant in PARTICIPANTS}
    negative_flow = {participant.key: 0.0 for participant in PARTICIPANTS}
    excess_sell = {participant.key: 0.0 for participant in PARTICIPANTS}
    realized_pnl = {participant.key: 0.0 for participant in PARTICIPANTS}
    realized_sell_amount = {participant.key: 0.0 for participant in PARTICIPANTS}
    daily_rows: list[dict[str, float | str]] = []

    for participant in PARTICIPANTS:
        shares = initial_inventory[participant.key]
        if shares > SHARE_EPSILON:
            lots[participant.key].append(
                {
                    "Date": f"Before {warmup_start_date}",
                    "Shares": shares,
                    "EntryPrice": initial_cost_price,
                    "LotType": "最低期初庫存估計",
                }
            )

    for row in metrics.to_dict("records"):
        date = str(row["Date"])
        entry_price = float(row["EntryPrice"]) if finite_float(row["EntryPrice"]) else float(row["Close"])
        in_report_window = bool(row.get("InReportWindow", True))
        daily: dict[str, float | str] = {
            "Code": code,
            "Name": name,
            "MethodKey": method.key,
            "Method": method.label,
            "Date": date,
            "Close": float(row["Close"]),
            "EntryPrice": entry_price,
        }

        for participant in PARTICIPANTS:
            key = participant.key
            net = float(row.get(f"{key}_net", 0.0) or 0.0)
            cumulative[key] += net
            if in_report_window:
                report_cumulative[key] += net
            if net > 0:
                if in_report_window:
                    positive_flow[key] += net
                lots[key].append({"Date": date, "Shares": net, "EntryPrice": entry_price, "LotType": "區間淨買"})
            elif net < 0:
                sell_shares = -net
                if in_report_window:
                    negative_flow[key] += sell_shares
                excess, realized, realized_amount = consume_sell(lots[key], sell_shares, entry_price, method.key)
                excess_sell[key] += excess
                if in_report_window:
                    realized_pnl[key] += realized
                    realized_sell_amount[key] += realized_amount

            open_shares = lot_shares(lots[key])
            open_cost = lot_cost(lots[key])
            daily[f"{key}_net_shares"] = net
            daily[f"{key}_cumulative_net_shares"] = report_cumulative[key]
            daily[f"{key}_full_history_net_shares"] = cumulative[key]
            daily[f"{key}_open_shares"] = open_shares
            daily[f"{key}_average_cost"] = open_cost / open_shares if open_shares else 0.0

        if in_report_window:
            daily_rows.append(daily)

    summary_rows: list[dict[str, float | str]] = []
    open_lot_rows: list[dict[str, float | str]] = []
    for participant in PARTICIPANTS:
        key = participant.key
        open_shares = lot_shares(lots[key])
        open_cost = lot_cost(lots[key])
        residual_sell_gap = 0.0 if abs(excess_sell[key]) < RESIDUAL_OUTPUT_EPSILON else excess_sell[key]
        market_value = open_shares * current_close
        unrealized_pnl = market_value - open_cost
        if open_shares > 0:
            state = "區間內仍有留倉"
        elif cumulative[key] < 0:
            state = "區間淨流出，可能來自期初庫存"
        else:
            state = "區間買賣大致沖銷"

        summary_rows.append(
            {
                "Code": code,
                "Name": name,
                "WarmupStartDate": warmup_start_date,
                "StartDate": report_start_date,
                "EndDate": latest_date,
                "CurrentClose": current_close,
                "MethodKey": method.key,
                "Method": method.label,
                "MethodDescription": method.description,
                "ParticipantKey": key,
                "Participant": participant.label,
                "CumulativeNetShares": report_cumulative[key],
                "CumulativeNetLots": report_cumulative[key] / 1000,
                "FullHistoryNetShares": cumulative[key],
                "FullHistoryNetLots": cumulative[key] / 1000,
                "InitialInventoryShares": initial_inventory[key],
                "InitialInventoryLots": initial_inventory[key] / 1000,
                "InitialInventoryCostPrice": initial_cost_price if initial_inventory[key] else 0.0,
                "PositiveNetFlowShares": positive_flow[key],
                "NegativeNetFlowShares": negative_flow[key],
                "ResidualSellGapShares": residual_sell_gap,
                "OpenShares": open_shares,
                "OpenLots": open_shares / 1000,
                "OpenCostAmount": open_cost,
                "AverageCost": open_cost / open_shares if open_shares else 0.0,
                "MarketValue": market_value,
                "UnrealizedPnl": unrealized_pnl,
                "UnrealizedPnlPct": unrealized_pnl / open_cost if open_cost else 0.0,
                "RealizedPnlOnWindowLots": realized_pnl[key],
                "RealizedSellAmountOnWindowLots": realized_sell_amount[key],
                "PositionState": state,
            }
        )

        for lot in lots[key]:
            shares = float(lot["Shares"])
            entry_price = float(lot["EntryPrice"])
            cost_amount = shares * entry_price
            lot_market_value = shares * current_close
            lot_pnl = lot_market_value - cost_amount
            open_lot_rows.append(
                {
                    "Code": code,
                    "Name": name,
                    "WarmupStartDate": warmup_start_date,
                    "StartDate": report_start_date,
                    "EndDate": latest_date,
                    "MethodKey": method.key,
                    "Method": method.label,
                    "ParticipantKey": key,
                    "Participant": participant.label,
                    "LotDate": str(lot["Date"]),
                    "LotType": str(lot.get("LotType", "區間淨買")),
                    "Shares": shares,
                    "Lots": shares / 1000,
                    "EntryPrice": entry_price,
                    "CostAmount": cost_amount,
                    "CurrentClose": current_close,
                    "MarketValue": lot_market_value,
                    "UnrealizedPnl": lot_pnl,
                    "UnrealizedPnlPct": lot_pnl / cost_amount if cost_amount else 0.0,
                }
            )

    open_lots = pd.DataFrame(open_lot_rows)
    cost_bins = build_cost_bins(code, name, report_start_date, latest_date, current_close, open_lots, bin_width)
    if not cost_bins.empty:
        cost_bins["MethodKey"] = method.key
        cost_bins["Method"] = method.label
        cost_bins["MethodDescription"] = method.description
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows), open_lots, cost_bins


def estimate_inventory(
    code: str,
    name: str,
    metrics: pd.DataFrame,
    bin_width: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    open_lot_frames: list[pd.DataFrame] = []
    cost_bin_frames: list[pd.DataFrame] = []
    for method in INVENTORY_METHODS:
        summary, daily, open_lots, cost_bins = estimate_inventory_for_method(code, name, metrics, bin_width, method)
        summary_frames.append(summary)
        daily_frames.append(daily)
        if not open_lots.empty:
            open_lot_frames.append(open_lots)
        if not cost_bins.empty:
            cost_bin_frames.append(cost_bins)

    return (
        pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
        pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(),
        pd.concat(open_lot_frames, ignore_index=True) if open_lot_frames else pd.DataFrame(),
        pd.concat(cost_bin_frames, ignore_index=True) if cost_bin_frames else pd.DataFrame(),
    )


def build_cost_bins(
    code: str,
    name: str,
    start_date: str,
    latest_date: str,
    current_close: float,
    open_lots: pd.DataFrame,
    bin_width: float,
) -> pd.DataFrame:
    if open_lots.empty:
        return pd.DataFrame()
    frame = open_lots.copy()
    frame["EntryPrice"] = pd.to_numeric(frame["EntryPrice"], errors="coerce")
    frame["Shares"] = pd.to_numeric(frame["Shares"], errors="coerce").fillna(0.0)
    frame["CostAmount"] = pd.to_numeric(frame["CostAmount"], errors="coerce").fillna(0.0)
    frame = frame[frame["EntryPrice"].notna() & frame["Shares"].gt(0)].copy()
    if frame.empty:
        return pd.DataFrame()

    lower, _upper, width, bin_count = cost_bin_plan(frame["EntryPrice"], current_close, bin_width)
    participant_label = {participant.key: participant.label for participant in PARTICIPANTS}
    participant_order = {participant.key: index for index, participant in enumerate(PARTICIPANTS)}

    frame["BucketIndex"] = frame["EntryPrice"].map(
        lambda price: min(
            bin_count - 1,
            max(0, int(math.floor((float(price) - lower) / width + 1e-9))),
        )
    )
    frame["BucketStart"] = lower + frame["BucketIndex"] * width
    frame["BucketEnd"] = frame["BucketStart"] + width
    frame["BucketLabel"] = frame.apply(
        lambda row: (
            f"{format_bucket_price(float(row['BucketStart']), width)}-"
            f"{format_bucket_price(float(row['BucketEnd']), width)}"
        ),
        axis=1,
    )
    grouped = (
        frame.groupby(["ParticipantKey", "Participant", "BucketIndex"], dropna=False)
        .agg(Shares=("Shares", "sum"), CostAmount=("CostAmount", "sum"))
        .reset_index()
    )
    lookup = {
        (str(row["ParticipantKey"]), int(row["BucketIndex"])): (float(row["Shares"]), float(row["CostAmount"]))
        for _, row in grouped.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for bucket_index in range(bin_count):
        bucket_start = lower + bucket_index * width
        bucket_end = bucket_start + width
        bucket_label = f"{format_bucket_price(bucket_start, width)}-{format_bucket_price(bucket_end, width)}"
        for participant in PARTICIPANTS:
            shares, cost_amount = lookup.get((participant.key, bucket_index), (0.0, 0.0))
            rows.append(
                {
                    "ParticipantKey": participant.key,
                    "Participant": participant_label[participant.key],
                    "ParticipantOrder": participant_order[participant.key],
                    "BucketIndex": bucket_index,
                    "BucketStart": bucket_start,
                    "BucketEnd": bucket_end,
                    "BucketLabel": bucket_label,
                    "BinWidth": width,
                    "BinCount": bin_count,
                    "Shares": shares,
                    "CostAmount": cost_amount,
                }
            )
    grouped = pd.DataFrame(rows).sort_values(["BucketIndex", "ParticipantOrder"]).drop(columns=["ParticipantOrder"])
    grouped["Code"] = code
    grouped["Name"] = name
    grouped["StartDate"] = start_date
    grouped["EndDate"] = latest_date
    grouped["CurrentClose"] = current_close
    grouped["Lots"] = grouped["Shares"] / 1000
    grouped["AverageCost"] = grouped["CostAmount"] / grouped["Shares"].where(grouped["Shares"].ne(0))
    grouped["MarketValue"] = grouped["Shares"] * current_close
    grouped["UnrealizedPnl"] = grouped["MarketValue"] - grouped["CostAmount"]
    grouped["UnrealizedPnlPct"] = grouped["UnrealizedPnl"] / grouped["CostAmount"].where(grouped["CostAmount"].ne(0))
    grouped = grouped.replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
    ordered = [
        "Code",
        "Name",
        "StartDate",
        "EndDate",
        "CurrentClose",
        "ParticipantKey",
        "Participant",
        "BucketIndex",
        "BucketStart",
        "BucketEnd",
        "BucketLabel",
        "BinWidth",
        "BinCount",
        "Shares",
        "Lots",
        "AverageCost",
        "CostAmount",
        "MarketValue",
        "UnrealizedPnl",
        "UnrealizedPnlPct",
    ]
    return grouped[ordered]


def svg_polyline(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    segments = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    segments.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(segments)


def render_cumulative_chart(daily: pd.DataFrame) -> str:
    if daily.empty:
        return "<div class=\"empty\">沒有可畫的累積資料</div>"

    width, height = 980, 360
    left, right, top, bottom = 72, 152, 36, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    date_count = len(daily)
    series = {
        participant.key: daily[f"{participant.key}_cumulative_net_shares"].astype(float) / 1000
        for participant in PARTICIPANTS
    }
    values = [float(value) for data in series.values() for value in data]
    min_y = min(values + [0.0])
    max_y = max(values + [0.0])
    if math.isclose(min_y, max_y):
        min_y -= 1
        max_y += 1
    padding = (max_y - min_y) * 0.08
    min_y -= padding
    max_y += padding

    def x_at(index: int) -> float:
        if date_count <= 1:
            return left
        return left + index / (date_count - 1) * plot_width

    def y_at(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_height

    grid_lines = []
    for tick in range(5):
        value = min_y + (max_y - min_y) * tick / 4
        y = y_at(value)
        grid_lines.append(
            f"<line x1=\"{left}\" x2=\"{width - right}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\" class=\"grid\" />"
            f"<text x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"axis\">{html.escape(fmt_number(value, 0))}</text>"
        )

    paths = []
    label_positions: list[dict[str, Any]] = []
    for participant in PARTICIPANTS:
        points = [(x_at(index), y_at(float(value))) for index, value in enumerate(series[participant.key])]
        paths.append(
            f"<path d=\"{svg_polyline(points)}\" fill=\"none\" stroke=\"{participant.color}\" "
            "stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\" />"
        )
        last_value = float(series[participant.key].iloc[-1])
        label_positions.append(
            {
                "participant": participant,
                "value": last_value,
                "x": width - right + 12,
                "y": y_at(last_value),
            }
        )

    label_positions.sort(key=lambda item: item["y"])
    minimum_gap = 18
    for index in range(1, len(label_positions)):
        if label_positions[index]["y"] - label_positions[index - 1]["y"] < minimum_gap:
            label_positions[index]["y"] = label_positions[index - 1]["y"] + minimum_gap
    for index in range(len(label_positions) - 2, -1, -1):
        if label_positions[index + 1]["y"] > top + plot_height:
            label_positions[index + 1]["y"] = top + plot_height
        if label_positions[index + 1]["y"] - label_positions[index]["y"] < minimum_gap:
            label_positions[index]["y"] = label_positions[index + 1]["y"] - minimum_gap

    labels = []
    for item in label_positions:
        participant = item["participant"]
        labels.append(
            f"<text x=\"{item['x']:.1f}\" y=\"{item['y'] + 4:.1f}\" class=\"line-label\" "
            f"fill=\"{participant.color}\">{html.escape(participant.label)} {html.escape(fmt_number(item['value'], 0))}</text>"
        )

    dates = daily["Date"].astype(str).tolist()
    x_labels = []
    for index in sorted({0, len(dates) // 2, len(dates) - 1}):
        x = x_at(index)
        x_labels.append(
            f"<text x=\"{x:.1f}\" y=\"{height - 16}\" text-anchor=\"middle\" class=\"axis\">{html.escape(dates[index])}</text>"
        )

    zero_y = y_at(0)
    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"累積淨流量折線圖\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\" rx=\"8\" />"
        f"{''.join(grid_lines)}"
        f"<line x1=\"{left}\" x2=\"{width - right}\" y1=\"{zero_y:.1f}\" y2=\"{zero_y:.1f}\" class=\"zero\" />"
        f"{''.join(paths)}"
        f"{''.join(labels)}"
        f"{''.join(x_labels)}"
        f"<text x=\"{left}\" y=\"22\" class=\"chart-title\">累積淨流量（張）</text>"
        "</svg>"
    )


def render_cost_distribution(cost_bins: pd.DataFrame) -> str:
    if cost_bins.empty:
        return "<div class=\"empty\">區間內沒有可歸屬於仍留倉的成本籌碼</div>"

    width, height = 980, 390
    left, right, top, bottom = 72, 36, 42, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    bucket_order = (
        cost_bins[["BucketIndex", "BucketStart", "BucketEnd", "BucketLabel"]]
        .drop_duplicates()
        .sort_values("BucketIndex")
        .to_dict("records")
    )
    totals = cost_bins.groupby("BucketLabel")["Shares"].sum().to_dict()
    max_lots = max([shares / 1000 for shares in totals.values()] + [1.0])
    current_close = float(cost_bins["CurrentClose"].iloc[0])
    min_price = min(float(row["BucketStart"]) for row in bucket_order + [{"BucketStart": current_close}])
    max_price = max(float(row["BucketEnd"]) for row in bucket_order + [{"BucketEnd": current_close}])
    if math.isclose(min_price, max_price):
        min_price -= 1
        max_price += 1
    price_span = max_price - min_price

    def x_at_price(price: float) -> float:
        return left + (price - min_price) / price_span * plot_width

    def y_at(lots: float) -> float:
        return top + (1 - lots / max_lots) * plot_height

    grid = []
    for tick in range(5):
        value = max_lots * tick / 4
        y = y_at(value)
        grid.append(
            f"<line x1=\"{left}\" x2=\"{width - right}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\" class=\"grid\" />"
            f"<text x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"axis\">{html.escape(fmt_number(value, 0))}</text>"
        )

    bars = []
    label_step = max(1, math.ceil(len(bucket_order) / 8))
    for bucket_index, bucket in enumerate(bucket_order):
        x = x_at_price(float(bucket["BucketStart"]))
        x_end = x_at_price(float(bucket["BucketEnd"]))
        bar_width = max(2, x_end - x - 2)
        y_cursor = top + plot_height
        bucket_rows = cost_bins[cost_bins["BucketLabel"].eq(bucket["BucketLabel"])]
        for participant in PARTICIPANTS:
            row = bucket_rows[bucket_rows["ParticipantKey"].eq(participant.key)]
            if row.empty:
                continue
            lots = float(row["Shares"].iloc[0]) / 1000
            if lots <= 0:
                continue
            bar_height = lots / max_lots * plot_height
            y_cursor -= bar_height
            bars.append(
                f"<rect x=\"{x:.1f}\" y=\"{y_cursor:.1f}\" width=\"{bar_width:.1f}\" height=\"{bar_height:.1f}\" "
                f"fill=\"{participant.color}\" rx=\"2\"><title>{html.escape(participant.label)} "
                f"{html.escape(str(bucket['BucketLabel']))}：{html.escape(fmt_number(lots, 0))} 張</title></rect>"
            )
        if bucket_index % label_step == 0 or bucket_index == len(bucket_order) - 1:
            label = str(bucket["BucketLabel"])
            bars.append(
                f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{height - 38}\" text-anchor=\"middle\" class=\"axis\" "
                f"transform=\"rotate(-32 {x + bar_width / 2:.1f} {height - 38})\">{html.escape(label)}</text>"
            )

    current_x = x_at_price(current_close)
    current_line = (
        f"<line x1=\"{current_x:.1f}\" x2=\"{current_x:.1f}\" y1=\"{top}\" y2=\"{top + plot_height}\" "
        "stroke=\"#dc2626\" stroke-width=\"2\" stroke-dasharray=\"6 5\" />"
        f"<text x=\"{current_x + 6:.1f}\" y=\"{top + 16}\" class=\"bar-value\" fill=\"#dc2626\">"
        f"現價 {html.escape(fmt_price(current_close))}</text>"
    )

    legend = []
    legend_x = left
    for participant in PARTICIPANTS:
        legend.append(
            f"<rect x=\"{legend_x}\" y=\"{height - 20}\" width=\"12\" height=\"12\" fill=\"{participant.color}\" rx=\"2\" />"
            f"<text x=\"{legend_x + 18}\" y=\"{height - 10}\" class=\"axis\">{html.escape(participant.label)}</text>"
        )
        legend_x += 126

    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"成本分布堆疊圖\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\" rx=\"8\" />"
        f"{''.join(grid)}"
        f"{''.join(bars)}"
        f"{current_line}"
        f"{''.join(legend)}"
        f"<text x=\"{left}\" y=\"24\" class=\"chart-title\">仍留倉成本分布（張）</text>"
        "</svg>"
    )


def render_pnl_chart(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<div class=\"empty\">沒有可畫的損益資料</div>"

    width, height = 980, 280
    left, right, top, bottom = 156, 46, 42, 32
    plot_width = width - left - right
    row_height = 42
    values = {row["ParticipantKey"]: float(row["UnrealizedPnl"]) / 100_000_000 for _, row in summary.iterrows()}
    max_abs = max([abs(value) for value in values.values()] + [1.0])

    def x_at(value: float) -> float:
        return left + (value + max_abs) / (max_abs * 2) * plot_width

    zero_x = x_at(0)
    rows = []
    for index, participant in enumerate(PARTICIPANTS):
        value = values.get(participant.key, 0.0)
        y = top + index * row_height
        x0 = zero_x if value >= 0 else x_at(value)
        bar_width = abs(x_at(value) - zero_x)
        fill = "#059669" if value >= 0 else "#dc2626"
        label_x = x0 + bar_width + 8 if value >= 0 else x0 - 8
        anchor = "start" if value >= 0 else "end"
        rows.append(
            f"<text x=\"{left - 18}\" y=\"{y + 24}\" text-anchor=\"end\" class=\"bar-label\">{html.escape(participant.label)}</text>"
            f"<rect x=\"{x0:.1f}\" y=\"{y + 8}\" width=\"{bar_width:.1f}\" height=\"24\" fill=\"{fill}\" rx=\"4\" />"
            f"<text x=\"{label_x:.1f}\" y=\"{y + 25}\" text-anchor=\"{anchor}\" class=\"bar-value\">{html.escape(fmt_number(value, 2))} 億</text>"
        )

    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"未實現損益圖\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\" rx=\"8\" />"
        f"<line x1=\"{zero_x:.1f}\" x2=\"{zero_x:.1f}\" y1=\"{top - 6}\" y2=\"{height - bottom}\" class=\"zero\" />"
        f"{''.join(rows)}"
        f"<text x=\"{left}\" y=\"24\" class=\"chart-title\">仍留倉未實現損益估計（億元）</text>"
        "</svg>"
    )


def render_summary_table(summary: pd.DataFrame) -> str:
    participant_color = {participant.key: participant.color for participant in PARTICIPANTS}
    rows = []
    for _, row in summary.iterrows():
        color = participant_color.get(str(row["ParticipantKey"]), "#334155")
        rows.append(
            "<tr>"
            f"<td><span class=\"dot\" style=\"background:{color}\"></span>{html.escape(str(row['Participant']))}</td>"
            f"<td>{html.escape(fmt_lots(row['CumulativeNetShares']))}</td>"
            f"<td>{html.escape(fmt_lots(row['OpenShares']))}</td>"
            f"<td>{html.escape(fmt_price(row['AverageCost']))}</td>"
            f"<td>{html.escape(fmt_money(row['MarketValue']))}</td>"
            f"<td class=\"{'positive' if float(row['UnrealizedPnl']) >= 0 else 'negative'}\">{html.escape(fmt_money(row['UnrealizedPnl']))}</td>"
            f"<td>{html.escape(fmt_percent(row['UnrealizedPnlPct']))}</td>"
            f"<td>{html.escape(fmt_lots(row.get('InitialInventoryShares', 0)))}</td>"
            f"<td>{html.escape(str(row['PositionState']))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>群體</th><th>區間累積淨流量（張）</th><th>仍留倉（張）</th><th>平均成本</th>"
        "<th>市值估計</th><th>未實現損益</th><th>損益率</th><th>最低期初庫存（張）</th><th>解讀</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    code: str,
    name: str,
    metadata: dict[str, str],
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    cost_bins: pd.DataFrame,
    output_paths: dict[str, Path],
    report_window_label: str,
) -> str:
    if summary.empty or daily.empty:
        raise ValueError("Cannot render empty report")

    start_date = str(summary["StartDate"].iloc[0])
    end_date = str(summary["EndDate"].iloc[0])
    warmup_start_date = str(summary["WarmupStartDate"].iloc[0])
    current_close = float(summary["CurrentClose"].iloc[0])
    baseline_summary = summary[summary["MethodKey"].eq(DEFAULT_METHOD_KEY)].copy()
    if baseline_summary.empty:
        baseline_summary = summary[summary["MethodKey"].eq(str(summary["MethodKey"].iloc[0]))].copy()
    baseline_daily = daily[daily["MethodKey"].eq(DEFAULT_METHOD_KEY)].copy()
    if baseline_daily.empty:
        baseline_daily = daily[daily["MethodKey"].eq(str(daily["MethodKey"].iloc[0]))].copy()
    method_totals = (
        summary.groupby(["MethodKey", "Method"], dropna=False)
        .agg(
            OpenLots=("OpenLots", "sum"),
            MarketValue=("MarketValue", "sum"),
            UnrealizedPnl=("UnrealizedPnl", "sum"),
        )
        .reset_index()
    )
    total_open_lots = float(baseline_summary["OpenLots"].sum())
    total_market_value = float(baseline_summary["MarketValue"].sum())
    pnl_min = float(method_totals["UnrealizedPnl"].min())
    pnl_max = float(method_totals["UnrealizedPnl"].max())
    industry = str(metadata.get("產業群組", ""))

    cards = [
        ("報告區間", f"{start_date} 到 {end_date}"),
        ("暖機起點", warmup_start_date),
        ("最新收盤價", f"{fmt_price(current_close)} 元"),
        ("仍留倉合計", f"{fmt_number(total_open_lots, 0)} 張"),
        ("留倉市值估計", fmt_money(total_market_value)),
        ("損益估計區間", f"{fmt_money(pnl_min)} 到 {fmt_money(pnl_max)}"),
    ]
    card_html = "".join(
        f"<div class=\"metric\"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )

    csv_links = "".join(
        f"<li>{html.escape(label)}：<code>{html.escape(str(path.relative_to(PROJECT_ROOT)))}</code></li>"
        for label, path in output_paths.items()
    )
    method_sections = []
    for method in INVENTORY_METHODS:
        method_summary = summary[summary["MethodKey"].eq(method.key)].copy()
        method_cost_bins = cost_bins[cost_bins["MethodKey"].eq(method.key)].copy() if not cost_bins.empty else cost_bins
        if method_summary.empty:
            continue
        method_sections.append(
            "<section>"
            f"<h2>{html.escape(method.label)} 成本分布</h2>"
            f"<p class=\"note\">{html.escape(method.description)}</p>"
            f"{render_summary_table(method_summary)}"
            f"{render_cost_distribution(method_cost_bins)}"
            f"{render_pnl_chart(method_summary)}"
            "</section>"
        )
    method_sections_html = "".join(method_sections)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(code)} {html.escape(name)} {html.escape(report_window_label)}籌碼庫存與成本分布估計</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #64748b;
      --line: rgb(217, 226, 239);
      --band: #f6f8fb;
      --panel: #ffffff;
      --positive: #047857;
      --negative: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft JhengHei", "Noto Sans TC", "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef3f8;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 44px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }}
    .badge {{
      border: 1px solid var(--line);
      background: white;
      border-radius: 6px;
      padding: 8px 12px;
      color: #334155;
      white-space: nowrap;
      font-size: 14px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 15px;
      min-height: 84px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .metric strong {{
      display: block;
      font-size: 20px;
      line-height: 1.25;
    }}
    section {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-top: 14px;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 19px;
      letter-spacing: 0;
    }}
    .note {{
      color: #475569;
      font-size: 14px;
      line-height: 1.7;
      margin: 10px 0 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 9px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      color: #475569;
      background: var(--band);
      font-weight: 600;
    }}
    th:first-child, td:first-child, th:last-child, td:last-child {{
      text-align: left;
    }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: 0;
    }}
    .positive {{ color: var(--positive); font-weight: 700; }}
    .negative {{ color: var(--negative); font-weight: 700; }}
    .chart {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    .chart-title {{
      font-size: 17px;
      font-weight: 700;
      fill: #0f172a;
    }}
    .axis {{
      font-size: 12px;
      fill: #64748b;
    }}
    .grid {{
      stroke: #e5eaf2;
      stroke-width: 1;
    }}
    .zero {{
      stroke: #94a3b8;
      stroke-width: 1.2;
      stroke-dasharray: 4 5;
    }}
    .line-label {{
      font-size: 13px;
      font-weight: 700;
    }}
    .bar-label {{
      font-size: 14px;
      fill: #334155;
      font-weight: 700;
    }}
    .bar-value {{
      font-size: 13px;
      fill: #334155;
      font-weight: 700;
    }}
    .empty {{
      padding: 36px;
      text-align: center;
      color: var(--muted);
      background: var(--band);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    code {{
      color: #334155;
      background: #f1f5f9;
      border-radius: 4px;
      padding: 2px 5px;
      white-space: normal;
      word-break: break-all;
    }}
    ul {{
      margin: 6px 0 0;
      padding-left: 20px;
      color: #475569;
      line-height: 1.8;
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      main {{ padding: 18px 12px 30px; }}
      header {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      section {{ padding: 12px; overflow-x: auto; }}
      h1 {{ font-size: 24px; }}
      table {{ min-width: 980px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{html.escape(code)} {html.escape(name)} {html.escape(report_window_label)}籌碼庫存與成本分布估計</h1>
        <p class="subtitle">模型先從最早可用的法人與價格共同日期暖機，再展示報告區間內的籌碼狀態；若完整歷史仍出現賣超缺口，會反推最低期初庫存補足，避免負庫存。成本採當日成交均價，最低期初庫存成本暫以暖機起點成交均價估計，並同時計算 FIFO、FILO/LIFO、平均成本法三種沖銷假設。這不是官方持股資料。</p>
      </div>
      <div class="badge">{html.escape(industry or "未分類")}</div>
    </header>

    <div class="metrics">{card_html}</div>

    <section>
      <h2>累積淨流量</h2>
      {render_cumulative_chart(baseline_daily)}
      <p class="note">此圖顯示報告區間內的累積淨流量；暖機歷史與最低期初庫存會影響留倉成本與分布，但不改變報告區間每日淨流量本身。</p>
    </section>

    {method_sections_html}

    <section>
      <h2>輸出檔案</h2>
      <ul>{csv_links}</ul>
    </section>
  </main>
</body>
</html>
"""


def participant_market_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    grouped = (
        summary.groupby(["MethodKey", "Method", "ParticipantKey", "Participant"], dropna=False)
        .agg(
            StockCount=("Code", "nunique"),
            CumulativeNetShares=("CumulativeNetShares", "sum"),
            PositiveNetFlowShares=("PositiveNetFlowShares", "sum"),
            NegativeNetFlowShares=("NegativeNetFlowShares", "sum"),
            InitialInventoryShares=("InitialInventoryShares", "sum"),
            ResidualSellGapShares=("ResidualSellGapShares", "sum"),
            OpenShares=("OpenShares", "sum"),
            OpenCostAmount=("OpenCostAmount", "sum"),
            MarketValue=("MarketValue", "sum"),
            UnrealizedPnl=("UnrealizedPnl", "sum"),
            RealizedPnlOnWindowLots=("RealizedPnlOnWindowLots", "sum"),
        )
        .reset_index()
    )
    grouped["CumulativeNetLots"] = grouped["CumulativeNetShares"] / 1000
    grouped["OpenLots"] = grouped["OpenShares"] / 1000
    grouped["InitialInventoryLots"] = grouped["InitialInventoryShares"] / 1000
    grouped["ResidualSellGapLots"] = grouped["ResidualSellGapShares"] / 1000
    grouped["AverageCost"] = grouped["OpenCostAmount"] / grouped["OpenShares"].where(grouped["OpenShares"].ne(0))
    grouped["UnrealizedPnlPct"] = grouped["UnrealizedPnl"] / grouped["OpenCostAmount"].where(grouped["OpenCostAmount"].ne(0))
    grouped = grouped.replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
    method_order = {method.key: index for index, method in enumerate(INVENTORY_METHODS)}
    order = {participant.key: index for index, participant in enumerate(PARTICIPANTS)}
    grouped["MethodOrder"] = grouped["MethodKey"].map(method_order).fillna(99)
    grouped["ParticipantOrder"] = grouped["ParticipantKey"].map(order).fillna(99)
    return grouped.sort_values(["MethodOrder", "ParticipantOrder"]).drop(columns=["MethodOrder", "ParticipantOrder"])


def build_stock_summary(summary: pd.DataFrame, page_href_by_code: dict[str, str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (code, name, method_key, method_label), group in summary.groupby(
        ["Code", "Name", "MethodKey", "Method"], sort=True, dropna=False
    ):
        group = group.copy()
        dominant = group.sort_values("MarketValue", ascending=False).iloc[0]
        row: dict[str, Any] = {
            "Code": str(code),
            "Name": str(name),
            "MethodKey": str(method_key),
            "Method": str(method_label),
            "產業群組": str(group["產業群組"].iloc[0]) if "產業群組" in group.columns else "",
            "WarmupStartDate": str(group["WarmupStartDate"].iloc[0]) if "WarmupStartDate" in group.columns else "",
            "StartDate": str(group["StartDate"].iloc[0]),
            "EndDate": str(group["EndDate"].iloc[0]),
            "CurrentClose": float(group["CurrentClose"].iloc[0]),
            "DominantParticipantKey": str(dominant["ParticipantKey"]),
            "DominantParticipant": str(dominant["Participant"]),
            "TotalCumulativeNetLots": float(group["CumulativeNetLots"].sum()),
            "TotalInitialInventoryLots": float(group["InitialInventoryLots"].sum()),
            "TotalOpenLots": float(group["OpenLots"].sum()),
            "TotalMarketValue": float(group["MarketValue"].sum()),
            "TotalUnrealizedPnl": float(group["UnrealizedPnl"].sum()),
            "TotalOpenCostAmount": float(group["OpenCostAmount"].sum()),
            "TotalResidualSellGapLots": float(group["ResidualSellGapShares"].sum()) / 1000,
            "PageHref": page_href_by_code.get(str(code), ""),
        }
        row["TotalUnrealizedPnlPct"] = (
            row["TotalUnrealizedPnl"] / row["TotalOpenCostAmount"] if row["TotalOpenCostAmount"] else 0.0
        )
        for participant in PARTICIPANTS:
            part = group[group["ParticipantKey"].eq(participant.key)]
            prefix = participant.key.capitalize()
            if part.empty:
                row[f"{prefix}CumulativeNetLots"] = 0.0
                row[f"{prefix}OpenLots"] = 0.0
                row[f"{prefix}UnrealizedPnl"] = 0.0
                row[f"{prefix}AverageCost"] = 0.0
                row[f"{prefix}InitialInventoryLots"] = 0.0
                row[f"{prefix}ResidualSellGapLots"] = 0.0
                continue
            item = part.iloc[0]
            row[f"{prefix}CumulativeNetLots"] = float(item["CumulativeNetLots"])
            row[f"{prefix}OpenLots"] = float(item["OpenLots"])
            row[f"{prefix}UnrealizedPnl"] = float(item["UnrealizedPnl"])
            row[f"{prefix}AverageCost"] = float(item["AverageCost"])
            row[f"{prefix}InitialInventoryLots"] = float(item["InitialInventoryLots"])
            row[f"{prefix}ResidualSellGapLots"] = float(item["ResidualSellGapShares"]) / 1000
        rows.append(row)
    method_order = {method.key: index for index, method in enumerate(INVENTORY_METHODS)}
    result = pd.DataFrame(rows)
    result["MethodOrder"] = result["MethodKey"].map(method_order).fillna(99)
    return result.sort_values(["Code", "MethodOrder"]).drop(columns=["MethodOrder"]).reset_index(drop=True)


def stock_anchor(row: pd.Series) -> str:
    label = f"{row['Code']} {row['Name']}"
    href = str(row.get("PageHref", ""))
    if href:
        return f"<a href=\"{html.escape(href)}\">{html.escape(label)}</a>"
    return html.escape(label)


def render_stock_rank_table(frame: pd.DataFrame, title: str, sort_column: str, ascending: bool = False) -> str:
    if frame.empty:
        return ""
    chunk = frame.sort_values(sort_column, ascending=ascending).head(30)
    rows = []
    for _, row in chunk.iterrows():
        value = float(row[sort_column])
        if "Pnl" in sort_column or "Value" in sort_column:
            formatted_value = fmt_money(value)
        elif "Cost" in sort_column or "Close" in sort_column:
            formatted_value = fmt_price(value)
        else:
            formatted_value = f"{fmt_number(value, 0)} 張"
        rows.append(
            "<tr>"
            f"<td>{stock_anchor(row)}</td>"
            f"<td>{html.escape(str(row.get('產業群組', '')))}</td>"
            f"<td>{html.escape(str(row.get('DominantParticipant', '')))}</td>"
            f"<td>{html.escape(fmt_number(row.get('TotalOpenLots', 0), 0))}</td>"
            f"<td class=\"{'positive' if value >= 0 else 'negative'}\">{html.escape(formatted_value)}</td>"
            "</tr>"
        )
    return (
        "<section>"
        f"<h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>股票</th><th>產業</th><th>主要留倉群體</th><th>總留倉（張）</th><th>排序值</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</section>"
    )


def render_market_group_table(participant_summary: pd.DataFrame) -> str:
    rows = []
    for _, row in participant_summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['Method']))}</td>"
            f"<td>{html.escape(str(row['Participant']))}</td>"
            f"<td>{html.escape(fmt_number(row['StockCount'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['CumulativeNetLots'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['OpenLots'], 0))}</td>"
            f"<td>{html.escape(fmt_price(row['AverageCost']))}</td>"
            f"<td>{html.escape(fmt_money(row['MarketValue']))}</td>"
            f"<td class=\"{'positive' if float(row['UnrealizedPnl']) >= 0 else 'negative'}\">{html.escape(fmt_money(row['UnrealizedPnl']))}</td>"
            f"<td>{html.escape(fmt_percent(row['UnrealizedPnlPct']))}</td>"
            f"<td>{html.escape(fmt_number(row['InitialInventoryLots'], 0))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>方法</th><th>群體</th><th>股票數</th><th>累積淨流量（張）</th><th>仍留倉（張）</th><th>平均成本</th>"
        "<th>留倉市值</th><th>未實現損益</th><th>損益率</th><th>最低期初庫存（張）</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_method_histogram(
    participant_summary: pd.DataFrame,
    title: str,
    value_column: str,
    scale: float,
    value_kind: str,
    axis_unit: str,
) -> str:
    if participant_summary.empty:
        return ""

    method_labels = {method.key: method.label for method in INVENTORY_METHODS}
    participant_labels = {participant.key: participant.label for participant in PARTICIPANTS}
    values: dict[tuple[str, str], float] = {}
    for _, row in participant_summary.iterrows():
        participant_key = str(row["ParticipantKey"])
        method_key = str(row["MethodKey"])
        values[(participant_key, method_key)] = float(row.get(value_column, 0.0) or 0.0) * scale

    ordered_values = [
        values.get((participant.key, method.key), 0.0)
        for participant in PARTICIPANTS
        for method in INVENTORY_METHODS
    ]
    min_value = min(ordered_values + [0.0])
    max_value = max(ordered_values + [0.0])
    if math.isclose(min_value, max_value):
        padding = abs(max_value) * 0.2 or 1.0
        min_value -= padding
        max_value += padding
    else:
        padding = (max_value - min_value) * 0.1
        min_value -= padding
        max_value += padding
    if min_value > 0:
        min_value = 0.0
    if max_value < 0:
        max_value = 0.0

    def format_hist_value(value: float) -> str:
        if value_kind == "price":
            return fmt_price(value)
        if value_kind == "percent":
            return f"{fmt_number(value, 1)}%"
        return fmt_number(value, 1)

    width, height = 980, 360
    left, right, top, bottom = 82, 30, 48, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    span = max_value - min_value

    def y_at(value: float) -> float:
        return top + (max_value - value) / span * plot_height

    grid = []
    for index in range(5):
        value = min_value + span * index / 4
        y = y_at(value)
        grid.append(
            f"<line x1=\"{left}\" x2=\"{width - right}\" y1=\"{y:.1f}\" y2=\"{y:.1f}\" class=\"grid\" />"
            f"<text x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"axis\">"
            f"{html.escape(format_hist_value(value))}</text>"
        )

    zero_y = y_at(0.0)
    group_width = plot_width / len(PARTICIPANTS)
    bar_gap = 6
    bar_width = min(42, (group_width - 36) / len(INVENTORY_METHODS) - bar_gap)
    total_bar_width = bar_width * len(INVENTORY_METHODS) + bar_gap * (len(INVENTORY_METHODS) - 1)
    bars = []
    for participant_index, participant in enumerate(PARTICIPANTS):
        group_center = left + group_width * participant_index + group_width / 2
        x_start = group_center - total_bar_width / 2
        bars.append(
            f"<text x=\"{group_center:.1f}\" y=\"{height - 42}\" text-anchor=\"middle\" class=\"axis\">"
            f"{html.escape(participant.label)}</text>"
        )
        for method_index, method in enumerate(INVENTORY_METHODS):
            value = values.get((participant.key, method.key), 0.0)
            x = x_start + method_index * (bar_width + bar_gap)
            y_value = y_at(value)
            if value >= 0:
                y = y_value
                bar_height = max(1.5, zero_y - y_value)
                label_y = max(top + 12, y - 7)
                label_class = "positive" if value_kind != "price" else "neutral"
            else:
                y = zero_y
                bar_height = max(1.5, y_value - zero_y)
                label_y = min(top + plot_height - 4, y + bar_height + 14)
                label_class = "negative"
            color = METHOD_COLORS.get(method.key, "#334155")
            bars.append(
                f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width:.1f}\" height=\"{bar_height:.1f}\" "
                f"fill=\"{color}\" rx=\"3\"><title>{html.escape(participant_labels[participant.key])} "
                f"{html.escape(method_labels[method.key])}：{html.escape(format_hist_value(value))} "
                f"{html.escape(axis_unit)}</title></rect>"
                f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{label_y:.1f}\" text-anchor=\"middle\" "
                f"class=\"bar-value {label_class}\">{html.escape(format_hist_value(value))}</text>"
            )

    legend = []
    legend_x = left
    for method in INVENTORY_METHODS:
        color = METHOD_COLORS.get(method.key, "#334155")
        legend.append(
            f"<rect x=\"{legend_x}\" y=\"{height - 22}\" width=\"12\" height=\"12\" fill=\"{color}\" rx=\"2\" />"
            f"<text x=\"{legend_x + 18}\" y=\"{height - 12}\" class=\"axis\">{html.escape(method.label)}</text>"
        )
        legend_x += 142

    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)}\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"white\" rx=\"8\" />"
        f"{''.join(grid)}"
        f"<line x1=\"{left}\" x2=\"{width - right}\" y1=\"{zero_y:.1f}\" y2=\"{zero_y:.1f}\" class=\"zero\" />"
        f"{''.join(bars)}"
        f"{''.join(legend)}"
        f"<text x=\"{left}\" y=\"26\" class=\"chart-title\">{html.escape(title)}</text>"
        "</svg>"
    )


def render_market_method_histograms(participant_summary: pd.DataFrame) -> str:
    return (
        "<div class=\"chart-grid\">"
        f"{render_method_histogram(participant_summary, '方法差異 Histogram：未實現損益（億元）', 'UnrealizedPnl', 1 / 100_000_000, 'money', '億元')}"
        f"{render_method_histogram(participant_summary, '方法差異 Histogram：損益率', 'UnrealizedPnlPct', 100, 'percent', '')}"
        f"{render_method_histogram(participant_summary, '方法差異 Histogram：平均成本', 'AverageCost', 1, 'price', '元')}"
        "</div>"
    )


def render_all_stock_rows(stock_summary: pd.DataFrame) -> str:
    rows = []
    for _, row in stock_summary.sort_values(["Code"]).iterrows():
        search_text = (
            f"{row['Code']} {row['Name']} {row.get('產業群組', '')} {row.get('DominantParticipant', '')}"
        )
        rows.append(
            f"<tr data-search=\"{html.escape(search_text.lower())}\">"
            f"<td>{stock_anchor(row)}</td>"
            f"<td>{html.escape(str(row.get('產業群組', '')))}</td>"
            f"<td>{html.escape(str(row['EndDate']))}</td>"
            f"<td>{html.escape(fmt_price(row['CurrentClose']))}</td>"
            f"<td>{html.escape(str(row['DominantParticipant']))}</td>"
            f"<td>{html.escape(fmt_number(row['TotalOpenLots'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['TotalInitialInventoryLots'], 0))}</td>"
            f"<td class=\"{'positive' if float(row['TotalUnrealizedPnl']) >= 0 else 'negative'}\">{html.escape(fmt_money(row['TotalUnrealizedPnl']))}</td>"
            f"<td>{html.escape(fmt_number(row['ForeignCumulativeNetLots'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['TrustCumulativeNetLots'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['DealerCumulativeNetLots'], 0))}</td>"
            f"<td>{html.escape(fmt_number(row['OtherCumulativeNetLots'], 0))}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_market_index_html(
    stock_summary: pd.DataFrame,
    participant_summary: pd.DataFrame,
    skipped: pd.DataFrame,
    csv_paths: dict[str, Path],
    report_window_label: str,
) -> str:
    stock_count = int(stock_summary["Code"].nunique()) if not stock_summary.empty else 0
    method_count = int(stock_summary["MethodKey"].nunique()) if not stock_summary.empty else 0
    skipped_count = int(len(skipped))
    warmup_start_date = str(stock_summary["WarmupStartDate"].min()) if not stock_summary.empty else ""
    start_date = str(stock_summary["StartDate"].min()) if not stock_summary.empty else ""
    end_date = str(stock_summary["EndDate"].max()) if not stock_summary.empty else ""
    baseline = stock_summary[stock_summary["MethodKey"].eq(DEFAULT_METHOD_KEY)].copy() if not stock_summary.empty else stock_summary
    if baseline.empty and not stock_summary.empty:
        baseline = stock_summary[stock_summary["MethodKey"].eq(str(stock_summary["MethodKey"].iloc[0]))].copy()
    fifo_stock_summary = baseline
    method_totals = (
        stock_summary.groupby(["MethodKey", "Method"], dropna=False)
        .agg(TotalUnrealizedPnl=("TotalUnrealizedPnl", "sum"))
        .reset_index()
        if not stock_summary.empty
        else pd.DataFrame()
    )
    total_open_lots = float(baseline["TotalOpenLots"].sum()) if not baseline.empty else 0.0
    total_initial_lots = float(baseline["TotalInitialInventoryLots"].sum()) if not baseline.empty else 0.0
    total_market_value = float(baseline["TotalMarketValue"].sum()) if not baseline.empty else 0.0
    pnl_min = float(method_totals["TotalUnrealizedPnl"].min()) if not method_totals.empty else 0.0
    pnl_max = float(method_totals["TotalUnrealizedPnl"].max()) if not method_totals.empty else 0.0
    csv_links = "".join(
        f"<li>{html.escape(label)}：<code>{html.escape(str(path.relative_to(PROJECT_ROOT)))}</code></li>"
        for label, path in csv_paths.items()
    )
    cards = [
        ("完成股票數", f"{stock_count:,} 檔"),
        ("成本方法數", f"{method_count:,} 種"),
        ("暖機起點", warmup_start_date),
        ("報告區間", f"{start_date} 到 {end_date}"),
        ("FIFO 留倉合計", f"{fmt_number(total_open_lots, 0)} 張"),
        ("FIFO 最低期初庫存", f"{fmt_number(total_initial_lots, 0)} 張"),
        ("FIFO 留倉市值", fmt_money(total_market_value)),
        ("三法損益估計區間", f"{fmt_money(pnl_min)} 到 {fmt_money(pnl_max)}"),
        ("跳過股票數", f"{skipped_count:,} 檔"),
    ]
    card_html = "".join(
        f"<div class=\"metric\"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>全上市普通股{html.escape(report_window_label)}籌碼庫存與成本分布估計</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #64748b;
      --line: rgb(217, 226, 239);
      --band: #f6f8fb;
      --positive: #047857;
      --negative: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft JhengHei", "Noto Sans TC", "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef3f8;
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 28px 24px 46px; }}
    header {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: end; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; line-height: 1.25; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 19px; letter-spacing: 0; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 15px; line-height: 1.7; }}
    .badge {{ border: 1px solid var(--line); background: white; border-radius: 6px; padding: 8px 12px; color: #334155; white-space: nowrap; font-size: 14px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid var(--line); border-radius: 8px; padding: 14px 15px; min-height: 82px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 10px; }}
    .metric strong {{ display: block; font-size: 20px; line-height: 1.25; }}
    section {{ background: white; border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin-top: 14px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: right; vertical-align: middle; white-space: nowrap; }}
    th {{ color: #475569; background: var(--band); font-weight: 600; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .positive {{ color: var(--positive); font-weight: 700; }}
    .negative {{ color: var(--negative); font-weight: 700; }}
    .chart .positive {{ fill: var(--positive); color: var(--positive); }}
    .chart .negative {{ fill: var(--negative); color: var(--negative); }}
    .chart .neutral {{ fill: #334155; }}
    .note {{ color: #475569; font-size: 14px; line-height: 1.7; margin: 10px 0 0; }}
    .chart {{ display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: white; margin-top: 10px; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin: 12px 0 14px; }}
    .chart-title {{ font-size: 17px; font-weight: 700; fill: #0f172a; }}
    .zero {{ stroke: #94a3b8; stroke-width: 1.2; stroke-dasharray: 4 5; }}
    .bar-label {{ font-size: 14px; fill: #334155; font-weight: 700; }}
    .bar-value {{ font-size: 13px; fill: #334155; font-weight: 700; }}
    .filter {{ width: 100%; max-width: 420px; border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font-size: 15px; margin: 0 0 12px; }}
    code {{ color: #334155; background: #f1f5f9; border-radius: 4px; padding: 2px 5px; white-space: normal; word-break: break-all; }}
    ul {{ margin: 6px 0 0; padding-left: 20px; color: #475569; line-height: 1.8; font-size: 14px; }}
    @media (max-width: 980px) {{
      main {{ padding: 18px 12px 30px; }}
      header {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      h1 {{ font-size: 24px; }}
      table {{ min-width: 1180px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>全上市普通股{html.escape(report_window_label)}籌碼庫存與成本分布估計</h1>
        <p class="subtitle">以 `data/metadata.csv` 的上市普通股為宇宙，逐檔從最早可用的法人與價格共同日期暖機，再展示報告區間內的籌碼狀態；若完整歷史仍出現賣超缺口，會反推最低期初庫存補足，避免負庫存。成本採當日成交均價，最低期初庫存成本暫以暖機起點成交均價估計，並同時計算 FIFO、FILO/LIFO、平均成本法三種成本分布與未實現損益。這是流量推估模型，不是官方股權庫存。</p>
      </div>
      <div class="badge">TWSE 上市普通股</div>
    </header>
    <div class="metrics">{card_html}</div>
    <section>
      <h2>四群體全市場摘要</h2>
      {render_market_method_histograms(participant_summary)}
      {render_market_group_table(participant_summary)}
      <p class="note">「其他市場參與者」是扣除三大法人後的殘差流量；最低期初庫存是用完整暖機歷史的累積淨流量最低點反推，代表為了讓每日沖銷過程不出現負庫存，模型至少需要假設期初已存在的庫存。三種方法的留倉張數通常相同，差異主要落在成本分布與未實現損益。</p>
    </section>
    {render_stock_rank_table(fifo_stock_summary, "未實現獲利估計前三十名（FIFO）", "TotalUnrealizedPnl", False)}
    {render_stock_rank_table(fifo_stock_summary, "未實現虧損估計前三十名（FIFO）", "TotalUnrealizedPnl", True)}
    {render_stock_rank_table(fifo_stock_summary, "外資累積淨買前三十名（FIFO）", "ForeignCumulativeNetLots", False)}
    {render_stock_rank_table(fifo_stock_summary, "外資累積淨賣前三十名（FIFO）", "ForeignCumulativeNetLots", True)}
    <section>
      <h2>全股票清單（FIFO）</h2>
      <input id="stockFilter" class="filter" type="search" placeholder="輸入代號、公司簡稱、產業或群體">
      <table>
        <thead><tr>
          <th>股票</th><th>產業</th><th>最新日期</th><th>收盤價</th><th>主要留倉群體</th>
          <th>總留倉（張）</th><th>最低期初庫存（張）</th><th>未實現損益</th><th>外資淨流量（張）</th><th>投信淨流量（張）</th><th>自營商淨流量（張）</th><th>其他淨流量（張）</th>
        </tr></thead>
        <tbody id="stockRows">{render_all_stock_rows(fifo_stock_summary)}</tbody>
      </table>
    </section>
    <section>
      <h2>輸出檔案</h2>
      <ul>{csv_links}</ul>
    </section>
  </main>
  <script>
    const filter = document.getElementById('stockFilter');
    const rows = Array.from(document.querySelectorAll('#stockRows tr'));
    filter.addEventListener('input', () => {{
      const term = filter.value.trim().toLowerCase();
      for (const row of rows) {{
        row.style.display = !term || row.dataset.search.includes(term) ? '' : 'none';
      }}
    }});
  </script>
</body>
</html>
"""


def write_outputs(
    code: str,
    name: str,
    metadata: dict[str, str],
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    open_lots: pd.DataFrame,
    cost_bins: pd.DataFrame,
    report_window_label: str,
) -> tuple[Path, dict[str, Path]]:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    file_base = safe_filename_part(f"{code}_chip_inventory")
    csv_paths = {
        "群體摘要": OUTPUT_ROOT / f"{file_base}_summary.csv",
        "每日追蹤": OUTPUT_ROOT / f"{file_base}_daily.csv",
        "開放批次": OUTPUT_ROOT / f"{file_base}_open_lots.csv",
        "成本分布": OUTPUT_ROOT / f"{file_base}_cost_bins.csv",
    }
    summary.to_csv(csv_paths["群體摘要"], index=False, encoding="utf-8-sig")
    daily.to_csv(csv_paths["每日追蹤"], index=False, encoding="utf-8-sig")
    open_lots.to_csv(csv_paths["開放批次"], index=False, encoding="utf-8-sig")
    cost_bins.to_csv(csv_paths["成本分布"], index=False, encoding="utf-8-sig")

    html_path = DATA_VIZ_ROOT / f"{file_base}.html"
    html_path.write_text(
        render_html(code, name, metadata, summary, daily, cost_bins, csv_paths, report_window_label),
        encoding="utf-8",
    )
    return html_path, csv_paths


def write_stock_page(
    code: str,
    name: str,
    metadata: dict[str, str],
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    cost_bins: pd.DataFrame,
    output_paths: dict[str, Path],
    report_window_label: str,
) -> Path:
    STOCK_PAGES_ROOT.mkdir(parents=True, exist_ok=True)
    html_path = STOCK_PAGES_ROOT / f"{safe_filename_part(code)}_chip_inventory.html"
    html_path.write_text(
        render_html(code, name, metadata, summary, daily, cost_bins, output_paths, report_window_label),
        encoding="utf-8",
    )
    return html_path


def run_single_stock(args: argparse.Namespace) -> None:
    code = str(args.code).strip()
    metadata = listed_common_metadata(code)
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    if code not in price_paths:
        raise FileNotFoundError(f"Missing price CSV for {code}")
    if code not in institutional_paths:
        raise FileNotFoundError(f"Missing institutional CSV for {code}")

    name = str(metadata.get("Name") or stock_name_from_path(price_paths[code], code))
    metrics = prepare_metrics(
        price_paths[code],
        institutional_paths[code],
        args.start_date,
        args.end_date,
        args.years,
    )
    if metrics.empty:
        raise ValueError(f"No overlapping price and institutional rows for {code}")

    summary, daily, open_lots, cost_bins = estimate_inventory(code, name, metrics, args.bin_width)
    html_path, csv_paths = write_outputs(
        code,
        name,
        metadata,
        summary,
        daily,
        open_lots,
        cost_bins,
        window_label(args.years),
    )
    print(f"report={html_path}")
    for label, path in csv_paths.items():
        print(f"{label}={path}")
    print(f"rows={len(daily)} start={daily['Date'].iloc[0]} end={daily['Date'].iloc[-1]}")


def run_all_stocks(args: argparse.Namespace) -> None:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    metadata = listed_common_metadata_frame(args.limit)
    label = window_label(args.years)
    slug = window_slug(args.years)
    csv_paths = {
        "全市場群體摘要": OUTPUT_ROOT / f"all_stock_{slug}_participant_summary.csv",
        "全市場股票摘要": OUTPUT_ROOT / f"all_stock_{slug}_stock_summary.csv",
        "全市場成本分布": OUTPUT_ROOT / f"all_stock_{slug}_cost_bins.csv",
        "跳過清單": OUTPUT_ROOT / f"all_stock_{slug}_skipped.csv",
    }

    all_summary: list[pd.DataFrame] = []
    all_cost_bins: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, str]] = []
    page_href_by_code: dict[str, str] = {}
    total = len(metadata)

    for index, row in metadata.iterrows():
        code = str(row["Code"])
        name = str(row.get("Name") or "")
        if code not in price_paths:
            skipped_rows.append({"Code": code, "Name": name, "Reason": "missing_price_csv"})
            continue
        if code not in institutional_paths:
            skipped_rows.append({"Code": code, "Name": name, "Reason": "missing_institutional_csv"})
            continue
        try:
            metrics = prepare_metrics(
                price_paths[code],
                institutional_paths[code],
                args.start_date,
                args.end_date,
                args.years,
            )
            if metrics.empty:
                skipped_rows.append({"Code": code, "Name": name, "Reason": "no_overlapping_rows"})
                continue
            stock_summary, daily, _open_lots, cost_bins = estimate_inventory(code, name, metrics, args.bin_width)
            stock_summary["產業群組"] = str(row.get("產業群組", ""))
            if not cost_bins.empty:
                cost_bins["產業群組"] = str(row.get("產業群組", ""))
                all_cost_bins.append(cost_bins)
            all_summary.append(stock_summary)
            if not args.skip_stock_pages:
                page_path = write_stock_page(
                    code,
                    name,
                    row.to_dict(),
                    stock_summary,
                    daily,
                    cost_bins,
                    csv_paths,
                    label,
                )
                page_href_by_code[code] = str(page_path.relative_to(DATA_VIZ_ROOT)).replace("\\", "/")
        except Exception as exc:  # noqa: BLE001 - batch report should continue and preserve skip reason.
            skipped_rows.append({"Code": code, "Name": name, "Reason": f"error:{type(exc).__name__}:{exc}"})
            continue

        processed = len(all_summary)
        if processed % 50 == 0 or index + 1 == total:
            print(f"processed={processed} scanned={index + 1}/{total} skipped={len(skipped_rows)}")

    participant_rows = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    cost_bins_all = pd.concat(all_cost_bins, ignore_index=True) if all_cost_bins else pd.DataFrame()
    skipped = pd.DataFrame(skipped_rows, columns=["Code", "Name", "Reason"])
    stock_summary = build_stock_summary(participant_rows, page_href_by_code)
    market_summary = participant_market_summary(participant_rows)

    market_summary.to_csv(csv_paths["全市場群體摘要"], index=False, encoding="utf-8-sig")
    stock_summary.to_csv(csv_paths["全市場股票摘要"], index=False, encoding="utf-8-sig")
    cost_bins_all.to_csv(csv_paths["全市場成本分布"], index=False, encoding="utf-8-sig")
    skipped.to_csv(csv_paths["跳過清單"], index=False, encoding="utf-8-sig")

    index_path = DATA_VIZ_ROOT / "index.html"
    index_path.write_text(
        render_market_index_html(stock_summary, market_summary, skipped, csv_paths, label),
        encoding="utf-8",
    )
    print(f"index={index_path}")
    for output_label, path in csv_paths.items():
        print(f"{output_label}={path}")
    print(f"completed_stocks={stock_summary['Code'].nunique() if not stock_summary.empty else 0} skipped={len(skipped)}")


def main() -> None:
    args = parse_args()
    if args.all:
        run_all_stocks(args)
    else:
        run_single_stock(args)


if __name__ == "__main__":
    main()
