"""Rank stocks by current margin-regime state for a target trading date."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import csv_columns_canonical, read_csv_canonical  # noqa: E402

DEFAULT_TARGET_DATE = "2026-06-22"
BY_STOCK_OUTPUT = PROJECT_ROOT / "output" / "margin_patterns" / "by_stock" / "all_stock_regime_summary.csv"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "margin_patterns"
VIZ_ROOT = PROJECT_ROOT / "data_viz" / "margin_patterns"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"


MARGIN_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "MarginCurrentBalance",
    "MarginBalance20DayChangeRate",
    "MarginFinancingUsageRate",
    "MarginMarketValue",
    "MarginMarketValueTo20DayAvgTurnover",
    "ShortCurrentBalance",
    "ShortMarginBalanceRatio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank stocks by current margin state.")
    parser.add_argument("--date", default=DEFAULT_TARGET_DATE)
    parser.add_argument("--summary", type=Path, default=BY_STOCK_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--viz-root", type=Path, default=VIZ_ROOT)
    return parser.parse_args()


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{digits}f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{digits}f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.{digits}f}K"
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def find_stock_file(folder: Path, code: str) -> Path | None:
    matches = sorted(folder.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def load_margin_state(path: Path, target_date: str) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = csv_columns_canonical(path)
    usecols = [column for column in MARGIN_COLUMNS if column in columns]
    if not {"Date", "MarginCurrentBalance"}.issubset(usecols):
        raise ValueError("margin_missing_required_columns")
    df = read_csv_canonical(path, usecols=usecols, dtype={"Code": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for column in df.columns:
        if column not in {"Date", "Code", "Name"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    target_ts = pd.Timestamp(target_date)
    current = df[df["Date"].eq(target_ts)]
    if current.empty:
        raise ValueError("missing_target_margin_date")
    current_row = current.iloc[-1].copy()
    if SIGNAL_COLUMN not in df.columns:
        df[SIGNAL_COLUMN] = np.nan
    computed_change = np.nan
    if current_row.name >= 20:
        previous_balance = df.loc[current_row.name - 20, "MarginCurrentBalance"]
        current_balance = current_row["MarginCurrentBalance"]
        if pd.notna(previous_balance) and pd.notna(current_balance) and float(previous_balance) != 0:
            computed_change = float(current_balance) / float(previous_balance) - 1
    if json_float(current_row.get(SIGNAL_COLUMN)) is None:
        current_row[SIGNAL_COLUMN] = computed_change
    history = df[df["Date"].le(target_ts)].copy()
    return current_row.to_dict(), history


def load_price_state(path: Path, target_date: str) -> dict[str, Any]:
    columns = csv_columns_canonical(path)
    close_column = "close_adj" if "close_adj" in columns else "Close"
    usecols = [column for column in ["Date", "Close", "close_adj", "Capacity", "Turnover"] if column in columns]
    if "Date" not in usecols or close_column not in usecols:
        raise ValueError("price_missing_required_columns")
    df = read_csv_canonical(path, usecols=usecols)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    for column in df.columns:
        if column != "Date":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    target_ts = pd.Timestamp(target_date)
    current = df[df["Date"].eq(target_ts)]
    if current.empty:
        raise ValueError("missing_target_price_date")
    idx = int(current.index[-1])
    row = df.loc[idx].copy()
    adjusted = row[close_column]
    row["PriceReturn20D"] = adjusted / df.loc[idx - 20, close_column] - 1 if idx >= 20 and df.loc[idx - 20, close_column] else np.nan
    row["PriceReturn60D"] = adjusted / df.loc[idx - 60, close_column] - 1 if idx >= 60 and df.loc[idx - 60, close_column] else np.nan
    row["AdjustedClose"] = adjusted
    return row.to_dict()


def margin_percentile(history: pd.DataFrame, current_balance: float) -> float | None:
    values = pd.to_numeric(history["MarginCurrentBalance"], errors="coerce").dropna()
    if values.empty or pd.isna(current_balance):
        return None
    return float(values.le(float(current_balance)).mean())


def state_labels(row: dict[str, Any]) -> str:
    labels = []
    if row["IsMarginSurge"]:
        labels.append("融資大漲")
    if row["IsMarginDrop"]:
        labels.append("融資大跌")
    if row["IsMarginHighLevel"]:
        labels.append("高水位")
    if row["IsMarginLowLevel"]:
        labels.append("低水位")
    if row["IsMarginSurgeHigh"]:
        labels.append("大漲且高水位")
    return " / ".join(labels) if labels else "一般"


def scores(row: dict[str, Any]) -> tuple[float, float]:
    change = row.get("MarginBalance20DayChangeRate")
    top = row.get("MarginSurgeThreshold")
    bottom = row.get("MarginDropThreshold")
    percentile = row.get("MarginBalancePercentile")
    price20 = row.get("PriceReturn20D")
    short_ratio = row.get("ShortMarginBalanceRatio")
    pressure = 0.0
    if row["IsMarginSurgeHigh"]:
        pressure += 100
    if row["IsMarginHighLevel"]:
        pressure += 45
    if row["IsMarginSurge"]:
        pressure += 35
    if percentile is not None and not pd.isna(percentile):
        pressure += 20 * float(percentile)
    if change is not None and top is not None and not pd.isna(change) and not pd.isna(top) and abs(top) > 1e-12:
        pressure += max(0.0, min(25.0, (float(change) / abs(float(top))) * 8))
    if price20 is not None and not pd.isna(price20):
        pressure += max(0.0, min(15.0, float(price20) * 100))
    if short_ratio is not None and not pd.isna(short_ratio):
        pressure += max(0.0, min(10.0, float(short_ratio) * 20))
    if row["IsMarginLowLevel"]:
        pressure -= 25
    if row["IsMarginDrop"]:
        pressure -= 15

    opportunity = 0.0
    if row["IsMarginLowLevel"]:
        opportunity += 60
    if row["IsMarginDrop"]:
        opportunity += 35
    if percentile is not None and not pd.isna(percentile):
        opportunity += 25 * (1 - float(percentile))
    if change is not None and bottom is not None and not pd.isna(change) and not pd.isna(bottom) and abs(bottom) > 1e-12:
        opportunity += max(0.0, min(20.0, abs(min(0.0, float(change))) / abs(float(bottom)) * 8))
    if price20 is not None and not pd.isna(price20):
        opportunity += max(0.0, min(10.0, -float(price20) * 80))
    if row["IsMarginHighLevel"]:
        opportunity -= 25
    if row["IsMarginSurge"]:
        opportunity -= 15
    return pressure, opportunity


def current_record(summary_row: pd.Series, target_date: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    code = str(summary_row["Code"])
    try:
        margin_path = find_stock_file(PROJECT_ROOT / "data" / "margin", code)
        price_path = find_stock_file(PROJECT_ROOT / "data" / "price", code)
        if margin_path is None:
            raise ValueError("missing_margin_csv")
        if price_path is None:
            raise ValueError("missing_price_csv")
        margin, history = load_margin_state(margin_path, target_date)
        price = load_price_state(price_path, target_date)
        balance = json_float(margin.get("MarginCurrentBalance"))
        change = json_float(margin.get(SIGNAL_COLUMN))
        row = {
            "Date": target_date,
            "Code": code,
            "Name": str(summary_row["Name"]),
            "Close": json_float(price.get("Close")),
            "AdjustedClose": json_float(price.get("AdjustedClose")),
            "PriceReturn20D": json_float(price.get("PriceReturn20D")),
            "PriceReturn60D": json_float(price.get("PriceReturn60D")),
            "Capacity": json_float(price.get("Capacity")),
            "Turnover": json_float(price.get("Turnover")),
            "MarginCurrentBalance": balance,
            "MarginBalance20DayChangeRate": change,
            "MarginFinancingUsageRate": json_float(margin.get("MarginFinancingUsageRate")),
            "MarginMarketValue": json_float(margin.get("MarginMarketValue")),
            "MarginMarketValueTo20DayAvgTurnover": json_float(margin.get("MarginMarketValueTo20DayAvgTurnover")),
            "ShortCurrentBalance": json_float(margin.get("ShortCurrentBalance")),
            "ShortMarginBalanceRatio": json_float(margin.get("ShortMarginBalanceRatio")),
            "MarginSurgeThreshold": json_float(summary_row.get("MarginSurgeThreshold")),
            "MarginDropThreshold": json_float(summary_row.get("MarginDropThreshold")),
            "MarginHighLevelThreshold": json_float(summary_row.get("MarginHighLevelThreshold")),
            "MarginLowLevelThreshold": json_float(summary_row.get("MarginLowLevelThreshold")),
            "HistoricalHighFutureAvgReturn20D": json_float(summary_row.get("HighFutureAvgReturn20D")),
            "HistoricalLowFutureAvgReturn20D": json_float(summary_row.get("LowFutureAvgReturn20D")),
            "HistoricalSurgeHighFutureAvgReturn20D": json_float(summary_row.get("SurgeHighFutureAvgReturn20D")),
            "HistoricalAllFutureAvgReturn20D": json_float(summary_row.get("AllFutureAvgReturn20D")),
            "ByStockReport": str(summary_row.get("Report", "")),
        }
        row["MarginBalancePercentile"] = margin_percentile(history, balance)
        row["IsMarginSurge"] = bool(change is not None and row["MarginSurgeThreshold"] is not None and change >= row["MarginSurgeThreshold"])
        row["IsMarginDrop"] = bool(change is not None and row["MarginDropThreshold"] is not None and change <= row["MarginDropThreshold"])
        row["IsMarginHighLevel"] = bool(balance is not None and row["MarginHighLevelThreshold"] is not None and balance >= row["MarginHighLevelThreshold"])
        row["IsMarginLowLevel"] = bool(balance is not None and row["MarginLowLevelThreshold"] is not None and balance <= row["MarginLowLevelThreshold"])
        row["IsMarginSurgeHigh"] = bool(row["IsMarginSurge"] and row["IsMarginHighLevel"])
        row["StateLabels"] = state_labels(row)
        pressure, opportunity = scores(row)
        row["PressureScore"] = pressure
        row["OpportunityScore"] = opportunity
        row["MarginPath"] = str(margin_path.relative_to(PROJECT_ROOT))
        row["PricePath"] = str(price_path.relative_to(PROJECT_ROOT))
        return row, None
    except Exception as exc:  # noqa: BLE001 - keep batch report complete.
        return None, {"Code": code, "Name": str(summary_row.get("Name", "")), "Reason": str(exc)}


def display_columns() -> list[str]:
    return [
        "Code",
        "Name",
        "StateLabels",
        "PressureScore",
        "OpportunityScore",
        "Close",
        "PriceReturn20D",
        "PriceReturn60D",
        "MarginBalance20DayChangeRate",
        "MarginSurgeThreshold",
        "MarginCurrentBalance",
        "MarginBalancePercentile",
        "MarginFinancingUsageRate",
        "ShortMarginBalanceRatio",
        "HistoricalHighFutureAvgReturn20D",
        "HistoricalLowFutureAvgReturn20D",
        "ByStockReport",
    ]


PCT_COLUMNS = {
    "PriceReturn20D",
    "PriceReturn60D",
    "MarginBalance20DayChangeRate",
    "MarginSurgeThreshold",
    "MarginDropThreshold",
    "MarginBalancePercentile",
    "MarginFinancingUsageRate",
    "ShortMarginBalanceRatio",
    "HistoricalHighFutureAvgReturn20D",
    "HistoricalLowFutureAvgReturn20D",
    "HistoricalSurgeHighFutureAvgReturn20D",
    "HistoricalAllFutureAvgReturn20D",
}


def table_html(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    if df.empty:
        return "<p class=\"muted\">沒有符合條件的資料。</p>"
    data = df[display_columns()].head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in data.columns)
    body = []
    for _, row in data.iterrows():
        cells = []
        for column, value in row.items():
            if column == "ByStockReport":
                href = f"by_stock/{html.escape(str(value))}"
                text = '<a href="' + href + '">個股報告</a>' if value else ""
            elif column in PCT_COLUMNS:
                text = fmt_pct(value)
            elif column in {"PressureScore", "OpportunityScore"}:
                text = fmt_num(value, 1)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                text = fmt_num(value, 2)
            else:
                text = "" if pd.isna(value) else html.escape(str(value))
            if column != "ByStockReport":
                text = html.escape(text)
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def css() -> str:
    return """
body { margin: 0; font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif; color: #172033; background: #f7f9fc; }
main { max-width: 1280px; margin: 0 auto; padding: 28px 22px 48px; }
h1 { margin: 0 0 8px; font-size: 30px; }
h2 { margin: 30px 0 12px; font-size: 21px; }
p { line-height: 1.65; }
a { color: #0f766e; text-decoration: none; }
.muted { color: #617086; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0 22px; }
.card { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 14px 16px; }
.card .label { color: #617086; font-size: 13px; }
.card .value { display: block; margin-top: 7px; font-size: 21px; font-weight: 700; }
table { border-collapse: collapse; width: 100%; background: white; border: 1px solid #d9e2ef; margin: 10px 0 20px; }
th, td { border-bottom: 1px solid #e6edf5; padding: 8px 9px; text-align: right; white-space: nowrap; font-size: 13px; }
th:nth-child(1), th:nth-child(2), th:nth-child(3), td:nth-child(1), td:nth-child(2), td:nth-child(3) { text-align: left; }
th { background: #eef4fb; color: #1f2a3d; position: sticky; top: 0; }
.note { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 12px 14px; }
"""


def write_report(df: pd.DataFrame, skipped: pd.DataFrame, target_date: str, output_dir: Path, viz_path: Path) -> None:
    pressure = df.sort_values("PressureScore", ascending=False)
    opportunity = df.sort_values("OpportunityScore", ascending=False)
    surge = df[df["IsMarginSurge"]].sort_values("MarginBalance20DayChangeRate", ascending=False)
    high = df[df["IsMarginHighLevel"]].sort_values(["MarginBalancePercentile", "MarginBalance20DayChangeRate"], ascending=False)
    surge_high = df[df["IsMarginSurgeHigh"]].sort_values("PressureScore", ascending=False)
    low = df[df["IsMarginLowLevel"]].sort_values("OpportunityScore", ascending=False)
    cards = {
        "可評估股票": len(df),
        "融資大漲": int(df["IsMarginSurge"].sum()),
        "融資大跌": int(df["IsMarginDrop"].sum()),
        "融資高水位": int(df["IsMarginHighLevel"].sum()),
        "融資低水位": int(df["IsMarginLowLevel"].sum()),
        "大漲且高水位": int(df["IsMarginSurgeHigh"].sum()),
    }
    card_html = "".join(
        f'<div class="card"><span class="label">{html.escape(label)}</span><span class="value">{value:,}</span></div>'
        for label, value in cards.items()
    )
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{target_date} 融資狀態排序報告</title>
<style>{css()}</style>
</head>
<body>
<main>
<h1>{target_date} 融資狀態排序報告</h1>
<p class="muted">使用每檔股票自己的歷史門檻：融資 20 日變化率 top/bottom 10%，融資餘額 top/bottom 20%。資料來源為同日 data/margin 與 data/price。產生時間 {datetime.now().isoformat(timespec="seconds")}。</p>
<div class="cards">{card_html}</div>
<div class="note">PressureScore 偏向找融資擁擠與上方壓力：大漲且高水位、高水位、大漲、融資水位分位、20日價格漲幅與券資比會加分。OpportunityScore 偏向找低水位與去槓桿：低水位、大跌、低分位與近期價格回檔會加分。</div>
<h2>融資壓力排序</h2>
{table_html(pressure, max_rows=80)}
<h2>融資大漲且高水位</h2>
{table_html(surge_high, max_rows=80)}
<h2>融資高水位</h2>
{table_html(high, max_rows=80)}
<h2>融資大漲</h2>
{table_html(surge, max_rows=80)}
<h2>融資低水位 / 去槓桿排序</h2>
{table_html(opportunity, max_rows=80)}
<h2>融資低水位股票</h2>
{table_html(low, max_rows=80)}
<h2>完整資料</h2>
<p>完整排序資料已輸出到 {html.escape(str((output_dir / "current_margin_state.csv").relative_to(PROJECT_ROOT)))}。</p>
</main>
</body>
</html>
"""
    viz_path.write_text(html_text, encoding="utf-8")
    df.to_csv(output_dir / "current_margin_state.csv", index=False, encoding="utf-8-sig")
    pressure.head(200).to_csv(output_dir / "pressure_top.csv", index=False, encoding="utf-8-sig")
    opportunity.head(200).to_csv(output_dir / "opportunity_top.csv", index=False, encoding="utf-8-sig")
    surge_high.to_csv(output_dir / "surge_high.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(output_dir / "skipped.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "date": target_date,
        "rows": int(len(df)),
        "skipped": int(len(skipped)),
        "counts": cards,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "report": str(viz_path.relative_to(PROJECT_ROOT)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_date = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    if not args.summary.exists():
        raise FileNotFoundError(f"Missing by-stock summary: {args.summary}")
    output_dir = args.output_root / f"current_state_{target_date}"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_path = args.viz_root / f"current_state_{target_date}.html"
    summary = pd.read_csv(args.summary, encoding="utf-8-sig", dtype={"Code": str})
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        record, skip = current_record(row, target_date)
        if record is not None:
            records.append(record)
        if skip is not None:
            skipped.append(skip)
    df = pd.DataFrame(records)
    skipped_df = pd.DataFrame(skipped)
    if df.empty:
        raise ValueError(f"No current rows available for {target_date}")
    write_report(df.sort_values("PressureScore", ascending=False), skipped_df, target_date, output_dir, viz_path)
    payload = {
        "date": target_date,
        "rows": int(len(df)),
        "skipped": int(len(skipped_df)),
        "report": str(viz_path),
        "pressure_top": df.sort_values("PressureScore", ascending=False)[["Code", "Name", "StateLabels", "PressureScore"]].head(10).to_dict("records"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
