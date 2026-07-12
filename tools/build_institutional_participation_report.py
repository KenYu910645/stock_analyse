"""Build market-participant volume share reports from price and T86 data."""

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

from column_schema import read_csv_canonical
from viz.generate_dataset_viz import write_price_webgl_page


DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_participation"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_participation"

GROUPS = [
    ("foreign", "外資", "#2563eb"),
    ("trust", "投信", "#d97706"),
    ("dealer", "自營商", "#7c3aed"),
    ("other", "其他", "#64748b"),
]

METRIC_SUFFIXES = [
    ("participation", "p", "\u6210\u4ea4\u53c3\u8207\u7387"),
    ("net_ratio", "n", "\u6de8\u6d41\u91cf\u6bd4"),
    ("purity", "u", "\u65b9\u5411\u7d14\u5ea6"),
]

REQUIRED_INSTITUTIONAL_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "ForeignBuyExDealer",
    "ForeignSellExDealer",
    "ForeignDealerBuy",
    "ForeignDealerSell",
    "InvestmentTrustBuy",
    "InvestmentTrustSell",
    "DealerSelfBuy",
    "DealerSelfSell",
    "DealerHedgeBuy",
    "DealerHedgeSell",
]

REQUIRED_PRICE_COLUMNS = ["Date", "Capacity", "Open", "High", "Low", "Close"]


@dataclass
class StockResult:
    code: str
    name: str
    rows: int
    first_date: str
    last_date: str
    latest: dict[str, Any]
    output_path: Path
    skipped_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-stock institutional participation reports."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N stock limit for testing.")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; reports are always rewritten.")
    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "")).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.strip("._ ") or "Unknown"


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


def number_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def clean_ratio(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def compute_metrics(price_df: pd.DataFrame, institutional_df: pd.DataFrame) -> pd.DataFrame:
    price_missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in price_df.columns]
    inst_missing = [column for column in REQUIRED_INSTITUTIONAL_COLUMNS if column not in institutional_df.columns]
    if price_missing:
        raise ValueError(f"price_missing_columns:{price_missing}")
    if inst_missing:
        raise ValueError(f"institutional_missing_columns:{inst_missing}")

    optional_price_columns = ["Change", "open_adj", "high_adj", "low_adj", "close_adj"]
    price_columns = ["Date", "Capacity", "Open", "High", "Low", "Close"] + [
        column for column in optional_price_columns if column in price_df.columns
    ]
    price = price_df[price_columns].copy()
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price["Capacity"] = pd.to_numeric(price["Capacity"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", *optional_price_columns]:
        if column not in price.columns:
            continue
        price[column] = pd.to_numeric(price[column], errors="coerce")
    price = price.dropna(subset=["Date", "Capacity", "Open", "High", "Low", "Close"])
    price = price[price["Capacity"].gt(0)]
    price = price.sort_values("Date").drop_duplicates("Date", keep="last")

    inst = institutional_df.copy()
    inst["Date"] = pd.to_datetime(inst["Date"], errors="coerce")
    inst = inst.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")

    merged = price.merge(inst, on="Date", how="left", suffixes=("", "_inst"))
    for column in REQUIRED_INSTITUTIONAL_COLUMNS:
        if column in ["Date", "Code", "Name"]:
            continue
        merged[column] = number_series(merged, column)

    merged["foreign_buy"] = merged["ForeignBuyExDealer"] + merged["ForeignDealerBuy"]
    merged["foreign_sell"] = merged["ForeignSellExDealer"] + merged["ForeignDealerSell"]
    merged["trust_buy"] = merged["InvestmentTrustBuy"]
    merged["trust_sell"] = merged["InvestmentTrustSell"]
    merged["dealer_buy"] = merged["DealerSelfBuy"] + merged["DealerHedgeBuy"]
    merged["dealer_sell"] = merged["DealerSelfSell"] + merged["DealerHedgeSell"]

    institutional_buy = merged["foreign_buy"] + merged["trust_buy"] + merged["dealer_buy"]
    institutional_sell = merged["foreign_sell"] + merged["trust_sell"] + merged["dealer_sell"]
    merged["other_buy_raw"] = merged["Capacity"] - institutional_buy
    merged["other_sell_raw"] = merged["Capacity"] - institutional_sell
    merged["other_buy"] = merged["other_buy_raw"].clip(lower=0)
    merged["other_sell"] = merged["other_sell_raw"].clip(lower=0)

    for key, _label, _color in GROUPS:
        buy = merged[f"{key}_buy"]
        sell = merged[f"{key}_sell"]
        activity = (buy + sell) / 2.0
        net = buy - sell
        denominator = buy + sell
        merged[f"{key}_volume"] = activity
        merged[f"{key}_participation"] = activity / merged["Capacity"]
        merged[f"{key}_net_ratio"] = net / merged["Capacity"]
        merged[f"{key}_purity"] = net.where(denominator.ne(0), pd.NA) / denominator.where(denominator.ne(0), pd.NA)

    participation_columns = [f"{key}_participation" for key, _label, _color in GROUPS]
    merged["dominant_group_key"] = merged[participation_columns].idxmax(axis=1).str.replace("_participation", "", regex=False)
    label_by_key = {key: label for key, label, _color in GROUPS}
    merged["dominant_group"] = merged["dominant_group_key"].map(label_by_key).fillna("")
    merged["dominant_participation"] = [
        row.get(f"{row['dominant_group_key']}_participation", pd.NA)
        for _, row in merged.iterrows()
    ]
    merged["negative_other_side"] = (merged["other_buy_raw"].lt(-1e-6) | merged["other_sell_raw"].lt(-1e-6))
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    return merged


def format_percent(value: Any) -> str:
    number = clean_ratio(value)
    if number is None:
        return ""
    return f"{number * 100:.2f}%"


def format_number(value: Any) -> str:
    number = clean_ratio(value)
    if number is None:
        return ""
    return f"{number:,.0f}"


def compact_float(value: Any, digits: int = 8) -> float | None:
    number = clean_ratio(value)
    if number is None:
        return None
    return round(number, digits)



def metric_payload_by_date(metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    specs = institutional_metric_specs()
    for row in metrics.to_dict("records"):
        date = str(row.get("Date", ""))
        if not date:
            continue
        values: dict[str, float] = {}
        for source_key, payload_key, _label in specs:
            value = compact_float(row.get(source_key))
            if value is not None:
                values[payload_key] = value
        if values:
            payload[date] = values
    return payload


def volume_segments_by_date(metrics: pd.DataFrame) -> dict[str, list[list[Any]]]:
    payload: dict[str, list[list[Any]]] = {}
    for row in metrics.to_dict("records"):
        date = str(row.get("Date", ""))
        if not date:
            continue
        segments: list[list[Any]] = []
        for key, label, color in GROUPS:
            volume = clean_ratio(row.get(f"{key}_volume")) or 0.0
            segments.append([int(round(volume))])
        payload[date] = segments
    return payload


def volume_segment_groups() -> list[dict[str, str]]:
    return [{"label": label, "color": color} for _key, label, color in GROUPS]


def institutional_metric_specs() -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []
    for suffix, compact_suffix, metric_label in METRIC_SUFFIXES:
        for key, group_label, _color in GROUPS:
            specs.append((f"{key}_{suffix}", f"{key[0]}{compact_suffix}", f"{group_label}{metric_label}"))
    return specs


def institutional_metric_definitions() -> list[dict[str, str]]:
    return [
        {"key": payload_key, "label": label}
        for _source_key, payload_key, label in institutional_metric_specs()
    ]


def stock_summary_html(metrics: pd.DataFrame) -> str:
    latest = metrics.iloc[-1]
    cards = [
        ("\u6700\u65b0\u65e5\u671f", str(latest["Date"])),
        ("\u4e3b\u8981\u8ca2\u737b", str(latest["dominant_group"])),
        ("\u4e3b\u8981\u53c3\u8207\u7387", format_percent(latest["dominant_participation"])),
        ("\u6210\u4ea4\u91cf", format_number(latest["Capacity"])),
    ]
    card_html = "\n".join(
        f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    legend_html = "".join(
        f'<span><i class="swatch" style="background:{html.escape(color)}"></i>{html.escape(label)}</span>'
        for _key, label, color in GROUPS
    )
    return f"""
<section class="summary">{card_html}</section>
<div class="volume-legend"><strong>\u6210\u4ea4\u91cf\u5806\u758a</strong>{legend_html}</div>
"""


def write_stock_html(
    metrics: pd.DataFrame,
    output_path: Path,
    code: str,
    name: str,
    source_paths: list[Path],
) -> None:
    extra_styles = """
.summary { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }
.metric { background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }
.metric .label { color: #64748b; font-size: 12px; }
.metric .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
.volume-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 8px 0 10px; color: #334155; font-size: 13px; }
.volume-legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
@media (max-width: 760px) { .summary { grid-template-columns: 1fr 1fr; } }
"""
    ok = write_price_webgl_page(
        source_paths[0],
        output_path,
        f"{code} {name}",
        metrics,
        source_paths=source_paths,
        margin_by_date=metric_payload_by_date(metrics),
        margin_metrics=institutional_metric_definitions(),
        volume_segments_by_date=volume_segments_by_date(metrics),
        volume_segment_groups=volume_segment_groups(),
        duplicate_auxiliary_payload_to_adjusted=False,
        page_suffix="\u7c4c\u78bc\u6210\u4ea4\u7d50\u69cb",
        metric_control_label="\u7c4c\u78bc\u6307\u6a19",
        extra_body_before_chart=stock_summary_html(metrics),
        extra_styles=extra_styles,
    )
    if not ok:
        raise ValueError("webgl_price_payload_empty")

def latest_summary_row(code: str, name: str, metrics: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    latest = metrics.iloc[-1]
    row = {
        "Code": code,
        "Name": name,
        "Date": latest["Date"],
        "Close": latest["Close"],
        "Volume": latest["Capacity"],
        "DominantGroup": latest["dominant_group"],
        "DominantParticipation": latest["dominant_participation"],
        "ReportPath": str(output_path.relative_to(PROJECT_ROOT)),
    }
    for key, label, _color in GROUPS:
        row[f"{label}成交參與率"] = latest[f"{key}_participation"]
        row[f"{label}淨流量比"] = latest[f"{key}_net_ratio"]
        row[f"{label}方向純度"] = latest[f"{key}_purity"]
    return row


def aggregate_market_rows(all_metrics: list[pd.DataFrame]) -> pd.DataFrame:
    if not all_metrics:
        return pd.DataFrame()
    keep_columns = ["Date", "Capacity"] + [
        item
        for key, _label, _color in GROUPS
        for item in [f"{key}_buy", f"{key}_sell", f"{key}_volume"]
    ]
    combined = pd.concat([metrics[keep_columns] for metrics in all_metrics], ignore_index=True)
    grouped = combined.groupby("Date", as_index=False).sum(numeric_only=True)
    for key, _label, _color in GROUPS:
        buy = grouped[f"{key}_buy"]
        sell = grouped[f"{key}_sell"]
        grouped[f"{key}_participation"] = grouped[f"{key}_volume"] / grouped["Capacity"]
        grouped[f"{key}_net_ratio"] = (buy - sell) / grouped["Capacity"]
        grouped[f"{key}_purity"] = (buy - sell) / (buy + sell).replace(0, pd.NA)
    return grouped.sort_values("Date")


def write_index(results: list[StockResult], market_summary: pd.DataFrame) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    latest_market = market_summary.iloc[-1] if not market_summary.empty else {}
    dominant_group_order = {"外資": 0}
    rows = sorted(
        [result for result in results if not result.skipped_reason],
        key=lambda item: (
            dominant_group_order.get(str(item.latest.get("DominantGroup", "")), 1),
            -(clean_ratio(item.latest.get("DominantParticipation")) or 0.0),
            str(item.latest.get("Date", "")),
            item.code,
        ),
    )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(result.code)}</td>"
        f"<td>{html.escape(result.name)}</td>"
        f"<td>{html.escape(str(result.latest.get('Date', '')))}</td>"
        f"<td>{html.escape(str(result.latest.get('DominantGroup', '')))}</td>"
        f"<td>{html.escape(format_percent(result.latest.get('DominantParticipation')))}</td>"
        f"<td><a href=\"{html.escape(result.output_path.relative_to(DATA_VIZ_ROOT).as_posix())}\">開啟</a></td>"
        "</tr>"
        for result in rows
    )
    group_summary = "".join(
        f"<div class=\"metric\"><div class=\"label\">{label}市場參與率</div><div class=\"value\">{format_percent(latest_market.get(key + '_participation'))}</div></div>"
        for key, label, _color in GROUPS
    )
    index_path = DATA_VIZ_ROOT / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>籌碼成交結構總覽</title>
<style>
body {{ margin: 20px; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 14px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<h1>籌碼成交結構總覽</h1>
<div class="meta">股票數 {len(rows):,}，最新市場日期 {html.escape(str(latest_market.get("Date", "")))}</div>
<section class="summary">{group_summary}</section>
<table>
<thead><tr><th>代號</th><th>名稱</th><th>最新日期</th><th>主要貢獻</th><th>主要參與率</th><th>報告</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def write_summaries(results: list[StockResult], market_summary: pd.DataFrame) -> tuple[Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    latest_rows = [result.latest for result in results if not result.skipped_reason]
    latest_path = OUTPUT_ROOT / "latest_summary.csv"
    market_path = OUTPUT_ROOT / "market_daily_summary.csv"
    pd.DataFrame(latest_rows).to_csv(latest_path, index=False, encoding="utf-8-sig")
    market_summary.to_csv(market_path, index=False, encoding="utf-8-sig")
    return latest_path, market_path


def build_report(limit: int | None = None) -> tuple[list[StockResult], Path, Path, Path]:
    price_paths = path_by_code(PRICE_DIR)
    inst_paths = path_by_code(INSTITUTIONAL_DIR)
    codes = sorted(set(price_paths) & set(inst_paths))
    if limit is not None:
        codes = codes[:limit]

    results: list[StockResult] = []
    all_metrics: list[pd.DataFrame] = []
    for index, code in enumerate(codes, start=1):
        price_path = price_paths[code]
        inst_path = inst_paths[code]
        name = stock_name_from_path(inst_path, stock_name_from_path(price_path, code))
        output_path = DATA_VIZ_ROOT / "stocks" / f"{code}_{safe_filename_part(name)}.html"
        try:
            price_df = read_csv_canonical(price_path, dtype={"Code": str})
            inst_df = read_csv_canonical(inst_path, dtype={"Code": str})
            metrics = compute_metrics(price_df, inst_df)
            if metrics.empty:
                raise ValueError("empty_metrics")
            write_stock_html(metrics, output_path, code, name, [price_path, inst_path])
            latest = latest_summary_row(code, name, metrics, output_path)
            results.append(
                StockResult(
                    code=code,
                    name=name,
                    rows=len(metrics),
                    first_date=str(metrics["Date"].iloc[0]),
                    last_date=str(metrics["Date"].iloc[-1]),
                    latest=latest,
                    output_path=output_path,
                )
            )
            all_metrics.append(metrics)
        except Exception as exc:
            results.append(
                StockResult(
                    code=code,
                    name=name,
                    rows=0,
                    first_date="",
                    last_date="",
                    latest={},
                    output_path=output_path,
                    skipped_reason=str(exc),
                )
            )
        if index % 100 == 0 or index == len(codes):
            print(f"processed {index}/{len(codes)}")

    market_summary = aggregate_market_rows(all_metrics)
    latest_path, market_path = write_summaries(results, market_summary)
    index_path = write_index(results, market_summary)
    return results, latest_path, market_path, index_path


def main() -> None:
    args = parse_args()
    results, latest_path, market_path, index_path = build_report(limit=args.limit)
    ok = sum(1 for result in results if not result.skipped_reason)
    skipped = [result for result in results if result.skipped_reason]
    print("Report summary:")
    print(f"stocks_written={ok}")
    print(f"stocks_skipped={len(skipped)}")
    print(f"latest_summary={latest_path}")
    print(f"market_daily_summary={market_path}")
    print(f"index={index_path}")
    if skipped:
        print("Skipped sample:")
        for result in skipped[:10]:
            print(f"{result.code}: {result.skipped_reason}")


if __name__ == "__main__":
    main()
