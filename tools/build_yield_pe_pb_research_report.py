"""Build yield/PER/PBR valuation research reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from bisect import bisect_right, insort
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from downloader import price as price_downloader
from viz.generate_dataset_viz import (
    load_ex_right_events_for_stock,
    overlay_payload_by_date,
    write_price_webgl_page,
)

DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
VALUATION_DIR = DATA_DIR / "yield_pe_pb"
METADATA_PATH = DATA_DIR / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "yield_pe_pb_research"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "yield_pe_pb_research"

HORIZONS = [1, 5, 20, 60]
QUANTILES = 5
ROLLING_WINDOW = 756
ROLLING_MIN_PERIODS = 120
MIN_STOCKS_PER_DATE = 30

BASE_FACTOR_LABELS = {
    "dividend_yield": "殖利率",
    "earnings_yield": "盈餘殖利率(1/本益比)",
    "book_to_market": "淨值市價比(1/股價淨值比)",
}
VARIANT_LABELS = {
    "raw": "全市場原始值",
    "industry_rank": "產業中性排名",
    "own_percentile": "個股三年分位",
}
FACTOR_LABELS = {
    f"{factor}_{variant}": f"{factor_label} - {VARIANT_LABELS[variant]}"
    for factor, factor_label in BASE_FACTOR_LABELS.items()
    for variant in VARIANT_LABELS
}

OVERLAY_METRICS = [
    ("PricePercentile", "price_pct", "股價三年分位"),
    ("DividendYield", "dividend_yield", "殖利率"),
    ("PEratio", "pe_ratio", "本益比"),
    ("PBratio", "pb_ratio", "股價淨值比"),
    ("EarningsYield", "earnings_yield", "盈餘殖利率"),
    ("BookToMarket", "book_to_market", "淨值市價比"),
    ("DividendYieldPercentile", "dividend_yield_pct", "殖利率三年分位"),
    ("EarningsYieldPercentile", "earnings_yield_pct", "盈餘殖利率三年分位"),
    ("BookToMarketPercentile", "book_to_market_pct", "淨值市價比三年分位"),
]


@dataclass
class StockPanel:
    code: str
    name: str
    group: str
    price_path: Path
    valuation_path: Path
    panel: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build yield/PER/PBR valuation research report.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--horizons", default="1,5,20,60")
    parser.add_argument("--quantiles", type=int, default=QUANTILES)
    parser.add_argument("--rolling-window", type=int, default=ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=ROLLING_MIN_PERIODS)
    parser.add_argument("--min-stocks-per-date", type=int, default=MIN_STOCKS_PER_DATE)
    parser.add_argument("--stock-page-limit", type=int, default=0, help="0 means all stocks.")
    return parser.parse_args()


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must be positive integers")
    return horizons


def path_by_code(directory: Path) -> dict[str, Path]:
    paths = {}
    for path in sorted(directory.glob("*.csv")):
        code = path.stem.split("_", 1)[0]
        if code and code not in paths:
            paths[code] = path
    return paths


def safe_filename_component(value: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid or ord(char) < 32 else char for char in str(value))
    return cleaned.strip().rstrip(".") or "unknown"


def load_universe() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    required = {"Code", "Name", "Type", "Market", "Group"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    mask = (
        metadata["Type"].astype(str).isin([price_downloader.COMMON_STOCK_TYPE, "STOCK", "股票"])
        & metadata["Market"].astype(str).eq(price_downloader.TWSE_MARKET)
    )
    universe = metadata.loc[mask, ["Code", "Name", "Group"]].copy()
    universe["Code"] = universe["Code"].astype(str).str.strip()
    universe["Name"] = universe["Name"].astype(str).str.strip()
    universe["Group"] = universe["Group"].astype(str).str.strip().replace("", "未分類")
    return universe[universe["Code"].ne("")].drop_duplicates("Code")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def rolling_percentile(values: Iterable[float], window: int, min_periods: int) -> list[float]:
    ordered: list[float] = []
    queue: list[float] = []
    output: list[float] = []
    for raw_value in values:
        value = float(raw_value) if pd.notna(raw_value) else math.nan
        if math.isfinite(value):
            insort(ordered, value)
            queue.append(value)
        else:
            queue.append(math.nan)
        while len(queue) > window:
            old = queue.pop(0)
            if math.isfinite(old):
                pos = bisect_right(ordered, old) - 1
                del ordered[pos]
        if len(ordered) >= min_periods and math.isfinite(value):
            output.append(bisect_right(ordered, value) / len(ordered))
        else:
            output.append(math.nan)
    return output


def load_stock_panel(
    code: str,
    name: str,
    group: str,
    price_path: Path,
    valuation_path: Path,
    horizons: list[int],
    rolling_window: int,
    rolling_min_periods: int,
) -> StockPanel | None:
    try:
        price_df = read_csv_canonical(price_path, dtype=str).fillna("")
        value_df = read_csv_canonical(valuation_path, dtype=str).fillna("")
    except Exception as exc:
        print(f"skip {code}: {exc}")
        return None
    if price_df.empty or value_df.empty:
        return None

    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "open_adj", "high_adj", "low_adj", "close_adj"]:
        if column in price_df.columns:
            price_df[column] = numeric(price_df[column])
    price_df = (
        price_df.dropna(subset=["Date", "open_adj", "close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if price_df.empty:
        return None
    for horizon in horizons:
        entry = price_df["open_adj"].shift(-1)
        exit_ = price_df["open_adj"].shift(-(horizon + 1))
        price_df[f"ForwardReturn{horizon}D"] = exit_ / entry - 1
    price_df["PricePercentile"] = rolling_percentile(
        price_df["close_adj"], rolling_window, rolling_min_periods
    )

    value_df["Date"] = pd.to_datetime(value_df["Date"], errors="coerce")
    for column in ["Close", "DividendYield", "PEratio", "PBratio"]:
        if column in value_df.columns:
            value_df[column] = numeric(value_df[column])
    value_df = (
        value_df.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if value_df.empty:
        return None

    value_df["EarningsYield"] = np.where(value_df["PEratio"].gt(0), 1 / value_df["PEratio"], np.nan)
    value_df["BookToMarket"] = np.where(value_df["PBratio"].gt(0), 1 / value_df["PBratio"], np.nan)
    value_df["LogPE"] = np.where(value_df["PEratio"].gt(0), np.log(value_df["PEratio"]), np.nan)
    value_df["LogPB"] = np.where(value_df["PBratio"].gt(0), np.log(value_df["PBratio"]), np.nan)
    value_df["DividendYieldPercentile"] = rolling_percentile(
        value_df["DividendYield"], rolling_window, rolling_min_periods
    )
    value_df["EarningsYieldPercentile"] = rolling_percentile(
        value_df["EarningsYield"], rolling_window, rolling_min_periods
    )
    value_df["BookToMarketPercentile"] = rolling_percentile(
        value_df["BookToMarket"], rolling_window, rolling_min_periods
    )

    keep_price_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "PricePercentile",
        *[f"ForwardReturn{h}D" for h in horizons],
    ]
    merged = value_df.merge(price_df[keep_price_columns], on="Date", how="inner", suffixes=("", "_price"))
    if merged.empty:
        return None
    merged = merged.drop(columns=[column for column in ["Code", "Name", "Group"] if column in merged.columns])
    merged.insert(0, "Code", code)
    merged.insert(1, "Name", name)
    merged.insert(2, "Group", group)
    return StockPanel(code, name, group, price_path, valuation_path, merged)


def build_panel(args: argparse.Namespace, horizons: list[int]) -> tuple[pd.DataFrame, list[StockPanel]]:
    universe = load_universe()
    price_paths = path_by_code(PRICE_DIR)
    valuation_paths = path_by_code(VALUATION_DIR)
    panels: list[StockPanel] = []
    for row in universe.itertuples(index=False):
        code = str(row.Code)
        price_path = price_paths.get(code)
        valuation_path = valuation_paths.get(code)
        if not price_path or not valuation_path:
            continue
        stock_panel = load_stock_panel(
            code,
            str(row.Name),
            str(row.Group),
            price_path,
            valuation_path,
            horizons,
            args.rolling_window,
            args.rolling_min_periods,
        )
        if stock_panel is not None:
            panels.append(stock_panel)
    if not panels:
        raise ValueError("No overlapping price and yield_pe_pb data found.")
    return pd.concat([item.panel for item in panels], ignore_index=True), panels


def add_cross_sectional_factor_variants(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    base_columns = {
        "dividend_yield": "DividendYield",
        "earnings_yield": "EarningsYield",
        "book_to_market": "BookToMarket",
    }
    own_columns = {
        "dividend_yield": "DividendYieldPercentile",
        "earnings_yield": "EarningsYieldPercentile",
        "book_to_market": "BookToMarketPercentile",
    }
    for factor, column in base_columns.items():
        result[f"{factor}_raw"] = result[column]
        result[f"{factor}_industry_rank"] = result.groupby(["Date", "Group"])[column].rank(pct=True)
        result[f"{factor}_own_percentile"] = result[own_columns[factor]]
    return result


def _corr_by_group(sample: pd.DataFrame, factor_col: str, return_col: str) -> pd.DataFrame:
    clean = sample[["Date", factor_col, return_col]].dropna().copy()
    if clean.empty:
        return pd.DataFrame(columns=["Date", "IC", "RankIC", "SampleCount"])

    def daily_pearson(frame: pd.DataFrame, x_col: str, y_col: str, value_col: str) -> pd.DataFrame:
        work = frame[["Date", x_col, y_col]].copy()
        work["_xy"] = work[x_col] * work[y_col]
        work["_x2"] = work[x_col] * work[x_col]
        work["_y2"] = work[y_col] * work[y_col]
        stats = work.groupby("Date", as_index=False).agg(
            SampleCount=(x_col, "size"),
            SumX=(x_col, "sum"),
            SumY=(y_col, "sum"),
            SumXY=("_xy", "sum"),
            SumX2=("_x2", "sum"),
            SumY2=("_y2", "sum"),
        )
        n = stats["SampleCount"].astype(float)
        numerator = n * stats["SumXY"] - stats["SumX"] * stats["SumY"]
        denom_x = n * stats["SumX2"] - stats["SumX"] * stats["SumX"]
        denom_y = n * stats["SumY2"] - stats["SumY"] * stats["SumY"]
        denominator = np.sqrt(denom_x * denom_y)
        stats[value_col] = np.where(denominator.gt(0), numerator / denominator, np.nan)
        stats.loc[stats["SampleCount"] < MIN_STOCKS_PER_DATE, value_col] = np.nan
        return stats[["Date", "SampleCount", value_col]]

    ic = daily_pearson(clean, factor_col, return_col, "IC")
    clean["_factor_rank"] = clean.groupby("Date")[factor_col].rank(method="average")
    clean["_return_rank"] = clean.groupby("Date")[return_col].rank(method="average")
    rank_ic = daily_pearson(clean, "_factor_rank", "_return_rank", "RankIC")[["Date", "RankIC"]]
    result = ic.merge(rank_ic, on="Date", how="left")
    return result.dropna(subset=["IC", "RankIC"], how="all")


def compute_ic(panel: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeseries_frames = []
    summary_rows = []
    factor_columns = list(FACTOR_LABELS)
    for factor_col in factor_columns:
        for horizon in horizons:
            return_col = f"ForwardReturn{horizon}D"
            ts = _corr_by_group(panel, factor_col, return_col)
            if ts.empty:
                continue
            ts.insert(0, "Factor", factor_col)
            ts.insert(1, "FactorLabel", FACTOR_LABELS[factor_col])
            ts.insert(2, "Horizon", horizon)
            timeseries_frames.append(ts)

            rank_ic = ts["RankIC"].dropna()
            ic = ts["IC"].dropna()
            rank_std = rank_ic.std(ddof=1)
            ic_std = ic.std(ddof=1)
            summary_rows.append(
                {
                    "Factor": factor_col,
                    "FactorLabel": FACTOR_LABELS[factor_col],
                    "Horizon": horizon,
                    "DateCount": int(ts["Date"].nunique()),
                    "AverageSampleCount": float(ts["SampleCount"].mean()),
                    "MeanIC": float(ic.mean()) if not ic.empty else np.nan,
                    "MeanRankIC": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                    "PositiveRankICRate": float((rank_ic > 0).mean()) if not rank_ic.empty else np.nan,
                    "ICTStat": float(ic.mean() / (ic_std / math.sqrt(len(ic)))) if len(ic) > 1 and ic_std else np.nan,
                    "RankICTStat": float(rank_ic.mean() / (rank_std / math.sqrt(len(rank_ic))))
                    if len(rank_ic) > 1 and rank_std
                    else np.nan,
                }
            )
    timeseries = pd.concat(timeseries_frames, ignore_index=True) if timeseries_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows).sort_values(["Horizon", "MeanRankIC"], ascending=[True, False])
    return timeseries, summary


def compute_quantile_returns(panel: pd.DataFrame, horizons: list[int], quantiles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    summary_rows = []
    for factor_col, factor_label in FACTOR_LABELS.items():
        return_columns = [f"ForwardReturn{horizon}D" for horizon in horizons]
        work = panel[["Date", "Code", factor_col, *return_columns]].dropna(subset=[factor_col]).copy()
        if work.empty:
            continue
        grouped = work.groupby("Date")[factor_col]
        ranks = grouped.rank(method="first")
        counts = grouped.transform("size")
        work["Quantile"] = np.ceil(ranks * quantiles / counts).clip(1, quantiles)
        work.loc[counts < MIN_STOCKS_PER_DATE, "Quantile"] = np.nan
        work = work.dropna(subset=["Quantile"]).copy()
        if work.empty:
            continue
        work["Quantile"] = work["Quantile"].astype(int)
        for horizon in horizons:
            return_col = f"ForwardReturn{horizon}D"
            horizon_work = work.dropna(subset=[return_col])
            if horizon_work.empty:
                continue
            daily = (
                horizon_work.groupby(["Date", "Quantile"], as_index=False)
                .agg(MeanReturn=(return_col, "mean"), StockCount=("Code", "count"))
            )
            daily.insert(0, "Factor", factor_col)
            daily.insert(1, "FactorLabel", factor_label)
            daily.insert(2, "Horizon", horizon)
            daily_rows.extend(daily.to_dict("records"))

            top = daily[daily["Quantile"].eq(quantiles)][["Date", "MeanReturn"]].rename(
                columns={"MeanReturn": "TopReturn"}
            )
            bottom = daily[daily["Quantile"].eq(1)][["Date", "MeanReturn"]].rename(
                columns={"MeanReturn": "BottomReturn"}
            )
            long_short = top.merge(bottom, on="Date", how="inner")
            long_short["MeanReturn"] = long_short["TopReturn"] - long_short["BottomReturn"]
            for quantile, group in daily.groupby("Quantile", sort=True):
                summary_rows.append(
                    {
                        "Factor": factor_col,
                        "FactorLabel": factor_label,
                        "Horizon": horizon,
                        "Quantile": str(int(quantile)),
                        "DateCount": int(group["Date"].nunique()),
                        "MeanReturn": float(group["MeanReturn"].mean()),
                        "PositiveRate": float((group["MeanReturn"] > 0).mean()),
                    }
                )
            if not long_short.empty:
                summary_rows.append(
                    {
                        "Factor": factor_col,
                        "FactorLabel": factor_label,
                        "Horizon": horizon,
                        "Quantile": "多空(高-低)",
                        "DateCount": int(long_short["Date"].nunique()),
                        "MeanReturn": float(long_short["MeanReturn"].mean()),
                        "PositiveRate": float((long_short["MeanReturn"] > 0).mean()),
                    }
                )
    return pd.DataFrame(daily_rows), pd.DataFrame(summary_rows)


def latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    latest_rows = panel.sort_values("Date").groupby("Code", as_index=False).tail(1).copy()
    rank_columns = ["dividend_yield_raw", "earnings_yield_raw", "book_to_market_raw"]
    for column in rank_columns:
        latest_rows[f"{column}_Rank"] = latest_rows[column].rank(ascending=False, method="min")
    return latest_rows[
        [
            "Date",
            "Code",
            "Name",
            "Group",
            "Close",
            "DividendYield",
            "PEratio",
            "PBratio",
            "EarningsYield",
            "BookToMarket",
            "DividendYieldPercentile",
            "EarningsYieldPercentile",
            "BookToMarketPercentile",
            *[f"{column}_Rank" for column in rank_columns],
        ]
    ].sort_values("earnings_yield_raw_Rank")


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):,.{digits}f}"


def table_html(df: pd.DataFrame, columns: list[tuple[str, str, str]], limit: int | None = None) -> str:
    rows = df.head(limit).to_dict("records") if limit else df.to_dict("records")
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _label, kind in columns:
            value = row.get(key, "")
            if kind == "pct":
                text = pct(value)
            elif kind == "num":
                text = num(value)
            elif kind == "int":
                text = "" if pd.isna(value) else f"{int(value):,}"
            elif kind == "html":
                cells.append(f"<td>{'' if pd.isna(value) else str(value)}</td>")
                continue
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def bar_svg(df: pd.DataFrame, label_col: str, value_col: str, title: str, width: int = 980, height: int = 340) -> str:
    sample = df[[label_col, value_col]].dropna().head(20).copy()
    if sample.empty:
        return ""
    labels = sample[label_col].astype(str).tolist()
    values = sample[value_col].astype(float).tolist()
    max_abs = max(abs(v) for v in values) or 1
    left, right, top, bottom = 260, 32, 36, 30
    plot_w = width - left - right
    bar_h = max(12, (height - top - bottom) / len(values) - 5)
    zero_x = left + plot_w / 2
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="22" font-size="16" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{height-bottom}" stroke="#94a3b8"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + 5)
        length = abs(value) / max_abs * (plot_w / 2)
        x = zero_x if value >= 0 else zero_x - length
        color = "#0f766e" if value >= 0 else "#dc2626"
        parts.append(f'<text x="8" y="{y + bar_h * 0.72:.1f}" font-size="12">{html.escape(label[:34])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{length:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + length + 4 if value >= 0 else x - 58:.1f}" y="{y + bar_h * 0.72:.1f}" font-size="12">{value:.4f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    commands = [f"M {first_x:.1f} {first_y:.1f}"]
    commands.extend(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    return " ".join(commands)


def valuation_price_butterfly_svg(history: pd.DataFrame, width: int = 1120, height: int = 360) -> str:
    columns = [
        ("PricePercentile", "股價三年分位", "#334155"),
        ("DividendYieldPercentile", "殖利率便宜度", "#0f766e"),
        ("EarningsYieldPercentile", "盈餘殖利率便宜度", "#2563eb"),
        ("BookToMarketPercentile", "淨值市價比便宜度", "#9333ea"),
    ]
    work = history[["Date", *[key for key, _label, _color in columns]]].dropna(how="all").copy()
    work = work.dropna(subset=["Date"]).sort_values("Date")
    if work.empty:
        return ""
    if len(work) > 520:
        positions = np.linspace(0, len(work) - 1, 520).round().astype(int)
        work = work.iloc[np.unique(positions)].copy()

    left, right, top, bottom = 72, 30, 34, 50
    plot_w = width - left - right
    plot_h = height - top - bottom
    dates = pd.to_datetime(work["Date"], errors="coerce")
    if len(work) <= 1:
        xs = [left + plot_w / 2]
    else:
        xs = [left + i * plot_w / (len(work) - 1) for i in range(len(work))]

    parts = [
        f'<svg class="butterfly-chart" viewBox="0 0 {width} {height}" role="img" aria-label="股價估值蝶圖">',
        f'<text x="{left}" y="22" font-size="16" font-weight="700">股價估值蝶圖</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h * 0.2}" fill="#fee2e2" opacity="0.75"/>',
        f'<rect x="{left}" y="{top + plot_h * 0.2}" width="{plot_w}" height="{plot_h * 0.6}" fill="#f8fafc"/>',
        f'<rect x="{left}" y="{top + plot_h * 0.8}" width="{plot_w}" height="{plot_h * 0.2}" fill="#dcfce7" opacity="0.75"/>',
    ]
    for pct_tick in [0, 25, 50, 75, 100]:
        y = top + plot_h * (1 - pct_tick / 100)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#d7dee9"/>')
        parts.append(f'<text x="18" y="{y + 4:.1f}" font-size="12" fill="#64748b">{pct_tick}%</text>')
    parts.append(f'<text x="{left + 8}" y="{top + 18}" font-size="12" fill="#991b1b">高分位</text>')
    parts.append(f'<text x="{left + 8}" y="{top + plot_h - 8}" font-size="12" fill="#166534">低分位</text>')

    for key, label, color in columns:
        points = []
        for x, raw in zip(xs, work[key].tolist()):
            if pd.isna(raw):
                continue
            value = max(0, min(1, float(raw)))
            points.append((x, top + plot_h * (1 - value)))
        if points:
            parts.append(
                f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            last_x, last_y = points[-1]
            parts.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" fill="{color}"/>')

    valid_dates = dates.dropna()
    if not valid_dates.empty:
        parts.append(f'<text x="{left}" y="{height - 16}" font-size="12" fill="#64748b">{valid_dates.iloc[0].strftime("%Y-%m-%d")}</text>')
        parts.append(
            f'<text x="{left + plot_w - 76}" y="{height - 16}" font-size="12" fill="#64748b">{valid_dates.iloc[-1].strftime("%Y-%m-%d")}</text>'
        )

    legend_x = left + 160
    for index, (_key, label, color) in enumerate(columns):
        x = legend_x + index * 180
        parts.append(f'<line x1="{x}" y1="18" x2="{x + 24}" y2="18" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 30}" y="22" font-size="12" fill="#334155">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def stock_butterfly_html(row: pd.Series, history: pd.DataFrame) -> str:
    metrics = [
        ("股價三年分位", row.get("PricePercentile")),
        ("殖利率分位", row.get("DividendYieldPercentile")),
        ("盈餘殖利率分位", row.get("EarningsYieldPercentile")),
        ("淨值市價比分位", row.get("BookToMarketPercentile")),
    ]
    bars = []
    for label, value in metrics:
        pct_value = 0 if pd.isna(value) else max(0, min(1, float(value)))
        bars.append(
            f"""
            <div class="butterfly-row">
              <span>{html.escape(label)}</span>
              <div class="butterfly-track"><div style="width:{pct_value * 100:.1f}%"></div></div>
              <b>{pct(pct_value, 1)}</b>
            </div>
            """
        )
    return f"""
<section class="stock-summary">
  <h2>股價與估值蝶圖</h2>
  <p>最新日期 {html.escape(str(row.get("Date", ""))[:10])}，收盤價 {html.escape(num(row.get("Close"), 2))}，
  殖利率 {html.escape(num(row.get("DividendYield"), 2))}，本益比 {html.escape(num(row.get("PEratio"), 2))}，
  股價淨值比 {html.escape(num(row.get("PBratio"), 2))}。下方以三年歷史分位把股價與估值便宜度放在同一個 0-100% 尺度；股價線偏高且估值便宜度線偏低時，代表價格相對高、估值相對不便宜。</p>
  {''.join(bars)}
  {valuation_price_butterfly_svg(history)}
</section>
"""


def stock_extra_styles() -> str:
    return """
.stock-summary { margin: 16px 0; padding: 14px 16px; border: 1px solid #d7dee9; background: #fff; }
.stock-summary h2 { font-size: 18px; margin: 0 0 8px; }
.stock-summary p { color: #334155; margin: 0 0 10px; }
.butterfly-row { display: grid; grid-template-columns: 140px 1fr 70px; gap: 10px; align-items: center; margin: 7px 0; font-size: 13px; }
.butterfly-track { height: 12px; background: #e2e8f0; position: relative; }
.butterfly-track div { height: 100%; background: #0f766e; }
.butterfly-chart { width: 100%; height: auto; margin-top: 14px; background: white; border: 1px solid #d7dee9; }
"""


def write_stock_pages(stock_panels: list[StockPanel], latest: pd.DataFrame, args: argparse.Namespace) -> int:
    latest_by_code = latest.set_index("Code")
    written = 0
    stock_pages_dir = args.viz_dir / "stocks"
    iterable = stock_panels if args.stock_page_limit <= 0 else stock_panels[: args.stock_page_limit]
    for item in iterable:
        value_df = item.panel.copy()
        value_df["Date"] = value_df["Date"].dt.strftime("%Y-%m-%d")
        overlay_by_date, overlay_metrics = overlay_payload_by_date(value_df, OVERLAY_METRICS, ["Date"])
        price_df = read_csv_canonical(item.price_path, dtype=str).fillna("")
        events_by_date, event_csv = load_ex_right_events_for_stock(item.price_path, price_df)
        sources = [item.price_path, item.valuation_path]
        if event_csv is not None:
            sources.append(event_csv)
        latest_row = latest_by_code.loc[item.code] if item.code in latest_by_code.index else item.panel.iloc[-1]
        if write_price_webgl_page(
            item.price_path,
            stock_pages_dir / f"{safe_filename_component(item.code)}_{safe_filename_component(item.name)}.html",
            f"{item.code} {item.name}",
            price_df,
            source_paths=sources,
            margin_by_date=overlay_by_date,
            margin_metrics=overlay_metrics,
            events_by_date=events_by_date,
            page_suffix="價格與估值研究",
            metric_control_label="估值指標",
            extra_body_after_chart=stock_butterfly_html(latest_row, item.panel),
            extra_styles=stock_extra_styles(),
        ):
            written += 1
    return written


def write_index_html(
    args: argparse.Namespace,
    panel: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    latest: pd.DataFrame,
    stock_page_count: int,
) -> Path:
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    latest_date = panel["Date"].max().strftime("%Y-%m-%d")
    start_date = panel["Date"].min().strftime("%Y-%m-%d")
    stock_count = panel["Code"].nunique()
    row_count = len(panel)
    best_ic = ic_summary.sort_values("MeanRankIC", ascending=False).head(12)
    q20 = quantile_summary[
        quantile_summary["Horizon"].eq(20) & quantile_summary["Quantile"].eq("多空(高-低)")
    ].sort_values("MeanReturn", ascending=False)
    latest_top = latest.sort_values("earnings_yield_raw_Rank").head(30).copy()
    latest_top["StockLink"] = latest_top.apply(
        lambda row: f'<a href="stocks/{html.escape(safe_filename_component(str(row["Code"])))}_{html.escape(safe_filename_component(str(row["Name"])))}.html">{html.escape(str(row["Code"]))} {html.escape(str(row["Name"]))}</a>',
        axis=1,
    )

    index_path = args.viz_dir / "index.html"
    index_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>殖利率本益比股淨比研究報告</title>
<style>
body {{ margin: 24px; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 28px; font-size: 20px; }}
p, li {{ color: #334155; line-height: 1.6; }}
.meta {{ color: #64748b; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }}
.stat {{ border: 1px solid #d7dee9; background: white; padding: 12px 14px; }}
.stat b {{ display: block; font-size: 22px; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 20px; font-size: 13px; }}
th, td {{ border: 1px solid #d7dee9; padding: 7px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #eef2f7; }}
a {{ color: #0f766e; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
svg {{ width: 100%; height: auto; background: white; border: 1px solid #d7dee9; margin: 12px 0; }}
.note {{ border-left: 4px solid #0f766e; padding: 8px 12px; background: white; }}
</style>
</head>
<body>
<h1>殖利率、本益比、股價淨值比研究報告</h1>
<div class="meta">資料期間：{start_date} 至 {latest_date}；報告產生於 {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
<div class="grid">
  <div class="stat">股票數<b>{stock_count:,}</b></div>
  <div class="stat">股票日樣本<b>{row_count:,}</b></div>
  <div class="stat">單股互動頁<b>{stock_page_count:,}</b></div>
  <div class="stat">檢驗 horizon<b>{', '.join(str(h) + '日' for h in parse_horizons(args.horizons))}</b></div>
</div>
<section class="note">
  <p>本報告把本益比轉成盈餘殖利率，股價淨值比轉成淨值市價比；數值越高代表越便宜。報酬標籤使用訊號日後的下一個復權開盤價作為進場價，再看到未來 N 日復權開盤價，避免盤後資料被當成同日可交易訊號。</p>
</section>
<h2>RankIC 最佳組合</h2>
{bar_svg(best_ic.assign(Label=best_ic["FactorLabel"] + " / " + best_ic["Horizon"].astype(str) + "日"), "Label", "MeanRankIC", "平均 RankIC")}
{table_html(best_ic, [("FactorLabel", "因子", "text"), ("Horizon", "天期", "int"), ("MeanIC", "平均IC", "num"), ("MeanRankIC", "平均RankIC", "num"), ("PositiveRankICRate", "正RankIC比例", "pct"), ("RankICTStat", "RankIC t值", "num"), ("DateCount", "日期數", "int"), ("AverageSampleCount", "平均樣本", "num")])}
<h2>20日多空分組報酬</h2>
{bar_svg(q20.assign(Label=q20["FactorLabel"]), "Label", "MeanReturn", "高分組減低分組平均報酬")}
{table_html(q20, [("FactorLabel", "因子", "text"), ("Horizon", "天期", "int"), ("MeanReturn", "平均報酬", "pct"), ("PositiveRate", "正報酬比例", "pct"), ("DateCount", "日期數", "int")])}
<h2>最新便宜股快照</h2>
{table_html(latest_top, [("StockLink", "股票", "html"), ("Group", "產業", "text"), ("Close", "收盤價", "num"), ("DividendYield", "殖利率", "num"), ("PEratio", "本益比", "num"), ("PBratio", "股價淨值比", "num"), ("EarningsYield", "盈餘殖利率", "num"), ("BookToMarket", "淨值市價比", "num"), ("DividendYieldPercentile", "殖利率分位", "pct"), ("EarningsYieldPercentile", "盈餘殖利率分位", "pct"), ("BookToMarketPercentile", "淨值市價比分位", "pct")])}
<h2>輸出檔案</h2>
<ul>
  <li><a href="../../output/yield_pe_pb_research/factor_ic_summary.csv">IC 摘要 CSV</a></li>
  <li><a href="../../output/yield_pe_pb_research/factor_ic_timeseries.csv">IC 時序 CSV</a></li>
  <li><a href="../../output/yield_pe_pb_research/quantile_return_summary.csv">分組報酬摘要 CSV</a></li>
  <li><a href="../../output/yield_pe_pb_research/latest_factor_snapshot.csv">最新因子快照 CSV</a></li>
</ul>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    print("build panel...")
    panel, stock_panels = build_panel(args, horizons)
    panel = add_cross_sectional_factor_variants(panel)
    print(f"panel rows={len(panel):,}, stocks={panel['Code'].nunique():,}")
    print("compute IC...")
    ic_timeseries, ic_summary = compute_ic(panel, horizons)
    print("compute quantile returns...")
    quantile_daily, quantile_summary = compute_quantile_returns(panel, horizons, args.quantiles)
    print("build latest snapshot...")
    latest = latest_snapshot(panel)

    print("write CSV outputs...")
    csv_panel = panel.copy()
    csv_panel["Date"] = csv_panel["Date"].dt.strftime("%Y-%m-%d")
    csv_panel.to_csv(args.output_dir / "research_panel.csv", index=False, encoding="utf-8-sig")
    ic_timeseries.assign(Date=ic_timeseries["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        args.output_dir / "factor_ic_timeseries.csv", index=False, encoding="utf-8-sig"
    )
    ic_summary.to_csv(args.output_dir / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    if not quantile_daily.empty:
        quantile_daily.assign(Date=quantile_daily["Date"].dt.strftime("%Y-%m-%d")).to_csv(
            args.output_dir / "quantile_returns_daily.csv", index=False, encoding="utf-8-sig"
        )
    quantile_summary.to_csv(args.output_dir / "quantile_return_summary.csv", index=False, encoding="utf-8-sig")
    latest.assign(Date=latest["Date"].dt.strftime("%Y-%m-%d")).to_csv(
        args.output_dir / "latest_factor_snapshot.csv", index=False, encoding="utf-8-sig"
    )

    print("write stock pages...")
    stock_page_count = write_stock_pages(stock_panels, latest, args)
    print("write index...")
    index_path = write_index_html(args, panel, ic_summary, quantile_summary, latest, stock_page_count)
    metadata = {
        "stock_count": int(panel["Code"].nunique()),
        "row_count": int(len(panel)),
        "start_date": panel["Date"].min().strftime("%Y-%m-%d"),
        "end_date": panel["Date"].max().strftime("%Y-%m-%d"),
        "stock_page_count": int(stock_page_count),
        "index_path": str(index_path),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
