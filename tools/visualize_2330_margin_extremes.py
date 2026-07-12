"""Build an interactive 2330 price/margin overlay with margin extreme markers."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from tools.analyze_2330_margin_patterns import Config as PatternConfig
from tools.analyze_2330_margin_patterns import load_panel
from viz.generate_dataset_viz import load_ex_right_events_for_stock, write_price_webgl_page

OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "2330"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_patterns"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"

MARGIN_METRICS = [
    ("MarginCurrentBalance", "margin_balance", "融資今日餘額"),
    (SIGNAL_COLUMN, "margin_20d_change", "融資餘額20日變化率"),
    ("MarginFinancingUsageRate", "margin_usage", "融資使用率"),
    ("MarginMarketValueTo20DayAvgTurnover", "margin_value_turnover", "融資市值20日均成交值比"),
    ("ShortCurrentBalance", "short_balance", "融券今日餘額"),
    ("ShortMarginBalanceRatio", "short_margin_ratio", "券資比"),
]

HIGHLIGHT_RULES = [
    {
        "key": "margin_surge_zone",
        "label": "融資大漲",
        "panel": "price",
        "color": "#f97316",
        "opacity": 0.14,
        "marker": "top",
    },
    {
        "key": "margin_drop_zone",
        "label": "融資大跌",
        "panel": "price",
        "color": "#22c55e",
        "opacity": 0.11,
        "marker": "bottom",
    },
    {
        "key": "margin_high_level_zone",
        "label": "融資高水位",
        "panel": "margin",
        "color": "#8b5cf6",
        "opacity": 0.14,
    },
    {
        "key": "margin_low_level_zone",
        "label": "融資低水位",
        "panel": "margin",
        "color": "#38bdf8",
        "opacity": 0.16,
    },
]


@dataclass
class ExtremeConfig:
    code: str
    change_top_quantile: float
    change_bottom_quantile: float
    level_high_quantile: float
    level_low_quantile: float
    output_dir: Path
    viz_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize margin extremes on an interactive price/margin chart.")
    parser.add_argument("--code", default="2330")
    parser.add_argument("--change-top-quantile", type=float, default=0.90)
    parser.add_argument("--change-bottom-quantile", type=float, default=0.10)
    parser.add_argument("--level-high-quantile", type=float, default=0.80)
    parser.add_argument("--level-low-quantile", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def format_num(value: Any, digits: int = 0) -> str:
    number = json_float(value)
    if number is None:
        return ""
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def build_margin_payload(
    panel: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    available_metrics = [(column, key, label) for column, key, label in MARGIN_METRICS if column in panel.columns]
    payload: dict[str, dict[str, float]] = {}
    for row in panel.itertuples(index=False):
        values: dict[str, float] = {}
        for column, key, _label in available_metrics:
            value = json_float(getattr(row, column))
            if value is not None:
                values[key] = value
        signal_value = json_float(getattr(row, SIGNAL_COLUMN, None))
        margin_balance = json_float(getattr(row, "MarginCurrentBalance", None))
        if signal_value is not None:
            if signal_value >= thresholds["change_top"]:
                values["margin_surge_zone"] = 1.0
            if signal_value <= thresholds["change_bottom"]:
                values["margin_drop_zone"] = 1.0
        if margin_balance is not None:
            if margin_balance >= thresholds["level_high"]:
                values["margin_high_level_zone"] = 1.0
            if margin_balance <= thresholds["level_low"]:
                values["margin_low_level_zone"] = 1.0
        if values:
            values["margin_extreme_any"] = 1.0 if any(key.endswith("_zone") for key in values) else 0.0
            payload[row.Date.strftime("%Y-%m-%d")] = values
    metrics = [{"key": key, "label": label} for _column, key, label in available_metrics]
    return payload, metrics


def event_table(panel: pd.DataFrame, mask: pd.Series, sort_column: str, ascending: bool, limit: int = 12) -> str:
    rows = panel[mask].sort_values(sort_column, ascending=ascending).head(limit)
    columns = [
        ("Date", "日期", "date"),
        (SIGNAL_COLUMN, "融資20日變化", "pct"),
        ("MarginCurrentBalance", "融資餘額", "number"),
        ("close_adj", "復權收盤價", "number"),
    ]
    head = "".join(f"<th>{html.escape(label)}</th>" for _column, label, _kind in columns)
    body = []
    for _index, row in rows.iterrows():
        cells = []
        for column, _label, kind in columns:
            value = row[column]
            if kind == "date":
                text = value.strftime("%Y-%m-%d")
            elif kind == "pct":
                text = format_pct(float(value))
            else:
                text = format_num(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_summary_html(panel: pd.DataFrame, thresholds: dict[str, float]) -> tuple[str, str, str]:
    surge = panel[SIGNAL_COLUMN].ge(thresholds["change_top"])
    drop = panel[SIGNAL_COLUMN].le(thresholds["change_bottom"])
    high_level = panel["MarginCurrentBalance"].ge(thresholds["level_high"])
    latest = panel.iloc[-1]
    surge_table = event_table(panel, surge, SIGNAL_COLUMN, False)
    drop_table = event_table(panel, drop, SIGNAL_COLUMN, True)

    extra_styles = """
.summary-cards { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin: 14px 0 12px; }
.metric-card, .viz-section { background: white; border: 1px solid #dfe5ef; border-radius: 6px; }
.metric-card { padding: 12px 14px; }
.metric-label { color: #59677c; font-size: 12px; }
.metric-value { margin-top: 4px; font-weight: 700; font-size: 19px; }
.viz-section { margin: 18px 0; padding: 16px; overflow-x: auto; }
.viz-section h2 { margin: 0 0 10px; font-size: 17px; }
.note { color: #4b5870; font-size: 13px; line-height: 1.65; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { border-bottom: 1px solid #e5eaf2; padding: 8px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { background: #f2f5f9; color: #334155; }
@media (max-width: 960px) { .summary-cards, .grid2 { grid-template-columns: 1fr; } }
"""
    before_chart = f"""
<div class="summary-cards">
  <div class="metric-card"><div class="metric-label">融資大漲門檻</div><div class="metric-value">>= {format_pct(thresholds['change_top'])}</div></div>
  <div class="metric-card"><div class="metric-label">融資大跌門檻</div><div class="metric-value">&lt;= {format_pct(thresholds['change_bottom'])}</div></div>
  <div class="metric-card"><div class="metric-label">融資高水位門檻</div><div class="metric-value">>= {format_num(thresholds['level_high'])}</div></div>
  <div class="metric-card"><div class="metric-label">融資低水位門檻</div><div class="metric-value">&lt;= {format_num(thresholds['level_low'])}</div></div>
</div>
<div class="summary-cards">
  <div class="metric-card"><div class="metric-label">融資大漲交易日</div><div class="metric-value">{int(surge.sum()):,}</div></div>
  <div class="metric-card"><div class="metric-label">融資大跌交易日</div><div class="metric-value">{int(drop.sum()):,}</div></div>
  <div class="metric-card"><div class="metric-label">融資高水位交易日</div><div class="metric-value">{int(high_level.sum()):,}</div></div>
  <div class="metric-card"><div class="metric-label">最新融資變化 / 餘額</div><div class="metric-value">{format_pct(latest[SIGNAL_COLUMN])} / {format_num(latest.MarginCurrentBalance)}</div></div>
</div>
"""
    after_chart = f"""
<section class="viz-section">
  <h2>讀圖方式</h2>
  <div class="note">
    上方 K 線用紅色區塊標出融資20日變化率的歷史 top 10%，綠色區塊標出 bottom 10%。
    下方融資指標區用粉紅色表示融資餘額高於自身歷史 80% 分位，淺藍色表示低於 20% 分位。
    互動方式和 price 視覺化一致：滾輪縮放、拖曳平移、滑鼠移動游標、左右鍵逐日切換。
  </div>
</section>
<div class="grid2">
  <section class="viz-section"><h2>融資20日大漲代表日</h2>{surge_table}</section>
  <section class="viz-section"><h2>融資20日大跌代表日</h2>{drop_table}</section>
</div>
"""
    return before_chart, after_chart, extra_styles


def write_report(
    panel: pd.DataFrame,
    thresholds: dict[str, float],
    config: ExtremeConfig,
    price_path: Path,
    margin_path: Path,
) -> Path:
    config.viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.viz_dir / f"{config.code}_margin_extreme_overlay.html"
    price_df = read_csv_canonical(price_path)
    margin_by_date, margin_metrics = build_margin_payload(panel, thresholds)
    events_by_date, event_csv = load_ex_right_events_for_stock(price_path, price_df)
    source_paths = [price_path, margin_path]
    if event_csv is not None:
        source_paths.append(event_csv)
    before_chart, after_chart, extra_styles = build_summary_html(panel, thresholds)
    title = price_path.stem.replace("_", " ")
    ok = write_price_webgl_page(
        price_path,
        output_path,
        title,
        price_df,
        source_paths=source_paths,
        margin_by_date=margin_by_date,
        margin_metrics=margin_metrics,
        events_by_date=events_by_date,
        page_suffix="價格與融資極端區",
        metric_control_label="融資指標",
        highlight_rules=HIGHLIGHT_RULES,
        extra_body_before_chart=before_chart,
        extra_body_after_chart=after_chart,
        extra_styles=extra_styles,
    )
    if not ok:
        raise RuntimeError(f"Price CSV did not contain renderable OHLC rows: {price_path}")
    return output_path


def main() -> None:
    args = parse_args()
    config = ExtremeConfig(
        code=str(args.code),
        change_top_quantile=args.change_top_quantile,
        change_bottom_quantile=args.change_bottom_quantile,
        level_high_quantile=args.level_high_quantile,
        level_low_quantile=args.level_low_quantile,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
    )
    pattern_config = PatternConfig(
        code=config.code,
        window=20,
        top_quantile=0.90,
        bottom_quantile=0.10,
        near_high_band=0.05,
        breakout_threshold=0.03,
        plateau_band=0.02,
        output_dir=config.output_dir,
        viz_dir=config.viz_dir,
    )
    panel, price_path, margin_path = load_panel(pattern_config)
    panel = panel.reset_index(drop=True)
    thresholds = {
        "change_top": float(panel[SIGNAL_COLUMN].quantile(config.change_top_quantile)),
        "change_bottom": float(panel[SIGNAL_COLUMN].quantile(config.change_bottom_quantile)),
        "level_high": float(panel["MarginCurrentBalance"].quantile(config.level_high_quantile)),
        "level_low": float(panel["MarginCurrentBalance"].quantile(config.level_low_quantile)),
    }
    report_path = write_report(panel, thresholds, config, price_path, margin_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "code": config.code,
        "rows": int(len(panel)),
        "start": panel["Date"].iloc[0].strftime("%Y-%m-%d"),
        "end": panel["Date"].iloc[-1].strftime("%Y-%m-%d"),
        **thresholds,
        "price_path": str(price_path.relative_to(PROJECT_ROOT)),
        "margin_path": str(margin_path.relative_to(PROJECT_ROOT)),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
    }
    summary_path = config.output_dir / "margin_extreme_overlay_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
