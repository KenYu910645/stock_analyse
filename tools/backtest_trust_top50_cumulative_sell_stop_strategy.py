from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from backtest_institutional_flow_strategy import (
    DATA_VIZ_ROOT,
    DEFAULT_MIN_DAILY_TURNOVER,
    DEFAULT_STOP_LOSS_RATE,
    DEFAULT_TOP_N,
    OUTPUT_ROOT,
    PRICE_DIR,
    INSTITUTIONAL_DIR,
    add_signals_and_returns,
    active_position_counts,
    fmt_num,
    fmt_pct,
    histogram_svg,
    latest_price_date,
    listed_common_codes,
    load_stock_panel,
    monthly_dynamic_summary,
    path_by_code,
    select_trust_cumulative_sell_stop_trades,
    stock_name_from_path,
    summarize_trust_cumulative_sell_stop,
    table_rows,
    to_float,
    write_summary_report,
)


SELECTION_HOLDING_DAYS = 60
SELECTION_MIN_TRADES = 30
TOP_N_STOCKS = 50
EXIT_REASON_LABELS = {
    "quota_depleted": "\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6",
    "cumulative_sell": "\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6",
    "stop_loss": "\u505c\u640d",
    "stop_loss_and_quota_depleted": "\u505c\u640d\u4e14 quota \u6b78\u96f6",
    "stop_loss_and_cumulative_sell": "\u505c\u640d\u4e14 quota \u6b78\u96f6",
    "data_end_mark": "\u8cc7\u6599\u7d50\u675f",
}


def profit_factor(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    gains = clean[clean.gt(0)].sum()
    losses = clean[clean.lt(0)].sum()
    return float(gains / abs(losses)) if losses < 0 else None


def add_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_csv(PROJECT_ROOT / "data" / "metadata.csv", dtype={"Code": str}, encoding="utf-8-sig")
    columns = ["Code", "\u7522\u696d\u7fa4\u7d44", "\u985e\u578b", "\u5e02\u5834"]
    available = [column for column in columns if column in metadata.columns]
    merged = frame.merge(metadata[available].drop_duplicates("Code"), on="Code", how="left")
    if "\u7522\u696d\u7fa4\u7d44" in merged.columns:
        merged = merged.rename(columns={"\u7522\u696d\u7fa4\u7d44": "IndustryGroup"})
    return merged


def build_full_history_trust_top_stocks(
    *,
    top_n_stocks: int = TOP_N_STOCKS,
    min_trades: int = SELECTION_MIN_TRADES,
    min_daily_turnover: float = DEFAULT_MIN_DAILY_TURNOVER,
) -> pd.DataFrame:
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    allowed = listed_common_codes(limit=None)
    codes = sorted((set(price_paths) & set(institutional_paths)) & allowed)
    rows: list[dict[str, Any]] = []
    start_buffer = pd.Timestamp("1900-01-01")
    net_col = f"net_return_{SELECTION_HOLDING_DAYS}d"
    score_col = "score_trust_1d"
    turnover_col = "turnover_1d"
    min_window_turnover = float(min_daily_turnover)

    for index, code in enumerate(codes, start=1):
        try:
            frame = load_stock_panel(code, price_paths[code], institutional_paths[code], start_buffer)
            if frame.empty:
                continue
            frame["Name"] = stock_name_from_path(institutional_paths[code], stock_name_from_path(price_paths[code], code))
            frame = add_signals_and_returns(frame)
            candidates = frame[
                frame[score_col].gt(0)
                & frame["trust_net"].gt(0)
                & frame[turnover_col].ge(min_window_turnover)
                & frame[net_col].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
            ].copy()
            if len(candidates) < min_trades:
                continue
            returns = candidates[net_col].astype(float)
            rows.append(
                {
                    "Code": code,
                    "Name": frame["Name"].iloc[0],
                    "SelectionTradeCount": int(len(candidates)),
                    "SelectionAvgNetReturn60D": float(returns.mean()),
                    "SelectionMedianNetReturn60D": float(returns.median()),
                    "SelectionWinRate60D": float(returns.gt(0).mean()),
                    "SelectionProfitFactor60D": profit_factor(returns),
                    "SelectionAvgScore": float(candidates[score_col].mean()),
                    "FirstSignalDate": pd.to_datetime(candidates["Date"], errors="coerce").min().strftime("%Y-%m-%d"),
                    "LastSignalDate": pd.to_datetime(candidates["Date"], errors="coerce").max().strftime("%Y-%m-%d"),
                }
            )
        except Exception as exc:
            print(f"skip_selection {code}: {exc}")
        if index % 100 == 0 or index == len(codes):
            print(f"selection processed {index}/{len(codes)}")

    if not rows:
        raise SystemExit("no_trust_top_stock_selection_rows")
    ranking = pd.DataFrame(rows).sort_values(
        ["SelectionAvgNetReturn60D", "SelectionTradeCount", "SelectionWinRate60D"],
        ascending=[False, False, False],
    )
    ranking = add_metadata_columns(ranking).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    return ranking.head(top_n_stocks).copy()


def build_recent_panel_for_codes(codes: set[str], *, lookback_years: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    allowed = listed_common_codes(limit=None) & set(codes)
    valid_codes = sorted((set(price_paths) & set(institutional_paths)) & allowed)
    latest = latest_price_date(price_paths, valid_codes)
    start = latest - pd.DateOffset(years=lookback_years)
    start_buffer = start - pd.Timedelta(days=120)

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for index, code in enumerate(valid_codes, start=1):
        try:
            frame = load_stock_panel(code, price_paths[code], institutional_paths[code], start_buffer)
            if frame.empty:
                skipped.append({"Code": code, "Reason": "empty_panel"})
                continue
            frame["Name"] = stock_name_from_path(institutional_paths[code], stock_name_from_path(price_paths[code], code))
            frame = add_signals_and_returns(frame)
            frame = frame[frame["Date"].ge(start)].copy()
            if frame.empty:
                skipped.append({"Code": code, "Reason": "no_rows_after_start"})
                continue
            rows.append(frame)
        except Exception as exc:
            skipped.append({"Code": code, "Reason": str(exc)})
        if index % 10 == 0 or index == len(valid_codes):
            print(f"top50 panel processed {index}/{len(valid_codes)}")

    if not rows:
        raise SystemExit("no_top50_signal_rows_built")
    panel = pd.concat(rows, ignore_index=True)
    meta = {
        "latest_date": latest.strftime("%Y-%m-%d"),
        "start_date": start.strftime("%Y-%m-%d"),
        "stock_count": len(valid_codes),
        "loaded_stock_count": int(panel["Code"].nunique()),
        "skipped": skipped,
    }
    return panel, meta


def comparison_table(summary: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    row = summary.iloc[0]
    rows.append(
        {
            "StrategyName": "\u524d50\u6a94\uff1a\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6+10%\u505c\u640d",
            "TradeCount": row.TradeCount,
            "AvgNetReturn": row.AvgNetReturn,
            "MedianNetReturn": row.MedianNetReturn,
            "WinRate": row.WinRate,
            "ProfitFactor": row.ProfitFactor,
            "AvgHoldingTradingDays": row.AvgHoldingTradingDays,
            "AverageActivePositions": row.AverageActivePositions,
        }
    )
    full_path = OUTPUT_ROOT / "trust_cumulative_sell_stop_summary.csv"
    if full_path.exists():
        full = pd.read_csv(full_path, encoding="utf-8-sig").iloc[0]
        rows.append(
            {
                "StrategyName": "\u5168\u5e02\u5834\uff1a\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6+10%\u505c\u640d",
                "TradeCount": full.TradeCount,
                "AvgNetReturn": full.AvgNetReturn,
                "MedianNetReturn": full.MedianNetReturn,
                "WinRate": full.WinRate,
                "ProfitFactor": full.ProfitFactor,
                "AvgHoldingTradingDays": full.AvgHoldingTradingDays,
                "AverageActivePositions": full.AverageActivePositions,
            }
        )
    columns = [
        ("StrategyName", "\u7b56\u7565", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("MedianNetReturn", "\u4e2d\u4f4d\u6578", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("ProfitFactor", "\u7372\u5229\u56e0\u5b50", "num"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
        ("AverageActivePositions", "\u5e73\u5747\u6d3b\u8e8d\u6301\u80a1", "num"),
    ]
    heads = "".join(f"<th>{label}</th>" for _column, label, _kind in columns)
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{table_rows(pd.DataFrame(rows), columns)}</tbody></table>"


def timeline_date(value: Any) -> str:
    if pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.notna(timestamp):
        return timestamp.strftime("%Y-%m-%d")
    text = str(value)
    return text[:10] if text else ""


def timeline_float(value: Any, digits: int = 6) -> float | None:
    number = to_float(value)
    if number is None or not math.isfinite(number):
        return None
    return round(number, digits)


def timeline_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None or not math.isfinite(number):
        return None
    return int(round(number))


def build_strategy_timeline(trades: pd.DataFrame, active_counts: pd.DataFrame) -> list[dict[str, Any]]:
    active = active_counts.copy()
    active["DateText"] = active["Date"].map(timeline_date)
    active = active[active["DateText"].ne("")].sort_values("DateText")
    active = active.drop_duplicates("DateText", keep="last")
    dates = active["DateText"].tolist()

    buy_events: dict[str, list[dict[str, Any]]] = {}
    exit_events: dict[str, list[dict[str, Any]]] = {}
    trade_rows = trades.copy()
    trade_rows["EntryDateText"] = trade_rows["EntryDate"].map(timeline_date)
    trade_rows["ExitDateText"] = trade_rows["ExitDate"].map(timeline_date)
    trade_rows = trade_rows.sort_values(["EntryDateText", "BuyRank", "Code"])

    for row in trade_rows.to_dict("records"):
        code = str(row.get("Code", ""))
        name = "" if pd.isna(row.get("Name")) else str(row.get("Name", ""))
        entry_date = row.get("EntryDateText", "")
        if entry_date:
            buy_events.setdefault(entry_date, []).append(
                {
                    "type": "buy",
                    "action": "\u8cb7\u9032",
                    "code": code,
                    "name": name,
                    "signalDate": timeline_date(row.get("EntrySignalDate")),
                    "entryDate": entry_date,
                    "rank": timeline_int(row.get("BuyRank")),
                    "buyScore": timeline_float(row.get("BuyScore"), 6),
                    "entryOpen": timeline_float(row.get("EntryOpen"), 4),
                    "entryBuyNetShares": timeline_int(row.get("EntryBuyNetShares")),
                }
            )
        exit_date = row.get("ExitDateText", "")
        if exit_date:
            reason = "" if pd.isna(row.get("ExitReason")) else str(row.get("ExitReason", ""))
            exit_events.setdefault(exit_date, []).append(
                {
                    "type": "exit",
                    "action": "\u51fa\u5834",
                    "code": code,
                    "name": name,
                    "entryDate": entry_date,
                    "triggerDate": timeline_date(row.get("ExitTriggerDate")),
                    "exitDate": exit_date,
                    "reason": EXIT_REASON_LABELS.get(reason, reason),
                    "netReturn": timeline_float(row.get("NetReturn"), 6),
                    "holdingTradingDays": timeline_int(row.get("HoldingTradingDays")),
                    "triggerQuotaShares": timeline_int(row.get("TriggerTrustQuotaShares")),
                    "quotaRatio": timeline_float(row.get("TrustQuotaToInitialBuyRatio"), 6),
                    "netFlowAfterSignalShares": timeline_int(row.get("TrustNetFlowAfterSignalShares")),
                    "exitOpen": timeline_float(row.get("ExitOpen"), 4),
                }
            )

    timeline: list[dict[str, Any]] = []
    cumulative_return_sum = 0.0
    cumulative_exit_count = 0
    active_by_date = dict(zip(active["DateText"], active["ActivePositions"], strict=False))
    for date in dates:
        exits = exit_events.get(date, [])
        buys = buy_events.get(date, [])
        exit_returns = [
            float(event["netReturn"])
            for event in exits
            if event.get("netReturn") is not None and math.isfinite(float(event["netReturn"]))
        ]
        if exit_returns:
            cumulative_return_sum += sum(exit_returns)
            cumulative_exit_count += len(exit_returns)
        timeline.append(
            {
                "date": date,
                "activePositions": timeline_int(active_by_date.get(date)) or 0,
                "buyCount": len(buys),
                "exitCount": len(exits),
                "dailyExitAvgReturn": round(sum(exit_returns) / len(exit_returns), 6) if exit_returns else None,
                "cumulativeClosedAvgReturn": (
                    round(cumulative_return_sum / cumulative_exit_count, 6) if cumulative_exit_count else None
                ),
                "events": buys + exits,
            }
        )
    return timeline


def render_strategy_timeline_chart(timeline: list[dict[str, Any]]) -> str:
    timeline_json = json.dumps(timeline, ensure_ascii=True, separators=(",", ":"))
    return """
<style>
.timeline-shell { position: relative; min-width: 760px; }
.timeline-legend { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 4px 0 10px; color: #334155; font-size: 13px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.legend-swatch { width: 18px; height: 3px; border-radius: 999px; display: inline-block; }
.timeline-wrap { position: relative; height: 470px; min-width: 760px; border: 1px solid #d7dee9; border-radius: 6px; background: #ffffff; }
#strategyTimelineCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
#strategyTimelineTooltip { position: absolute; z-index: 5; max-width: 430px; min-width: 260px; pointer-events: none; opacity: 0; transform: translate(12px, 12px); background: rgba(15, 23, 42, 0.95); color: #f8fafc; border-radius: 6px; padding: 10px 12px; box-shadow: 0 18px 50px rgba(15, 23, 42, 0.24); font-size: 12px; line-height: 1.45; }
.tooltip-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
.tooltip-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 10px; margin-bottom: 8px; color: #cbd5e1; }
.tooltip-events { max-height: 270px; overflow: hidden; border-top: 1px solid rgba(203, 213, 225, 0.25); padding-top: 6px; }
.tooltip-event { margin-top: 5px; }
.tooltip-event strong { color: #ffffff; }
.tooltip-pos { color: #86efac; font-weight: 700; }
.tooltip-neg { color: #fca5a5; font-weight: 700; }
</style>
<div class="timeline-shell">
  <div class="timeline-legend">
    <span class="legend-item"><span class="legend-swatch" style="background:#2563eb"></span>累積已出場平均報酬</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#f59e0b"></span>活躍持股數</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#10b981"></span>買進筆數</span>
    <span class="legend-item"><span class="legend-swatch" style="background:#ef4444"></span>出場筆數</span>
  </div>
  <div class="timeline-wrap">
    <canvas id="strategyTimelineCanvas" tabindex="0"></canvas>
    <div id="strategyTimelineTooltip"></div>
  </div>
</div>
<script>
(() => {
  const timeline = __TIMELINE_JSON__;
  const canvas = document.getElementById("strategyTimelineCanvas");
  const tooltip = document.getElementById("strategyTimelineTooltip");
  if (!canvas || !tooltip || !timeline.length) return;
  const ctx = canvas.getContext("2d");
  let width = 1;
  let height = 1;
  let dpr = 1;
  let hoverIndex = timeline.length - 1;
  let viewStart = 0;
  let viewEnd = timeline.length - 1;
  let dragging = false;
  let dragX = 0;
  let dragStart = 0;
  let dragEnd = timeline.length - 1;

  function finite(value) {
    return Number.isFinite(Number(value));
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatNumber(value, digits = 1) {
    if (!finite(value)) return "-";
    return Number(value).toLocaleString("zh-TW", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  }

  function formatInt(value) {
    if (!finite(value)) return "-";
    return Math.round(Number(value)).toLocaleString("zh-TW");
  }

  function formatPct(value, digits = 1) {
    if (!finite(value)) return "-";
    return `${(Number(value) * 100).toFixed(digits)}%`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, match => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    })[match]);
  }

  function layout() {
    return {
      left: 78,
      right: 76,
      top: 28,
      lineBottom: height - 130,
      barTop: height - 106,
      barBottom: height - 52,
      dateY: height - 22,
      plotWidth: Math.max(1, width - 154)
    };
  }

  function visibleRows() {
    return timeline.slice(viewStart, viewEnd + 1);
  }

  function xForIndex(index, box) {
    if (viewEnd === viewStart) return box.left + box.plotWidth / 2;
    return box.left + ((index - viewStart) / (viewEnd - viewStart)) * box.plotWidth;
  }

  function indexFromX(x, box) {
    const ratio = clamp((x - box.left) / box.plotWidth, 0, 1);
    return clamp(Math.round(viewStart + ratio * (viewEnd - viewStart)), viewStart, viewEnd);
  }

  function scaledY(value, min, max, top, bottom) {
    if (!finite(value) || max === min) return bottom;
    return bottom - ((Number(value) - min) / (max - min)) * (bottom - top);
  }

  function drawLine(points, color, box, yMin, yMax, key, widthPx = 2) {
    ctx.beginPath();
    let started = false;
    for (let index = viewStart; index <= viewEnd; index += 1) {
      const value = timeline[index][key];
      if (!finite(value)) continue;
      const x = xForIndex(index, box);
      const y = scaledY(value, yMin, yMax, box.top, box.lineBottom);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = widthPx;
    ctx.stroke();
  }

  function draw() {
    const box = layout();
    const rows = visibleRows();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const returns = rows.map(row => row.cumulativeClosedAvgReturn).filter(finite).map(Number);
    const retMinRaw = Math.min(0, ...returns);
    const retMaxRaw = Math.max(0, ...returns);
    const retPad = Math.max(0.01, (retMaxRaw - retMinRaw) * 0.12);
    const retMin = retMinRaw - retPad;
    const retMax = retMaxRaw + retPad;
    const activeMax = Math.max(1, ...rows.map(row => Number(row.activePositions || 0)));
    const activeYMax = Math.ceil(activeMax * 1.12);
    const eventMax = Math.max(1, ...rows.map(row => Math.max(Number(row.buyCount || 0), Number(row.exitCount || 0))));

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "12px Microsoft JhengHei, Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = box.top + ((box.lineBottom - box.top) * tick) / 4;
      ctx.beginPath();
      ctx.moveTo(box.left, y);
      ctx.lineTo(width - box.right, y);
      ctx.stroke();
      const retValue = retMax - ((retMax - retMin) * tick) / 4;
      ctx.fillText(formatPct(retValue, 1), box.left - 10, y);
      ctx.textAlign = "left";
      const activeValue = activeYMax - (activeYMax * tick) / 4;
      ctx.fillText(formatNumber(activeValue, 0), width - box.right + 10, y);
      ctx.textAlign = "right";
    }

    ctx.fillStyle = "#334155";
    ctx.save();
    ctx.translate(18, (box.top + box.lineBottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("累積已出場平均報酬", 0, 0);
    ctx.restore();
    ctx.save();
    ctx.translate(width - 18, (box.top + box.lineBottom) / 2);
    ctx.rotate(Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("活躍持股數", 0, 0);
    ctx.restore();

    drawLine(rows, "#2563eb", box, retMin, retMax, "cumulativeClosedAvgReturn", 2.4);
    drawLine(rows, "#f59e0b", box, 0, activeYMax, "activePositions", 2);

    const slot = box.plotWidth / Math.max(1, viewEnd - viewStart + 1);
    const barWidth = clamp(slot * 0.68, 1, 8);
    const barMid = (box.barTop + box.barBottom) / 2;
    const barHalf = (box.barBottom - box.barTop) / 2 - 2;
    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(box.left, barMid);
    ctx.lineTo(width - box.right, barMid);
    ctx.stroke();
    for (let index = viewStart; index <= viewEnd; index += 1) {
      const row = timeline[index];
      const x = xForIndex(index, box) - barWidth / 2;
      const buyHeight = (Number(row.buyCount || 0) / eventMax) * barHalf;
      const exitHeight = (Number(row.exitCount || 0) / eventMax) * barHalf;
      if (buyHeight > 0) {
        ctx.fillStyle = "#10b981";
        ctx.fillRect(x, barMid - buyHeight, barWidth, buyHeight);
      }
      if (exitHeight > 0) {
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(x, barMid, barWidth, exitHeight);
      }
    }
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "right";
    ctx.fillText("買進", box.left - 10, barMid - barHalf * 0.62);
    ctx.fillText("出場", box.left - 10, barMid + barHalf * 0.62);

    const tickCount = Math.min(7, Math.max(2, Math.floor(box.plotWidth / 150)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let tick = 0; tick < tickCount; tick += 1) {
      const index = Math.round(viewStart + ((viewEnd - viewStart) * tick) / (tickCount - 1 || 1));
      const x = xForIndex(index, box);
      ctx.fillStyle = "#64748b";
      ctx.fillText(timeline[index].date, x, box.dateY);
    }

    if (hoverIndex >= viewStart && hoverIndex <= viewEnd) {
      const row = timeline[hoverIndex];
      const x = xForIndex(hoverIndex, box);
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, box.top);
      ctx.lineTo(x, box.barBottom);
      ctx.stroke();
      if (finite(row.cumulativeClosedAvgReturn)) {
        const y = scaledY(row.cumulativeClosedAvgReturn, retMin, retMax, box.top, box.lineBottom);
        ctx.fillStyle = "#2563eb";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function eventHtml(event) {
    const title = `${escapeHtml(event.action)} ${escapeHtml(event.code)} ${escapeHtml(event.name)}`;
    if (event.type === "buy") {
      return `<div class="tooltip-event"><strong>${title}</strong><br>訊號日 ${escapeHtml(event.signalDate)} ｜ 排名 ${formatInt(event.rank)} ｜ 買超 ${formatInt(event.entryBuyNetShares)} 股 ｜ 進場價 ${formatNumber(event.entryOpen, 2)}</div>`;
    }
    const cls = Number(event.netReturn || 0) >= 0 ? "tooltip-pos" : "tooltip-neg";
    return `<div class="tooltip-event"><strong>${title}</strong><br>原因 ${escapeHtml(event.reason)} ｜ 淨報酬 <span class="${cls}">${formatPct(event.netReturn, 2)}</span> ｜ 持有 ${formatInt(event.holdingTradingDays)} 日 ｜ 觸發 quota ${formatInt(event.triggerQuotaShares)} 股 ｜ quota/初買 ${formatPct(event.quotaRatio, 0)}</div>`;
  }

  function updateTooltip(clientX, clientY) {
    const row = timeline[hoverIndex];
    const events = row.events || [];
    const shown = events.slice(0, 16);
    const more = events.length > shown.length ? `<div class="tooltip-event">另有 ${events.length - shown.length} 筆交易</div>` : "";
    tooltip.innerHTML = `
      <div class="tooltip-title">${escapeHtml(row.date)}</div>
      <div class="tooltip-grid">
        <div>活躍持股 ${formatInt(row.activePositions)}</div>
        <div>累積均報酬 ${formatPct(row.cumulativeClosedAvgReturn, 2)}</div>
        <div>買進 ${formatInt(row.buyCount)} 筆</div>
        <div>出場 ${formatInt(row.exitCount)} 筆</div>
        <div>當日出場均報酬 ${formatPct(row.dailyExitAvgReturn, 2)}</div>
        <div>交易內容 ${formatInt(events.length)} 筆</div>
      </div>
      <div class="tooltip-events">${shown.map(eventHtml).join("") || "<div class='tooltip-event'>當日沒有交易</div>"}${more}</div>
    `;
    const wrap = canvas.parentElement.getBoundingClientRect();
    const maxLeft = wrap.width - tooltip.offsetWidth - 14;
    const maxTop = wrap.height - tooltip.offsetHeight - 14;
    tooltip.style.left = `${clamp(clientX - wrap.left + 12, 8, Math.max(8, maxLeft))}px`;
    tooltip.style.top = `${clamp(clientY - wrap.top + 12, 8, Math.max(8, maxTop))}px`;
    tooltip.style.opacity = "1";
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    width = Math.max(1, Math.floor(rect.width));
    height = Math.max(1, Math.floor(rect.height));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  canvas.addEventListener("mousemove", event => {
    const box = layout();
    const rect = canvas.getBoundingClientRect();
    if (dragging) {
      const span = dragEnd - dragStart;
      const delta = Math.round(((event.clientX - dragX) / box.plotWidth) * span);
      viewStart = clamp(dragStart - delta, 0, Math.max(0, timeline.length - 1 - span));
      viewEnd = viewStart + span;
    }
    hoverIndex = indexFromX(event.clientX - rect.left, box);
    draw();
    updateTooltip(event.clientX, event.clientY);
  });
  canvas.addEventListener("mouseleave", () => {
    tooltip.style.opacity = "0";
    dragging = false;
    draw();
  });
  canvas.addEventListener("mousedown", event => {
    dragging = true;
    dragX = event.clientX;
    dragStart = viewStart;
    dragEnd = viewEnd;
    canvas.focus();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const box = layout();
    const rect = canvas.getBoundingClientRect();
    const centerIndex = indexFromX(event.clientX - rect.left, box);
    const oldSpan = viewEnd - viewStart;
    const factor = event.deltaY > 0 ? 1.25 : 0.8;
    const newSpan = clamp(Math.round(oldSpan * factor), 30, timeline.length - 1);
    const leftRatio = oldSpan <= 0 ? 0.5 : (centerIndex - viewStart) / oldSpan;
    viewStart = clamp(Math.round(centerIndex - newSpan * leftRatio), 0, Math.max(0, timeline.length - 1 - newSpan));
    viewEnd = viewStart + newSpan;
    hoverIndex = clamp(centerIndex, viewStart, viewEnd);
    draw();
    updateTooltip(event.clientX, event.clientY);
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    viewStart = 0;
    viewEnd = timeline.length - 1;
    hoverIndex = timeline.length - 1;
    draw();
  });
  canvas.addEventListener("keydown", event => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const currentSpan = viewEnd - viewStart;
    hoverIndex = clamp(hoverIndex + (event.key === "ArrowRight" ? 1 : -1), 0, timeline.length - 1);
    if (hoverIndex < viewStart) {
      viewStart = hoverIndex;
      viewEnd = Math.min(timeline.length - 1, viewStart + currentSpan);
    } else if (hoverIndex > viewEnd) {
      viewEnd = hoverIndex;
      viewStart = Math.max(0, viewEnd - currentSpan);
    }
    draw();
  });
  window.addEventListener("resize", resize);
  resize();
})();
</script>
""".replace("__TIMELINE_JSON__", timeline_json)


def write_report(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    selection: pd.DataFrame,
    active_counts: pd.DataFrame,
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "trust_top50_cumulative_sell_stop_strategy_report.html"
    row = summary.iloc[0]
    timeline_chart = render_strategy_timeline_chart(build_strategy_timeline(trades, active_counts))
    reason_summary = (
        trades.groupby("ExitReason", dropna=False)
        .agg(
            TradeCount=("Code", "count"),
            AvgNetReturn=("NetReturn", "mean"),
            AvgHoldingTradingDays=("HoldingTradingDays", "mean"),
        )
        .reset_index()
    )
    reason_summary["ExitReasonName"] = reason_summary["ExitReason"].map(EXIT_REASON_LABELS).fillna(reason_summary["ExitReason"])
    reason_summary["ExitRate"] = reason_summary["TradeCount"] / max(len(trades), 1)
    reason_summary = reason_summary[["ExitReasonName", "TradeCount", "ExitRate", "AvgNetReturn", "AvgHoldingTradingDays"]]

    selection_view = selection.head(50).copy()
    selection_columns = [
        ("Rank", "\u6392\u540d", "int"),
        ("Code", "\u4ee3\u865f", "text"),
        ("Name", "\u540d\u7a31", "text"),
        ("IndustryGroup", "\u7522\u696d\u7fa4\u7d44", "text"),
        ("SelectionTradeCount", "\u5168\u6b77\u53f2\u8a0a\u865f\u6578", "int"),
        ("SelectionAvgNetReturn60D", "\u5168\u6b77\u53f260\u65e5\u5e73\u5747", "pct"),
        ("SelectionWinRate60D", "\u52dd\u7387", "pct"),
        ("SelectionProfitFactor60D", "\u7372\u5229\u56e0\u5b50", "num"),
    ]
    reason_columns = [
        ("ExitReasonName", "\u51fa\u5834\u539f\u56e0", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("ExitRate", "\u5360\u6bd4", "pct"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    monthly_columns = [
        ("ExitMonth", "\u51fa\u5834\u6708\u4efd", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("MedianNetReturn", "\u4e2d\u4f4d\u6578", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    recent = trades.sort_values("ExitDate", ascending=False).head(100).copy()
    recent["ExitReason"] = recent["ExitReason"].map(EXIT_REASON_LABELS).fillna(recent["ExitReason"])
    recent_columns = [
        ("EntrySignalDate", "\u8cb7\u8a0a\u65e5", "text"),
        ("EntryDate", "\u9032\u5834\u65e5", "text"),
        ("ExitTriggerDate", "\u89f8\u767c\u65e5", "text"),
        ("ExitDate", "\u51fa\u5834\u65e5", "text"),
        ("Code", "\u4ee3\u865f", "text"),
        ("Name", "\u540d\u7a31", "text"),
        ("NetReturn", "\u6de8\u5831\u916c", "pct"),
        ("HoldingTradingDays", "\u6301\u6709\u4ea4\u6613\u65e5", "int"),
        ("InitialTrustQuotaShares", "\u521d\u59cb quota", "int"),
        ("TriggerTrustQuotaShares", "\u89f8\u767c quota", "int"),
        ("TrustQuotaToInitialBuyRatio", "quota/\u521d\u59cb\u8cb7\u8d85", "pct"),
        ("TrustNetFlowAfterSignalShares", "\u8a0a\u865f\u5f8c\u6de8\u8cb7\u8ce3", "int"),
        ("ExitReason", "\u51fa\u5834\u539f\u56e0", "text"),
    ]
    selection_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in selection_columns)
    reason_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in reason_columns)
    monthly_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in monthly_columns)
    recent_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in recent_columns)
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u6295\u4fe1\u524d50\u6a94\u6de8\u8cb7\u8ce3 quota \u505c\u640d\u56de\u6e2c</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 19px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; overflow-x: auto; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6),
th:nth-child(1), th:nth-child(2), th:nth-child(3), th:nth-child(4), th:nth-child(5), th:nth-child(6) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<main>
<h1>\u6295\u4fe1\u5168\u6b77\u53f2\u5f37\u80a1\u524d50\u6a94\u56de\u6e2c</h1>
<div class="meta">\u9078\u80a1\uff1a\u5168\u6b77\u53f2\u6295\u4fe11\u65e5\u8cb7\u8d85\u5f8c\u56fa\u5b9a60\u65e5\u5e73\u5747\u6de8\u5831\u916c\u524d50\u6a94\uff1b\u4ea4\u6613\uff1a\u8fd1\u4e94\u5e74\u300150\u6a94\u5167\u6bcf\u65e5\u6700\u591a\u9078\u524d {DEFAULT_TOP_N} \u6a94\uff1bquota \u5c0f\u65bc\u7b49\u65bc 0 \u9694\u5929\u51fa\u5834\uff1b\u505c\u640d {fmt_pct(DEFAULT_STOP_LOSS_RATE, 0)}</div>
<p>\u9019\u662f\u6a23\u672c\u5167\u9078\u80a1\uff1a\u7528\u5168\u6b77\u53f2\u5148\u6311\u51fa\u6295\u4fe1\u8cb7\u8d85\u5f8c\u8868\u73fe\u6700\u597d\u7684\u80a1\u7968\uff0c\u518d\u56de\u6e2c\u8fd1\u4e94\u5e74\u7b56\u7565\u3002\u521d\u59cb quota \u662f\u8cb7\u8a0a\u865f\u7576\u5929\u6295\u4fe1\u6de8\u8cb7\u8d85\u80a1\u6578\uff1b\u9032\u5834\u5f8c\u6bcf\u5929\u7684\u6295\u4fe1\u6de8\u8cb7\u6703\u589e\u52a0 quota\uff0c\u6de8\u8ce3\u6703\u964d\u4f4e quota\uff0cquota \u5c0f\u65bc\u7b49\u65bc 0 \u5247\u9694\u5929\u958b\u76e4\u51fa\u5834\u3002\u9019\u500b\u9078\u80a1\u65b9\u5f0f\u662f\u6a23\u672c\u5167\u9078\u80a1\uff0c\u9069\u5408\u6aa2\u67e5\u300c\u82e5\u53ea\u4ea4\u6613\u6295\u4fe1\u64c5\u9577\u80a1\u7968\u300d\u7684\u4e0a\u9650\uff0c\u4e0d\u80fd\u76f4\u63a5\u8996\u70ba\u771f\u5be6\u672a\u4f86\u7e3e\u6548\u3002</p>
<section class="summary">
<div class="metric"><div class="label">\u4ea4\u6613\u6578</div><div class="value">{int(row.TradeCount):,}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6de8\u5831\u916c</div><div class="value">{fmt_pct(row.AvgNetReturn)}</div></div>
<div class="metric"><div class="label">\u4e2d\u4f4d\u6578\u6de8\u5831\u916c</div><div class="value">{fmt_pct(row.MedianNetReturn)}</div></div>
<div class="metric"><div class="label">\u52dd\u7387</div><div class="value">{fmt_pct(row.WinRate)}</div></div>
<div class="metric"><div class="label">\u505c\u640d\u6bd4\u4f8b</div><div class="value">{fmt_pct(row.StopLossExitRate)}</div></div>
<div class="metric"><div class="label">quota \u6b78\u96f6\u51fa\u5834\u6bd4\u4f8b</div><div class="value">{fmt_pct(row.QuotaDepletedExitRate)}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6301\u6709\u4ea4\u6613\u65e5</div><div class="value">{fmt_num(row.AvgHoldingTradingDays, 1)}</div></div>
<div class="metric"><div class="label">\u7372\u5229\u56e0\u5b50</div><div class="value">{fmt_num(row.ProfitFactor, 2)}</div></div>
</section>
<section class="panel">
<h2>\u7b56\u7565\u4ea4\u6613\u904e\u7a0b</h2>
{timeline_chart}
</section>
<section class="panel">
<h2>\u8207\u5168\u5e02\u5834\u7248\u672c\u5c0d\u7167</h2>
{comparison_table(summary)}
</section>
<section class="panel">
<h2>\u5168\u6b77\u53f2\u9078\u51fa\u7684\u524d50\u6a94</h2>
<table><thead><tr>{selection_heads}</tr></thead><tbody>{table_rows(selection_view, selection_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u51fa\u5834\u539f\u56e0\u7d71\u8a08</h2>
<table><thead><tr>{reason_heads}</tr></thead><tbody>{table_rows(reason_summary, reason_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u55ae\u7b46\u5831\u916c\u5206\u5e03</h2>
{histogram_svg(trades, "NetReturn", "\u55ae\u7b46\u6de8\u5831\u916c\u5206\u5e03")}
</section>
<section class="panel">
<h2>\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03</h2>
{histogram_svg(trades, "HoldingTradingDays", "\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03", percent=False)}
</section>
<section class="panel">
<h2>\u6708\u7d71\u8a08</h2>
<table><thead><tr>{monthly_heads}</tr></thead><tbody>{table_rows(monthly, monthly_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u6700\u8fd1\u51fa\u5834\u4ea4\u6613</h2>
<table><thead><tr>{recent_heads}</tr></thead><tbody>{table_rows(recent, recent_columns)}</tbody></table>
</section>
<p><a href="summary.html">\u56de\u5230\u7b56\u7565\u7d71\u6574\u5831\u544a</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    selection = build_full_history_trust_top_stocks()
    top_codes = set(selection["Code"].astype(str))
    panel, meta = build_recent_panel_for_codes(top_codes, lookback_years=5)
    trades = select_trust_cumulative_sell_stop_trades(
        panel,
        top_n=DEFAULT_TOP_N,
        min_daily_turnover=DEFAULT_MIN_DAILY_TURNOVER,
        stop_loss_rate=DEFAULT_STOP_LOSS_RATE,
    )
    if trades.empty:
        raise SystemExit("no_trust_top50_cumulative_trades")
    active_counts = active_position_counts(trades, panel)
    summary = summarize_trust_cumulative_sell_stop(
        trades,
        active_counts,
        meta=meta,
        top_n=DEFAULT_TOP_N,
        min_daily_turnover=DEFAULT_MIN_DAILY_TURNOVER,
        stop_loss_rate=DEFAULT_STOP_LOSS_RATE,
    )
    summary["UniverseSelection"] = "full_history_trust_1d_buy_fixed_60d_top50"
    monthly = monthly_dynamic_summary(trades)

    paths = {
        "selection": OUTPUT_ROOT / "trust_top50_selection_by_stock.csv",
        "trades": OUTPUT_ROOT / "trust_top50_cumulative_sell_stop_trades.csv",
        "summary": OUTPUT_ROOT / "trust_top50_cumulative_sell_stop_summary.csv",
        "monthly": OUTPUT_ROOT / "trust_top50_cumulative_sell_stop_monthly.csv",
        "active": OUTPUT_ROOT / "trust_top50_cumulative_sell_stop_active_positions.csv",
    }
    selection.to_csv(paths["selection"], index=False, encoding="utf-8-sig")
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    monthly.to_csv(paths["monthly"], index=False, encoding="utf-8-sig")
    active_counts.to_csv(paths["active"], index=False, encoding="utf-8-sig")
    report = write_report(trades, summary, monthly, selection, active_counts)
    metrics_path = OUTPUT_ROOT / "strategy_metrics.csv"
    if metrics_path.exists():
        base_summary = pd.read_csv(metrics_path, encoding="utf-8-sig")
        write_summary_report(base_summary)
    print("meta=", meta)
    for key, path in paths.items():
        print(f"{key}={path}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
