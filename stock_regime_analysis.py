"""Interactive stock price regime analysis.

The regime map uses adjusted prices:
- x: rolling regression slope of log close.
- y: rolling ATR%, min/max normalized per stock and regime.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRICE_DIR = PROJECT_ROOT / "data" / "adj_price"
DEFAULT_TAIEX_PATH = PROJECT_ROOT / "data" / "price" / "TAIEX_202001_to_202606.csv"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "regime" / "absolute"
DEFAULT_BETA_OUTPUT_DIR = PROJECT_ROOT / "output" / "regime" / "taiex_beta"

COMMON_STOCK_TYPE = "\u80a1\u7968"
TWSE_MARKET = "\u4e0a\u5e02"
REQUIRED_PRICE_COLUMNS = [
    "Date",
    "Open_adj",
    "High_adj",
    "Low_adj",
    "Close_adj",
    "Capacity",
]
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
REGIME_CONFIGS = [
    {
        "id": "short",
        "label": "Short-term",
        "trend_window": 20,
        "atr_window": 10,
        "color": "#dc2626",
        "description": "20D trend / 10D volatility",
    },
    {
        "id": "medium",
        "label": "Medium-term",
        "trend_window": 60,
        "atr_window": 20,
        "color": "#2563eb",
        "description": "60D trend / 20D volatility",
    },
    {
        "id": "long",
        "label": "Long-term",
        "trend_window": 120,
        "atr_window": 60,
        "color": "#16a34a",
        "description": "120D trend / 60D volatility",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adjusted-price regime trajectory HTML reports."
    )
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_DIR))
    parser.add_argument("--taiex", default=str(DEFAULT_TAIEX_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--beta-output-dir", default=str(DEFAULT_BETA_OUTPUT_DIR))
    parser.add_argument(
        "--stocks",
        default=None,
        help="Optional comma-separated stock codes for a smaller run.",
    )
    parser.add_argument(
        "--stock-limit",
        type=int,
        default=None,
        help="Optional first-N stock cap for smoke generation.",
    )
    return parser.parse_args()


def stock_code_from_path(csv_path: Path) -> str:
    return csv_path.stem.split("_", 1)[0]


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", str(value)).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Unknown"


def load_listed_common_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_path, dtype={"Code": str})
    required = {"Code", "Name", "Type", "Market", "Group"}
    missing = required - set(metadata_df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    filtered = metadata_df[
        metadata_df["Type"].eq(COMMON_STOCK_TYPE)
        & metadata_df["Market"].eq(TWSE_MARKET)
    ].copy()
    filtered["Code"] = filtered["Code"].astype(str)
    filtered["Name"] = filtered["Name"].fillna("").astype(str)
    filtered["Group"] = filtered["Group"].fillna("Unknown").astype(str)
    return filtered.drop_duplicates("Code").set_index("Code", drop=False)


def latest_price_csvs(price_dir: Path) -> dict[str, Path]:
    latest_by_code: dict[str, Path] = {}
    for csv_path in sorted(price_dir.glob("*_to_*.csv")):
        code = stock_code_from_path(csv_path)
        current = latest_by_code.get(code)
        if current is None or csv_path.name > current.name:
            latest_by_code[code] = csv_path
    return latest_by_code


def clean_price_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    df = df[REQUIRED_PRICE_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open_adj", "High_adj", "Low_adj", "Close_adj", "Capacity"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return (
        df.dropna(subset=["Date", "Open_adj", "High_adj", "Low_adj", "Close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def clean_taiex_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = ["Date", "Open", "High", "Low", "Close"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    df = df[required_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["Capacity"] = 0
    df = df.rename(
        columns={
            "Open": "Open_adj",
            "High": "High_adj",
            "Low": "Low_adj",
            "Close": "Close_adj",
        }
    )
    return (
        df.dropna(subset=["Date", "Open_adj", "High_adj", "Low_adj", "Close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def regression_log_slope(close: pd.Series, window: int) -> pd.Series:
    """Return rolling regression slope of log(close), using past/current rows."""
    numeric_close = pd.to_numeric(close, errors="coerce")
    log_close = np.log(numeric_close.where(numeric_close > 0))
    x_values = np.arange(window, dtype=float)
    x_centered = x_values - x_values.mean()
    denominator = float((x_centered**2).sum())

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_centered = values - values.mean()
        return float((x_centered * y_centered).sum() / denominator)

    return log_close.rolling(window=window, min_periods=window).apply(slope, raw=True)


def calculate_atr_percent(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["Close_adj"].shift(1)
    true_range = pd.concat(
        [
            df["High_adj"] - df["Low_adj"],
            (df["High_adj"] - prev_close).abs(),
            (df["Low_adj"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr / df["Close_adj"] * 100


def minmax_normalize(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)

    low = float(valid.min())
    high = float(valid.max())
    if math.isclose(low, high):
        return pd.Series(0.5, index=series.index).where(series.notna())
    return (series - low) / (high - low)


def compute_regime_dataframe(
    df: pd.DataFrame,
    trend_window: int = 30,
    atr_window: int = 10,
) -> pd.DataFrame:
    result = df.copy()
    result["trend_slope"] = regression_log_slope(result["Close_adj"], trend_window)
    result["atr_pct"] = calculate_atr_percent(result, atr_window)
    result["vol_norm"] = minmax_normalize(result["atr_pct"]).clip(0, 1)
    return result.dropna(subset=["trend_slope", "vol_norm", "atr_pct"]).reset_index(drop=True)


def stock_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "x": round(float(row.trend_slope), 8),
            "y": round(float(row.vol_norm), 6),
            "trend_slope": round(float(row.trend_slope), 10),
            "atr_pct": round(float(row.atr_pct), 6),
            "close": round(float(row.Close_adj), 4),
            "capacity": float(row.Capacity) if pd.notna(row.Capacity) else None,
        }
        for row in df.itertuples(index=False)
    ]


def price_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "date": row.Date.strftime("%Y-%m-%d"),
            "close": round(float(row.Close_adj), 4),
        }
        for row in df.itertuples(index=False)
        if pd.notna(row.Close_adj)
    ]


def build_stock_regime_sets(df: pd.DataFrame) -> list[dict[str, Any]]:
    regimes = []
    for config in REGIME_CONFIGS:
        regime_df = compute_regime_dataframe(
            df,
            trend_window=int(config["trend_window"]),
            atr_window=int(config["atr_window"]),
        )
        if regime_df.empty:
            continue

        regimes.append(
            {
                **config,
                "records": stock_records(regime_df),
                "median_vol": float(regime_df["vol_norm"].median()),
            }
        )
    return regimes


def build_taiex_regime_lookup(taiex_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {regime["id"]: regime for regime in build_stock_regime_sets(taiex_df)}


def build_beta_regime_sets(
    stock_regimes: list[dict[str, Any]],
    taiex_regimes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    beta_regimes = []
    for stock_regime in stock_regimes:
        taiex_regime = taiex_regimes_by_id.get(stock_regime["id"])
        if not taiex_regime:
            continue

        taiex_by_date = {
            record["date"]: record
            for record in taiex_regime["records"]
        }
        records = []
        for stock_record in stock_regime["records"]:
            taiex_record = taiex_by_date.get(stock_record["date"])
            if not taiex_record:
                continue
            records.append(
                {
                    "date": stock_record["date"],
                    "x": round(float(stock_record["x"]) - float(taiex_record["x"]), 8),
                    "y": round(float(stock_record["y"]) - float(taiex_record["y"]), 6),
                    "stock_x": stock_record["x"],
                    "stock_y": stock_record["y"],
                    "taiex_x": taiex_record["x"],
                    "taiex_y": taiex_record["y"],
                    "atr_pct": stock_record["atr_pct"],
                    "taiex_atr_pct": taiex_record["atr_pct"],
                    "close": stock_record["close"],
                }
            )

        if records:
            beta_regimes.append(
                {
                    **{key: stock_regime[key] for key in ["id", "label", "trend_window", "atr_window", "color", "description"]},
                    "records": records,
                }
            )
    return beta_regimes


def union_dates_from_regimes(regimes: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            record["date"]
            for regime in regimes
            for record in regime["records"]
        }
    )


def json_script_data(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def html_page(title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="{PLOTLY_CDN}"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f7f4; color: #1f2933; }}
    header {{ padding: 18px 24px 8px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ padding: 0 18px 22px; }}
    .plot {{ width: 100%; height: 460px; }}
    .price-plot {{ width: 100%; height: 330px; }}
    .controls {{ display: flex; align-items: center; gap: 14px; padding: 10px 6px 16px; }}
    input[type=range] {{ width: min(920px, 80vw); }}
    .date-label {{ min-width: 110px; font-weight: 700; }}
    .meta {{ color: #56616f; font-size: 13px; margin-top: 4px; }}
    a {{ color: #0f766e; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f1; }}
  </style>
</head>
<body>
{body}
<script>
{script}
</script>
</body>
</html>
"""


def write_stock_html(
    output_path: Path,
    code: str,
    name: str,
    group: str,
    regimes: list[dict[str, Any]],
    dates: list[str],
    prices: list[dict[str, Any]],
    trend_axis_min: float,
    trend_axis_max: float,
) -> None:
    title = f"{code} {name} Regime Analysis"
    body = f"""
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Group: {html.escape(group)} | Adjusted-price regime trajectory</div>
</header>
<main>
  <div id="regimePlot" class="plot"></div>
  <div class="controls">
    <input id="dateSlider" type="range" min="0" max="{max(len(dates) - 1, 0)}" value="{max(len(dates) - 1, 0)}" step="1">
    <span id="dateLabel" class="date-label"></span>
  </div>
  <div id="pricePlot" class="price-plot"></div>
</main>
"""
    script = f"""
const regimes = {json_script_data(regimes)};
const dates = {json_script_data(dates)};
const prices = {json_script_data(prices)};
const trendAxisMin = {trend_axis_min:.10f};
const trendAxisMax = {trend_axis_max:.10f};
const closeByDate = new Map(prices.map(row => [row.date, row.close]));

function rowsUntil(regime, selectedDate) {{
  return regime.records.filter(row => row.date <= selectedDate);
}}

function hoverText(regime, row) {{
  return `${{regime.label}}<br>${{regime.description}}<br>Date: ${{row.date}}<br>Slope x: ${{row.x.toFixed(5)}}<br>Vol y: ${{row.y.toFixed(3)}}<br>ATR%: ${{row.atr_pct.toFixed(3)}}<br>Close_adj: ${{row.close.toFixed(2)}}`;
}}

const priceTrace = {{
  x: prices.map(row => row.date),
  y: prices.map(row => row.close),
  type: "scatter",
  mode: "lines",
  name: "Close_adj",
  line: {{color: "#111827", width: 1.5}}
}};

const regimeTraces = [];
regimes.forEach(regime => {{
  regimeTraces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "lines",
    name: regime.label,
    line: {{color: regime.color, width: 2.2}},
    opacity: 0.55,
    hoverinfo: "skip"
  }});
}});
regimes.forEach(regime => {{
  regimeTraces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    name: `${{regime.label}} current`,
    marker: {{color: regime.color, size: 11, line: {{color: "#ffffff", width: 1}}}},
    hovertext: [],
    hoverinfo: "text"
  }});
}});

const medianShapes = regimes.map(regime => ({{
  type: "line",
  x0: trendAxisMin,
  x1: trendAxisMax,
  y0: regime.median_vol,
  y1: regime.median_vol,
  line: {{color: regime.color, width: 1, dash: "dot"}}
}}));

Plotly.newPlot("regimePlot", regimeTraces, {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 28, b: 48}},
  xaxis: {{title: "Trend: regression slope of log adjusted close", range: [trendAxisMin, trendAxisMax], zeroline: true, zerolinewidth: 2, zerolinecolor: "#334155"}},
  yaxis: {{title: "Volatility: normalized ATR%", range: [0, 1], zeroline: false}},
  shapes: [
    {{type: "line", x0: 0, x1: 0, y0: 0, y1: 1, line: {{color: "#334155", width: 1.5}}}},
    ...medianShapes
  ],
  showlegend: true
}}, {{responsive: true}});

Plotly.newPlot("pricePlot", [
  priceTrace,
  {{x: [], y: [], type: "scatter", mode: "markers", name: "Selected Date", marker: {{color: "#dc2626", size: 9}}}}
], {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 18, b: 48}},
  xaxis: {{title: "Date"}},
  yaxis: {{title: "Adjusted Close"}}
}}, {{responsive: true}});

function update(index) {{
  if (!dates.length) return;
  const selectedDate = dates[index];
  regimes.forEach((regime, regimeIndex) => {{
    const rows = rowsUntil(regime, selectedDate);
    Plotly.restyle("regimePlot", {{
      x: [rows.map(row => row.x)],
      y: [rows.map(row => row.y)]
    }}, [regimeIndex]);
    const latest = rows.length ? rows[rows.length - 1] : null;
    Plotly.restyle("regimePlot", {{
      x: [latest ? [latest.x] : []],
      y: [latest ? [latest.y] : []],
      hovertext: [latest ? [hoverText(regime, latest)] : []]
    }}, [regimes.length + regimeIndex]);
  }});
  const close = closeByDate.get(selectedDate);
  Plotly.restyle("pricePlot", {{
    x: [close === undefined ? [] : [selectedDate]],
    y: [close === undefined ? [] : [close]]
  }}, [1]);
  document.getElementById("dateLabel").textContent = selectedDate;
}}

const slider = document.getElementById("dateSlider");
slider.addEventListener("input", event => update(Number(event.target.value)));
update(Number(slider.value));
"""
    output_path.write_text(html_page(title, body, script), encoding="utf-8")


def write_group_members_html(
    output_path: Path,
    group: str,
    members: list[dict[str, Any]],
    date_values: list[str],
    trend_axis_min: float,
    trend_axis_max: float,
) -> None:
    title = f"{group} Members Regime Analysis"
    body = f"""
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">All member trajectories use adjusted prices; trails show history through the selected date.</div>
</header>
<main>
  <div id="regimePlot" class="plot"></div>
  <div class="controls">
    <input id="dateSlider" type="range" min="0" max="{max(len(date_values) - 1, 0)}" value="{max(len(date_values) - 1, 0)}" step="1">
    <span id="dateLabel" class="date-label"></span>
  </div>
  <div id="pricePlot" class="price-plot"></div>
</main>
"""
    script = f"""
const members = {json_script_data(members)};
const dates = {json_script_data(date_values)};
const trendAxisMin = {trend_axis_min:.10f};
const trendAxisMax = {trend_axis_max:.10f};

function rowsUntil(regime, selectedDate) {{
  return regime.records.filter(row => row.date <= selectedDate);
}}

function memberHover(member, regime, row) {{
  return `${{member.code}} ${{member.name}}<br>${{regime.label}}<br>${{regime.description}}<br>Date: ${{row.date}}<br>Slope x: ${{row.x.toFixed(5)}}<br>Vol y: ${{row.y.toFixed(3)}}<br>ATR%: ${{row.atr_pct.toFixed(3)}}`;
}}

const regimeTraces = [];
const priceTraces = [];
const traceMap = [];
members.forEach(member => {{
  member.regimes.forEach(regime => {{
    traceMap.push({{member, regime}});
    regimeTraces.push({{
      x: [],
      y: [],
      type: "scatter",
      mode: "lines",
      name: `${{member.code}} ${{member.name}} ${{regime.label}}`,
      line: {{color: regime.color, width: 1}},
      opacity: 0.16,
      hoverinfo: "skip",
      showlegend: false
    }});
  }});
  priceTraces.push({{
    x: member.prices.map(row => row.date),
    y: member.prices.map(row => row.close),
    type: "scatter",
    mode: "lines",
    name: `${{member.code}} ${{member.name}}`,
    line: {{color: "rgba(37,99,235,0.13)", width: 1}},
    hoverinfo: "skip",
    showlegend: false
  }});
}});
const regimeLabels = [];
members.forEach(member => {{
  member.regimes.forEach(regime => {{
    if (!regimeLabels.some(item => item.id === regime.id)) {{
      regimeLabels.push(regime);
    }}
  }});
}});
regimeLabels.forEach(regime => {{
  regimeTraces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    name: `${{regime.label}} current`,
    marker: {{color: regime.color, size: 5, opacity: 0.72}},
    hovertext: [],
    hoverinfo: "text"
  }});
}});
priceTraces.push({{
  x: [],
  y: [],
  type: "scatter",
  mode: "markers",
  name: "Current closes",
  marker: {{color: "#dc2626", size: 5, opacity: 0.75}},
  hovertext: [],
  hoverinfo: "text"
}});

Plotly.newPlot("regimePlot", regimeTraces, {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 28, b: 48}},
  xaxis: {{title: "Trend: regression slope of log adjusted close", range: [trendAxisMin, trendAxisMax], zeroline: true, zerolinewidth: 2, zerolinecolor: "#334155"}},
  yaxis: {{title: "Volatility: normalized ATR%", range: [0, 1]}},
  shapes: [{{type: "line", x0: 0, x1: 0, y0: 0, y1: 1, line: {{color: "#334155", width: 1.5}}}}]
}}, {{responsive: true}});

Plotly.newPlot("pricePlot", priceTraces, {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 18, b: 48}},
  xaxis: {{title: "Date"}},
  yaxis: {{title: "Adjusted Close"}}
}}, {{responsive: true}});

function update(index) {{
  if (!dates.length) return;
  const selectedDate = dates[index];
  const currentByRegime = new Map(regimeLabels.map(regime => [regime.id, {{x: [], y: [], hover: []}}]));
  const priceX = [];
  const priceY = [];
  const priceHover = [];
  traceMap.forEach((item, traceIndex) => {{
    const rows = rowsUntil(item.regime, selectedDate);
    Plotly.restyle("regimePlot", {{
      x: [rows.map(row => row.x)],
      y: [rows.map(row => row.y)]
    }}, [traceIndex]);
    const latest = rows.length ? rows[rows.length - 1] : null;
    if (latest) {{
      const current = currentByRegime.get(item.regime.id);
      current.x.push(latest.x);
      current.y.push(latest.y);
      current.hover.push(memberHover(item.member, item.regime, latest));
    }}
  }});
  regimeLabels.forEach((regime, regimeIndex) => {{
    const current = currentByRegime.get(regime.id);
    Plotly.restyle("regimePlot", {{
      x: [current.x],
      y: [current.y],
      hovertext: [current.hover]
    }}, [traceMap.length + regimeIndex]);
  }});
  members.forEach(member => {{
    const priceRows = member.prices.filter(row => row.date <= selectedDate);
    const latestPrice = priceRows.length ? priceRows[priceRows.length - 1] : null;
    if (latestPrice) {{
      priceX.push(latestPrice.date);
      priceY.push(latestPrice.close);
      priceHover.push(`${{member.code}} ${{member.name}}<br>Date: ${{latestPrice.date}}<br>Close_adj: ${{latestPrice.close.toFixed(2)}}`);
    }}
  }});
  Plotly.restyle("pricePlot", {{x: [priceX], y: [priceY], hovertext: [priceHover]}}, [members.length]);
  document.getElementById("dateLabel").textContent = selectedDate;
}}

const slider = document.getElementById("dateSlider");
slider.addEventListener("input", event => update(Number(event.target.value)));
update(Number(slider.value));
"""
    output_path.write_text(html_page(title, body, script), encoding="utf-8")


def write_group_average_html(
    output_path: Path,
    group: str,
    regimes: list[dict[str, Any]],
    dates: list[str],
    prices: list[dict[str, Any]],
    trend_axis_min: float,
    trend_axis_max: float,
) -> None:
    title = f"{group} Average Regime Analysis"
    body = f"""
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Daily average of member normalized regime coordinates.</div>
</header>
<main>
  <div id="regimePlot" class="plot"></div>
  <div class="controls">
    <input id="dateSlider" type="range" min="0" max="{max(len(dates) - 1, 0)}" value="{max(len(dates) - 1, 0)}" step="1">
    <span id="dateLabel" class="date-label"></span>
  </div>
  <div id="pricePlot" class="price-plot"></div>
</main>
"""
    script = f"""
const regimes = {json_script_data(regimes)};
const dates = {json_script_data(dates)};
const prices = {json_script_data(prices)};
const trendAxisMin = {trend_axis_min:.10f};
const trendAxisMax = {trend_axis_max:.10f};
const avgCloseByDate = new Map(prices.map(row => [row.date, row.avg_close]));

function rowsUntil(regime, selectedDate) {{
  return regime.records.filter(row => row.date <= selectedDate);
}}

function hoverText(regime, row) {{
  return `${{regime.label}}<br>${{regime.description}}<br>Date: ${{row.date}}<br>Avg slope x: ${{row.x.toFixed(5)}}<br>Avg vol y: ${{row.y.toFixed(3)}}<br>Members: ${{row.member_count}}`;
}}

const regimeTraces = [];
regimes.forEach(regime => {{
  regimeTraces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "lines",
    name: `${{regime.label}} average`,
    line: {{color: regime.color, width: 2.4}},
    opacity: 0.62,
    hoverinfo: "skip"
  }});
}});
regimes.forEach(regime => {{
  regimeTraces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    name: `${{regime.label}} current`,
    marker: {{color: regime.color, size: 11, line: {{color: "#ffffff", width: 1}}}},
    hovertext: [],
    hoverinfo: "text"
  }});
}});

Plotly.newPlot("regimePlot", regimeTraces, {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 28, b: 48}},
  xaxis: {{title: "Average log-slope trend", range: [trendAxisMin, trendAxisMax], zeroline: true, zerolinewidth: 2, zerolinecolor: "#334155"}},
  yaxis: {{title: "Average normalized volatility", range: [0, 1]}},
  shapes: [{{type: "line", x0: 0, x1: 0, y0: 0, y1: 1, line: {{color: "#334155", width: 1.5}}}}]
}}, {{responsive: true}});
Plotly.newPlot("pricePlot", [
  {{x: prices.map(row => row.date), y: prices.map(row => row.avg_close), type: "scatter", mode: "lines", name: "Average Close_adj", line: {{color: "#111827", width: 1.5}}}},
  {{x: [], y: [], type: "scatter", mode: "markers", name: "Selected Date", marker: {{color: "#dc2626", size: 9}}}}
], {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 18, b: 48}},
  xaxis: {{title: "Date"}},
  yaxis: {{title: "Average Adjusted Close"}}
}}, {{responsive: true}});
function update(index) {{
  if (!dates.length) return;
  const selectedDate = dates[index];
  regimes.forEach((regime, regimeIndex) => {{
    const rows = rowsUntil(regime, selectedDate);
    const latest = rows.length ? rows[rows.length - 1] : null;
    Plotly.restyle("regimePlot", {{
      x: [rows.map(row => row.x)],
      y: [rows.map(row => row.y)]
    }}, [regimeIndex]);
    Plotly.restyle("regimePlot", {{
      x: [latest ? [latest.x] : []],
      y: [latest ? [latest.y] : []],
      hovertext: [latest ? [hoverText(regime, latest)] : []]
    }}, [regimes.length + regimeIndex]);
  }});
  const close = avgCloseByDate.get(selectedDate);
  Plotly.restyle("pricePlot", {{
    x: [close === undefined ? [] : [selectedDate]],
    y: [close === undefined ? [] : [close]]
  }}, [1]);
  document.getElementById("dateLabel").textContent = selectedDate;
}}
const slider = document.getElementById("dateSlider");
slider.addEventListener("input", event => update(Number(event.target.value)));
update(Number(slider.value));
"""
    output_path.write_text(html_page(title, body, script), encoding="utf-8")


def write_beta_stock_html(
    output_path: Path,
    code: str,
    name: str,
    group: str,
    regimes: list[dict[str, Any]],
    dates: list[str],
    prices: list[dict[str, Any]],
    x_axis_min: float,
    x_axis_max: float,
    y_axis_min: float,
    y_axis_max: float,
) -> None:
    title = f"{code} {name} Regime Analysis vs TAIEX"
    body = f"""
<header>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Group: {html.escape(group)} | Coordinates are stock regime minus TAIEX regime. Origin means equal to TAIEX.</div>
</header>
<main>
  <div id="regimePlot" class="plot"></div>
  <div class="controls">
    <input id="dateSlider" type="range" min="0" max="{max(len(dates) - 1, 0)}" value="{max(len(dates) - 1, 0)}" step="1">
    <span id="dateLabel" class="date-label"></span>
  </div>
  <div id="pricePlot" class="price-plot"></div>
</main>
"""
    script = f"""
const regimes = {json_script_data(regimes)};
const dates = {json_script_data(dates)};
const prices = {json_script_data(prices)};
const xAxisMin = {x_axis_min:.10f};
const xAxisMax = {x_axis_max:.10f};
const yAxisMin = {y_axis_min:.10f};
const yAxisMax = {y_axis_max:.10f};
const closeByDate = new Map(prices.map(row => [row.date, row.close]));

function rowsUntil(regime, selectedDate) {{
  return regime.records.filter(row => row.date <= selectedDate);
}}

function hoverText(regime, row) {{
  return `${{regime.label}} vs TAIEX<br>${{regime.description}}<br>Date: ${{row.date}}<br>Trend diff x: ${{row.x.toFixed(5)}}<br>Vol diff y: ${{row.y.toFixed(3)}}<br>Stock slope: ${{row.stock_x.toFixed(5)}}<br>TAIEX slope: ${{row.taiex_x.toFixed(5)}}<br>Stock vol: ${{row.stock_y.toFixed(3)}}<br>TAIEX vol: ${{row.taiex_y.toFixed(3)}}`;
}}

const traces = [];
regimes.forEach(regime => {{
  traces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "lines",
    name: regime.label,
    line: {{color: regime.color, width: 2.2}},
    opacity: 0.58,
    hoverinfo: "skip"
  }});
}});
regimes.forEach(regime => {{
  traces.push({{
    x: [],
    y: [],
    type: "scatter",
    mode: "markers",
    name: `${{regime.label}} current`,
    marker: {{color: regime.color, size: 11, line: {{color: "#ffffff", width: 1}}}},
    hovertext: [],
    hoverinfo: "text"
  }});
}});

Plotly.newPlot("regimePlot", traces, {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 28, b: 48}},
  xaxis: {{title: "Trend diff: stock slope - TAIEX slope", range: [xAxisMin, xAxisMax], zeroline: true, zerolinewidth: 2, zerolinecolor: "#334155"}},
  yaxis: {{title: "Volatility diff: stock normalized ATR - TAIEX normalized ATR", range: [yAxisMin, yAxisMax], zeroline: true, zerolinewidth: 2, zerolinecolor: "#334155"}},
  shapes: [
    {{type: "line", x0: 0, x1: 0, y0: yAxisMin, y1: yAxisMax, line: {{color: "#334155", width: 1.2}}}},
    {{type: "line", x0: xAxisMin, x1: xAxisMax, y0: 0, y1: 0, line: {{color: "#334155", width: 1.2}}}}
  ],
  showlegend: true
}}, {{responsive: true}});

Plotly.newPlot("pricePlot", [
  {{x: prices.map(row => row.date), y: prices.map(row => row.close), type: "scatter", mode: "lines", name: "Close_adj", line: {{color: "#111827", width: 1.5}}}},
  {{x: [], y: [], type: "scatter", mode: "markers", name: "Selected Date", marker: {{color: "#dc2626", size: 9}}}}
], {{
  template: "plotly_white",
  margin: {{l: 54, r: 24, t: 18, b: 48}},
  xaxis: {{title: "Date"}},
  yaxis: {{title: "Adjusted Close"}}
}}, {{responsive: true}});

function update(index) {{
  if (!dates.length) return;
  const selectedDate = dates[index];
  regimes.forEach((regime, regimeIndex) => {{
    const rows = rowsUntil(regime, selectedDate);
    const latest = rows.length ? rows[rows.length - 1] : null;
    Plotly.restyle("regimePlot", {{x: [rows.map(row => row.x)], y: [rows.map(row => row.y)]}}, [regimeIndex]);
    Plotly.restyle("regimePlot", {{
      x: [latest ? [latest.x] : []],
      y: [latest ? [latest.y] : []],
      hovertext: [latest ? [hoverText(regime, latest)] : []]
    }}, [regimes.length + regimeIndex]);
  }});
  const close = closeByDate.get(selectedDate);
  Plotly.restyle("pricePlot", {{
    x: [close === undefined ? [] : [selectedDate]],
    y: [close === undefined ? [] : [close]]
  }}, [1]);
  document.getElementById("dateLabel").textContent = selectedDate;
}}

const slider = document.getElementById("dateSlider");
slider.addEventListener("input", event => update(Number(event.target.value)));
update(Number(slider.value));
"""
    output_path.write_text(html_page(title, body, script), encoding="utf-8")


def build_group_average_regimes(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for member in members:
        for regime in member["regimes"]:
            for record in regime["records"]:
                row = dict(record)
                row["code"] = member["code"]
                row["regime_id"] = regime["id"]
                rows.append(row)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["regime_id", "date"], as_index=False)
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            member_count=("code", "nunique"),
        )
        .sort_values(["regime_id", "date"])
    )
    average_regimes = []
    for config in REGIME_CONFIGS:
        regime_rows = grouped[grouped["regime_id"].eq(config["id"])]
        if regime_rows.empty:
            continue
        average_regimes.append(
            {
                **config,
                "records": [
                    {
                        "date": row.date,
                        "x": round(float(row.x), 8),
                        "y": round(float(row.y), 6),
                        "member_count": int(row.member_count),
                    }
                    for row in regime_rows.itertuples(index=False)
                ],
            }
        )
    return average_regimes


def build_group_average_prices(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for member in members:
        for record in member["prices"]:
            row = dict(record)
            row["code"] = member["code"]
            rows.append(row)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("date", as_index=False)
        .agg(avg_close=("close", "mean"), member_count=("code", "nunique"))
        .sort_values("date")
    )
    return [
        {
            "date": row.date,
            "avg_close": round(float(row.avg_close), 4),
            "member_count": int(row.member_count),
        }
        for row in grouped.itertuples(index=False)
    ]


def get_trend_axis_range_from_records(records: list[dict[str, Any]]) -> tuple[float, float]:
    slopes = [
        float(record["x"])
        for record in records
        if record.get("x") is not None and math.isfinite(float(record["x"]))
    ]
    if not slopes:
        raise ValueError("No finite trend slopes are available for x-axis range.")

    trend_min = min(slopes)
    trend_max = max(slopes)
    if math.isclose(trend_min, trend_max):
        padding = abs(trend_min) * 0.05 or 0.001
        trend_min -= padding
        trend_max += padding
    return trend_min, trend_max


def get_trend_axis_range_from_members(members: list[dict[str, Any]]) -> tuple[float, float]:
    slopes = [
        float(record["x"])
        for member in members
        for regime in member["regimes"]
        for record in regime["records"]
        if record.get("x") is not None and math.isfinite(float(record["x"]))
    ]
    if not slopes:
        raise ValueError("No finite trend slopes are available for x-axis range.")

    trend_min = min(slopes)
    trend_max = max(slopes)
    if math.isclose(trend_min, trend_max):
        padding = abs(trend_min) * 0.05 or 0.001
        trend_min -= padding
        trend_max += padding
    return trend_min, trend_max


def get_trend_axis_range_from_regimes(regimes: list[dict[str, Any]]) -> tuple[float, float]:
    slopes = [
        float(record["x"])
        for regime in regimes
        for record in regime["records"]
        if record.get("x") is not None and math.isfinite(float(record["x"]))
    ]
    if not slopes:
        raise ValueError("No finite trend slopes are available for x-axis range.")

    trend_min = min(slopes)
    trend_max = max(slopes)
    if math.isclose(trend_min, trend_max):
        padding = abs(trend_min) * 0.05 or 0.001
        trend_min -= padding
        trend_max += padding
    return trend_min, trend_max


def get_value_axis_range_from_regimes(regimes: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = [
        float(record[field])
        for regime in regimes
        for record in regime["records"]
        if record.get(field) is not None and math.isfinite(float(record[field]))
    ]
    if not values:
        raise ValueError(f"No finite {field} values are available for axis range.")

    axis_min = min(values)
    axis_max = max(values)
    if math.isclose(axis_min, axis_max):
        padding = abs(axis_min) * 0.05 or 0.001
        axis_min -= padding
        axis_max += padding
    return axis_min, axis_max


def write_index_html(output_path: Path, stock_links: list[dict[str, str]], group_links: list[dict[str, str]]) -> None:
    stock_rows = "\n".join(
        f"<tr><td>{html.escape(item['code'])}</td><td>{html.escape(item['name'])}</td>"
        f"<td>{html.escape(item['group'])}</td><td><a href=\"{html.escape(item['href'])}\">open</a></td></tr>"
        for item in stock_links
    )
    group_rows = "\n".join(
        f"<tr><td>{html.escape(item['group'])}</td>"
        f"<td><a href=\"{html.escape(item['members_href'])}\">members</a></td>"
        f"<td><a href=\"{html.escape(item['average_href'])}\">average</a></td></tr>"
        for item in group_links
    )
    body = f"""
<header>
  <h1>Stock Price Regime Analysis</h1>
  <div class="meta">Adjusted-price 2D trend/volatility trajectory reports.</div>
</header>
<main>
  <h2>Groups</h2>
  <table><thead><tr><th>Group</th><th>All stocks</th><th>Group average</th></tr></thead><tbody>{group_rows}</tbody></table>
  <h2>Stocks</h2>
  <table><thead><tr><th>Code</th><th>Name</th><th>Group</th><th>Report</th></tr></thead><tbody>{stock_rows}</tbody></table>
</main>
"""
    output_path.write_text(html_page("Stock Price Regime Analysis", body, ""), encoding="utf-8")


def write_beta_index_html(output_path: Path, stock_links: list[dict[str, str]]) -> None:
    stock_rows = "\n".join(
        f"<tr><td>{html.escape(item['code'])}</td><td>{html.escape(item['name'])}</td>"
        f"<td>{html.escape(item['group'])}</td><td><a href=\"{html.escape(item['href'])}\">open</a></td></tr>"
        for item in stock_links
    )
    body = f"""
<header>
  <h1>Regime Analysis Compare With TAIEX</h1>
  <div class="meta">Each point is stock regime minus TAIEX regime. Origin means equal trend and volatility to TAIEX.</div>
</header>
<main>
  <h2>Stocks</h2>
  <table><thead><tr><th>Code</th><th>Name</th><th>Group</th><th>Report</th></tr></thead><tbody>{stock_rows}</tbody></table>
</main>
"""
    output_path.write_text(html_page("Regime Analysis Compare With TAIEX", body, ""), encoding="utf-8")


def load_regime_members(
    price_dir: Path,
    metadata_df: pd.DataFrame,
    stock_codes: set[str] | None = None,
    stock_limit: int | None = None,
) -> list[dict[str, Any]]:
    latest = latest_price_csvs(price_dir)
    allowed_codes = [code for code in sorted(metadata_df.index) if code in latest]
    if stock_codes is not None:
        allowed_codes = [code for code in allowed_codes if code in stock_codes]
    if stock_limit is not None:
        allowed_codes = allowed_codes[:stock_limit]

    members = []
    skipped = []
    for code in allowed_codes:
        try:
            price_df = clean_price_csv(latest[code])
            regimes = build_stock_regime_sets(price_df)
        except Exception as exc:
            skipped.append(f"{code}: {exc}")
            continue
        if not regimes:
            skipped.append(f"{code}: no valid regime rows")
            continue

        info = metadata_df.loc[code]
        members.append(
            {
                "code": code,
                "name": str(info["Name"]),
                "group": str(info["Group"]),
                "regimes": regimes,
                "prices": price_records(price_df),
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} stocks. First skipped entries: {skipped[:5]}")
    return members


def generate_beta_reports(
    members: list[dict[str, Any]],
    taiex_path: Path = DEFAULT_TAIEX_PATH,
    output_dir: Path = DEFAULT_BETA_OUTPUT_DIR,
) -> dict[str, int | str]:
    taiex_df = clean_taiex_csv(taiex_path)
    taiex_regimes_by_id = build_taiex_regime_lookup(taiex_df)
    stock_dir = output_dir / "stocks"
    stock_dir.mkdir(parents=True, exist_ok=True)

    stock_links = []
    skipped = []
    for member in members:
        beta_regimes = build_beta_regime_sets(member["regimes"], taiex_regimes_by_id)
        if not beta_regimes:
            skipped.append(member["code"])
            continue

        beta_dates = union_dates_from_regimes(beta_regimes)
        x_axis_min, x_axis_max = get_value_axis_range_from_regimes(beta_regimes, "x")
        y_axis_min, y_axis_max = get_value_axis_range_from_regimes(beta_regimes, "y")
        filename = f"{member['code']}.html"
        write_beta_stock_html(
            stock_dir / filename,
            member["code"],
            member["name"],
            member["group"],
            beta_regimes,
            beta_dates,
            member["prices"],
            x_axis_min,
            x_axis_max,
            y_axis_min,
            y_axis_max,
        )
        stock_links.append(
            {
                "code": member["code"],
                "name": member["name"],
                "group": member["group"],
                "href": f"stocks/{filename}",
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} beta comparison stocks without matched TAIEX regimes.")
    write_beta_index_html(output_dir / "index.html", stock_links)
    return {
        "output_dir": str(output_dir),
        "stocks": len(stock_links),
        "index": str(output_dir / "index.html"),
    }


def generate_reports(
    price_dir: Path = DEFAULT_PRICE_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    taiex_path: Path = DEFAULT_TAIEX_PATH,
    beta_output_dir: Path = DEFAULT_BETA_OUTPUT_DIR,
    stocks: str | None = None,
    stock_limit: int | None = None,
) -> dict[str, int | str]:
    metadata_df = load_listed_common_metadata(metadata_path)
    stock_codes = None
    if stocks:
        stock_codes = {code.strip() for code in stocks.split(",") if code.strip()}

    members = load_regime_members(
        price_dir,
        metadata_df,
        stock_codes=stock_codes,
        stock_limit=stock_limit,
    )
    if not members:
        raise ValueError("No stock regime data was generated.")

    stock_dir = output_dir / "stocks"
    group_dir = output_dir / "groups"
    stock_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)

    stock_links = []
    members_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for member in members:
        filename = f"{member['code']}.html"
        output_path = stock_dir / filename
        regime_dates = union_dates_from_regimes(member["regimes"])
        trend_axis_min, trend_axis_max = get_trend_axis_range_from_regimes(member["regimes"])
        write_stock_html(
            output_path,
            member["code"],
            member["name"],
            member["group"],
            member["regimes"],
            regime_dates,
            member["prices"],
            trend_axis_min,
            trend_axis_max,
        )
        stock_links.append(
            {
                "code": member["code"],
                "name": member["name"],
                "group": member["group"],
                "href": f"stocks/{filename}",
            }
        )
        members_by_group[member["group"]].append(member)

    group_links = []
    for group_index, group in enumerate(sorted(members_by_group), start=1):
        group_members = sorted(members_by_group[group], key=lambda item: item["code"])
        slug = f"{group_index:02d}_{safe_filename(group)}"
        member_dates = sorted(
            {
                record["date"]
                for member in group_members
                for regime in member["regimes"]
                for record in regime["records"]
            }
        )
        member_axis_min, member_axis_max = get_trend_axis_range_from_members(group_members)
        average_regimes = build_group_average_regimes(group_members)
        average_prices = build_group_average_prices(group_members)
        average_dates = union_dates_from_regimes(average_regimes)
        average_axis_min, average_axis_max = get_trend_axis_range_from_regimes(average_regimes)
        members_filename = f"{slug}_members.html"
        average_filename = f"{slug}_average.html"
        write_group_members_html(
            group_dir / members_filename,
            group,
            group_members,
            member_dates,
            member_axis_min,
            member_axis_max,
        )
        write_group_average_html(
            group_dir / average_filename,
            group,
            average_regimes,
            average_dates,
            average_prices,
            average_axis_min,
            average_axis_max,
        )
        group_links.append(
            {
                "group": group,
                "members_href": f"groups/{members_filename}",
                "average_href": f"groups/{average_filename}",
            }
        )

    write_index_html(output_dir / "index.html", stock_links, group_links)
    beta_result = generate_beta_reports(
        members,
        taiex_path=taiex_path,
        output_dir=beta_output_dir,
    )
    return {
        "output_dir": str(output_dir),
        "stocks": len(stock_links),
        "groups": len(group_links),
        "index": str(output_dir / "index.html"),
        "beta_stocks": beta_result["stocks"],
        "beta_index": beta_result["index"],
    }


def main() -> None:
    args = parse_args()
    result = generate_reports(
        price_dir=Path(args.price_dir),
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        taiex_path=Path(args.taiex),
        beta_output_dir=Path(args.beta_output_dir),
        stocks=args.stocks,
        stock_limit=args.stock_limit,
    )
    print(
        "Regime reports written: "
        f"{result['stocks']} stocks, {result['groups']} groups."
    )
    print(f"Index: {result['index']}")
    print(f"TAIEX comparison reports written: {result['beta_stocks']} stocks.")
    print(f"TAIEX comparison index: {result['beta_index']}")


if __name__ == "__main__":
    main()
