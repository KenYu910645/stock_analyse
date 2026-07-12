"""Create interactive 5-year pair-trading candidate charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from column_schema import read_csv_canonical
from pair_trading_backtest import calculate_metrics, summarize_trades
from stock_correlation_analysis import DEFAULT_PRICE_PATH
from strategies.pair_trading import PairTradingStrategy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "output" / "pair_trading" / "cointegration_gt_0_5" / "pair_trading_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_viz" / "pair_trading" / "candidate_visuals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interactive 5-year pair-trading candidate charts.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chart-subdir", default="pair_charts_interactive_5y")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--rolling-corr-window", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--stop-z", type=float, default=3.5)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def latest_price_path(price_dir: Path, stock_id: str) -> Path:
    paths = sorted(
        path for path in price_dir.glob(f"{stock_id}_*.csv")
        if not path.name.startswith("twse_price_")
    )
    if not paths:
        raise FileNotFoundError(f"No adjusted price CSV found for {stock_id}.")
    return paths[-1]


def load_adjusted_close(price_dir: Path, stock_id: str) -> pd.DataFrame:
    path = latest_price_path(price_dir, stock_id)
    df = read_csv_canonical(path)
    close_col = "close_adj" if "close_adj" in df.columns else "Close_adj" if "Close_adj" in df.columns else "Close"
    df = df[["Date", close_col]].copy()
    df.columns = ["Date", stock_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[stock_id] = pd.to_numeric(df[stock_id], errors="coerce")
    return df.dropna().sort_values("Date")


def build_pair_data(price_dir: Path, row: pd.Series, start_date: pd.Timestamp, rolling_corr_window: int) -> tuple[pd.DataFrame, dict]:
    stock_a = str(row["stock_id_1"])
    stock_b = str(row["stock_id_2"])
    price_a = load_adjusted_close(price_dir, stock_a)
    price_b = load_adjusted_close(price_dir, stock_b)
    df = price_a.merge(price_b, on="Date", how="inner")
    df = df[df["Date"] >= start_date].copy()
    df = df.rename(columns={stock_a: "price_a", stock_b: "price_b"})
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    alpha = float(row["eg_alpha"])
    beta = float(row["eg_beta"])
    df["log_a"] = np.log(df["price_a"].where(df["price_a"] > 0))
    df["log_b"] = np.log(df["price_b"].where(df["price_b"] > 0))
    df["spread"] = df["log_a"] - alpha - beta * df["log_b"]

    spread_mean = float(df["spread"].mean())
    spread_std = float(df["spread"].std(ddof=0))
    df["equilibrium_price_a"] = np.exp(alpha + beta * df["log_b"] + spread_mean)
    df["norm_a"] = df["price_a"] / df["price_a"].iloc[0] * 100
    df["norm_b"] = df["price_b"] / df["price_b"].iloc[0] * 100
    df["norm_equilibrium_a"] = df["equilibrium_price_a"] / df["price_a"].iloc[0] * 100
    df["return_a"] = df["price_a"].pct_change()
    df["return_b"] = df["price_b"].pct_change()
    df["rolling_corr"] = df["return_a"].rolling(rolling_corr_window).corr(df["return_b"])

    stats = {
        "start_date": df["Date"].iloc[0].date().isoformat(),
        "end_date": df["Date"].iloc[-1].date().isoformat(),
        "spread_mean": spread_mean,
        "spread_std": spread_std,
    }
    return df, stats


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    strategy = PairTradingStrategy(
        initial_capital=initial_capital,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
    )
    equity_df, trade_df = strategy.backtest(df[["Date", "price_a", "price_b"]])
    metrics = calculate_metrics(equity_df, initial_capital)
    metrics.update(summarize_trades(trade_df))
    return equity_df, trade_df, metrics


def position_windows(equity_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    if equity_df.empty:
        return []
    windows = []
    active_start = None
    active_position = None
    for row in equity_df.itertuples(index=False):
        date = getattr(row, "Date")
        position = getattr(row, "position")
        if position != "flat" and active_start is None:
            active_start = date
            active_position = position
        elif active_start is not None and position != active_position:
            windows.append((active_start, date, active_position))
            active_start = date if position != "flat" else None
            active_position = position if position != "flat" else None
    if active_start is not None:
        windows.append((active_start, equity_df["Date"].iloc[-1], active_position))
    return windows


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def safe_chart_filename(rank: int, row: pd.Series) -> str:
    filename = f"{rank:03d}_{row['stock_id_1']}_{row['stock_id_2']}_{row['stock_name_1']}_{row['stock_name_2']}.html"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)


def autoscale_y_post_script() -> str:
    """Return Plotly JS that rescales each y-axis to the visible x-window."""
    return r"""
const gd = document.getElementById('{plot_id}');
function toMillis(value) {
  if (value === undefined || value === null) return NaN;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Number(value) : parsed;
}
function traceAxisName(trace) {
  const axis = trace.yaxis || 'y';
  return axis === 'y' ? 'yaxis' : `yaxis${axis.slice(1)}`;
}
function fullDataXRange() {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  gd.data.forEach(trace => {
    if (!trace.x) return;
    trace.x.forEach(xValue => {
      const x = toMillis(xValue);
      if (!Number.isFinite(x)) return;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
    });
  });
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY];
  }
  return [minX, maxX];
}
function visibleXRange(eventData) {
  if (eventData && eventData['xaxis.autorange']) return fullDataXRange();
  if (eventData && eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
    return [toMillis(eventData['xaxis.range[0]']), toMillis(eventData['xaxis.range[1]'])];
  }
  if (eventData && eventData['xaxis.range'] && eventData['xaxis.range'].length === 2) {
    return [toMillis(eventData['xaxis.range'][0]), toMillis(eventData['xaxis.range'][1])];
  }
  const range = gd.layout.xaxis && gd.layout.xaxis.range;
  if (range && range.length === 2 && !(gd.layout.xaxis && gd.layout.xaxis.autorange)) {
    return [toMillis(range[0]), toMillis(range[1])];
  }
  return fullDataXRange();
}
function autoscaleVisibleY(eventData) {
  if (gd.__autoscalingY) return;
  const [x0, x1] = visibleXRange(eventData);
  const ranges = {};
  gd.data.forEach(trace => {
    if (!trace.visible || trace.visible === 'legendonly') return;
    if (!trace.x || !trace.y || trace.y.length === 0) return;
    const yaxis = traceAxisName(trace);
    trace.x.forEach((xValue, idx) => {
      const x = toMillis(xValue);
      if (Number.isNaN(x) || x < x0 || x > x1) return;
      const y = Number(trace.y[idx]);
      if (!Number.isFinite(y)) return;
      if (!ranges[yaxis]) ranges[yaxis] = [y, y];
      ranges[yaxis][0] = Math.min(ranges[yaxis][0], y);
      ranges[yaxis][1] = Math.max(ranges[yaxis][1], y);
    });
  });
  const update = {};
  Object.entries(ranges).forEach(([axis, range]) => {
    const min = range[0];
    const max = range[1];
    const span = max - min;
    const pad = span === 0 ? Math.max(Math.abs(max) * 0.05, 1) : span * 0.08;
    update[`${axis}.range`] = [min - pad, max + pad];
  });
  if (Object.keys(update).length) {
    gd.__autoscalingY = true;
    Plotly.relayout(gd, update).then(() => { gd.__autoscalingY = false; });
  }
}
let autoscaleTimer = null;
gd.on('plotly_relayout', (eventData) => {
  if (gd.__autoscalingY) return;
  clearTimeout(autoscaleTimer);
  autoscaleTimer = setTimeout(() => autoscaleVisibleY(eventData), 80);
});
gd.on('plotly_doubleclick', () => setTimeout(() => autoscaleVisibleY({'xaxis.autorange': true}), 120));
gd.on('plotly_legendclick', () => setTimeout(() => autoscaleVisibleY(), 120));
gd.on('plotly_legenddoubleclick', () => setTimeout(() => autoscaleVisibleY(), 120));
setTimeout(autoscaleVisibleY, 200);
"""


def add_position_shapes(fig: go.Figure, equity_df: pd.DataFrame, stock_a_name: str, stock_b_name: str) -> None:
    legend_added = set()
    colors = {
        "long_a_short_b": "rgba(35, 139, 69, 0.11)",
        "short_a_long_b": "rgba(203, 24, 29, 0.10)",
    }
    labels = {
        "long_a_short_b": f"持倉: 買 {stock_a_name} / 空 {stock_b_name}",
        "short_a_long_b": f"持倉: 空 {stock_a_name} / 買 {stock_b_name}",
    }
    for start, end, position in position_windows(equity_df):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=colors.get(position, "rgba(150,150,150,0.10)"),
            opacity=1,
            layer="below",
            line_width=0,
        )
        label = labels.get(position)
        if label and label not in legend_added:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=11, color=colors[position].replace("0.11", "0.45").replace("0.10", "0.45")),
                    name=label,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            legend_added.add(label)


def add_trade_markers(fig: go.Figure, trade_df: pd.DataFrame, stock_a_name: str, stock_b_name: str) -> None:
    if trade_df.empty:
        return
    styles = {
        "enter_long_a_short_b": ("triangle-up", "#238b45", f"建倉 買 {stock_a_name} / 空 {stock_b_name}"),
        "enter_short_a_long_b": ("triangle-down", "#cb181d", f"建倉 空 {stock_a_name} / 買 {stock_b_name}"),
        "exit_long_a_short_b": ("circle", "#08519c", "平倉"),
        "exit_short_a_long_b": ("circle", "#08519c", "平倉"),
        "forced_exit": ("x", "#54278f", "強制平倉"),
    }
    for action, (symbol, color, label) in styles.items():
        subset = trade_df[trade_df["action"].eq(action)].copy()
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["Date"],
                y=subset["zscore"],
                mode="markers",
                marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color="white")),
                name=label,
                customdata=np.column_stack([subset["action"], subset["equity"]]),
                hovertemplate="%{x|%Y-%m-%d}<br>z=%{y:.2f}<br>%{customdata[0]}<br>equity=%{customdata[1]:,.0f}<extra></extra>",
            ),
            row=3,
            col=1,
        )


def make_chart(
    rank: int,
    row: pd.Series,
    df: pd.DataFrame,
    stats: dict,
    equity_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    metrics: dict,
) -> go.Figure:
    stock_a = f"{row['stock_name_1']}({row['stock_id_1']})"
    stock_b = f"{row['stock_name_2']}({row['stock_id_2']})"
    stock_a_name = str(row["stock_name_1"])
    stock_b_name = str(row["stock_name_2"])

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.36, 0.24, 0.24, 0.16],
        subplot_titles=(
            "標準化股價與均值回歸線",
            "Spread 與五年平均/標準差",
            "策略 rolling z-score 與交易點",
            "60日 rolling correlation",
        ),
    )

    title = (
        f"{rank:03d} {stock_a} / {stock_b} | corr={row['raw_correlation']:.3f}, "
        f"ADF p={row['adf_pvalue_approx']:.3g}, half-life={row['half_life_days']:.1f}d, "
        f"Hurst={row['hurst_exponent']:.3f}"
    )
    metric_text = (
        f"總報酬：{pct(metrics.get('total_return', np.nan))}<br>"
        f"年化報酬：{pct(metrics.get('annual_return', np.nan))}<br>"
        f"最大回撤：{pct(metrics.get('max_drawdown', np.nan))}<br>"
        f"來回交易：{int(metrics.get('round_trips', 0))} 次<br>"
        f"勝率：{pct(metrics.get('win_rate', np.nan))}"
    )

    fig.add_trace(go.Scatter(x=df["Date"], y=df["norm_a"], name=stock_a, line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["norm_b"], name=stock_b, line=dict(color="#ff7f0e")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["norm_equilibrium_a"],
            name=f"{stock_a_name} 均值回歸線",
            line=dict(color="#2ca25f", width=2),
        ),
        row=1,
        col=1,
    )

    mean = stats["spread_mean"]
    std = stats["spread_std"]
    fig.add_trace(go.Scatter(x=df["Date"], y=df["spread"], name="Spread", line=dict(color="#4c4c4c")), row=2, col=1)
    for value, label, color, dash in [
        (mean, "5年 spread mean", "#222222", "solid"),
        (mean + std, "+1σ", "#9ecae1", "dash"),
        (mean - std, "-1σ", "#9ecae1", "dash"),
        (mean + 2 * std, "+2σ", "#fdae6b", "dash"),
        (mean - 2 * std, "-2σ", "#fdae6b", "dash"),
    ]:
        fig.add_hline(y=value, line=dict(color=color, dash=dash, width=1), row=2, col=1, annotation_text=label if label in ["5年 spread mean", "+2σ", "-2σ"] else None)

    fig.add_trace(
        go.Scatter(
            x=equity_df["Date"],
            y=equity_df["zscore"],
            name="策略 rolling z-score",
            line=dict(color="#08519c", width=2),
        ),
        row=3,
        col=1,
    )
    for value, label, color, dash in [
        (0, "0", "#222222", "solid"),
        (2, "+2 建倉門檻", "#d95f0e", "dash"),
        (-2, "-2 建倉門檻", "#d95f0e", "dash"),
        (0.25, "+0.25 平倉區", "#31a354", "dot"),
        (-0.25, "-0.25 平倉區", "#31a354", "dot"),
        (3.5, "+3.5 停損", "#756bb1", "dot"),
        (-3.5, "-3.5 停損", "#756bb1", "dot"),
    ]:
        fig.add_hline(y=value, line=dict(color=color, dash=dash, width=1), row=3, col=1, annotation_text=label if abs(value) in [2, 3.5] else None)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["rolling_corr"], name="60日 rolling corr", line=dict(color="#08519c")), row=4, col=1)
    fig.add_hline(y=float(row["raw_correlation"]), line=dict(color="#636363", dash="dash", width=1), row=4, col=1, annotation_text="全期 raw corr")

    add_position_shapes(fig, equity_df, stock_a_name, stock_b_name)
    add_trade_markers(fig, trade_df, stock_a_name, stock_b_name)

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.985,
        xanchor="right",
        yanchor="top",
        align="right",
        text=f"<b>Backtest</b><br>{metric_text}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#bdbdbd",
        borderwidth=1,
        font=dict(size=12),
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1050,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=105, b=55),
        dragmode="zoom",
    )
    fig.update_yaxes(title_text="標準化股價<br>起點=100", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    fig.update_yaxes(title_text="Rolling corr", range=[-1.05, 1.05], row=4, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.045),
        rangeselector=dict(
            buttons=[
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="5Y"),
            ]
        ),
        row=4,
        col=1,
    )
    return fig


def write_index_html(index_df: pd.DataFrame, output_dir: Path, chart_subdir: str) -> None:
    rows = []
    for row in index_df.itertuples(index=False):
        chart_name = Path(row.plot_path).name
        rows.append(
            "<tr>"
            f"<td>{row.visual_rank}</td>"
            f"<td><a href='{chart_subdir}/{chart_name}'>{row.stock_name_1} / {row.stock_name_2}</a></td>"
            f"<td>{row.industry_1} / {row.industry_2}</td>"
            f"<td>{row.raw_correlation:.3f}</td>"
            f"<td>{row.adf_pvalue_approx:.4g}</td>"
            f"<td>{row.half_life_days:.1f}</td>"
            f"<td>{row.backtest_total_return:.2%}</td>"
            f"<td>{row.backtest_max_drawdown:.2%}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>Pair Trading Interactive Charts</title>
<style>
body {{ font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
th {{ background: #f5f5f5; position: sticky; top: 0; }}
a {{ color: #08519c; text-decoration: none; }}
</style>
</head>
<body>
<h1>Pair Trading Interactive Charts</h1>
<p>每張圖都是近五年資料，可以用滑鼠框選放大、拖曳平移、雙擊還原，底部 range slider 可快速縮放時間。</p>
<table>
<thead><tr><th>Rank</th><th>Pair</th><th>Industry</th><th>Corr</th><th>ADF p</th><th>Half-life</th><th>Backtest return</th><th>Max drawdown</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</body>
</html>
"""
    (output_dir / "interactive_5y_index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    chart_dir = output_dir / args.chart_subdir
    chart_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, dtype={"stock_id_1": str, "stock_id_2": str})
    candidates = candidates.sort_values(
        ["adf_pvalue_approx", "half_life_days", "hit_rate_z2_cross_mean_20d"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    if args.max_pairs:
        candidates = candidates.head(args.max_pairs).copy()

    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)
    rows = []
    for rank, candidate in enumerate(candidates.itertuples(index=False), start=1):
        row = pd.Series(candidate._asdict())
        df, stats = build_pair_data(Path(args.price_dir), row, start_date, args.rolling_corr_window)
        equity_df, trade_df, metrics = run_backtest(
            df,
            args.initial_capital,
            args.lookback,
            args.entry_z,
            args.exit_z,
            args.stop_z,
        )
        fig = make_chart(rank, row, df, stats, equity_df, trade_df, metrics)
        output_path = chart_dir / safe_chart_filename(rank, row)
        fig.write_html(
            output_path,
            include_plotlyjs="directory",
            config={"scrollZoom": True, "displaylogo": False},
            post_script=autoscale_y_post_script(),
        )
        rows.append(
            {
                **row.to_dict(),
                "visual_rank": rank,
                "plot_path": str(output_path),
                "start_date": stats["start_date"],
                "end_date": stats["end_date"],
                "backtest_total_return": metrics.get("total_return", np.nan),
                "backtest_annual_return": metrics.get("annual_return", np.nan),
                "backtest_max_drawdown": metrics.get("max_drawdown", np.nan),
                "backtest_round_trips": metrics.get("round_trips", 0),
                "backtest_win_rate": metrics.get("win_rate", np.nan),
            }
        )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_dir / "interactive_5y_index.csv", index=False, encoding="utf-8-sig")
    write_index_html(index_df, output_dir, args.chart_subdir)
    print(f"Interactive 5-year charts written to {chart_dir}.")
    print(f"Charts: {len(index_df)}")
    print(f"Index HTML: {output_dir / 'interactive_5y_index.html'}")
    print(f"Index CSV: {output_dir / 'interactive_5y_index.csv'}")


if __name__ == "__main__":
    main()
