'''
Modular stock strategy backtesting.
'''
import argparse
import glob
import importlib.util
import inspect
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_viz import (
    get_axis_range,
    get_doji_points,
    get_missing_trading_dates,
    get_wick_points,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
PLOT_DIR = PROJECT_ROOT / 'plot'
STRATEGY_DIR = PROJECT_ROOT / 'strategies'
REQUIRED_COLUMNS = [
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Capacity',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Backtest a modular trading strategy on cached stock data.'
    )
    parser.add_argument('--stock', default='3105', help='Stock code to backtest.')
    parser.add_argument(
        '--start',
        default=None,
        help='Start date in YYYY-MM-DD format. Defaults to recent one year.',
    )
    parser.add_argument(
        '--end',
        default=None,
        help='End date in YYYY-MM-DD format. Defaults to latest cached date.',
    )
    parser.add_argument(
        '--fund',
        type=float,
        default=3000000,
        help='Starting cash fund. Defaults to 3000000.',
    )
    parser.add_argument(
        '--strategy',
        default='NaiveStrategy',
        help='Strategy class name to load from strategies/*.py.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output HTML path. Defaults to plot/backtest_*.html.',
    )
    return parser.parse_args()


def find_latest_stock_csv(stock):
    patterns = [
        str(DATA_DIR / f'{stock}_*_to_*.csv'),
        str(DATA_DIR / 'price' / f'{stock}_*_to_*.csv'),
    ]
    stock_files = sorted(
        stock_file
        for pattern in patterns
        for stock_file in glob.glob(pattern)
    )
    if not stock_files:
        raise FileNotFoundError(
            f'No cached CSV found for stock {stock}. Expected one of: '
            f'{", ".join(patterns)}'
        )
    return Path(stock_files[-1])


def load_stock_data(stock):
    csv_path = find_latest_stock_csv(stock)
    df_stock = pd.read_csv(csv_path, parse_dates=['Date'])

    missing_cols = [
        column for column in REQUIRED_COLUMNS
        if column not in df_stock.columns
    ]
    if missing_cols:
        raise ValueError(f'{csv_path} is missing required columns: {missing_cols}')

    df_stock = df_stock.copy()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

    for column in ['Open', 'High', 'Low', 'Close', 'Capacity']:
        df_stock[column] = pd.to_numeric(df_stock[column], errors='coerce')

    df_stock = (
        df_stock
        .dropna(subset=REQUIRED_COLUMNS)
        .sort_values('Date')
        .reset_index(drop=True)
    )

    if df_stock.empty:
        raise ValueError(f'{csv_path} does not contain usable stock data.')

    return df_stock, csv_path


def parse_date_arg(value, arg_name):
    if value is None:
        return None

    try:
        return pd.to_datetime(value, format='%Y-%m-%d')
    except ValueError as exc:
        raise ValueError(f'{arg_name} must use YYYY-MM-DD format.') from exc


def filter_stock_data(df_stock, start_arg, end_arg):
    latest_date = df_stock['Date'].max()
    earliest_date = df_stock['Date'].min()
    end_date = parse_date_arg(end_arg, '--end') or latest_date
    start_date = parse_date_arg(start_arg, '--start') or (latest_date - timedelta(days=365))

    if start_date > end_date:
        raise ValueError(
            f'Invalid date range: start {start_date.date()} is after '
            f'end {end_date.date()}.'
        )

    start_date = max(start_date, earliest_date)
    end_date = min(end_date, latest_date)
    filtered_df = df_stock[
        (df_stock['Date'] >= start_date)
        & (df_stock['Date'] <= end_date)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f'No stock data between {start_date.date()} and {end_date.date()}.'
        )

    return filtered_df.reset_index(drop=True), start_date, end_date


def import_module_from_path(module_path):
    module_name = f'_backtesting_strategy_{module_path.stem}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import strategy module {module_path}.')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_available_strategy_classes():
    strategy_classes = {}
    if not STRATEGY_DIR.exists():
        return strategy_classes

    for module_path in sorted(STRATEGY_DIR.glob('*.py')):
        if module_path.name == '__init__.py':
            continue

        module = import_module_from_path(module_path)
        for name, value in inspect.getmembers(module, inspect.isclass):
            if value.__module__ == module.__name__:
                strategy_classes[name] = value

    return strategy_classes


def load_strategy_class(strategy_name):
    strategy_classes = get_available_strategy_classes()
    if strategy_name not in strategy_classes:
        available = ', '.join(sorted(strategy_classes)) or 'none'
        raise ValueError(
            f'Unknown strategy class {strategy_name}. Available classes: {available}'
        )
    return strategy_classes[strategy_name]


def run_backtest(df_stock, strategy):
    equity_rows = []
    trade_rows = []

    for row in df_stock.itertuples(index=False):
        price = float(row.Close)
        before_trade_count = len(strategy.trades)
        action = strategy.run(price)
        equity = strategy.total_equity(price)

        if len(strategy.trades) > before_trade_count:
            trade = strategy.trades[-1].copy()
            trade['date'] = row.Date
            trade['equity'] = equity
            trade_rows.append(trade)

        equity_rows.append({
            'Date': row.Date,
            'Close': price,
            'Action': action,
            'Cash': strategy.fund,
            'Shares': strategy.shares,
            'Equity': equity,
        })

    equity_df = pd.DataFrame(equity_rows)
    trade_df = pd.DataFrame(trade_rows)
    return equity_df, trade_df


def calculate_max_drawdown(equity_series):
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1
    return drawdown.min(), drawdown


def calculate_round_trip_profits(trade_df):
    if trade_df.empty:
        return []

    open_buys = []
    round_trip_profits = []

    for trade in trade_df.itertuples(index=False):
        if trade.action == 'buy':
            open_buys.append(float(trade.price))
        elif trade.action == 'sell' and open_buys:
            buy_price = open_buys.pop(0)
            shares = int(trade.shares)
            round_trip_profits.append((float(trade.price) - buy_price) * shares)

    return round_trip_profits


def calculate_win_rate(round_trip_profits):
    if not round_trip_profits:
        return None

    wins = sum(1 for profit in round_trip_profits if profit > 0)
    return wins / len(round_trip_profits)


def calculate_profit_factor(round_trip_profits):
    profits = [profit for profit in round_trip_profits if profit > 0]
    losses = [-profit for profit in round_trip_profits if profit < 0]

    if not profits or not losses:
        if profits and not losses:
            return math.inf
        return None

    return (sum(profits) / len(profits)) / (sum(losses) / len(losses))


def calculate_sharpe_ratio(equity_series):
    daily_returns = equity_series.pct_change().dropna()
    daily_returns = daily_returns[daily_returns.notna()]

    if daily_returns.empty:
        return None

    return_std = daily_returns.std(ddof=1)
    if return_std == 0 or pd.isna(return_std):
        return None

    return float((daily_returns.mean() / return_std) * math.sqrt(252))


def calculate_metrics(initial_fund, equity_df, trade_df):
    final_equity = float(equity_df['Equity'].iloc[-1])
    final_profit = final_equity - initial_fund
    profit_pct = final_profit / initial_fund if initial_fund else 0
    max_drawdown, drawdown = calculate_max_drawdown(equity_df['Equity'])
    buy_count = int((trade_df['action'] == 'buy').sum()) if not trade_df.empty else 0
    sell_count = int((trade_df['action'] == 'sell').sum()) if not trade_df.empty else 0
    round_trip_profits = calculate_round_trip_profits(trade_df)

    metrics = {
        'initial_fund': float(initial_fund),
        'final_equity': final_equity,
        'final_profit': final_profit,
        'profit_pct': profit_pct,
        'max_drawdown': float(max_drawdown),
        'total_trades': int(len(trade_df)),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'ending_cash': float(equity_df['Cash'].iloc[-1]),
        'ending_shares': int(equity_df['Shares'].iloc[-1]),
        'completed_trades': len(round_trip_profits),
        'win_rate': calculate_win_rate(round_trip_profits),
        'profit_factor': calculate_profit_factor(round_trip_profits),
        'sharpe_ratio': calculate_sharpe_ratio(equity_df['Equity']),
    }
    return metrics, drawdown


def format_pct(value):
    if value is None:
        return 'N/A'
    return f'{value:.2%}'


def format_number(value):
    if value is None:
        return 'N/A'
    if value == math.inf:
        return 'Infinity'
    return f'{value:.2f}'


def print_metrics(metrics, output_path):
    print('Backtest summary')
    print(f"Initial fund: {metrics['initial_fund']:,.2f}")
    print(f"Final equity: {metrics['final_equity']:,.2f}")
    print(
        f"Final profit: {metrics['final_profit']:,.2f} "
        f"({format_pct(metrics['profit_pct'])})"
    )
    print(f"Max drawdown: {format_pct(metrics['max_drawdown'])}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Buy count: {metrics['buy_count']}")
    print(f"Sell count: {metrics['sell_count']}")
    print(f"Completed trades: {metrics['completed_trades']}")
    print(f"Ending cash: {metrics['ending_cash']:,.2f}")
    print(f"Ending shares: {metrics['ending_shares']:,}")
    print(f"Win rate: {format_pct(metrics['win_rate'])}")
    print(f"Profit factor: {format_number(metrics['profit_factor'])}")
    print(f"Sharpe ratio: {format_number(metrics['sharpe_ratio'])}")
    print(f'HTML report: {output_path}')


def get_default_output_path(stock, strategy_name, start_date, end_date):
    PLOT_DIR.mkdir(exist_ok=True)
    return PLOT_DIR / (
        f'backtest_{stock}_{strategy_name}_'
        f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.html'
    )


def add_trade_markers(fig, trade_df, df_stock, row=1):
    if trade_df.empty:
        return

    marker_df = trade_df.merge(
        df_stock[['Date', 'High', 'Low']],
        left_on='date',
        right_on='Date',
        how='left',
    )
    price_span = df_stock['High'].max() - df_stock['Low'].min()
    marker_offset = price_span * 0.025 if price_span else 1

    marker_styles = {
        'buy': dict(
            symbol='triangle-up',
            color='#f59e0b',
            name='Buy',
            y_column='Low',
            offset=-marker_offset,
        ),
        'sell': dict(
            symbol='triangle-down',
            color='#2563eb',
            name='Sell',
            y_column='High',
            offset=marker_offset,
        ),
    }

    for action, marker in marker_styles.items():
        rows = marker_df[marker_df['action'] == action].copy()
        if rows.empty:
            continue

        rows['marker_y'] = rows[marker['y_column']] + marker['offset']

        fig.add_trace(
            go.Scatter(
                x=rows['date'],
                y=rows['marker_y'],
                mode='markers',
                name=marker['name'],
                marker=dict(
                    symbol=marker['symbol'],
                    color=marker['color'],
                    size=12,
                    line=dict(color='white', width=1),
                ),
                customdata=rows[['shares', 'value', 'price']],
                hovertemplate=(
                    'Date=%{x}<br>'
                    f'{marker["name"]} price=%{{customdata[2]:.2f}}<br>'
                    'Shares=%{customdata[0]:,}<br>'
                    'Value=%{customdata[1]:,.2f}<extra></extra>'
                ),
            ),
            row=row,
            col=1,
        )


def add_solid_candle_traces(fig, df_stock, row=1):
    rising_df = df_stock[df_stock['Close'] > df_stock['Open']]
    falling_df = df_stock[df_stock['Close'] < df_stock['Open']]
    rising_wick_x, rising_wick_y = get_wick_points(df_stock, 'rising')
    falling_wick_x, falling_wick_y = get_wick_points(df_stock, 'falling')
    doji_wick_x, doji_wick_y = get_wick_points(df_stock, 'doji')
    doji_x, doji_y = get_doji_points(df_stock)

    fig.add_trace(
        go.Scatter(
            x=rising_wick_x,
            y=rising_wick_y,
            mode='lines',
            line=dict(color='#d62728', width=1),
            hoverinfo='skip',
            showlegend=False,
            name='Rising wick',
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=falling_wick_x,
            y=falling_wick_y,
            mode='lines',
            line=dict(color='#2ca02c', width=1),
            hoverinfo='skip',
            showlegend=False,
            name='Falling wick',
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=doji_wick_x,
            y=doji_wick_y,
            mode='lines',
            line=dict(color='#666666', width=1),
            hoverinfo='skip',
            showlegend=False,
            name='Unchanged wick',
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=doji_x,
            y=doji_y,
            mode='lines',
            line=dict(color='#666666', width=2),
            hoverinfo='skip',
            showlegend=False,
            name='Unchanged price',
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=rising_df['Date'],
            y=(rising_df['Close'] - rising_df['Open']).abs(),
            base=rising_df[['Open', 'Close']].min(axis=1),
            name='Rising price',
            marker=dict(color='#d62728', line=dict(width=0)),
            customdata=rising_df[['Open', 'High', 'Low', 'Close']],
            hovertemplate=(
                'Date=%{x}<br>'
                'Open=%{customdata[0]}<br>'
                'High=%{customdata[1]}<br>'
                'Low=%{customdata[2]}<br>'
                'Close=%{customdata[3]}<extra></extra>'
            ),
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=falling_df['Date'],
            y=(falling_df['Close'] - falling_df['Open']).abs(),
            base=falling_df[['Open', 'Close']].min(axis=1),
            name='Falling price',
            marker=dict(color='#2ca02c', line=dict(width=0)),
            customdata=falling_df[['Open', 'High', 'Low', 'Close']],
            hovertemplate=(
                'Date=%{x}<br>'
                'Open=%{customdata[0]}<br>'
                'High=%{customdata[1]}<br>'
                'Low=%{customdata[2]}<br>'
                'Close=%{customdata[3]}<extra></extra>'
            ),
        ),
        row=row,
        col=1,
    )


def add_metrics_table(fig, metrics):
    labels = [
        'Final equity',
        'Final profit',
        'Max DD',
        'Total trades',
        'Win rate',
        'Profit factor',
        'Sharpe',
    ]
    values = [
        f"{metrics['final_equity']:,.2f}",
        f"{metrics['final_profit']:,.2f} ({format_pct(metrics['profit_pct'])})",
        format_pct(metrics['max_drawdown']),
        f"{metrics['total_trades']}",
        format_pct(metrics['win_rate']),
        format_number(metrics['profit_factor']),
        format_number(metrics['sharpe_ratio']),
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=labels,
                fill_color='#eef2ff',
                align='center',
                font=dict(color='#0f172a', size=12),
                height=26,
            ),
            cells=dict(
                values=[[value] for value in values],
                fill_color='white',
                align='center',
                font=dict(color='#0f172a', size=12),
                height=28,
            ),
        ),
        row=1,
        col=1,
    )


def build_report_figure(df_stock, equity_df, trade_df, drawdown, title, metrics):
    df_stock = df_stock.copy()
    missing_dates = get_missing_trading_dates(df_stock)
    hidden_date_breaks = [
        dict(bounds=['sat', 'mon']),
        dict(values=missing_dates),
    ]
    price_range = get_axis_range(df_stock['Low'].min(), df_stock['High'].max())
    equity_low = min(equity_df['Equity'].min(), (equity_df['Equity'] - metrics['initial_fund']).min())
    equity_high = max(equity_df['Equity'].max(), (equity_df['Equity'] - metrics['initial_fund']).max())
    equity_range = get_axis_range(equity_low, equity_high)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.11, 0.47, 0.25, 0.17],
        specs=[
            [{'type': 'table'}],
            [{'type': 'xy'}],
            [{'type': 'xy'}],
            [{'type': 'xy'}],
        ],
        subplot_titles=('', 'Price and trades', 'Equity and profit', 'Drawdown'),
    )

    add_metrics_table(fig, metrics)
    add_solid_candle_traces(fig, df_stock, row=2)
    add_trade_markers(fig, trade_df, df_stock, row=2)

    profit = equity_df['Equity'] - metrics['initial_fund']
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#2563eb', width=2),
            hovertemplate='Date=%{x}<br>Equity=%{y:,.2f}<extra></extra>',
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=profit,
            mode='lines',
            name='Profit',
            line=dict(color='#f59e0b', width=2),
            hovertemplate='Date=%{x}<br>Profit=%{y:,.2f}<extra></extra>',
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=drawdown,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#7c3aed', width=2),
            hovertemplate='Date=%{x}<br>Drawdown=%{y:.2%}<extra></extra>',
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=dict(text=title, x=0.02),
        height=980,
        template='plotly_white',
        hovermode='x unified',
        dragmode='zoom',
        bargap=0.18,
        margin=dict(l=60, r=32, t=86, b=105),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.12,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.85)',
        ),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            rangebreaks=hidden_date_breaks,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all', label='All'),
                ],
            ),
        ),
        xaxis2=dict(
            rangeslider=dict(visible=False),
            rangebreaks=hidden_date_breaks,
        ),
        xaxis3=dict(
            rangeslider=dict(visible=False),
            rangebreaks=hidden_date_breaks,
        ),
        yaxis=dict(range=price_range),
        yaxis2=dict(range=equity_range),
    )
    fig.update_yaxes(title_text='Price', row=2, col=1)
    fig.update_yaxes(title_text='Amount', row=3, col=1)
    fig.update_yaxes(title_text='Drawdown', tickformat='.0%', row=4, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor')
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor')
    return fig


def write_html_report(
    df_stock,
    equity_df,
    trade_df,
    drawdown,
    metrics,
    stock,
    strategy_name,
    output_path,
):
    title = (
        f'{stock} {strategy_name} backtest '
        f'({df_stock["Date"].min().date()} to {df_stock["Date"].max().date()})'
    )
    fig = build_report_figure(df_stock, equity_df, trade_df, drawdown, title, metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        config={'scrollZoom': True, 'displaylogo': False},
    )


def main():
    args = parse_args()
    if args.fund <= 0:
        raise ValueError('--fund must be greater than 0.')

    df_stock, csv_path = load_stock_data(args.stock)
    df_stock, start_date, end_date = filter_stock_data(
        df_stock,
        args.start,
        args.end,
    )
    strategy_class = load_strategy_class(args.strategy)
    strategy = strategy_class(fund=args.fund)
    equity_df, trade_df = run_backtest(df_stock, strategy)
    metrics, drawdown = calculate_metrics(args.fund, equity_df, trade_df)
    output_path = (
        Path(args.output)
        if args.output
        else get_default_output_path(args.stock, args.strategy, start_date, end_date)
    )

    write_html_report(
        df_stock,
        equity_df,
        trade_df,
        drawdown,
        metrics,
        args.stock,
        args.strategy,
        output_path,
    )
    print(f'Data source: {csv_path}')
    print_metrics(metrics, output_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
