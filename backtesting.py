'''
Modular stock strategy backtesting.
'''
import argparse
import glob
import importlib.util
import inspect
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_viz import (
    add_capacity_trace,
    add_stock_price_traces,
    get_axis_range,
    get_autorange_script,
    get_hidden_date_breaks,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
PLOT_DIR = PROJECT_ROOT / 'data_viz' / 'backtesting'
STRATEGY_DIR = PROJECT_ROOT / 'strategies'
STOCK_METADATA_PATH = DATA_DIR / 'metadata.csv'
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
        '--all',
        action='store_true',
        help='Backtest every cached listed-stock CSV in data/price.',
    )
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
        '--strategies',
        default=None,
        help=(
            'Comma-separated strategy aliases or class names, for example '
            'buy_and_hold,optimal,macd. Defaults to naive,buy_and_hold.'
        ),
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Output HTML path for one stock. In --all mode, this is treated as '
            'the run directory unless --run-dir is set.'
        ),
    )
    parser.add_argument(
        '--run-dir',
        default=None,
        help=(
            'Directory for one batch run. Defaults to '
            'data_viz/backtesting/backtest_run_<timestamp> in --all mode.'
        ),
    )
    return parser.parse_args()


def find_latest_stock_csv(stock):
    patterns = [str(DATA_DIR / 'price' / f'{stock}_*.csv')]
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


def get_stock_code_from_csv(csv_path):
    return csv_path.stem.split('_', 1)[0]


def get_all_cached_stock_codes():
    if not STOCK_METADATA_PATH.exists():
        raise FileNotFoundError(
            f'Metadata catalog is required for --all: {STOCK_METADATA_PATH}'
        )

    metadata_df = read_csv_canonical(STOCK_METADATA_PATH, dtype={'Code': str})
    required_columns = {'Code', 'Type', 'Market'}
    missing_columns = required_columns.difference(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            f'{STOCK_METADATA_PATH} missing required columns: '
            f'{sorted(missing_columns)}'
        )

    listed_codes = set(
        metadata_df.loc[
            metadata_df['Type'].eq('股票')
            & metadata_df['Market'].eq('上市')
            & metadata_df['Code'].astype(str).str.fullmatch(r'\d{4}'),
            'Code',
        ].astype(str)
    )
    csv_files = list((DATA_DIR / 'price').glob('*.csv'))
    latest_by_stock = {}
    for csv_path in csv_files:
        stock = get_stock_code_from_csv(csv_path)
        if stock not in listed_codes:
            continue
        current_path = latest_by_stock.get(stock)
        if current_path is None or csv_path.name > current_path.name:
            latest_by_stock[stock] = csv_path

    return sorted(latest_by_stock)


def load_stock_name_map():
    if not STOCK_METADATA_PATH.exists():
        return {}

    metadata_df = read_csv_canonical(STOCK_METADATA_PATH, dtype={'Code': str})
    if 'Code' not in metadata_df.columns or 'Name' not in metadata_df.columns:
        return {}

    metadata_df['Code'] = metadata_df['Code'].astype(str)
    metadata_df['Name'] = metadata_df['Name'].fillna('').astype(str)
    return dict(zip(metadata_df['Code'], metadata_df['Name']))


def load_stock_data(stock):
    csv_path = find_latest_stock_csv(stock)
    df_stock = read_csv_canonical(csv_path, parse_dates=['Date'])
    return clean_loaded_stock_csv(df_stock), csv_path


def clean_loaded_stock_csv(df_stock):
    missing_cols = [
        column for column in REQUIRED_COLUMNS
        if column not in df_stock.columns
    ]
    if missing_cols:
        raise ValueError(f'CSV is missing required columns: {missing_cols}')

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
        raise ValueError('CSV does not contain usable stock data.')

    return df_stock


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


def get_available_strategies():
    strategy_aliases = {}
    strategy_classes = {}
    if not STRATEGY_DIR.exists():
        return strategy_aliases, strategy_classes

    for module_path in sorted(STRATEGY_DIR.glob('*.py')):
        if module_path.name == '__init__.py':
            continue

        module = import_module_from_path(module_path)
        module_classes = []
        for name, value in inspect.getmembers(module, inspect.isclass):
            if value.__module__ == module.__name__:
                module_classes.append((name, value))
                strategy_classes[name] = value

        preferred_classes = [
            (name, value)
            for name, value in module_classes
            if name.endswith('Strategy')
        ]
        if preferred_classes:
            strategy_aliases[module_path.stem] = preferred_classes[0][1]
        elif module_classes:
            strategy_aliases[module_path.stem] = module_classes[0][1]

    return strategy_aliases, strategy_classes


def load_strategy_class(strategy_name):
    strategy_aliases, strategy_classes = get_available_strategies()
    if strategy_name in strategy_aliases:
        return strategy_aliases[strategy_name]
    if strategy_name in strategy_classes:
        return strategy_classes[strategy_name]

    available_aliases = ', '.join(sorted(strategy_aliases)) or 'none'
    available_classes = ', '.join(sorted(strategy_classes)) or 'none'
    raise ValueError(
        f'Unknown strategy {strategy_name}. Available aliases: '
        f'{available_aliases}. Available classes: {available_classes}'
    )


def parse_strategy_names(args):
    if args.strategies:
        names = [
            strategy_name.strip()
            for strategy_name in args.strategies.split(',')
            if strategy_name.strip()
        ]
        if not names:
            raise ValueError('--strategies must include at least one strategy.')
        return ensure_baseline_strategy(names)

    return ensure_baseline_strategy(['naive'])


def ensure_baseline_strategy(strategy_names):
    names = list(strategy_names)
    if 'buy_and_hold' not in names:
        names.append('buy_and_hold')
    return names


def get_strategy_display_name(strategy_name, strategy_class):
    return strategy_name if strategy_name else strategy_class.__name__


def run_strategy(df_stock, strategy_name, strategy_class, initial_fund):
    strategy = strategy_class(fund=initial_fund)
    equity_df, trade_df = run_backtest(df_stock, strategy)
    metrics, drawdown = calculate_metrics(initial_fund, equity_df, trade_df)
    display_name = get_strategy_display_name(strategy_name, strategy_class)
    return {
        'name': display_name,
        'strategy_class': strategy_class,
        'equity_df': equity_df,
        'trade_df': trade_df,
        'metrics': metrics,
        'drawdown': drawdown,
    }


def run_backtest(df_stock, strategy):
    equity_rows = []
    trade_rows = []

    if hasattr(strategy, 'prepare'):
        strategy.prepare(df_stock['Close'].tolist())

    for row in df_stock.itertuples(index=False):
        price = float(row.Close)
        before_trade_count = len(strategy.trades)
        action = strategy.run(price)
        equity = strategy.total_equity(price)

        for new_trade in strategy.trades[before_trade_count:]:
            trade = new_trade.copy()
            trade['date'] = row.Date
            trade['equity'] = equity
            trade.setdefault('fee', 0)
            trade.setdefault('tax', 0)
            trade.setdefault('transaction_cost', 0)
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
            open_buys.append({
                'value': float(trade.value),
                'transaction_cost': float(getattr(trade, 'transaction_cost', 0)),
            })
        elif trade.action == 'sell' and open_buys:
            buy_trade = open_buys.pop(0)
            sell_value = float(trade.value)
            sell_cost = float(getattr(trade, 'transaction_cost', 0))
            round_trip_profits.append(
                sell_value
                - buy_trade['value']
                - buy_trade['transaction_cost']
                - sell_cost
            )

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
    transaction_cost = (
        float(trade_df['transaction_cost'].sum())
        if not trade_df.empty and 'transaction_cost' in trade_df.columns
        else 0
    )
    round_trip_profits = calculate_round_trip_profits(trade_df)

    metrics = {
        'initial_fund': float(initial_fund),
        'final_equity': final_equity,
        'final_profit': final_profit,
        'profit_pct': profit_pct,
        'max_drawdown': float(max_drawdown),
        'total_trades': int(len(trade_df)),
        'transaction_cost': transaction_cost,
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


def build_strategy_comparison_row(name, metrics):
    return {
        'name': name,
        'available': True,
        'final_equity': metrics['final_equity'],
        'final_profit': metrics['final_profit'],
        'profit_pct': metrics['profit_pct'],
        'max_drawdown': metrics['max_drawdown'],
        'total_trades': metrics['total_trades'],
        'transaction_cost': metrics['transaction_cost'],
        'win_rate': metrics['win_rate'],
        'profit_factor': metrics['profit_factor'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'note': (
            f"cash={format_money(metrics['ending_cash'])}, "
            f"shares={format_int(metrics['ending_shares'])}"
        ),
    }


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


def format_money(value):
    if value is None:
        return 'N/A'
    return f'{value:,.2f}'


def format_int(value):
    if value is None:
        return 'N/A'
    return f'{int(value):,}'


def print_metrics(metrics, output_path):
    print(f"Initial fund: {metrics['initial_fund']:,.2f}")
    print(f"Final equity: {metrics['final_equity']:,.2f}")
    print(
        f"Final profit: {metrics['final_profit']:,.2f} "
        f"({format_pct(metrics['profit_pct'])})"
    )
    print(f"Max drawdown: {format_pct(metrics['max_drawdown'])}")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Transaction cost: {metrics['transaction_cost']:,.2f}")
    print(f"Buy count: {metrics['buy_count']}")
    print(f"Sell count: {metrics['sell_count']}")
    print(f"Completed trades: {metrics['completed_trades']}")
    print(f"Ending cash: {metrics['ending_cash']:,.2f}")
    print(f"Ending shares: {metrics['ending_shares']:,}")
    print(f"Win rate: {format_pct(metrics['win_rate'])}")
    print(f"Profit factor: {format_number(metrics['profit_factor'])}")
    print(f"Sharpe ratio: {format_number(metrics['sharpe_ratio'])}")

    print(f'HTML report: {output_path}')


def print_strategy_results(strategy_results, output_path):
    print('Backtest summary')
    for index, result in enumerate(strategy_results):
        if index:
            print('')
        print(f"Strategy: {result['name']}")
        print_metrics(result['metrics'], output_path)


def get_default_output_path(stock, strategy_names, start_date, end_date, output_dir=None):
    output_dir = Path(output_dir) if output_dir else PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_slug = '_'.join(strategy_names)
    return output_dir / (
        f'backtest_{stock}_{strategy_slug}_'
        f'{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.html'
    )


def add_trade_markers(
    fig,
    trade_df,
    df_stock,
    strategy_name=None,
    visible=True,
    row=1,
):
    if trade_df.empty:
        return []

    trace_indices = []

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
                name=(
                    f'{strategy_name} {marker["name"]}'
                    if strategy_name
                    else marker['name']
                ),
                visible=visible,
                marker=dict(
                    symbol=marker['symbol'],
                    color=marker['color'],
                    size=12,
                    line=dict(color='white', width=1),
                ),
                customdata=rows[['shares', 'value', 'price', 'transaction_cost']],
                hovertemplate=(
                    'Date=%{x}<br>'
                    f'{marker["name"]} price=%{{customdata[2]:.2f}}<br>'
                    'Shares=%{customdata[0]:,}<br>'
                    'Value=%{customdata[1]:,.2f}<br>'
                    'Cost=%{customdata[3]:,.2f}<extra></extra>'
                ),
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    return trace_indices


def comparison_cell(row, key, formatter):
    if not row.get('available'):
        return 'N/A'
    return formatter(row.get(key))


def add_metrics_table(fig, comparison_rows):
    labels = [
        'Name',
        'Final equity',
        'Final profit',
        'Return',
        'Max DD',
        'Trades',
        'Transaction cost',
        'Win rate',
        'Profit factor',
        'Sharpe',
        'Note',
    ]
    values = [
        [row['name'] for row in comparison_rows],
        [comparison_cell(row, 'final_equity', format_money) for row in comparison_rows],
        [comparison_cell(row, 'final_profit', format_money) for row in comparison_rows],
        [comparison_cell(row, 'profit_pct', format_pct) for row in comparison_rows],
        [comparison_cell(row, 'max_drawdown', format_pct) for row in comparison_rows],
        [comparison_cell(row, 'total_trades', format_int) for row in comparison_rows],
        [comparison_cell(row, 'transaction_cost', format_money) for row in comparison_rows],
        [comparison_cell(row, 'win_rate', format_pct) for row in comparison_rows],
        [comparison_cell(row, 'profit_factor', format_number) for row in comparison_rows],
        [comparison_cell(row, 'sharpe_ratio', format_number) for row in comparison_rows],
        [row.get('note', '') for row in comparison_rows],
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
                values=values,
                fill_color='white',
                align='center',
                font=dict(color='#0f172a', size=12),
                height=30,
            ),
        ),
        row=1,
        col=1,
    )


def get_strategy_plot_ranges(strategy_results):
    ranges = {}
    for result in strategy_results:
        equity_df = result['equity_df']
        initial_fund = result['metrics']['initial_fund']
        amount_values = []
        amount_values.extend(equity_df['Equity'].tolist())
        amount_values.extend((equity_df['Equity'] - initial_fund).tolist())
        ranges[result['name']] = {
            'amount': get_axis_range(min(amount_values), max(amount_values)),
            'drawdown': get_axis_range(result['drawdown'].min(), 0),
        }

    return ranges


def add_strategy_selector(fig, strategy_results, strategy_trace_indices, ranges):
    buttons = []
    for result in strategy_results:
        strategy_name = result['name']
        visible = [True] * len(fig.data)

        for current_name, trace_indices in strategy_trace_indices.items():
            for trace_index in trace_indices:
                visible[trace_index] = current_name == strategy_name

        buttons.append(
            dict(
                label=strategy_name,
                method='update',
                args=[
                    {'visible': visible},
                    {
                        'yaxis3.range': ranges[strategy_name]['amount'],
                        'yaxis4.range': ranges[strategy_name]['drawdown'],
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=1,
                xanchor='right',
                y=1.08,
                yanchor='top',
                pad=dict(l=8, r=8, t=4, b=4),
            )
        ],
        annotations=[
            *fig.layout.annotations,
            dict(
                text='Shown strategy',
                x=0.82,
                xref='paper',
                y=1.08,
                yref='paper',
                showarrow=False,
                align='right',
            ),
        ],
    )


def build_report_figure(
    df_stock,
    strategy_results,
    title,
    comparison_rows,
):
    df_stock = df_stock.copy()
    hidden_date_breaks = get_hidden_date_breaks(df_stock)
    price_range = get_axis_range(df_stock['Low'].min(), df_stock['High'].max())
    capacity_range = get_axis_range(0, df_stock['Capacity'].max(), floor_zero=True)
    strategy_ranges = get_strategy_plot_ranges(strategy_results)
    selected_strategy = strategy_results[0]['name']
    table_height = 70 + len(comparison_rows) * 34
    plot_row_heights = [365, 160, 275, 180]
    subplot_gap_px = 28
    subplot_gap_count = 4
    figure_height = (
        table_height
        + sum(plot_row_heights)
        + subplot_gap_px * subplot_gap_count
    )
    vertical_spacing = subplot_gap_px / figure_height

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_heights=[table_height, *plot_row_heights],
        specs=[
            [{'type': 'table'}],
            [{'type': 'xy'}],
            [{'type': 'xy'}],
            [{'type': 'xy'}],
            [{'type': 'xy'}],
        ],
        subplot_titles=('', 'Price and trades', 'Capacity', 'Equity and profit', 'Drawdown'),
    )

    add_metrics_table(fig, comparison_rows)
    add_stock_price_traces(
        fig,
        df_stock,
        row=2,
        col=1,
        include_moving_averages=False,
    )
    add_capacity_trace(fig, df_stock, row=3, col=1)

    strategy_trace_indices = {
        result['name']: []
        for result in strategy_results
    }
    colors = [
        '#2563eb',
        '#f59e0b',
        '#16a34a',
        '#dc2626',
        '#7c3aed',
        '#0891b2',
        '#be123c',
    ]
    for index, result in enumerate(strategy_results):
        color = colors[index % len(colors)]
        visible = result['name'] == selected_strategy
        marker_indices = add_trade_markers(
            fig,
            result['trade_df'],
            df_stock,
            strategy_name=result['name'],
            visible=visible,
            row=2,
        )
        strategy_trace_indices[result['name']].extend(marker_indices)

        equity_df = result['equity_df']
        metrics = result['metrics']
        profit = equity_df['Equity'] - metrics['initial_fund']
        fig.add_trace(
            go.Scatter(
                x=equity_df['Date'],
                y=equity_df['Equity'],
                mode='lines',
                name=f"{result['name']} Equity",
                visible=visible,
                line=dict(color=color, width=2),
                hovertemplate='Date=%{x}<br>Equity=%{y:,.2f}<extra></extra>',
            ),
            row=4,
            col=1,
        )
        strategy_trace_indices[result['name']].append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=equity_df['Date'],
                y=profit,
                mode='lines',
                name=f"{result['name']} Profit",
                visible=visible,
                line=dict(color=color, width=2, dash='dot'),
                hovertemplate='Date=%{x}<br>Profit=%{y:,.2f}<extra></extra>',
            ),
            row=4,
            col=1,
        )
        strategy_trace_indices[result['name']].append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=equity_df['Date'],
                y=result['drawdown'],
                mode='lines',
                name=f"{result['name']} Drawdown",
                visible=visible,
                line=dict(color=color, width=2),
                hovertemplate='Date=%{x}<br>Drawdown=%{y:.2%}<extra></extra>',
            ),
            row=5,
            col=1,
        )
        strategy_trace_indices[result['name']].append(len(fig.data) - 1)

    fig.update_layout(
        title=dict(text=title, x=0.02),
        height=figure_height,
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
        xaxis4=dict(
            rangeslider=dict(visible=False),
            rangebreaks=hidden_date_breaks,
        ),
        yaxis=dict(range=price_range),
        yaxis2=dict(range=capacity_range),
        yaxis3=dict(range=strategy_ranges[selected_strategy]['amount']),
        yaxis4=dict(range=strategy_ranges[selected_strategy]['drawdown']),
    )
    add_strategy_selector(
        fig,
        strategy_results,
        strategy_trace_indices,
        strategy_ranges,
    )
    fig.update_yaxes(title_text='Price', row=2, col=1)
    fig.update_yaxes(title_text='Capacity', row=3, col=1)
    fig.update_yaxes(title_text='Amount', row=4, col=1)
    fig.update_yaxes(title_text='Drawdown', tickformat='.0%', row=5, col=1)
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor')
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor')
    return fig


def write_html_report(
    df_stock,
    strategy_results,
    comparison_rows,
    stock,
    strategy_names,
    output_path,
):
    strategy_label = ', '.join(strategy_names)
    title = (
        f'{stock} {strategy_label} backtest '
        f'({df_stock["Date"].min().date()} to {df_stock["Date"].max().date()})'
    )
    fig = build_report_figure(
        df_stock,
        strategy_results,
        title,
        comparison_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        config={'scrollZoom': True, 'displaylogo': False},
        post_script=get_autorange_script(df_stock),
    )


def run_backtest_for_stock(stock, args, strategy_names, output_dir=None):
    df_stock, csv_path = load_stock_data(stock)
    df_stock, start_date, end_date = filter_stock_data(
        df_stock,
        args.start,
        args.end,
    )
    strategy_results = [
        run_strategy(
            df_stock,
            strategy_name,
            load_strategy_class(strategy_name),
            args.fund,
        )
        for strategy_name in strategy_names
    ]
    comparison_rows = [
        build_strategy_comparison_row(result['name'], result['metrics'])
        for result in strategy_results
    ]

    output_path = (
        Path(args.output)
        if args.output and not args.all
        else get_default_output_path(stock, strategy_names, start_date, end_date, output_dir)
    )

    write_html_report(
        df_stock,
        strategy_results,
        comparison_rows,
        stock,
        strategy_names,
        output_path,
    )
    return {
        'stock': stock,
        'csv_path': csv_path,
        'start_date': start_date,
        'end_date': end_date,
        'output_path': output_path,
        'strategy_results': strategy_results,
        'comparison_rows': comparison_rows,
    }


def get_batch_run_dir(args):
    if args.run_dir:
        return Path(args.run_dir)
    if args.output:
        return Path(args.output)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return PLOT_DIR / f'backtest_run_{timestamp}'


def add_summary_metric_columns(row, strategy_name, metrics):
    prefix = strategy_name
    row[f'{prefix}_final_equity'] = metrics['final_equity']
    row[f'{prefix}_final_profit'] = metrics['final_profit']
    row[f'{prefix}_profit_pct'] = metrics['profit_pct']
    row[f'{prefix}_max_drawdown'] = metrics['max_drawdown']
    row[f'{prefix}_total_trades'] = metrics['total_trades']
    row[f'{prefix}_transaction_cost'] = metrics['transaction_cost']
    row[f'{prefix}_win_rate'] = metrics['win_rate']
    row[f'{prefix}_profit_factor'] = metrics['profit_factor']
    row[f'{prefix}_sharpe_ratio'] = metrics['sharpe_ratio']
    row[f'{prefix}_ending_cash'] = metrics['ending_cash']
    row[f'{prefix}_ending_shares'] = metrics['ending_shares']


def build_batch_summary_row(result, stock_name_by_code):
    row = {
        'stock': result['stock'],
        'name': stock_name_by_code.get(result['stock'], ''),
    }

    for strategy_result in result['strategy_results']:
        strategy_name = strategy_result['name']
        metrics = strategy_result['metrics']
        add_summary_metric_columns(row, strategy_name, metrics)
        if strategy_name == 'buy_and_hold':
            row['buy_and_hold_profit_pct'] = metrics['profit_pct']

    return row


def build_failed_summary_row(stock, stock_name_by_code):
    return {
        'stock': stock,
        'name': stock_name_by_code.get(stock, ''),
    }


def get_summary_sort_column(strategy_names, summary_df):
    for strategy_name in strategy_names:
        if strategy_name != 'buy_and_hold':
            column = f'{strategy_name}_profit_pct'
            if column in summary_df.columns:
                return column

    if 'buy_and_hold_profit_pct' in summary_df.columns:
        return 'buy_and_hold_profit_pct'

    profit_columns = [
        column for column in summary_df.columns
        if column.endswith('_profit_pct')
    ]
    return profit_columns[0] if profit_columns else None


def write_batch_summary(rows, run_dir, strategy_names):
    summary_path = run_dir / 'strategy_performance.csv'
    summary_df = pd.DataFrame(rows)
    sort_column = get_summary_sort_column(strategy_names, summary_df)
    if sort_column:
        summary_df = summary_df.sort_values(
            by=sort_column,
            ascending=False,
            na_position='last',
        )
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    return summary_path


def run_all_stocks(args, strategy_names):
    stock_codes = get_all_cached_stock_codes()
    run_dir = get_batch_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    stock_name_by_code = load_stock_name_map()
    rows = []

    print(f'Running {len(stock_codes)} stocks.')
    print(f'Run directory: {run_dir}')
    for index, stock in enumerate(stock_codes, start=1):
        try:
            print(f'[{index}/{len(stock_codes)}] {stock}')
            result = run_backtest_for_stock(stock, args, strategy_names, run_dir)
            rows.append(build_batch_summary_row(result, stock_name_by_code))
        except Exception as exc:
            rows.append(build_failed_summary_row(stock, stock_name_by_code))
            print(f'Error for {stock}: {exc}')

    summary_path = write_batch_summary(rows, run_dir, strategy_names)
    print(f'Summary CSV: {summary_path}')
    return run_dir, summary_path


def main():
    args = parse_args()
    if args.fund <= 0:
        raise ValueError('--fund must be greater than 0.')

    strategy_names = parse_strategy_names(args)

    if args.all:
        run_all_stocks(args, strategy_names)
        return

    result = run_backtest_for_stock(args.stock, args, strategy_names)
    print(f"Data source: {result['csv_path']}")
    print_strategy_results(result['strategy_results'], result['output_path'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}')
        raise SystemExit(1)
