'''
stock_viz.py

Interactive stock CSV visualization.
'''
import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from column_schema import read_csv_canonical
from plotly.subplots import make_subplots


PLOT_DIR = './data_viz/price_charts'
METADATA_PATH = './data/metadata.csv'
REQUIRED_COLUMNS = [
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Capacity',
]
MOVING_AVERAGES = [
    (5, '#f59e0b'),
    (10, '#2563eb'),
    (20, '#7c3aed'),
    (60, '#111827'),
]


def get_stock_code_from_csv_path(csv_path):
    '''
    Extract stock code from filenames like 2308_台達電.csv.
    '''
    return Path(csv_path).stem.split('_')[0]


def read_stock_metadata(metadata_path=METADATA_PATH):
    '''
    Read stock metadata when available.
    '''
    if not os.path.exists(metadata_path):
        return pd.DataFrame()

    return read_csv_canonical(metadata_path, dtype={'Code': str})


def get_stock_title(csv_path, metadata_path=METADATA_PATH):
    '''
    Build a chart title that includes stock code, Chinese name, and group.
    '''
    stock_code = get_stock_code_from_csv_path(csv_path)
    title_parts = [stock_code]
    metadata_df = read_stock_metadata(metadata_path)

    if not metadata_df.empty and 'Code' in metadata_df.columns:
        matched_rows = metadata_df[metadata_df['Code'] == stock_code]
        if not matched_rows.empty:
            stock_info = matched_rows.iloc[0]
            for column in ['Name', 'Group']:
                value = stock_info.get(column)
                if pd.notna(value) and value:
                    title_parts.append(str(value))

    return ' - '.join(title_parts)


def read_stock_csv(csv_path):
    '''
    Read a stock price CSV and parse the Date column.
    '''
    return read_csv_canonical(csv_path, parse_dates=['Date'])


def validate_stock_data(df_stock):
    '''
    Validate that the CSV has the columns needed for interactive plotting.
    '''
    missing_cols = [
        column for column in REQUIRED_COLUMNS
        if column not in df_stock.columns
    ]
    if missing_cols:
        raise ValueError(f'CSV is missing required columns: {missing_cols}')


def clean_stock_data(df_stock):
    '''
    Keep only rows with complete stock price data.
    '''
    validate_stock_data(df_stock)
    df_stock = df_stock.copy()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

    for column in ['Open', 'High', 'Low', 'Close', 'Capacity']:
        df_stock[column] = pd.to_numeric(df_stock[column], errors='coerce')

    return (
        df_stock
        .dropna(subset=REQUIRED_COLUMNS)
        .sort_values('Date')
        .reset_index(drop=True)
    )


def get_missing_trading_dates(df_stock):
    '''
    Return weekday dates that are absent from the CSV, such as holidays.
    '''
    if df_stock.empty:
        return []

    date_values = df_stock['Date'].dt.normalize()
    all_weekdays = pd.date_range(
        start=date_values.min(),
        end=date_values.max(),
        freq='B',
    )
    missing_dates = all_weekdays.difference(date_values)
    return [date.strftime('%Y-%m-%d') for date in missing_dates]


def get_axis_range(low, high, padding_ratio=0.05, floor_zero=False):
    '''
    Return a padded axis range.
    '''
    if pd.isna(low) or pd.isna(high):
        return None

    if low == high:
        padding = abs(high) * padding_ratio or 1
    else:
        padding = (high - low) * padding_ratio

    axis_low = low - padding
    axis_high = high + padding

    if floor_zero:
        axis_low = max(0, axis_low)

    return [axis_low, axis_high]


def get_autorange_script(df_stock):
    '''
    Return JavaScript that rescales y-axes to the currently visible x window.
    '''
    records = [
        {
            'date': row.Date.isoformat(),
            'low': float(row.Low),
            'high': float(row.High),
            'capacity': float(row.Capacity),
        }
        for row in df_stock.itertuples(index=False)
    ]

    return f'''
const graphDiv = document.getElementById('{{plot_id}}');
const stockRecords = {records!r};
const fullPriceLow = Math.min(...stockRecords.map(row => row.low));
const fullPriceHigh = Math.max(...stockRecords.map(row => row.high));
const fullCapacityHigh = Math.max(...stockRecords.map(row => row.capacity));
let isUpdatingAxes = false;

function paddedRange(minValue, maxValue, floorZero) {{
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {{
        return null;
    }}

    const spread = maxValue - minValue;
    const padding = spread === 0 ? Math.max(Math.abs(maxValue) * 0.05, 1) : spread * 0.05;
    let rangeMin = minValue - padding;
    const rangeMax = maxValue + padding;

    if (floorZero) {{
        rangeMin = Math.max(0, rangeMin);
    }}

    return [rangeMin, rangeMax];
}}

function asDate(value) {{
    return value instanceof Date ? value : new Date(value);
}}

function visibleRecords(xRange) {{
    if (!xRange || xRange.length < 2) {{
        return stockRecords;
    }}

    const start = asDate(xRange[0]);
    const end = asDate(xRange[1]);

    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {{
        return stockRecords;
    }}

    return stockRecords.filter(row => {{
        const rowDate = asDate(row.date);
        return rowDate >= start && rowDate <= end;
    }});
}}

function rescaleVisibleAxes(xRange) {{
    const rows = visibleRecords(xRange);
    if (rows.length === 0) {{
        return;
    }}

    const priceLow = Math.min(...rows.map(row => row.low));
    const priceHigh = Math.max(...rows.map(row => row.high));
    const capacityHigh = Math.max(...rows.map(row => row.capacity));

    const priceRange = paddedRange(priceLow, priceHigh, false);
    const capacityRange = paddedRange(0, capacityHigh, true);

    if (!priceRange || !capacityRange) {{
        return;
    }}

    isUpdatingAxes = true;
    Plotly.relayout(graphDiv, {{
        'yaxis.range': priceRange,
        'yaxis2.range': capacityRange
    }}).then(() => {{
        isUpdatingAxes = false;
    }});
}}

graphDiv.on('plotly_relayout', eventData => {{
    if (isUpdatingAxes) {{
        return;
    }}

    if (eventData['xaxis.range']) {{
        rescaleVisibleAxes(eventData['xaxis.range']);
        return;
    }}

    if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {{
        rescaleVisibleAxes([
            eventData['xaxis.range[0]'],
            eventData['xaxis.range[1]']
        ]);
        return;
    }}

    if (eventData['xaxis2.range']) {{
        rescaleVisibleAxes(eventData['xaxis2.range']);
        return;
    }}

    if (eventData['xaxis2.range[0]'] && eventData['xaxis2.range[1]']) {{
        rescaleVisibleAxes([
            eventData['xaxis2.range[0]'],
            eventData['xaxis2.range[1]']
        ]);
        return;
    }}

    if (eventData['xaxis.autorange'] || eventData['xaxis2.autorange']) {{
        isUpdatingAxes = true;
        Plotly.relayout(graphDiv, {{
            'yaxis.range': paddedRange(fullPriceLow, fullPriceHigh, false),
            'yaxis2.range': paddedRange(0, fullCapacityHigh, true)
        }}).then(() => {{
            isUpdatingAxes = false;
        }});
    }}
}});
'''


def get_wick_points(df_stock, direction):
    '''
    Return x/y points for vertical candle wick segments.
    '''
    wick_x = []
    wick_y = []

    if direction == 'rising':
        rows = df_stock[df_stock['Close'] > df_stock['Open']]
    elif direction == 'falling':
        rows = df_stock[df_stock['Close'] < df_stock['Open']]
    else:
        rows = df_stock[df_stock['Close'] == df_stock['Open']]

    for row in rows.itertuples(index=False):
        wick_x.extend([row.Date, row.Date, None])
        wick_y.extend([row.Low, row.High, None])

    return wick_x, wick_y


def get_doji_points(df_stock):
    '''
    Return x/y points for horizontal zero-change candle bodies.
    '''
    doji_x = []
    doji_y = []
    doji_df = df_stock[df_stock['Close'] == df_stock['Open']]

    if doji_df.empty:
        return doji_x, doji_y

    date_step = df_stock['Date'].diff().dropna().median()
    if pd.isna(date_step):
        date_step = pd.Timedelta(days=1)

    half_width = date_step * 0.28

    for row in doji_df.itertuples(index=False):
        doji_x.extend([row.Date - half_width, row.Date + half_width, None])
        doji_y.extend([row.Close, row.Close, None])

    return doji_x, doji_y


def add_moving_average_traces(fig, df_stock, row=1, col=1):
    '''
    Add moving average lines to the price chart.
    '''
    for window, color in MOVING_AVERAGES:
        ma_column = f'MA{window}'
        df_stock[ma_column] = df_stock['Close'].rolling(
            window=window,
            min_periods=window,
        ).mean()

        fig.add_trace(
            go.Scatter(
                x=df_stock['Date'],
                y=df_stock[ma_column],
                mode='lines',
                name=f'{window} day MA',
                line=dict(color=color, width=1.8),
                connectgaps=False,
                hovertemplate=(
                    'Date=%{x}<br>'
                    f'{window} day MA='
                    '%{y:.2f}<extra></extra>'
                ),
            ),
            row=row,
            col=col,
        )


def get_hidden_date_breaks(df_stock):
    '''
    Return range breaks for weekends and missing trading dates.
    '''
    return [
        dict(bounds=['sat', 'mon']),
        dict(values=get_missing_trading_dates(df_stock)),
    ]


def add_stock_price_traces(
    fig,
    df_stock,
    row=1,
    col=1,
    include_moving_averages=True,
):
    '''
    Add solid candle price traces to a caller-provided subplot.
    '''
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
        col=col,
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
        col=col,
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
        col=col,
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
        col=col,
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
        col=col,
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
        col=col,
    )

    if include_moving_averages:
        add_moving_average_traces(fig, df_stock, row=row, col=col)


def add_capacity_trace(fig, df_stock, row=1, col=1):
    '''
    Add capacity bars to a caller-provided subplot.
    '''
    fig.add_trace(
        go.Bar(
            x=df_stock['Date'],
            y=df_stock['Capacity'],
            name='Capacity',
            marker_color='#4c78a8',
        ),
        row=row,
        col=col,
    )


def build_stock_figure(df_stock, title):
    '''
    Build an interactive candlestick and capacity bar chart.
    '''
    df_stock = clean_stock_data(df_stock)
    hidden_date_breaks = get_hidden_date_breaks(df_stock)
    price_range = get_axis_range(df_stock['Low'].min(), df_stock['High'].max())
    capacity_range = get_axis_range(0, df_stock['Capacity'].max(), floor_zero=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.70, 0.30],
    )

    add_stock_price_traces(fig, df_stock, row=1, col=1)
    add_capacity_trace(fig, df_stock, row=2, col=1)

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        dragmode='zoom',
        bargap=0.18,
        height=780,
        margin=dict(l=60, r=32, t=64, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.07),
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
        yaxis_title='Price',
        yaxis2_title='Capacity',
        yaxis=dict(range=price_range),
        yaxis2=dict(range=capacity_range),
    )
    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor')
    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor')

    return fig, df_stock


def default_output_path(csv_path):
    '''
    Return the default HTML output path for a CSV.
    '''
    os.makedirs(PLOT_DIR, exist_ok=True)
    return f'{PLOT_DIR}/{Path(csv_path).stem}.html'


def write_stock_figure(fig, output_path, post_script=None):
    '''
    Write an interactive Plotly figure to HTML.
    '''
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    config = {
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape',
        ],
    }
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        config=config,
        post_script=post_script,
    )


def visualize_stock_csv(csv_path, output_path=None):
    '''
    Build and write an interactive stock visualization for a CSV file.
    '''
    if output_path is None:
        output_path = default_output_path(csv_path)

    # Read csv
    df_stock = read_stock_csv(csv_path)

    # Build stock
    fig, df_stock = build_stock_figure(df_stock, get_stock_title(csv_path))

    # Output Figure to HTML file 
    write_stock_figure(fig, output_path, get_autorange_script(df_stock))
    print(f'Interactive stock chart saved to {output_path}.')

    return output_path


def parse_args():
    '''
    Parse command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Visualize a stock price CSV as an interactive HTML chart.'
    )
    parser.add_argument('csv_path', help='Path to the stock CSV file.')
    parser.add_argument(
        '--output',
        help='Output HTML path. Defaults to data_viz/price_charts/<csv_stem>.html.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_stock_csv(args.csv_path, args.output)
