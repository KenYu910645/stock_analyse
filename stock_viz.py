'''
stock_viz.py

Interactive stock CSV visualization.
'''
import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOT_DIR = './plot'
REQUIRED_COLUMNS = [
    'Date',
    'Open',
    'High',
    'Low',
    'Close',
    'Capacity',
]


def read_stock_csv(csv_path):
    '''
    Read a stock price CSV and parse the Date column.
    '''
    return pd.read_csv(csv_path, parse_dates=['Date'])


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


def build_stock_figure(df_stock, title):
    '''
    Build an interactive candlestick and capacity bar chart.
    '''
    df_stock = clean_stock_data(df_stock)
    missing_dates = get_missing_trading_dates(df_stock)
    price_range = get_axis_range(df_stock['Low'].min(), df_stock['High'].max())
    capacity_range = get_axis_range(0, df_stock['Capacity'].max(), floor_zero=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=df_stock['Date'],
            open=df_stock['Open'],
            high=df_stock['High'],
            low=df_stock['Low'],
            close=df_stock['Close'],
            name='Price',
            increasing_line_color='#d62728',
            decreasing_line_color='#2ca02c',
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df_stock['Date'],
            y=df_stock['Capacity'],
            name='Capacity',
            marker_color='#4c78a8',
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=title,
        template='plotly_white',
        hovermode='x unified',
        dragmode='zoom',
        height=780,
        margin=dict(l=60, r=32, t=64, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangebreaks=[
                dict(bounds=['sat', 'mon']),
                dict(values=missing_dates),
            ],
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
    fig, df_stock = build_stock_figure(df_stock, Path(csv_path).stem)

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
        help='Output HTML path. Defaults to plot/<csv_stem>.html.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_stock_csv(args.csv_path, args.output)
