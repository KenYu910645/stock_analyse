'''
build_forward_adjusted_prices.py

Create forward-adjusted price CSVs from data/price into data/adj_price.
'''
import argparse
import glob
import os
import re

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRICE_DIR = os.path.join(PROJECT_ROOT, 'data', 'price')
ADJ_PRICE_DIR = os.path.join(PROJECT_ROOT, 'data', 'adj_price')
DIVIDEND_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'dividend',
    'twse_ex_right_dividend_calculation_20030505_to_20260605.csv',
)

PRICE_COLUMNS = ['Open', 'Close', 'Low', 'High']
ADJUSTED_COLUMNS = {
    'Open': 'Open_adj',
    'Close': 'Close_adj',
    'Low': 'Low_adj',
    'High': 'High_adj',
}


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Build forward-adjusted OHLC prices from TWSE factors.'
    )
    parser.add_argument(
        '--price-dir',
        default=PRICE_DIR,
        help='Input price CSV directory.',
    )
    parser.add_argument(
        '--output-dir',
        default=ADJ_PRICE_DIR,
        help='Output adjusted price CSV directory.',
    )
    parser.add_argument(
        '--dividend-path',
        default=DIVIDEND_PATH,
        help='TWSE ex-right/ex-dividend calculation CSV.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing adjusted CSVs.',
    )
    return parser.parse_args()


def extract_stock_id(path):
    '''
    Extract the stock id prefix from a price CSV filename.
    '''
    filename = os.path.basename(path)
    match = re.match(r'^([^_]+)_', filename)
    if not match:
        raise ValueError(f'Cannot extract stock id from filename: {filename}')

    return match.group(1)


def load_adjustment_events(dividend_path):
    '''
    Load valid per-stock forward-adjustment factors from TWSE calculation data.
    '''
    df = pd.read_csv(dividend_path, dtype={'stock_id': str})
    df['ex_date'] = pd.to_datetime(df['ex_date'], errors='coerce')
    df['previous_close'] = pd.to_numeric(df['previous_close'], errors='coerce')
    df['ex_reference_price'] = pd.to_numeric(
        df['ex_reference_price'],
        errors='coerce',
    )
    df['adjustment_factor'] = df['ex_reference_price'] / df['previous_close']

    df = df[
        df['ex_date'].notna()
        & df['stock_id'].notna()
        & df['previous_close'].gt(0)
        & df['ex_reference_price'].gt(0)
        & df['adjustment_factor'].gt(0)
    ].copy()

    return {
        stock_id: events.sort_values('ex_date')[['ex_date', 'adjustment_factor']]
        for stock_id, events in df.groupby('stock_id')
    }


def add_forward_adjusted_columns(price_df, events):
    '''
    Add forward-adjusted OHLC columns to a price dataframe.

    For an ex-date D, TWSE gives factor = ex_reference_price / previous_close.
    Forward adjustment applies that factor to every price date before D.
    '''
    df = price_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['AdjFactor'] = 1.0

    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    if events is not None and not events.empty:
        for _, event in events.iterrows():
            df.loc[df['Date'] < event['ex_date'], 'AdjFactor'] *= event[
                'adjustment_factor'
            ]

    for source_column, adjusted_column in ADJUSTED_COLUMNS.items():
        df[adjusted_column] = (df[source_column] * df['AdjFactor']).round(4)

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def convert_file(path, output_dir, events_by_stock, force=False):
    '''
    Convert one price CSV and return summary details.
    '''
    stock_id = extract_stock_id(path)
    output_path = os.path.join(output_dir, os.path.basename(path))
    if os.path.exists(output_path) and not force:
        return stock_id, output_path, 0, 'exists'

    price_df = pd.read_csv(path)
    adjusted_df = add_forward_adjusted_columns(
        price_df,
        events_by_stock.get(stock_id),
    )
    adjusted_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return stock_id, output_path, len(adjusted_df), 'saved'


def main():
    '''
    Convert every CSV in data/price to data/adj_price.
    '''
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    events_by_stock = load_adjustment_events(args.dividend_path)
    price_paths = sorted(glob.glob(os.path.join(args.price_dir, '*.csv')))
    if not price_paths:
        raise ValueError(f'No price CSV files found in {args.price_dir}')

    saved = 0
    skipped = 0
    for index, path in enumerate(price_paths, start=1):
        stock_id, output_path, row_count, status = convert_file(
            path,
            args.output_dir,
            events_by_stock,
            force=args.force,
        )
        if status == 'saved':
            saved += 1
            print(f'[{index}/{len(price_paths)}] saved {stock_id}: {row_count} rows')
        else:
            skipped += 1
            print(f'[{index}/{len(price_paths)}] skipped {stock_id}: {output_path}')

    print('Adjustment summary:')
    print(f'input_files={len(price_paths)}')
    print(f'saved_files={saved}')
    print(f'skipped_files={skipped}')
    print(f'output_dir={args.output_dir}')


if __name__ == '__main__':
    main()
