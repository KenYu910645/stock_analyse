'''
splitStockBasic.py

Split the TWSE historical basic valuation CSV into one CSV per stock.
'''
import argparse
import os
import re

import pandas as pd


DEFAULT_INPUT_PATH = './data/twse_basic_valuation_history_20200101_to_20260514.csv'
DEFAULT_OUTPUT_DIR = './data/dividend_pe_pb'

EXPECTED_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'Close',
    'DividendYield',
    'DividendYear',
    'PEratio',
    'PBratio',
    'FiscalYearQuarter',
]

WINDOWS_INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Split TWSE basic valuation history into per-stock CSV files.'
    )
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT_PATH,
        help=f'Input historical valuation CSV. Default: {DEFAULT_INPUT_PATH}',
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output folder for per-stock CSV files. Default: {DEFAULT_OUTPUT_DIR}',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files in the output folder.',
    )
    return parser.parse_args()


def sanitize_filename_part(value):
    '''
    Return text that is safe to use in a Windows filename.
    '''
    text = str(value).strip()
    text = re.sub(WINDOWS_INVALID_FILENAME_CHARS, '_', text)
    text = re.sub(r'\s+', '_', text)
    text = text.strip(' ._')
    return text or 'unknown'


def validate_input_columns(df):
    '''
    Ensure the input CSV has the expected historical valuation columns.
    '''
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Input CSV is missing columns: {missing_columns}')


def ensure_output_dir(output_dir, force):
    '''
    Create or validate the output directory.
    '''
    if os.path.isdir(output_dir):
        existing_csvs = [
            name for name in os.listdir(output_dir)
            if name.lower().endswith('.csv')
        ]
        if existing_csvs and not force:
            raise FileExistsError(
                f'Output folder already contains {len(existing_csvs)} CSV files: '
                f'{output_dir}. Use --force to overwrite.'
            )
    else:
        os.makedirs(output_dir, exist_ok=True)


def get_stock_output_path(output_dir, code, name):
    '''
    Return a readable and safe per-stock CSV path.
    '''
    safe_code = sanitize_filename_part(code)
    safe_name = sanitize_filename_part(name)
    return os.path.join(output_dir, f'{safe_code}_{safe_name}.csv')


def split_stock_basic(input_path, output_dir, force=False):
    '''
    Split one historical valuation CSV into one CSV per stock code.
    '''
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input CSV does not exist: {input_path}')

    ensure_output_dir(output_dir, force)

    df = pd.read_csv(input_path, dtype={'Code': str})
    validate_input_columns(df)

    df = df[EXPECTED_COLUMNS].copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Code'] = df['Code'].astype(str)
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)

    files_written = 0

    for code, stock_df in df.groupby('Code', sort=True):
        stock_df = stock_df.sort_values('Date').copy()
        stock_name = stock_df['Name'].dropna().iloc[-1]
        output_path = get_stock_output_path(output_dir, code, stock_name)
        stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')
        stock_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        files_written += 1

    return {
        'rows_read': len(df),
        'unique_stocks': df['Code'].nunique(),
        'files_written': files_written,
        'output_dir': output_dir,
    }


def main():
    '''
    Split TWSE basic valuation history into per-stock CSV files.
    '''
    args = parse_args()
    summary = split_stock_basic(args.input, args.output_dir, force=args.force)

    print('Split summary:')
    print(f"rows_read={summary['rows_read']}")
    print(f"unique_stocks={summary['unique_stocks']}")
    print(f"files_written={summary['files_written']}")
    print(f"output_dir={summary['output_dir']}")


if __name__ == '__main__':
    main()
