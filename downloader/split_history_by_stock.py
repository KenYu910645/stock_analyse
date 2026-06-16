'''
Split a historical stock dataset into one CSV per stock.

The project stores several long-form TWSE history files as one row per
date/code.  This helper writes Code_Name.csv files under a by_stock directory.
'''
import argparse
import os
import re

import pandas as pd


WINDOWS_INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split a Date/Code stock-history CSV into per-stock files.'
    )
    parser.add_argument('--input', required=True, help='Input long-form CSV path.')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory that will receive per-stock CSV files.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete existing CSVs in output-dir before writing.',
    )
    return parser.parse_args()


def sanitize_filename_part(value):
    text = str(value).strip()
    text = re.sub(WINDOWS_INVALID_FILENAME_CHARS, '_', text)
    text = re.sub(r'\s+', '_', text)
    text = text.strip(' ._')
    return text or 'unknown'


def ensure_output_dir(output_dir, force=False):
    os.makedirs(output_dir, exist_ok=True)
    existing_csvs = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith('.csv')
    ]
    if existing_csvs and not force:
        raise FileExistsError(
            f'Output directory already has {len(existing_csvs)} CSV files: '
            f'{output_dir}. Use --force to overwrite.'
        )
    if force:
        for path in existing_csvs:
            os.remove(path)


def validate_columns(df):
    required = ['Date', 'Code']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f'Input CSV is missing required columns: {missing}')


def normalize_date_column(series):
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format='%Y%m%d', errors='coerce')
    fallback_mask = parsed.isna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            text.loc[fallback_mask],
            errors='coerce',
        )
    return parsed


def output_path_for_stock(output_dir, code, name):
    safe_code = sanitize_filename_part(code)
    safe_name = sanitize_filename_part(name)
    if safe_name and safe_name != 'unknown':
        filename = f'{safe_code}_{safe_name}.csv'
    else:
        filename = f'{safe_code}.csv'
    return os.path.join(output_dir, filename)


def split_history_by_stock(input_path, output_dir, force=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input CSV does not exist: {input_path}')

    ensure_output_dir(output_dir, force=force)
    df = pd.read_csv(input_path, dtype={'Code': str}, keep_default_na=False)
    validate_columns(df)

    df['Code'] = df['Code'].astype(str).str.strip()
    if 'Date' in df.columns:
        df['Date'] = normalize_date_column(df['Date'])

    sort_columns = ['Code']
    if 'Date' in df.columns:
        sort_columns.append('Date')
    df = df.sort_values(sort_columns).reset_index(drop=True)

    files_written = 0
    for code, stock_df in df.groupby('Code', sort=True):
        stock_df = stock_df.copy()
        if 'Date' in stock_df.columns:
            stock_df = stock_df.sort_values('Date')
            stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')

        name = ''
        if 'Name' in stock_df.columns:
            names = stock_df['Name'].astype(str).str.strip()
            names = names[names != '']
            if not names.empty:
                name = names.iloc[-1]

        output_path = output_path_for_stock(output_dir, code, name)
        stock_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        files_written += 1

    return {
        'input_path': input_path,
        'output_dir': output_dir,
        'rows_read': len(df),
        'unique_stocks': df['Code'].nunique(),
        'files_written': files_written,
    }


def main():
    args = parse_args()
    summary = split_history_by_stock(
        args.input,
        args.output_dir,
        force=args.force,
    )
    print('Split summary:')
    for key, value in summary.items():
        print(f'{key}={value}')


if __name__ == '__main__':
    main()
