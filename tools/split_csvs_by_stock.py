'''
Split stock-keyed CSV datasets into per-stock CSV files.

This helper is for data folders whose source files are long-form snapshots or
histories.  It recognizes the stock-code column names used by TWSE/OpenAPI and
broker datasets, filters rows to codes in data/metadata.csv by default, and
writes one file per stock under a dataset-specific output directory.
'''
import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical, to_csv_storage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DEFAULT_METADATA_PATH = DATA_DIR / 'metadata.csv'

CODE_COLUMN_ALIASES = ('Code', 'stock_id', 'StockNo', '公司代號', '證券代號')
NAME_COLUMN_ALIASES = ('Name', 'stock_name', 'StockName', '公司簡稱', '公司名稱', '證券名稱')
WINDOWS_INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split one or more stock-keyed CSV files into per-stock CSVs.'
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Input CSV path(s).',
    )
    parser.add_argument(
        '--output-root',
        required=True,
        help='Root output directory. Each input gets a subdirectory by file stem.',
    )
    parser.add_argument(
        '--metadata',
        default=str(DEFAULT_METADATA_PATH),
        help=f'Metadata CSV used to filter TWSE listed stock codes. Default: {DEFAULT_METADATA_PATH}.',
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=100000,
        help='Rows per pandas chunk. Default: 100000.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete an input file stem output directory before writing.',
    )
    return parser.parse_args()


def sanitize_filename_part(value):
    text = str(value).strip()
    text = re.sub(WINDOWS_INVALID_FILENAME_CHARS, '_', text)
    text = re.sub(r'\s+', '_', text)
    return text.strip(' ._') or 'unknown'


def load_listed_codes(metadata_path):
    df = read_csv_canonical(metadata_path, dtype=str).fillna('')
    df['Code'] = df['Code'].astype(str).str.strip()
    mask = (
        (df['Market'] == '上市')
        & (df['Type'] == '股票')
        & df['Code'].str.match(r'^\d{4}$')
    )
    return set(df.loc[mask, 'Code'])


def detect_column(columns, aliases):
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def output_path_for_stock(output_dir, code, name):
    safe_code = sanitize_filename_part(code)
    safe_name = sanitize_filename_part(name)
    if safe_name and safe_name != 'unknown':
        return output_dir / f'{safe_code}_{safe_name}.csv'
    return output_dir / f'{safe_code}.csv'


def prepare_output_dir(output_dir, force):
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def split_one_csv(input_path, output_root, listed_codes, chunksize, force):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV does not exist: {input_path}')

    output_dir = Path(output_root) / input_path.stem
    prepare_output_dir(output_dir, force)

    header = pd.read_csv(input_path, nrows=0, dtype=str).columns.tolist()
    code_column = detect_column(header, CODE_COLUMN_ALIASES)
    name_column = detect_column(header, NAME_COLUMN_ALIASES)
    if not code_column:
        return {
            'input': str(input_path),
            'output_dir': str(output_dir),
            'status': 'skipped_no_code_column',
            'rows_read': 0,
            'rows_written': 0,
            'files_written': 0,
        }

    rows_read = 0
    rows_written = 0
    seen_paths = set()
    for chunk in pd.read_csv(input_path, dtype=str, keep_default_na=False, chunksize=chunksize):
        rows_read += len(chunk)
        chunk[code_column] = chunk[code_column].astype(str).str.strip()
        chunk = chunk[chunk[code_column].isin(listed_codes)].copy()
        if chunk.empty:
            continue

        for code, stock_df in chunk.groupby(code_column, sort=True):
            name = ''
            if name_column:
                names = stock_df[name_column].astype(str).str.strip()
                names = names[names != '']
                if not names.empty:
                    name = names.iloc[-1]

            path = output_path_for_stock(output_dir, code, name)
            write_header = path not in seen_paths and not path.exists()
            to_csv_storage(
                stock_df,
                path,
                mode='a',
                header=write_header,
                index=False,
                encoding='utf-8-sig',
            )
            seen_paths.add(path)
            rows_written += len(stock_df)

    return {
        'input': str(input_path),
        'output_dir': str(output_dir),
        'status': 'written',
        'rows_read': rows_read,
        'rows_written': rows_written,
        'files_written': len(seen_paths),
    }


def main():
    args = parse_args()
    listed_codes = load_listed_codes(args.metadata)
    for input_path in args.inputs:
        summary = split_one_csv(
            input_path,
            args.output_root,
            listed_codes,
            args.chunksize,
            args.force,
        )
        print(
            ' | '.join(
                f'{key}={value}'
                for key, value in summary.items()
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
