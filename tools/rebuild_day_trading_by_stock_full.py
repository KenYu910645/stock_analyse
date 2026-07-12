'''
Rebuild the canonical per-stock TWSE day-trading dataset.

The input is the long-form TWSE TWTB4U history CSV.  The output is one
UTF-8-BOM CSV per listed common stock under data/day_trading/by_stock_full/,
with Date normalized to ISO YYYY-MM-DD.
'''
import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical, to_csv_storage
from downloader.update_all_data import (
    DAY_TRADING_OUTPUT_COLUMNS,
    refresh_day_trading_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
DAY_TRADING_DIR = DATA_DIR / 'day_trading'
DEFAULT_SOURCE = DAY_TRADING_DIR / 'twse_day_trading_history_20140106_to_20260614.csv'
DEFAULT_METADATA = DATA_DIR / 'metadata.csv'
DEFAULT_TRADING_DAYS = DATA_DIR / 'trading_days.csv'
DEFAULT_OUTPUT_DIR = DAY_TRADING_DIR
DEFAULT_MANIFEST = DAY_TRADING_DIR / 'manifest.json'
DEFAULT_MISSING_DATES = DAY_TRADING_DIR / 'missing_dates.csv'
DEFAULT_SKIPPED_CODES = DAY_TRADING_DIR / 'skipped_codes.csv'
DEFAULT_LOGS = DAY_TRADING_DIR / 'day_trading.logs'

DAY_TRADING_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'SuspensionNote',
    'DayTradingVolume',
    'DayTradingBuyAmount',
    'DayTradingSellAmount',
]
WINDOWS_INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rebuild data/day_trading/by_stock_full from TWSE history.'
    )
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--metadata', default=str(DEFAULT_METADATA))
    parser.add_argument('--trading-days', default=str(DEFAULT_TRADING_DAYS))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--missing-dates', default=str(DEFAULT_MISSING_DATES))
    parser.add_argument('--skipped-codes', default=str(DEFAULT_SKIPPED_CODES))
    parser.add_argument('--logs', default=str(DEFAULT_LOGS))
    parser.add_argument('--chunksize', type=int, default=100000)
    return parser.parse_args()


def sanitize_filename_part(value):
    text = str(value).strip()
    text = re.sub(WINDOWS_INVALID_FILENAME_CHARS, '_', text)
    text = re.sub(r'\s+', '_', text)
    return text.strip(' ._') or 'unknown'


def normalize_date(value):
    text = str(value).strip()
    if re.fullmatch(r'\d{8}', text):
        return datetime.strptime(text, '%Y%m%d').date().isoformat()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        return datetime.strptime(text, '%Y-%m-%d').date().isoformat()
    return ''


def load_listed_common_metadata(path):
    df = read_csv_canonical(path, dtype=str, keep_default_na=False)
    required = {'Code', 'Name', 'Type', 'Market'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Metadata missing required columns: {sorted(missing)}')

    df['Code'] = df['Code'].astype(str).str.strip()
    mask = (
        df['Market'].eq('上市')
        & df['Type'].eq('股票')
        & df['Code'].str.match(r'^\d{4}$')
    )
    listed = df.loc[mask, ['Code', 'Name']].copy()
    return {
        row.Code: str(row.Name).strip()
        for row in listed.itertuples(index=False)
    }


def load_trading_days(path, start_date, end_date):
    if not path.exists():
        return []

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return []

    date_column = 'Date' if 'Date' in df.columns else df.columns[0]
    dates = []
    for value in df[date_column]:
        normalized = normalize_date(value)
        if start_date <= normalized <= end_date:
            dates.append(normalized)
    return sorted(set(dates))


def prepare_output_dir(path):
    resolved = path.resolve()
    allowed_parent = DAY_TRADING_DIR.resolve()
    if allowed_parent not in resolved.parents:
        raise ValueError(f'Refusing to clear output outside data/day_trading: {path}')
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def output_path_for_stock(output_dir, code, name):
    safe_code = sanitize_filename_part(code)
    safe_name = sanitize_filename_part(name)
    if safe_name and safe_name != 'unknown':
        return output_dir / f'{safe_code}_{safe_name}.csv'
    return output_dir / f'{safe_code}.csv'


def write_missing_dates(path, missing_dates):
    with path.open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['Date', 'Reason'])
        writer.writeheader()
        for date_text in missing_dates:
            writer.writerow({
                'Date': date_text,
                'Reason': 'no_rows_in_source_history',
            })


def write_skipped_codes(path, skipped_code_counts, skipped_code_names):
    with path.open('w', encoding='utf-8-sig', newline='') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['Code', 'Name', 'Rows', 'Reason'])
        writer.writeheader()
        for code, rows in sorted(skipped_code_counts.items()):
            writer.writerow({
                'Code': code,
                'Name': skipped_code_names.get(code, ''),
                'Rows': rows,
                'Reason': 'not_twse_listed_common_stock_in_metadata',
            })


def write_logs(path, manifest_path, missing_dates_path, skipped_codes_path):
    with path.open('w', encoding='utf-8-sig', newline='') as output:
        output.write('# manifest.json\n')
        output.write(manifest_path.read_text(encoding='utf-8'))
        output.write('\n\n# missing_dates.csv\n')
        output.write(missing_dates_path.read_text(encoding='utf-8-sig'))
        output.write('\n\n# skipped_codes.csv\n')
        output.write(skipped_codes_path.read_text(encoding='utf-8-sig'))


def rebuild(
    source,
    metadata,
    trading_days_path,
    output_dir,
    manifest_path,
    missing_dates_path,
    skipped_codes_path,
    logs_path,
    chunksize,
):
    if not source.exists():
        raise FileNotFoundError(f'Source CSV not found: {source}')

    code_to_name = load_listed_common_metadata(metadata)
    listed_codes = set(code_to_name)
    prepare_output_dir(output_dir)

    header = read_csv_canonical(source, nrows=0, dtype=str).columns.tolist()
    missing_columns = [column for column in DAY_TRADING_COLUMNS if column not in header]
    if missing_columns:
        raise ValueError(f'Source CSV missing required columns: {missing_columns}')

    rows_read = 0
    rows_written = 0
    skipped = Counter()
    per_stock_rows = Counter()
    dates_with_rows = set()
    seen_keys = set()
    written_files = set()
    skipped_code_counts = Counter()
    skipped_code_names = {}
    min_date = ''
    max_date = ''

    for chunk in read_csv_canonical(source, dtype=str, keep_default_na=False, chunksize=chunksize):
        rows_read += len(chunk)
        chunk = chunk[DAY_TRADING_COLUMNS].copy()
        chunk['Code'] = chunk['Code'].astype(str).str.strip()
        chunk['Date'] = chunk['Date'].map(normalize_date)

        invalid_date_mask = chunk['Date'].eq('')
        skipped['invalid_date_rows'] += int(invalid_date_mask.sum())
        chunk = chunk[~invalid_date_mask]

        unlisted_mask = ~chunk['Code'].isin(listed_codes)
        skipped['unlisted_or_non_common_stock_rows'] += int(unlisted_mask.sum())
        if unlisted_mask.any():
            for row in chunk.loc[unlisted_mask, ['Code', 'Name']].itertuples(index=False):
                skipped_code_counts[row.Code] += 1
                skipped_code_names.setdefault(row.Code, str(row.Name).strip())
        chunk = chunk[~unlisted_mask]
        if chunk.empty:
            continue

        chunk['Name'] = chunk['Code'].map(code_to_name).fillna(chunk['Name'])
        chunk['_key'] = chunk['Date'] + '\0' + chunk['Code']
        duplicate_mask = chunk['_key'].isin(seen_keys) | chunk.duplicated('_key')
        skipped['duplicate_date_code_rows'] += int(duplicate_mask.sum())
        chunk = chunk[~duplicate_mask]
        if chunk.empty:
            continue

        seen_keys.update(chunk['_key'])
        chunk = chunk.drop(columns=['_key'])

        min_chunk_date = chunk['Date'].min()
        max_chunk_date = chunk['Date'].max()
        min_date = min_chunk_date if not min_date else min(min_date, min_chunk_date)
        max_date = max_chunk_date if not max_date else max(max_date, max_chunk_date)
        dates_with_rows.update(chunk['Date'].unique())

        for code, stock_df in chunk.groupby('Code', sort=True):
            name = code_to_name.get(code, '')
            path = output_path_for_stock(output_dir, code, name)
            write_header = path not in written_files
            to_csv_storage(
                stock_df,
                path,
                mode='a',
                header=write_header,
                index=False,
                encoding='utf-8-sig',
            )
            written_files.add(path)
            rows_written += len(stock_df)
            per_stock_rows[code] += len(stock_df)

    refresh_day_trading_features(written_files)

    trading_days = load_trading_days(trading_days_path, min_date, max_date) if min_date and max_date else []
    missing_dates = sorted(set(trading_days) - dates_with_rows)
    write_missing_dates(missing_dates_path, missing_dates)
    write_skipped_codes(skipped_codes_path, skipped_code_counts, skipped_code_names)

    manifest = {
        'dataset': 'day_trading',
        'canonical_output_dir': str(output_dir),
        'source_csv': str(source),
        'source_endpoint': 'https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U',
        'metadata_csv': str(metadata),
        'trading_days_csv': str(trading_days_path),
        'missing_dates_csv': str(missing_dates_path),
        'skipped_codes_csv': str(skipped_codes_path),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'date_format': 'YYYY-MM-DD',
        'date_min': min_date,
        'date_max': max_date,
        'schema': DAY_TRADING_OUTPUT_COLUMNS,
        'listed_common_stock_count_in_metadata': len(listed_codes),
        'stock_files_written': len(written_files),
        'rows_read': rows_read,
        'rows_written': rows_written,
        'unique_dates_with_rows': len(dates_with_rows),
        'trading_days_in_range': len(trading_days),
        'missing_trading_dates': len(missing_dates),
        'skipped_codes': len(skipped_code_counts),
        'skipped_rows': dict(sorted(skipped.items())),
        'per_stock_min_rows': min(per_stock_rows.values()) if per_stock_rows else 0,
        'per_stock_max_rows': max(per_stock_rows.values()) if per_stock_rows else 0,
    }
    with manifest_path.open('w', encoding='utf-8') as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)
    write_logs(logs_path, manifest_path, missing_dates_path, skipped_codes_path)

    return manifest


def main():
    args = parse_args()
    manifest = rebuild(
        source=Path(args.source),
        metadata=Path(args.metadata),
        trading_days_path=Path(args.trading_days),
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        missing_dates_path=Path(args.missing_dates),
        skipped_codes_path=Path(args.skipped_codes),
        logs_path=Path(args.logs),
        chunksize=args.chunksize,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
