'''
yield_pe_pb.py

Download/cache TWSE historical daily valuation data:
P/E ratio, dividend yield, and price-to-book ratio.
'''
import argparse
import os
import random
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from column_schema import read_csv_canonical, to_csv_storage


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'yield_pe_pb')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
DEFAULT_START_DATE = '2020-01-01'
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
THROTTLE_MIN_SECONDS = 1.0
THROTTLE_MAX_SECONDS = 2.0
TWSE_BWIBBU_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d'

OUTPUT_COLUMNS = [
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

NUMERIC_COLUMNS = [
    'Close',
    'DividendYield',
    'DividendYear',
    'PEratio',
    'PBratio',
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

NO_DATA_STATUS_MARKERS = (
    'no data',
    'no records',
    'not found',
    '\u67e5\u7121',
    '\u7121\u8cc7\u6599',
    '\u6c92\u6709\u7b26\u5408\u689d\u4ef6',
)


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Download TWSE historical P/E, dividend yield, and P/B data.'
    )
    parser.add_argument(
        '--start-date',
        default=DEFAULT_START_DATE,
        help='Start date in YYYY-MM-DD format. Default: 2020-01-01.',
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date in YYYY-MM-DD format. Default: today.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    '''
    Return a date from YYYY-MM-DD text.
    '''
    return datetime.strptime(value, '%Y-%m-%d').date()


def format_twse_date(value):
    '''
    Return YYYYMMDD text for TWSE query parameters.
    '''
    return value.strftime('%Y%m%d')


def iter_dates(start_date, end_date):
    '''
    Yield every calendar date in the requested range.
    '''
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def sleep_between_requests():
    '''
    Sleep for a randomized polite throttle interval.
    '''
    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))


def get_json_response(query_date):
    '''
    Fetch one TWSE historical valuation response with retries.
    '''
    expected_date = format_twse_date(query_date)
    params = {
        'date': expected_date,
        'selectType': 'ALL',
        'response': 'json',
    }
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                TWSE_BWIBBU_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError(
                    f'Unexpected HTTP status {response.status_code}: '
                    f'{response.text[:120]}'
                )

            response.encoding = 'utf-8'
            if 'json' not in response.headers.get('content-type', '').lower():
                raise ValueError(
                    'TWSE returned a non-JSON response: '
                    f'{response.text[:120]}'
                )

            payload = response.json()

            payload_date = payload.get('date')
            if payload.get('stat') == 'OK' and payload_date != expected_date:
                raise ValueError(
                    f'Unexpected TWSE payload date: requested {expected_date}, '
                    f'got {payload_date}'
                )

            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            print(
                f'Fetch failed for {query_date} '
                f'(attempt {attempt}/{MAX_RETRIES}): {exc}'
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def is_no_data_response(payload):
    '''
    Return True when TWSE reports no records for the requested date.
    '''
    stat = str(payload.get('stat', ''))
    if stat == 'OK' or payload.get('data'):
        return False

    lower_stat = stat.lower()
    return any(marker in lower_stat for marker in NO_DATA_STATUS_MARKERS)


def parse_twse_rows(payload, query_date):
    '''
    Convert one TWSE payload into normalized row dictionaries.
    '''
    if is_no_data_response(payload):
        return []

    if payload.get('stat') != 'OK':
        raise ValueError(f"Unexpected TWSE response status: {payload.get('stat')}")

    data = payload.get('data')
    if not data:
        return []

    rows = []
    row_date = payload.get('date') or format_twse_date(query_date)

    for raw_row in data:
        values = raw_row.get('value', raw_row) if isinstance(raw_row, dict) else raw_row
        if len(values) >= 8:
            rows.append({
                'Date': row_date,
                'Code': str(values[0]),
                'Name': values[1],
                'Close': values[2],
                'DividendYield': values[3],
                'DividendYear': values[4],
                'PEratio': values[5],
                'PBratio': values[6],
                'FiscalYearQuarter': values[7],
            })
        elif len(values) >= 5:
            rows.append({
                'Date': row_date,
                'Code': str(values[0]),
                'Name': values[1],
                'Close': '',
                'DividendYield': values[3],
                'DividendYear': '',
                'PEratio': values[2],
                'PBratio': values[4],
                'FiscalYearQuarter': '',
            })
        else:
            raise ValueError(f'Unexpected row format: {values}')

    return rows


def normalize_dataframe(rows):
    '''
    Normalize raw rows into the final CSV schema.
    '''
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.replace({'': pd.NA, '-': pd.NA})

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    df['Code'] = df['Code'].astype(str)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column].astype('string').str.replace(',', '', regex=False),
            errors='coerce',
        )

    df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')

    return df[OUTPUT_COLUMNS]


def safe_filename_part(value):
    '''
    Return text that is safe to use in a Windows filename.
    '''
    text = str(value or '').strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text)
    text = re.sub(r'\s+', '_', text)
    return text.strip(' ._') or 'unknown'


def load_listed_common_metadata():
    '''
    Load the listed common-stock catalog used for per-stock outputs.
    '''
    metadata = read_csv_canonical(METADATA_PATH, dtype={'Code': str}).fillna('')
    required = {'Code', 'Name', 'Type', 'Market'}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f'{METADATA_PATH} is missing columns: {sorted(missing)}')

    metadata['Code'] = metadata['Code'].astype(str).str.strip()
    metadata = metadata[
        metadata['Code'].str.fullmatch(r'\d{4}', na=False)
        & metadata['Type'].eq('股票')
        & metadata['Market'].eq('上市')
    ]
    return metadata.drop_duplicates('Code').set_index('Code', drop=False)


def output_path_for_code(code, metadata):
    name = metadata.at[code, 'Name'] if code in metadata.index else code
    return os.path.join(OUTPUT_DIR, f'{code}_{safe_filename_part(name)}.csv')


def merge_existing(existing, updates):
    merged = pd.concat([existing, updates], ignore_index=True)
    merged['Date'] = merged['Date'].astype(str)
    merged['Code'] = merged['Code'].astype(str)
    return (
        merged.drop_duplicates(subset=['Date', 'Code'], keep='last')
        .sort_values(['Date', 'Code'])
        [OUTPUT_COLUMNS]
    )


def write_per_stock_csvs(df, metadata):
    '''
    Merge downloaded rows into canonical per-stock valuation CSVs.
    '''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = df[df['Code'].isin(metadata.index)].copy()

    written_files = 0
    written_rows = 0
    for code, stock_df in df.groupby('Code', sort=True):
        output_path = output_path_for_code(code, metadata)
        stock_df = stock_df[OUTPUT_COLUMNS].sort_values(['Date', 'Code'])
        if os.path.exists(output_path):
            existing = read_csv_canonical(output_path, dtype={'Date': str, 'Code': str})
            stock_df = merge_existing(existing, stock_df)
        to_csv_storage(stock_df, output_path, index=False, encoding='utf-8-sig')
        written_files += 1
        written_rows += len(stock_df)

    return written_files, written_rows


def download_history(start_date, end_date):
    '''
    Download historical TWSE valuation rows for all dates in range.
    '''
    all_rows = []
    stats = {
        'dates_checked': 0,
        'trading_days_downloaded': 0,
        'skipped_days': 0,
        'failed_days': 0,
    }
    failed_dates = []
    total_dates = (end_date - start_date).days + 1

    for index, query_date in enumerate(iter_dates(start_date, end_date), start=1):
        stats['dates_checked'] += 1
        print(f'[{index}/{total_dates}] Fetching {query_date}.')

        try:
            payload = get_json_response(query_date)
            rows = parse_twse_rows(payload, query_date)
        except Exception as exc:
            stats['failed_days'] += 1
            failed_dates.append((query_date.isoformat(), str(exc)))
            print(f'Failed {query_date}: {exc}')
        else:
            if rows:
                all_rows.extend(rows)
                stats['trading_days_downloaded'] += 1
                print(f'Downloaded {len(rows)} rows for {query_date}.')
            else:
                stats['skipped_days'] += 1
                print(f'No trading data for {query_date}; skipped.')
        finally:
            sleep_between_requests()

    return all_rows, stats, failed_dates


def validate_date_range(start_date, end_date):
    '''
    Validate command line date inputs.
    '''
    if start_date > end_date:
        raise ValueError('start-date must be earlier than or equal to end-date.')


def log_failed_dates(failed_dates, start_date, end_date):
    '''
    Save failed date details for rerun/debugging.
    '''
    if not failed_dates:
        return None

    os.makedirs(LOG_DIR, exist_ok=True)
    start_text = format_twse_date(start_date)
    end_text = format_twse_date(end_date)
    log_path = os.path.join(
        LOG_DIR,
        f'twse_basic_valuation_failed_dates_{start_text}_to_{end_text}.csv',
    )
    df_errors = pd.DataFrame(
        failed_dates,
        columns=['Date', 'Error'],
    )
    df_errors.to_csv(log_path, index=False, encoding='utf-8-sig')
    return log_path


def main():
    '''
    Download TWSE historical valuation data into per-stock CSVs.
    '''
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    validate_date_range(start_date, end_date)

    metadata = load_listed_common_metadata()
    rows, stats, failed_dates = download_history(start_date, end_date)

    if not rows:
        raise ValueError('No TWSE valuation rows were downloaded.')

    df = normalize_dataframe(rows)
    written_files, written_rows = write_per_stock_csvs(df, metadata)
    failed_log_path = log_failed_dates(failed_dates, start_date, end_date)

    print('Download summary:')
    print(f"dates_checked={stats['dates_checked']}")
    print(f"trading_days_downloaded={stats['trading_days_downloaded']}")
    print(f"downloaded_rows={len(df)}")
    print(f"per_stock_files_written={written_files}")
    print(f"per_stock_rows_after_merge={written_rows}")
    print(f"skipped_days={stats['skipped_days']}")
    print(f"failed_days={stats['failed_days']}")
    print(f'output_dir={OUTPUT_DIR}')
    if failed_log_path:
        print(f'failed_log_path={failed_log_path}')

    if failed_dates:
        print('Failed dates:')
        for failed_date, error in failed_dates:
            print(f'{failed_date}: {error}')


if __name__ == '__main__':
    main()
