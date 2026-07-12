'''
Download TWSE listed-stock margin trading balance data.

The TWSE source is the 融資融券餘額 report:
https://www.twse.com.tw/zh/trading/margin/mi-margn.html
'''
import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests

from column_schema import normalize_date_text, storage_fieldnames, storage_record


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'margin')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
DEFAULT_START_DATE = '2001-01-01'
TWSE_MARGIN_URL = 'https://www.twse.com.tw/exchangeReport/MI_MARGN'
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 8
THROTTLE_MIN_SECONDS = 0.08
THROTTLE_MAX_SECONDS = 0.25

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'stock_analyse/1.0'
    ),
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Referer': 'https://www.twse.com.tw/zh/trading/margin/mi-margn.html',
}

OUTPUT_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'MarginPurchase',
    'MarginSale',
    'MarginCashRepayment',
    'MarginPreviousBalance',
    'MarginCurrentBalance',
    'MarginNextDayLimit',
    'MarginFinancingUsageRate',
    'MarginBalance20DayChangeRate',
    'MarginMarketValue',
    'MarginMarketValueTo20DayAvgTurnover',
    'ShortPurchase',
    'ShortSale',
    'ShortStockRepayment',
    'ShortPreviousBalance',
    'ShortCurrentBalance',
    'ShortNextDayLimit',
    'ShortMarginBalanceRatio',
    'Offsetting',
    'Note',
]

MARGIN_FEATURE_COLUMNS = [
    'MarginFinancingUsageRate',
    'MarginBalance20DayChangeRate',
    'MarginMarketValue',
    'MarginMarketValueTo20DayAvgTurnover',
    'ShortMarginBalanceRatio',
]

RAW_OUTPUT_COLUMNS = [
    column for column in OUTPUT_COLUMNS
    if column not in MARGIN_FEATURE_COLUMNS
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download TWSE listed-stock margin trading data.'
    )
    parser.add_argument(
        '--start-date',
        default=DEFAULT_START_DATE,
        help='Start date in YYYY-MM-DD format. Default: 2001-01-01.',
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date in YYYY-MM-DD format. Default: today.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Refetch raw JSON even when a cached file already exists.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel workers for raw JSON fetches. Default: 1.',
    )
    parser.add_argument(
        '--include-weekends',
        action='store_true',
        help='Also query Saturday/Sunday dates. Default skips weekends.',
    )
    parser.add_argument(
        '--rebuild-only',
        action='store_true',
        help='Only rebuild the CSV from cached raw JSON; do not make network requests.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def format_twse_date(value):
    return value.strftime('%Y%m%d')


def iter_dates(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def iter_query_dates(start_date, end_date, include_weekends=False):
    for query_date in iter_dates(start_date, end_date):
        if not include_weekends and query_date.weekday() >= 5:
            continue
        yield query_date


def raw_path_for_date(query_date):
    year_dir = os.path.join(RAW_DIR, query_date.strftime('%Y'))
    return os.path.join(year_dir, f'{format_twse_date(query_date)}.json')


def output_path_for_range(start_date, end_date):
    filename = (
        f'twse_margin_stocks_{format_twse_date(start_date)}'
        f'_to_{format_twse_date(end_date)}.csv'
    )
    return os.path.join(DATA_DIR, filename)


def manifest_path_for_range(start_date, end_date):
    filename = (
        f'twse_margin_stocks_{format_twse_date(start_date)}'
        f'_to_{format_twse_date(end_date)}_manifest.json'
    )
    return os.path.join(DATA_DIR, filename)


def parse_number(value):
    text = str(value).strip().replace(',', '')
    if text in ('', '--'):
        return ''
    return int(text)


def fetch_payload(session, query_date):
    params = {
        'date': format_twse_date(query_date),
        'selectType': 'STOCK',
        'response': 'json',
    }
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                TWSE_MARGIN_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            print(
                f'Fetch failed for {query_date} '
                f'(attempt {attempt}/{MAX_RETRIES}): {exc}'
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise last_error


def load_or_fetch_payload(session, query_date, force=False):
    path = raw_path_for_date(query_date)
    if not force and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj), False

    payload = fetch_payload(session, query_date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)

    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))
    return payload, True


def fetch_and_cache_date(query_date, force=False):
    session = requests.Session()
    payload, downloaded = load_or_fetch_payload(session, query_date, force)
    return query_date, downloaded, payload.get('stat')


def is_no_data_response(payload):
    return payload.get('stat') != 'OK' and not payload.get('tables')


def stock_table_from_payload(payload):
    tables = payload.get('tables') or []
    for table in tables:
        title = table.get('title', '')
        if '融資融券彙總' in title and table.get('data'):
            return table
    return None


def parse_payload_rows(payload, query_date):
    if is_no_data_response(payload):
        return []

    if payload.get('stat') != 'OK':
        raise ValueError(f"Unexpected TWSE response status: {payload.get('stat')}")

    table = stock_table_from_payload(payload)
    if table is None:
        return []

    row_date = normalize_date_text(payload.get('date') or format_twse_date(query_date))
    rows = []
    for raw_row in table.get('data', []):
        if len(raw_row) < 16:
            raise ValueError(f'Unexpected row format: {raw_row}')

        code = str(raw_row[0]).strip()
        if not code or code == '　':
            continue

        rows.append({
            'Date': row_date,
            'Code': code,
            'Name': str(raw_row[1]).strip(),
            'MarginPurchase': parse_number(raw_row[2]),
            'MarginSale': parse_number(raw_row[3]),
            'MarginCashRepayment': parse_number(raw_row[4]),
            'MarginPreviousBalance': parse_number(raw_row[5]),
            'MarginCurrentBalance': parse_number(raw_row[6]),
            'MarginNextDayLimit': parse_number(raw_row[7]),
            'ShortPurchase': parse_number(raw_row[8]),
            'ShortSale': parse_number(raw_row[9]),
            'ShortStockRepayment': parse_number(raw_row[10]),
            'ShortPreviousBalance': parse_number(raw_row[11]),
            'ShortCurrentBalance': parse_number(raw_row[12]),
            'ShortNextDayLimit': parse_number(raw_row[13]),
            'Offsetting': parse_number(raw_row[14]),
            'Note': str(raw_row[15]).strip(),
        })

    return rows


def warm_raw_cache(start_date, end_date, force=False, workers=1, include_weekends=False):
    dates = list(iter_query_dates(start_date, end_date, include_weekends))
    if workers <= 1:
        downloaded_days = 0
        errors = []
        session = requests.Session()
        for index, query_date in enumerate(dates, start=1):
            try:
                _, downloaded = load_or_fetch_payload(session, query_date, force)
                if downloaded:
                    downloaded_days += 1
            except Exception as exc:
                errors.append({
                    'Date': format_twse_date(query_date),
                    'Error': str(exc),
                })
                print(f'ERROR {query_date}: {exc}')

            if index % 100 == 0:
                print(f'Cached {index}/{len(dates)} calendar days.')

        return downloaded_days, errors

    missing_or_forced = [
        query_date
        for query_date in dates
        if force or not os.path.exists(raw_path_for_date(query_date))
    ]
    downloaded_days = 0
    errors = []

    if not missing_or_forced:
        return downloaded_days, errors

    print(
        f'Fetching {len(missing_or_forced)} missing calendar days '
        f'with {workers} workers.'
    )
    batch_size = max(workers * 20, 100)
    completed = 0
    for batch_start in range(0, len(missing_or_forced), batch_size):
        batch = missing_or_forced[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_and_cache_date, query_date, force): query_date
                for query_date in batch
            }
            for future in as_completed(futures):
                query_date = futures[future]
                try:
                    _, downloaded, _ = future.result()
                    if downloaded:
                        downloaded_days += 1
                except Exception as exc:
                    errors.append({
                        'Date': format_twse_date(query_date),
                        'Error': str(exc),
                    })
                    print(f'ERROR {query_date}: {exc}')

                completed += 1
                if completed % 100 == 0:
                    print(
                        f'Fetched {completed}/{len(missing_or_forced)} '
                        f'missing days; {downloaded_days} downloaded.'
                    )

    return downloaded_days, errors


def read_cached_payload(query_date):
    path = raw_path_for_date(query_date)
    with open(path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def download_range(
    start_date,
    end_date,
    force=False,
    workers=1,
    include_weekends=False,
    rebuild_only=False,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    output_path = output_path_for_range(start_date, end_date)
    error_path = os.path.join(LOG_DIR, 'twse_margin_download_errors.csv')
    manifest_path = manifest_path_for_range(start_date, end_date)

    trading_days = 0
    skipped_days = 0
    if rebuild_only:
        downloaded_days = 0
        errors = []
    else:
        downloaded_days, errors = warm_raw_cache(
            start_date,
            end_date,
            force=force,
            workers=workers,
            include_weekends=include_weekends,
        )
    total_rows = 0

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=storage_fieldnames(OUTPUT_COLUMNS))
        writer.writeheader()

        for index, query_date in enumerate(
            iter_query_dates(start_date, end_date, include_weekends),
            start=1,
        ):
            try:
                payload = read_cached_payload(query_date)
                rows = parse_payload_rows(payload, query_date)
            except Exception as exc:
                errors.append({
                    'Date': format_twse_date(query_date),
                    'Error': str(exc),
                })
                print(f'ERROR {query_date}: {exc}')
                continue

            if rows:
                writer.writerows(storage_record(row) for row in rows)
                trading_days += 1
                total_rows += len(rows)
            else:
                skipped_days += 1

            if index % 100 == 0:
                print(
                    f'Processed {index} calendar days; '
                    f'{trading_days} trading days; {total_rows} rows.'
                )
                output_file.flush()

    if errors:
        with open(error_path, 'w', encoding='utf-8-sig', newline='') as error_file:
            writer = csv.DictWriter(error_file, fieldnames=['Date', 'Error'])
            writer.writeheader()
            writer.writerows(errors)

    manifest = {
        'source': TWSE_MARGIN_URL,
        'selectType': 'STOCK',
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'output_csv': output_path,
        'raw_json_dir': RAW_DIR,
        'trading_days': trading_days,
        'skipped_days': skipped_days,
        'downloaded_days': downloaded_days,
        'workers': workers,
        'include_weekends': include_weekends,
        'rebuild_only': rebuild_only,
        'total_rows': total_rows,
        'error_count': len(errors),
        'error_csv': error_path if errors else None,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    return manifest


def main():
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if end_date < start_date:
        raise ValueError('end-date must be on or after start-date')

    manifest = download_range(
        start_date,
        end_date,
        force=args.force,
        workers=args.workers,
        include_weekends=args.include_weekends,
        rebuild_only=args.rebuild_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
