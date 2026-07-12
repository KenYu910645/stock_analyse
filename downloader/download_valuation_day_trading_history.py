'''
Download long-history TWSE valuation and day-trading datasets.

The TWSE endpoints have hard lower bounds:
- valuation BWIBBU_d starts on 2005-09-02
- day-trading TWTB4U starts on 2014-01-06

The downloader is intentionally sequential with retry/backoff because TWSE
returns HTTP 428 when historical requests are sent too aggressively.
'''
import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta

import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from downloader import price


VALUATION_START = date(2005, 9, 2)
DAY_TRADING_START = date(2014, 1, 6)
VALUATION_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d'
DAY_TRADING_URL = 'https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'stock_analyse/1.0'
    ),
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
}

VALUATION_COLUMNS = [
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

DAY_TRADING_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'SuspensionNote',
    'DayTradingVolume',
    'DayTradingBuyAmount',
    'DayTradingSellAmount',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download TWSE valuation and day-trading history.'
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date in YYYY-MM-DD format. Default: today.',
    )
    parser.add_argument(
        '--dataset',
        choices=['all', 'valuation', 'day-trading'],
        default='all',
        help='Dataset to download. Default: all.',
    )
    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=0.45,
        help='Base sleep after each request. Default: 0.45.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing final CSVs and progress files.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def format_yyyymmdd(value):
    return value.strftime('%Y%m%d')


def yyyymmdd_to_iso(value):
    text = str(value or '').strip().replace('/', '').replace('-', '')
    if len(text) == 8 and text.isdigit():
        return f'{text[:4]}-{text[4:6]}-{text[6:8]}'
    if len(text) == 7 and text.isdigit():
        year = int(text[:3]) + 1911
        return f'{year:04d}-{text[3:5]}-{text[5:7]}'
    return str(value or '').strip()


def iter_weekdays(start_date, end_date):
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def clean_number(value):
    return str(value).replace(',', '').strip()


def load_listed_codes():
    catalog = price.build_stock_catalog()
    return set(catalog['Code'].astype(str))


def is_listed_common_code(code, listed_codes):
    text = str(code).strip()
    return text in listed_codes and text.isdigit() and len(text) == 4


def ensure_output_dir(name):
    if name == 'valuation':
        output_dir = os.path.join(PROJECT_ROOT, 'data', 'yield_pe_pb')
    else:
        output_dir = os.path.join(PROJECT_ROOT, 'data', 'day_trading')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_paths(name, start_date, end_date):
    output_dir = ensure_output_dir(name)
    start_text = format_yyyymmdd(start_date)
    end_text = format_yyyymmdd(end_date)
    base = f'twse_{name}_history_{start_text}_to_{end_text}'
    return {
        'csv': os.path.join(output_dir, f'{base}.csv'),
        'progress': os.path.join(output_dir, f'{base}_progress.json'),
        'errors': os.path.join(output_dir, f'{base}_errors.csv'),
        'manifest': os.path.join(output_dir, f'{base}_manifest.json'),
    }


def read_completed_dates(progress_path):
    if not os.path.exists(progress_path):
        return set()
    with open(progress_path, 'r', encoding='utf-8') as file_obj:
        payload = json.load(file_obj)
    return set(payload.get('completed_dates', []))


def write_progress(progress_path, completed_dates):
    with open(progress_path, 'w', encoding='utf-8') as file_obj:
        json.dump(
            {
                'completed_dates': sorted(completed_dates),
                'updated_at': datetime.now().isoformat(timespec='seconds'),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )


def append_error(error_path, row):
    write_header = not os.path.exists(error_path)
    with open(error_path, 'a', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=['Date', 'Error'])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def request_json(session, url, params):
    response = session.get(url, params=params, timeout=30)
    if response.status_code == 428:
        raise RuntimeError('TWSE returned HTTP 428 rate limit / precondition response')
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.json()


def request_json_with_retries(session, url, params, base_sleep):
    last_error = None
    for attempt in range(1, 6):
        try:
            return request_json(session, url, params)
        except Exception as exc:
            last_error = exc
            sleep_seconds = base_sleep * (2 ** attempt)
            print(
                f'Fetch failed for {params}: {exc}; '
                f'retry {attempt}/5 after {sleep_seconds:.1f}s',
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise last_error


def parse_valuation_payload(payload, query_date, listed_codes):
    if payload.get('stat') != 'OK':
        return []

    fields = payload.get('fields') or []
    row_date = yyyymmdd_to_iso(payload.get('date') or format_yyyymmdd(query_date))
    rows = []
    for raw in payload.get('data') or []:
        if len(raw) < 5:
            continue
        code = str(raw[0]).strip()
        if not is_listed_common_code(code, listed_codes):
            continue

        row = {
            'Date': row_date,
            'Code': code,
            'Name': str(raw[1]).strip(),
            'Close': '',
            'DividendYield': '',
            'DividendYear': '',
            'PEratio': '',
            'PBratio': '',
            'FiscalYearQuarter': '',
        }
        if len(fields) == 5:
            row.update({
                'PEratio': clean_number(raw[2]),
                'DividendYield': clean_number(raw[3]),
                'PBratio': clean_number(raw[4]),
            })
        elif len(fields) >= 8 and len(raw) >= 8:
            row.update({
                'Close': clean_number(raw[2]),
                'DividendYield': clean_number(raw[3]),
                'DividendYear': clean_number(raw[4]),
                'PEratio': clean_number(raw[5]),
                'PBratio': clean_number(raw[6]),
                'FiscalYearQuarter': str(raw[7]).strip(),
            })
        rows.append(row)
    return rows


def find_day_trading_stock_table(payload):
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if fields and fields[0] == '\u8b49\u5238\u4ee3\u865f':
            return table
    return None


def parse_day_trading_payload(payload, query_date, listed_codes):
    if payload.get('stat') != 'OK':
        return []

    table = find_day_trading_stock_table(payload)
    if table is None:
        return []

    row_date = yyyymmdd_to_iso(payload.get('date') or format_yyyymmdd(query_date))
    rows = []
    for raw in table.get('data') or []:
        if len(raw) < 6:
            continue
        code = str(raw[0]).strip()
        if not is_listed_common_code(code, listed_codes):
            continue

        row = {
            'Date': row_date,
            'Code': code,
            'Name': str(raw[1]).strip(),
            'SuspensionNote': '',
            'DayTradingVolume': '',
            'DayTradingBuyAmount': '',
            'DayTradingSellAmount': '',
        }
        if len(raw) == 6:
            row.update({
                'SuspensionNote': str(raw[2]).strip(),
                'DayTradingVolume': clean_number(raw[3]),
                'DayTradingBuyAmount': clean_number(raw[4]),
                'DayTradingSellAmount': clean_number(raw[5]),
            })
        elif len(raw) >= 11:
            row.update({
                'DayTradingVolume': clean_number(raw[2]),
                'DayTradingBuyAmount': clean_number(raw[3]),
                'DayTradingSellAmount': clean_number(raw[4]),
            })
        rows.append(row)
    return rows


def fetch_valuation_rows(session, query_date, listed_codes, base_sleep):
    payload = request_json_with_retries(
        session,
        VALUATION_URL,
        {
            'date': format_yyyymmdd(query_date),
            'selectType': 'ALL',
            'response': 'json',
        },
        base_sleep,
    )
    return parse_valuation_payload(payload, query_date, listed_codes)


def fetch_day_trading_rows(session, query_date, listed_codes, base_sleep):
    payload = request_json_with_retries(
        session,
        DAY_TRADING_URL,
        {
            'date': format_yyyymmdd(query_date),
            'response': 'json',
        },
        base_sleep,
    )
    return parse_day_trading_payload(payload, query_date, listed_codes)


def download_dataset(
    name,
    start_date,
    end_date,
    columns,
    fetcher,
    listed_codes,
    force=False,
    sleep_seconds=0.45,
):
    paths = output_paths(name, start_date, end_date)
    if force:
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)

    completed_dates = read_completed_dates(paths['progress'])
    append = os.path.exists(paths['csv']) and not force
    total_dates = list(iter_weekdays(start_date, end_date))
    pending_dates = [
        query_date
        for query_date in total_dates
        if format_yyyymmdd(query_date) not in completed_dates
    ]

    print(
        f'{name}: {len(completed_dates)} completed, '
        f'{len(pending_dates)} pending, output={paths["csv"]}',
        flush=True,
    )

    row_count = 0
    error_count = 0
    with requests.Session() as session:
        session.headers.update(HEADERS)
        with open(
            paths['csv'],
            'a' if append else 'w',
            newline='',
            encoding='utf-8-sig',
        ) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            if not append:
                writer.writeheader()

            for index, query_date in enumerate(pending_dates, start=1):
                date_text = format_yyyymmdd(query_date)
                try:
                    rows = fetcher(session, query_date, listed_codes, sleep_seconds)
                except Exception as exc:
                    error_count += 1
                    append_error(paths['errors'], {
                        'Date': date_text,
                        'Error': str(exc),
                    })
                    rows = []
                    print(f'{name}: ERROR {date_text}: {exc}', flush=True)

                if rows:
                    rows.sort(key=lambda row: row['Code'])
                    writer.writerows(rows)
                    row_count += len(rows)

                completed_dates.add(date_text)
                if index % 20 == 0 or index == len(pending_dates):
                    output_file.flush()
                    write_progress(paths['progress'], completed_dates)
                    print(
                        f'{name}: {index}/{len(pending_dates)} pending done; '
                        f'new_rows={row_count}; errors={error_count}',
                        flush=True,
                    )

                time.sleep(sleep_seconds)

    # Count final rows without loading the full CSV into memory.
    final_rows = 0
    if os.path.exists(paths['csv']):
        with open(paths['csv'], 'r', encoding='utf-8-sig', newline='') as file_obj:
            final_rows = max(sum(1 for _line in file_obj) - 1, 0)

    manifest = {
        'dataset': name,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'output_csv': paths['csv'],
        'progress_json': paths['progress'],
        'errors_csv': paths['errors'] if os.path.exists(paths['errors']) else None,
        'manifest_json': paths['manifest'],
        'completed_dates': len(completed_dates),
        'total_weekdays': len(total_dates),
        'rows': final_rows,
        'error_count': error_count,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(paths['manifest'], 'w', encoding='utf-8') as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main():
    args = parse_args()
    end_date = parse_iso_date(args.end_date)
    listed_codes = load_listed_codes()
    manifests = []

    if args.dataset in ('all', 'valuation'):
        manifests.append(download_dataset(
            'valuation',
            VALUATION_START,
            end_date,
            VALUATION_COLUMNS,
            fetch_valuation_rows,
            listed_codes,
            force=args.force,
            sleep_seconds=args.sleep_seconds,
        ))

    if args.dataset in ('all', 'day-trading'):
        manifests.append(download_dataset(
            'day_trading',
            DAY_TRADING_START,
            end_date,
            DAY_TRADING_COLUMNS,
            fetch_day_trading_rows,
            listed_codes,
            force=args.force,
            sleep_seconds=args.sleep_seconds,
        ))

    print('DONE')
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
