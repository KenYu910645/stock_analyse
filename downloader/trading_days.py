'''
trading_days.py

Download TWSE trading days from the official monthly FMTQIK market summary.

TWSE's online FMTQIK endpoint starts at 1990-01-04. Older exchange history is
not exposed by this endpoint, so this file records the official online range.
'''
import argparse
import os
import random
import time
import calendar
from datetime import date, datetime, timedelta

import pandas as pd
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'trading_days.csv')

TWSE_FMTQIK_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK'
TWSE_ONLINE_START_DATE = date(1990, 1, 4)
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 12
THROTTLE_MIN_SECONDS = 0.8
THROTTLE_MAX_SECONDS = 1.8

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'stock_analyse/1.0'
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download TWSE trading days to data/trading_days.csv.'
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help=(
            'Start date in YYYY-MM-DD. Default: refresh incrementally from '
            'the existing output, or from 1990-01-04 if no output exists.'
        ),
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date in YYYY-MM-DD. Default: today.',
    )
    parser.add_argument(
        '--output',
        default=OUTPUT_PATH,
        help='Output CSV path. Default: data/trading_days.csv.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def load_existing_trading_days(output_path):
    if not os.path.exists(output_path):
        return pd.DataFrame(columns=['date'])
    existing = pd.read_csv(output_path, dtype=str).fillna('')
    if 'date' not in existing.columns:
        existing = existing.rename(columns={existing.columns[0]: 'date'})
    existing['date'] = pd.to_datetime(existing['date'], errors='coerce').dt.date
    existing = existing.dropna(subset=['date'])
    existing['date'] = existing['date'].map(lambda value: value.isoformat())
    return existing[['date']].drop_duplicates().sort_values('date').reset_index(drop=True)


def default_start_date(output_path):
    existing = load_existing_trading_days(output_path)
    if existing.empty:
        return TWSE_ONLINE_START_DATE
    latest = pd.to_datetime(existing['date'], errors='coerce').dropna().max().date()
    return max(latest - timedelta(days=31), TWSE_ONLINE_START_DATE)


def month_start(value):
    return date(value.year, value.month, 1)


def iter_months(start_date, end_date):
    current = month_start(start_date)
    end_month = month_start(end_date)

    while current <= end_month:
        yield current
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = date(year, month, 1)


def format_twse_query_date(value):
    # Some old months fail on the 1st if that day was not a trading day.
    # Querying an in-month middate still returns the whole month.
    return value.strftime('%Y%m%d')


def get_twse_month_query_dates(month_date, end_date):
    last_day = calendar.monthrange(month_date.year, month_date.month)[1]
    days = [15, 1, 2, 5, 10, 20, 28, last_day]
    query_dates = []

    for day in days:
        if day > last_day:
            continue
        query_date = date(month_date.year, month_date.month, day)
        if query_date > end_date:
            continue
        if query_date not in query_dates:
            query_dates.append(query_date)

    return query_dates


def parse_roc_date(value):
    text = str(value).strip()
    year_text, month_text, day_text = text.split('/')
    year = int(year_text) + 1911
    return date(year, int(month_text), int(day_text))


def sleep_between_requests():
    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))


def fetch_month(session, month_date, end_date):
    errors = []

    for query_date in get_twse_month_query_dates(month_date, end_date):
        response = session.get(
            TWSE_FMTQIK_URL,
            params={
                'date': format_twse_query_date(query_date),
                'response': 'json',
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            errors.append(f'HTTP {response.status_code} on {query_date}')
            continue

        content_type = response.headers.get('content-type', '')
        if 'json' not in content_type.lower():
            snippet = response.text[:120].replace('\n', ' ')
            errors.append(f'non-JSON on {query_date}: {snippet}')
            continue

        try:
            payload = response.json()
        except ValueError:
            snippet = response.text[:120].replace('\n', ' ')
            errors.append(f'invalid JSON on {query_date}: {snippet}')
            continue

        if payload.get('stat') == 'OK':
            return payload.get('data', [])

        errors.append(f"{payload.get('stat')} on {query_date}")

    raise ValueError(
        f'TWSE FMTQIK failed for {month_date:%Y-%m}: ' + '; '.join(errors)
    )


def fetch_month_with_retries(session, month_date, end_date):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_month(session, month_date, end_date)
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            print(
                f'Fetch failed for {month_date:%Y-%m} '
                f'(attempt {attempt}/{MAX_RETRIES}): {exc}'
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def download_trading_days(start_date, end_date):
    if start_date < TWSE_ONLINE_START_DATE:
        print(
            'TWSE online FMTQIK starts at 1990-01-04; '
            f'clamping start date from {start_date} to {TWSE_ONLINE_START_DATE}.'
        )
        start_date = TWSE_ONLINE_START_DATE

    rows = []

    with requests.Session() as session:
        months = list(iter_months(start_date, end_date))
        for index, month_date in enumerate(months, start=1):
            print(f'[{index}/{len(months)}] Fetching TWSE trading days {month_date:%Y-%m}.')
            for raw_row in fetch_month_with_retries(session, month_date, end_date):
                trading_date = parse_roc_date(raw_row[0])
                if start_date <= trading_date <= end_date:
                    rows.append({
                        'date': trading_date.isoformat(),
                    })
            sleep_between_requests()

    if not rows:
        raise ValueError('No TWSE trading days were downloaded.')

    df = pd.DataFrame(rows).drop_duplicates().sort_values('date').reset_index(drop=True)
    return df


def main():
    args = parse_args()
    explicit_start = args.start_date is not None
    start_date = parse_iso_date(args.start_date) if explicit_start else default_start_date(args.output)
    end_date = parse_iso_date(args.end_date)

    if end_date < start_date:
        raise ValueError('end-date must be on or after start-date.')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = download_trading_days(start_date, end_date)
    if not explicit_start:
        existing = load_existing_trading_days(args.output)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates().sort_values('date').reset_index(drop=True)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(
        f'Saved {len(df)} TWSE trading days to {args.output}. '
        f'Range: {df["date"].iloc[0]} to {df["date"].iloc[-1]}.'
    )


if __name__ == '__main__':
    main()
