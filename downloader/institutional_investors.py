'''
institutional_investors.py

Download/cache TWSE 三大法人買賣超日報 data for listed common stocks.
'''
import argparse
import os
import random
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'institutional')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
DEFAULT_START_DATE = '2020-01-01'
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
THROTTLE_MIN_SECONDS = 1.0
THROTTLE_MAX_SECONDS = 2.0
TWSE_T86_URL = 'https://www.twse.com.tw/fund/T86'

COMMON_STOCK_TYPE = '\u80a1\u7968'
TWSE_MARKET = '\u4e0a\u5e02'

OUTPUT_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'ForeignBuyExDealer',
    'ForeignSellExDealer',
    'ForeignNetExDealer',
    'ForeignDealerBuy',
    'ForeignDealerSell',
    'ForeignDealerNet',
    'InvestmentTrustBuy',
    'InvestmentTrustSell',
    'InvestmentTrustNet',
    'DealerNet',
    'DealerSelfBuy',
    'DealerSelfSell',
    'DealerSelfNet',
    'DealerHedgeBuy',
    'DealerHedgeSell',
    'DealerHedgeNet',
    'InstitutionalNet',
]

NUMERIC_COLUMNS = OUTPUT_COLUMNS[3:]

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Download TWSE 三大法人 daily data for listed common stocks.'
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
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output CSV if it already exists.',
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


def get_output_path(start_date, end_date):
    '''
    Return the institutional-investor CSV output path.
    '''
    start_text = format_twse_date(start_date)
    end_text = format_twse_date(end_date)
    filename = f'twse_institutional_investors_{start_text}_to_{end_text}.csv'
    return os.path.join(OUTPUT_DIR, filename)


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


def load_listed_common_stock_codes():
    '''
    Load listed common-stock codes from the project metadata.
    '''
    df_metadata = pd.read_csv(STOCK_METADATA_PATH, dtype={'Code': str})
    df_metadata['Code'] = df_metadata['Code'].astype(str).str.strip()
    mask = (
        (df_metadata['Type'] == COMMON_STOCK_TYPE)
        & (df_metadata['Market'] == TWSE_MARKET)
    )
    return set(df_metadata.loc[mask, 'Code'])


def get_json_response(query_date):
    '''
    Fetch one TWSE 三大法人 response with retries.
    '''
    expected_date = format_twse_date(query_date)
    params = {
        'date': expected_date,
        'selectType': 'ALLBUT0999',
        'response': 'json',
    }
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                TWSE_T86_URL,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
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
    return '沒有符合條件' in stat


def parse_twse_rows(payload, query_date, listed_common_stock_codes):
    '''
    Convert one TWSE payload into normalized row dictionaries.
    '''
    if is_no_data_response(payload):
        return []

    if payload.get('stat') != 'OK':
        raise ValueError(f"Unexpected TWSE response status: {payload.get('stat')}")

    rows = []
    row_date = payload.get('date') or format_twse_date(query_date)

    for raw_row in payload.get('data', []):
        values = raw_row.get('value', raw_row) if isinstance(raw_row, dict) else raw_row
        if len(values) < 12:
            raise ValueError(f'Unexpected row format: {values}')

        code = str(values[0]).strip()
        if code not in listed_common_stock_codes:
            continue

        if len(values) >= 19:
            rows.append({
                'Date': row_date,
                'Code': code,
                'Name': str(values[1]).strip(),
                'ForeignBuyExDealer': values[2],
                'ForeignSellExDealer': values[3],
                'ForeignNetExDealer': values[4],
                'ForeignDealerBuy': values[5],
                'ForeignDealerSell': values[6],
                'ForeignDealerNet': values[7],
                'InvestmentTrustBuy': values[8],
                'InvestmentTrustSell': values[9],
                'InvestmentTrustNet': values[10],
                'DealerNet': values[11],
                'DealerSelfBuy': values[12],
                'DealerSelfSell': values[13],
                'DealerSelfNet': values[14],
                'DealerHedgeBuy': values[15],
                'DealerHedgeSell': values[16],
                'DealerHedgeNet': values[17],
                'InstitutionalNet': values[18],
            })
        elif len(values) >= 16:
            rows.append({
                'Date': row_date,
                'Code': code,
                'Name': str(values[1]).strip(),
                'ForeignBuyExDealer': values[2],
                'ForeignSellExDealer': values[3],
                'ForeignNetExDealer': values[4],
                'ForeignDealerBuy': 0,
                'ForeignDealerSell': 0,
                'ForeignDealerNet': 0,
                'InvestmentTrustBuy': values[5],
                'InvestmentTrustSell': values[6],
                'InvestmentTrustNet': values[7],
                'DealerNet': values[8],
                'DealerSelfBuy': values[9],
                'DealerSelfSell': values[10],
                'DealerSelfNet': values[11],
                'DealerHedgeBuy': values[12],
                'DealerHedgeSell': values[13],
                'DealerHedgeNet': values[14],
                'InstitutionalNet': values[15],
            })
        else:
            rows.append({
                'Date': row_date,
                'Code': code,
                'Name': str(values[1]).strip(),
                'ForeignBuyExDealer': values[2],
                'ForeignSellExDealer': values[3],
                'ForeignNetExDealer': values[4],
                'ForeignDealerBuy': 0,
                'ForeignDealerSell': 0,
                'ForeignDealerNet': 0,
                'InvestmentTrustBuy': values[5],
                'InvestmentTrustSell': values[6],
                'InvestmentTrustNet': values[7],
                'DealerNet': values[8],
                'DealerSelfBuy': values[9],
                'DealerSelfSell': values[10],
                'DealerSelfNet': values[8],
                'DealerHedgeBuy': 0,
                'DealerHedgeSell': 0,
                'DealerHedgeNet': 0,
                'InstitutionalNet': values[11],
            })

    return rows


def normalize_dataframe(rows):
    '''
    Normalize raw rows into the final CSV schema.
    '''
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.replace({'': pd.NA, '-': pd.NA})

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['Code'] = df['Code'].astype(str)

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column].astype('string').str.replace(',', '', regex=False),
            errors='coerce',
        )

    df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')
    df = df.sort_values(['Date', 'Code']).reset_index(drop=True)

    return df[OUTPUT_COLUMNS]


def download_history(start_date, end_date, listed_common_stock_codes):
    '''
    Download historical TWSE 三大法人 rows for all dates in range.
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
            rows = parse_twse_rows(payload, query_date, listed_common_stock_codes)
        except Exception as exc:
            stats['failed_days'] += 1
            failed_dates.append((query_date.isoformat(), str(exc)))
            print(f'Failed {query_date}: {exc}')
        else:
            if rows:
                all_rows.extend(rows)
                stats['trading_days_downloaded'] += 1
                print(f'Downloaded {len(rows)} listed stock rows for {query_date}.')
            else:
                stats['skipped_days'] += 1
                print(f'No listed stock data for {query_date}; skipped.')
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
        f'twse_institutional_failed_dates_{start_text}_to_{end_text}.csv',
    )
    df_errors = pd.DataFrame(failed_dates, columns=['Date', 'Error'])
    df_errors.to_csv(log_path, index=False, encoding='utf-8-sig')
    return log_path


def main():
    '''
    Download TWSE historical 三大法人 data and save a normalized CSV.
    '''
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    validate_date_range(start_date, end_date)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = get_output_path(start_date, end_date)

    if os.path.exists(output_path) and not args.force:
        print(f'Output already exists: {output_path}')
        print('Use --force to overwrite it.')
        return

    listed_common_stock_codes = load_listed_common_stock_codes()
    rows, stats, failed_dates = download_history(
        start_date,
        end_date,
        listed_common_stock_codes,
    )

    if not rows:
        raise ValueError('No TWSE institutional-investor rows were downloaded.')

    df = normalize_dataframe(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    failed_log_path = log_failed_dates(failed_dates, start_date, end_date)

    print('Download summary:')
    print(f"dates_checked={stats['dates_checked']}")
    print(f"trading_days_downloaded={stats['trading_days_downloaded']}")
    print(f"rows_saved={len(df)}")
    print(f"skipped_days={stats['skipped_days']}")
    print(f"failed_days={stats['failed_days']}")
    print(f'output_path={output_path}')
    if failed_log_path:
        print(f'failed_log_path={failed_log_path}')

    if failed_dates:
        print('Failed dates:')
        for failed_date, error in failed_dates:
            print(f'{failed_date}: {error}')


if __name__ == '__main__':
    main()
