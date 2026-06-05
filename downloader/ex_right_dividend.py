'''
ex_right_dividend.py

Download/cache TWSE listed-stock ex-right/ex-dividend data for price
adjustment.
'''
import argparse
import os
import random
import re
import time
from datetime import date, datetime

import pandas as pd
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'dividend')
DEFAULT_START_DATE = '2020-01-01'
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
THROTTLE_MIN_SECONDS = 0.5
THROTTLE_MAX_SECONDS = 1.5

TWSE_TWT48U_URL = 'https://www.twse.com.tw/rwd/zh/exRight/TWT48U'
TWSE_TWT49U_URL = 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U'
TWSE_TWT49U_DETAIL_URL = 'https://www.twse.com.tw/rwd/zh/exRight/TWT49UDetail'

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

FINAL_COLUMNS = [
    'ex_date',
    'stock_id',
    'stock_name',
    'previous_close',
    'ex_reference_price',
    'opening_reference_price',
    'opening_auction_base',
    'limit_up_price',
    'limit_down_price',
    'cash_dividend',
    'dividend_value',
    'stock_dividend_rate',
    'cash_capital_increase_price',
    'cash_capital_increase_rate',
    'right_or_dividend',
    'deducted_dividend_reference_price',
    'detail_key',
]

NUMERIC_COLUMNS = [
    'previous_close',
    'ex_reference_price',
    'opening_reference_price',
    'opening_auction_base',
    'limit_up_price',
    'limit_down_price',
    'cash_dividend',
    'dividend_value',
    'stock_dividend_rate',
    'cash_capital_increase_price',
    'cash_capital_increase_rate',
    'deducted_dividend_reference_price',
]


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Download TWSE listed-stock ex-right/ex-dividend data.'
    )
    parser.add_argument(
        '--start-date',
        default=DEFAULT_START_DATE,
        help='Start date for calculation results in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date for calculation results in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--source',
        choices=['merged', 'calculation', 'forecast'],
        default='merged',
        help='Dataset to save. Default: merged.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output CSV if it already exists.',
    )
    parser.add_argument(
        '--skip-details',
        action='store_true',
        help='Skip per-stock detail requests. Faster, but fewer dividend fields.',
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


def parse_twse_roc_date(value):
    '''
    Convert TWSE ROC date text, such as 115年06月01日, to YYYY-MM-DD.
    '''
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    match = re.match(r'^(\d{2,3})年(\d{1,2})月(\d{1,2})日$', text)
    if match:
        year, month, day = match.groups()
        return date(int(year) + 1911, int(month), int(day)).isoformat()

    match = re.match(r'^(\d{4})(\d{2})(\d{2})$', text)
    if match:
        year, month, day = match.groups()
        return date(int(year), int(month), int(day)).isoformat()

    return text


def clean_text(value):
    '''
    Strip simple HTML fragments and normalize empty values.
    '''
    if pd.isna(value):
        return pd.NA

    text = re.sub(r'<[^>]+>', '', str(value)).strip()
    text = text.replace('&nbsp;', '').strip()
    if text in {'', '-', 'N/A', '待公告實際收益分配金額'}:
        return pd.NA

    return text


def clean_number(value):
    '''
    Convert TWSE numeric text to a float, preserving missing values.
    '''
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA

    text = str(text).replace(',', '')
    return pd.to_numeric(text, errors='coerce')


def parse_first_number(value):
    '''
    Extract the first numeric token from TWSE detail text.
    '''
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA

    match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
    if not match:
        return pd.NA

    return clean_number(match.group(0))


def is_listed_common_stock_id(stock_id):
    '''
    Return True for TWSE listed common stock codes.

    TWSE ex-right/ex-dividend reports also include ETFs, bonds, and other
    instruments. This project only analyzes listed stocks, so keep ordinary
    4-digit numeric stock ids and exclude the 00xx product-code range.
    '''
    text = str(stock_id).strip()
    return bool(re.match(r'^\d{4}$', text)) and not text.startswith('0')


def get_output_path(source, start_date, end_date):
    '''
    Return the output CSV path for the requested source.
    '''
    os.makedirs(DATA_DIR, exist_ok=True)

    if source == 'forecast':
        return os.path.join(DATA_DIR, 'twse_ex_right_dividend_forecast.csv')

    start_text = format_twse_date(start_date)
    end_text = format_twse_date(end_date)
    return os.path.join(
        DATA_DIR,
        f'twse_ex_right_dividend_{source}_{start_text}_to_{end_text}.csv',
    )


def sleep_between_requests():
    '''
    Sleep for a randomized polite throttle interval.
    '''
    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))


def get_json_response(url, params):
    '''
    Fetch one TWSE JSON response with retries.
    '''
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            payload = response.json()
            return payload
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            print(f'Fetch failed (attempt {attempt}/{MAX_RETRIES}): {exc}')
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def require_ok(payload):
    '''
    Validate a TWSE payload and return an empty payload for no-data responses.
    '''
    stat = str(payload.get('stat', ''))
    if stat == 'OK':
        return

    no_data_markers = ['無資料', '查無資料', '很抱歉', '沒有符合條件']
    if any(marker in stat for marker in no_data_markers):
        payload['data'] = []
        return

    raise ValueError(f'Unexpected TWSE response status: {stat}')


def fetch_forecast_rows():
    '''
    Fetch current TWSE ex-right/ex-dividend forecast rows.
    '''
    payload = get_json_response(TWSE_TWT48U_URL, {'response': 'json'})
    require_ok(payload)
    return payload.get('data') or []


def fetch_calculation_rows(start_date, end_date):
    '''
    Fetch TWSE ex-right/ex-dividend calculation result rows for a date range.
    '''
    params = {
        'response': 'json',
        'startDate': format_twse_date(start_date),
        'endDate': format_twse_date(end_date),
    }
    payload = get_json_response(TWSE_TWT49U_URL, params)
    require_ok(payload)
    return payload.get('data') or []


def fetch_detail_row(detail_key):
    '''
    Fetch one TWT49U detail payload by the list report detail key.
    '''
    if pd.isna(detail_key):
        return None

    parts = str(detail_key).split(',')
    if len(parts) != 2:
        return None

    params = {
        'response': 'json',
        'STK_NO': parts[0],
        'T1': parts[1],
    }
    payload = get_json_response(TWSE_TWT49U_DETAIL_URL, params)
    stat = str(payload.get('stat') or payload.get('state') or payload.get('status'))
    if not re.match(r'^ok$', stat, flags=re.IGNORECASE):
        raise ValueError(f'Unexpected TWSE detail response status: {stat}')

    data = payload.get('data') or []
    if not data:
        return None

    return data[0]


def normalize_detail_row(raw):
    '''
    Normalize one TWT49U detail row.
    '''
    if not raw or len(raw) < 12:
        return {}

    free_stock_per_thousand = parse_first_number(raw[4])
    cash_capital_per_thousand = parse_first_number(raw[11])

    stock_dividend_rate = pd.NA
    if not pd.isna(free_stock_per_thousand):
        stock_dividend_rate = free_stock_per_thousand / 1000

    cash_capital_increase_rate = pd.NA
    if not pd.isna(cash_capital_per_thousand):
        cash_capital_increase_rate = cash_capital_per_thousand / 1000

    return {
        'cash_dividend': parse_first_number(raw[2]),
        'stock_dividend_rate': stock_dividend_rate,
        'cash_capital_increase_price': parse_first_number(raw[7]),
        'cash_capital_increase_rate': cash_capital_increase_rate,
    }


def enrich_calculation_details(calculation_df):
    '''
    Fill dividend detail fields from the per-stock TWT49U detail endpoint.
    '''
    if calculation_df.empty:
        return calculation_df

    detail_rows = []
    total_rows = len(calculation_df)
    for index, row in calculation_df.iterrows():
        detail_key = row.get('detail_key')
        if pd.isna(detail_key):
            continue

        print(f'Fetching detail {index + 1}/{total_rows}: {detail_key}')
        detail = normalize_detail_row(fetch_detail_row(detail_key))
        if detail:
            detail['ex_date'] = row['ex_date']
            detail['stock_id'] = row['stock_id']
            detail_rows.append(detail)

        sleep_between_requests()

    if not detail_rows:
        return calculation_df

    detail_df = pd.DataFrame(detail_rows)
    return calculation_df.merge(
        detail_df,
        how='left',
        on=['ex_date', 'stock_id'],
        suffixes=('', '_detail'),
    )


def normalize_forecast_rows(rows):
    '''
    Normalize TWT48U forecast rows.
    '''
    normalized = []
    for raw in rows:
        if len(raw) < 9 or not is_listed_common_stock_id(raw[1]):
            continue

        normalized.append({
            'ex_date': parse_twse_roc_date(raw[0]),
            'stock_id': str(raw[1]).strip(),
            'stock_name': clean_text(raw[2]),
            'right_or_dividend': clean_text(raw[3]),
            'stock_dividend_rate': clean_number(raw[4]),
            'cash_capital_increase_rate': clean_number(raw[5]),
            'cash_capital_increase_price': clean_number(raw[6]),
            'cash_dividend': clean_number(raw[7]),
            'detail_key': clean_text(raw[8]),
        })

    return pd.DataFrame(normalized)


def normalize_calculation_rows(rows):
    '''
    Normalize TWT49U calculation-result rows.
    '''
    normalized = []
    for raw in rows:
        if len(raw) < 12 or not is_listed_common_stock_id(raw[1]):
            continue

        normalized.append({
            'ex_date': parse_twse_roc_date(raw[0]),
            'stock_id': str(raw[1]).strip(),
            'stock_name': clean_text(raw[2]),
            'previous_close': clean_number(raw[3]),
            'ex_reference_price': clean_number(raw[4]),
            'dividend_value': clean_number(raw[5]),
            'right_or_dividend': clean_text(raw[6]),
            'limit_up_price': clean_number(raw[7]),
            'limit_down_price': clean_number(raw[8]),
            'opening_auction_base': clean_number(raw[9]),
            'deducted_dividend_reference_price': clean_number(raw[10]),
            'detail_key': clean_text(raw[11]),
        })

    return pd.DataFrame(normalized)


def merge_sources(calculation_df, forecast_df):
    '''
    Merge TWT49U calculation results with TWT48U dividend detail fields.
    '''
    if calculation_df.empty:
        return calculation_df
    if forecast_df.empty:
        return calculation_df

    merged = calculation_df.merge(
        forecast_df.drop(columns=['stock_name', 'right_or_dividend']),
        how='left',
        on=['ex_date', 'stock_id'],
        suffixes=('', '_forecast'),
    )
    return merged


def coalesce_columns(df, target, candidates):
    '''
    Fill a target column from candidate columns in order.
    '''
    if target not in df.columns:
        df[target] = pd.NA

    for candidate in candidates:
        if candidate in df.columns:
            df[target] = df[target].combine_first(df[candidate])


def finalize_dataframe(df):
    '''
    Return the final stable CSV schema.
    '''
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    if 'opening_reference_price' not in df.columns:
        df['opening_reference_price'] = pd.NA

    coalesce_columns(df, 'cash_dividend', [
        'cash_dividend_detail',
        'cash_dividend_forecast',
    ])
    coalesce_columns(df, 'stock_dividend_rate', [
        'stock_dividend_rate_detail',
        'stock_dividend_rate_forecast',
    ])
    coalesce_columns(df, 'cash_capital_increase_price', [
        'cash_capital_increase_price_detail',
        'cash_capital_increase_price_forecast',
    ])
    coalesce_columns(df, 'cash_capital_increase_rate', [
        'cash_capital_increase_rate_detail',
        'cash_capital_increase_rate_forecast',
    ])

    for column in FINAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.drop_duplicates(subset=['ex_date', 'stock_id'], keep='last')
    df = df.sort_values(['ex_date', 'stock_id']).reset_index(drop=True)
    return df[FINAL_COLUMNS]


def download(source, start_date, end_date, include_details=True):
    '''
    Download and normalize the requested source.
    '''
    if source == 'forecast':
        return finalize_dataframe(normalize_forecast_rows(fetch_forecast_rows()))

    calculation_df = normalize_calculation_rows(
        fetch_calculation_rows(start_date, end_date)
    )
    if include_details:
        calculation_df = enrich_calculation_details(calculation_df)

    if source == 'calculation':
        return finalize_dataframe(calculation_df)

    sleep_between_requests()
    forecast_df = normalize_forecast_rows(fetch_forecast_rows())
    return finalize_dataframe(merge_sources(calculation_df, forecast_df))


def validate_date_range(start_date, end_date):
    '''
    Validate command line date inputs.
    '''
    if start_date > end_date:
        raise ValueError('start-date must be earlier than or equal to end-date.')


def main():
    '''
    Download TWSE ex-right/ex-dividend data and save a normalized CSV.
    '''
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    validate_date_range(start_date, end_date)

    output_path = get_output_path(args.source, start_date, end_date)
    if os.path.exists(output_path) and not args.force:
        print(f'Output already exists: {output_path}')
        print('Use --force to overwrite it.')
        return

    df = download(args.source, start_date, end_date, include_details=not args.skip_details)
    if df.empty:
        raise ValueError('No TWSE ex-right/ex-dividend rows were downloaded.')

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'rows_saved={len(df)}')
    print(f'output_path={output_path}')


if __name__ == '__main__':
    main()
