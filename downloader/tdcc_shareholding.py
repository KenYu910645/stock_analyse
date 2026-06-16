'''
tdcc_shareholding.py

Download TDCC shareholding distribution data.
'''
import argparse
import csv
import io
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from html.parser import HTMLParser

import pandas as pd
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'shareholding')
RAW_DIR = os.path.join(OUTPUT_DIR, 'raw')
LISTED_DIR = os.path.join(OUTPUT_DIR, 'listed')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

TDCC_OPEN_DATA_URL = 'https://opendata.tdcc.com.tw/getOD.ashx?id=1-5'
TDCC_QRY_STOCK_URL = 'https://www.tdcc.com.tw/portal/zh/smWeb/qryStock'

COMMON_STOCK_TYPE = '\u80a1\u7968'
TWSE_MARKET = '\u4e0a\u5e02'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'stock_analyse/1.0'
    ),
}

OUTPUT_COLUMNS = [
    '資料日期',
    '證券代號',
    '證券名稱',
    '持股分級',
    '持股/單位數分級',
    '人數',
    '股數',
    '占集保庫存數比例%',
]

HOLDING_LEVEL_LABELS = {
    1: '1-999',
    2: '1,000-5,000',
    3: '5,001-10,000',
    4: '10,001-15,000',
    5: '15,001-20,000',
    6: '20,001-30,000',
    7: '30,001-40,000',
    8: '40,001-50,000',
    9: '50,001-100,000',
    10: '100,001-200,000',
    11: '200,001-400,000',
    12: '400,001-600,000',
    13: '600,001-800,000',
    14: '800,001-1,000,000',
    15: '1,000,001以上',
    16: '差異數調整（說明4）',
    17: '合計',
}

THREAD_STATE = threading.local()


class TdccFormParser(HTMLParser):
    '''
    Extract synchronizer token, available dates, and result table rows.
    '''
    def __init__(self):
        super().__init__()
        self.token = None
        self.dates = []
        self._in_option = False
        self._option_value = None
        self._in_td = False
        self._current_cell = []
        self._current_row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'input' and attrs.get('name') == 'SYNCHRONIZER_TOKEN':
            self.token = attrs.get('value')
        elif tag == 'option':
            self._in_option = True
            self._option_value = attrs.get('value')
        elif tag == 'tr':
            self._current_row = []
        elif tag == 'td':
            self._in_td = True
            self._current_cell = []

    def handle_endtag(self, tag):
        if tag == 'option' and self._in_option:
            value = self._option_value or ''.join(self._current_cell).strip()
            if re.fullmatch(r'\d{8}', value or ''):
                self.dates.append(value)
            self._in_option = False
            self._option_value = None
        elif tag == 'td' and self._in_td:
            text = ' '.join(''.join(self._current_cell).split())
            self._current_row.append(text)
            self._in_td = False
        elif tag == 'tr':
            if len(self._current_row) == 5 and self._current_row[0].isdigit():
                self.rows.append(self._current_row)

    def handle_data(self, data):
        if self._in_td or self._in_option:
            self._current_cell.append(data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download TDCC shareholding distribution data.'
    )
    parser.add_argument(
        '--dates',
        choices=('all', 'latest'),
        default='all',
        help='Download all dates currently exposed by TDCC, or only latest.',
    )
    parser.add_argument(
        '--skip-history',
        action='store_true',
        help='Only download the latest official open-data CSV.',
    )
    parser.add_argument(
        '--stock-limit',
        type=int,
        default=None,
        help='Limit listed stock count for a quick smoke test.',
    )
    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=0.15,
        help='Polite delay between per-stock form requests.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for per-stock history downloads.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing per-date CSV files.',
    )
    return parser.parse_args()


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LISTED_DIR, exist_ok=True)


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def read_parser(html):
    parser = TdccFormParser()
    parser.feed(html)
    return parser


def load_listed_stocks():
    df_metadata = pd.read_csv(STOCK_METADATA_PATH, dtype={'Code': str})
    df_metadata['Code'] = df_metadata['Code'].astype(str).str.strip()
    mask = (
        (df_metadata['Type'] == COMMON_STOCK_TYPE)
        & (df_metadata['Market'] == TWSE_MARKET)
    )
    df_listed = df_metadata.loc[mask, ['Code', 'Name']].sort_values('Code')
    return list(df_listed.itertuples(index=False, name=None))


def normalize_tdcc_code(value):
    '''
    Convert TDCC padded stock codes like 002330 back to project stock codes.
    '''
    text = str(value).strip()
    if len(text) == 6 and text.startswith('00'):
        return text[-4:]
    return text


def add_holding_level_labels(df):
    levels = pd.to_numeric(df['持股分級'], errors='coerce')
    df.insert(
        df.columns.get_loc('持股分級') + 1,
        '持股/單位數分級',
        levels.map(HOLDING_LEVEL_LABELS),
    )
    return df


def adjust_difference_rows(df):
    '''
    Recompute level 16 as level 17 total minus levels 1-15.

    TDCC open data exposes level 16 as the adjustment row, but some rows do not
    carry the sign shown on the query page.  The arithmetic definition is more
    useful for analysis.
    '''
    levels = pd.to_numeric(df['持股分級'], errors='coerce')
    for col in ('人數', '股數', '占集保庫存數比例%'):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for _code, group in df.groupby('證券代號', sort=False):
        detail_mask = group.index[levels.loc[group.index].between(1, 15)]
        diff_index = group.index[levels.loc[group.index] == 16]
        total_index = group.index[levels.loc[group.index] == 17]
        if len(diff_index) != 1 or len(total_index) != 1:
            continue

        diff_idx = diff_index[0]
        total_idx = total_index[0]
        holders_diff = df.at[total_idx, '人數'] - df.loc[detail_mask, '人數'].sum()
        shares_diff = df.at[total_idx, '股數'] - df.loc[detail_mask, '股數'].sum()
        total_shares = df.at[total_idx, '股數']
        ratio_diff = (shares_diff / total_shares * 100) if total_shares else 0

        df.at[diff_idx, '人數'] = holders_diff
        df.at[diff_idx, '股數'] = shares_diff
        df.at[diff_idx, '占集保庫存數比例%'] = round(ratio_diff, 2)

    df['人數'] = df['人數'].round().astype('Int64')
    df['股數'] = df['股數'].round().astype('Int64')
    return df


def load_page_state(session):
    response = session.get(TDCC_QRY_STOCK_URL, timeout=30)
    response.raise_for_status()
    response.encoding = 'utf-8'
    parser = read_parser(response.text)
    if not parser.token:
        raise RuntimeError('TDCC page did not include a synchronizer token.')
    if not parser.dates:
        raise RuntimeError('TDCC page did not include any available dates.')
    return parser.token, parser.dates


def download_latest_open_data(session):
    response = session.get(TDCC_OPEN_DATA_URL, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8-sig'

    df_raw = pd.read_csv(
        io.StringIO(response.text),
        dtype={'資料日期': str, '證券代號': str, '持股分級': str},
    )
    df_raw = add_holding_level_labels(df_raw)
    df_raw = adjust_difference_rows(df_raw)
    latest_date = str(df_raw['資料日期'].iloc[0])
    raw_path = os.path.join(RAW_DIR, f'tdcc_shareholding_all_{latest_date}.csv')
    listed_path = os.path.join(
        LISTED_DIR, f'tdcc_shareholding_listed_{latest_date}.csv'
    )

    df_raw.to_csv(raw_path, index=False, encoding='utf-8-sig')

    listed_codes = {code for code, _name in load_listed_stocks()}
    normalized_codes = df_raw['證券代號'].map(normalize_tdcc_code)
    df_listed = df_raw[normalized_codes.isin(listed_codes)].copy()
    df_listed['證券代號'] = normalized_codes[normalized_codes.isin(listed_codes)]
    df_listed.to_csv(listed_path, index=False, encoding='utf-8-sig')
    return latest_date, raw_path, listed_path, len(df_raw), len(df_listed)


def query_stock(session, token, query_date, stock_code, stock_name):
    payload = {
        'SYNCHRONIZER_TOKEN': token,
        'SYNCHRONIZER_URI': '/portal/zh/smWeb/qryStock',
        'method': 'submit',
        'scaDate': query_date,
        'sqlMethod': 'StockNo',
        'stockNo': stock_code,
        'stockName': '',
    }
    response = session.post(
        TDCC_QRY_STOCK_URL,
        data=payload,
        headers={'Referer': TDCC_QRY_STOCK_URL, **HEADERS},
        timeout=30,
    )
    response.raise_for_status()
    response.encoding = 'utf-8'
    if 'HTTP 403 Forbidden' in response.text or '<title>ERROR :' in response.text:
        raise RuntimeError('TDCC returned an error page for the form request.')

    parser = read_parser(response.text)
    if parser.token:
        token = parser.token

    records = []
    for level, level_text, holders, shares, ratio in parser.rows:
        records.append({
            '資料日期': query_date,
            '證券代號': stock_code,
            '證券名稱': stock_name,
            '持股分級': level,
            '持股/單位數分級': level_text,
            '人數': holders.replace(',', ''),
            '股數': shares.replace(',', ''),
            '占集保庫存數比例%': ratio,
        })
    return token, records


def query_stock_threaded(query_date, stock_code, stock_name):
    session = getattr(THREAD_STATE, 'session', None)
    token = getattr(THREAD_STATE, 'token', None)
    if session is None or token is None:
        session = make_session()
        token, _dates = load_page_state(session)
        THREAD_STATE.session = session
        THREAD_STATE.token = token

    token, records = query_stock(session, token, query_date, stock_code, stock_name)
    THREAD_STATE.token = token
    return records


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def download_history(session, dates, stocks, sleep_seconds, force, workers):
    completed = []
    errors = []
    token, _available_dates = load_page_state(session)

    for index, query_date in enumerate(dates, start=1):
        path = os.path.join(
            LISTED_DIR, f'tdcc_shareholding_listed_{query_date}.csv'
        )
        if os.path.exists(path) and not force:
            print(f'[{index}/{len(dates)}] Skip existing {query_date}: {path}')
            completed.append(path)
            continue

        print(
            f'[{index}/{len(dates)}] Downloading {query_date} '
            f'for {len(stocks)} listed stocks.'
        )
        rows = []
        if workers <= 1:
            for stock_index, (stock_code, stock_name) in enumerate(stocks, start=1):
                try:
                    token, stock_rows = query_stock(
                        session, token, query_date, stock_code, stock_name
                    )
                    rows.extend(stock_rows)
                except Exception as exc:
                    errors.append((query_date, stock_code, str(exc)))
                    print(f'  Error {query_date} {stock_code}: {exc}')

                if stock_index % 100 == 0:
                    print(f'  {query_date}: {stock_index}/{len(stocks)} stocks')

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        query_stock_threaded, query_date, stock_code, stock_name
                    ): stock_code
                    for stock_code, stock_name in stocks
                }
                for stock_index, future in enumerate(as_completed(futures), start=1):
                    stock_code = futures[future]
                    try:
                        rows.extend(future.result())
                    except Exception as exc:
                        errors.append((query_date, stock_code, str(exc)))
                        print(f'  Error {query_date} {stock_code}: {exc}')

                    if stock_index % 100 == 0:
                        print(f'  {query_date}: {stock_index}/{len(stocks)} stocks')

                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

        write_rows(path, rows)
        print(f'  Wrote {len(rows)} rows to {path}')
        completed.append(path)

    if errors:
        error_path = os.path.join(
            OUTPUT_DIR,
            f'tdcc_shareholding_errors_{date.today().strftime("%Y%m%d")}.csv',
        )
        with open(error_path, 'w', newline='', encoding='utf-8-sig') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(['資料日期', '證券代號', '錯誤'])
            writer.writerows(errors)
        print(f'Wrote {len(errors)} errors to {error_path}')

    return completed, errors


def main():
    args = parse_args()
    ensure_dirs()
    session = make_session()

    latest_date, raw_path, listed_path, raw_count, listed_count = (
        download_latest_open_data(session)
    )
    print(
        f'Downloaded latest open data {latest_date}: '
        f'{raw_count} raw rows, {listed_count} listed rows.'
    )
    print(f'Raw latest path: {raw_path}')
    print(f'Listed latest path: {listed_path}')
    if args.skip_history:
        return

    token, dates = load_page_state(session)
    del token
    if args.dates == 'latest':
        dates = dates[:1]

    stocks = load_listed_stocks()
    if args.stock_limit is not None:
        stocks = stocks[:args.stock_limit]

    completed, errors = download_history(
        session=session,
        dates=dates,
        stocks=stocks,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
        workers=args.workers,
    )
    print(f'Completed {len(completed)} historical date files.')
    print(f'Historical output directory: {LISTED_DIR}')
    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
