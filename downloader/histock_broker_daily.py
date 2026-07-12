'''
histock_broker_daily.py

Download free HiStock broker branch buy/sell daily rank pages for listed
Taiwan common stocks.

HiStock's free page exposes the top buy/sell branch ranks for a stock/date.
It is not a complete official all-branch daily report.
'''
import argparse
import os
import random
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from lxml import html

from column_schema import read_csv_canonical, to_csv_storage


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'broker', 'histock_daily')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
BASE_URL = 'https://histock.tw/stock/branch.aspx'
LISTED_MARKET = '\u4e0a\u5e02'
COMMON_STOCK_TYPE = '\u80a1\u7968'
DEFAULT_PROBE_CODES = ['2330', '2317', '2454']
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 4
THROTTLE_MIN_SECONDS = 2.0
THROTTLE_MAX_SECONDS = 4.0

OUTPUT_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'Side',
    'Rank',
    'BrokerName',
    'BrokerId',
    'Buy',
    'Sell',
    'Net',
    'AvgPrice',
    'SourceUrl',
    'FetchedAt',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download HiStock broker branch daily rank data.'
    )
    parser.add_argument(
        '--start-date',
        default=date.today().isoformat(),
        help='Start date in YYYY-MM-DD format. Default: today.',
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date in YYYY-MM-DD format. Default: start-date minus max-back-days.',
    )
    parser.add_argument(
        '--max-back-days',
        type=int,
        default=450,
        help='Maximum calendar days to walk backward when end-date is omitted.',
    )
    parser.add_argument(
        '--max-dates',
        type=int,
        default=None,
        help='Maximum available trading dates to download in this run.',
    )
    parser.add_argument(
        '--probe-codes',
        nargs='+',
        default=DEFAULT_PROBE_CODES,
        help='Stock codes used to detect whether a date has HiStock data.',
    )
    parser.add_argument(
        '--empty-probe-stop',
        type=int,
        default=30,
        help='Stop after this many consecutive probed weekdays have no data.',
    )
    parser.add_argument(
        '--probe-only',
        action='store_true',
        help='Only probe available dates; do not download all listed stocks.',
    )
    parser.add_argument(
        '--stock-codes',
        nargs='+',
        default=None,
        help='Optional stock code subset for testing or targeted downloads.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite date CSVs that already exist.',
    )
    parser.add_argument(
        '--throttle-min',
        type=float,
        default=THROTTLE_MIN_SECONDS,
        help='Minimum seconds to sleep between requests. Default: 2.0.',
    )
    parser.add_argument(
        '--throttle-max',
        type=float,
        default=THROTTLE_MAX_SECONDS,
        help='Maximum seconds to sleep between requests. Default: 4.0.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def histock_date(value):
    return value.strftime('%Y%m%d')


def output_path_for_date(value):
    return os.path.join(OUTPUT_DIR, f'histock_broker_daily_{histock_date(value)}.csv')


def iter_weekdays_desc(start_date, end_date):
    current = start_date
    while current >= end_date:
        if current.weekday() < 5:
            yield current
        current -= timedelta(days=1)


def load_listed_stocks(stock_codes=None):
    df = read_csv_canonical(STOCK_METADATA_PATH, dtype=str).fillna('')
    mask = (
        (df['Market'] == LISTED_MARKET)
        & (df['Type'] == COMMON_STOCK_TYPE)
        & df['Code'].str.match(r'^\d{4}$')
    )
    listed = df.loc[mask, ['Code', 'Name']].copy()
    listed['Code'] = listed['Code'].astype(str)

    if stock_codes:
        requested = set(str(code) for code in stock_codes)
        listed = listed[listed['Code'].isin(requested)]

    if listed.empty:
        raise ValueError('No listed stock codes matched the requested filters.')

    return listed.sort_values('Code').to_dict('records')


def make_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/125.0.0.0 Safari/537.36'
        ),
        'Accept': (
            'text/html,application/xhtml+xml,application/xml;q=0.9,'
            'image/avif,image/webp,*/*;q=0.8'
        ),
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
    })
    return session


def sleep_between_requests(throttle_min, throttle_max):
    time.sleep(random.uniform(throttle_min, throttle_max))


def parse_number(value, default=0):
    text = str(value or '').strip().replace(',', '')
    if text == '':
        return default
    return int(float(text))


def parse_float(value):
    text = str(value or '').strip().replace(',', '')
    if text == '':
        return None
    return float(text)


def extract_broker_id(href):
    if not href:
        return ''
    match = re.search(r'[?&]bno=([^&]+)', href)
    return match.group(1) if match else ''


def build_url(stock_code, query_date):
    day = histock_date(query_date)
    return f'{BASE_URL}?no={stock_code}&from={day}&to={day}'


def fetch_html(session, stock_code, query_date):
    url = build_url(stock_code, query_date)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return url, response.text
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def parse_branch_rows(html_text, stock_code, stock_name, query_date, source_url):
    doc = html.fromstring(html_text)
    table_rows = doc.xpath('//tr[.//a[contains(@href, "brokertrace")]]')
    output_rows = []
    ranks = {'sell': 0, 'buy': 0}
    fetched_at = datetime.now().isoformat(timespec='seconds')
    row_date = query_date.isoformat()

    for table_row in table_rows:
        cells = table_row.xpath('./td')
        if len(cells) < 10:
            continue

        left = parse_side_cells(cells[:5], 'sell')
        right = parse_side_cells(cells[5:10], 'buy')

        for side_row in (left, right):
            if side_row is None:
                continue
            side = side_row['Side']
            ranks[side] += 1
            output_rows.append({
                'Date': row_date,
                'Code': stock_code,
                'Name': stock_name,
                'Side': side,
                'Rank': ranks[side],
                'SourceUrl': source_url,
                'FetchedAt': fetched_at,
                **side_row,
            })

    return output_rows


def parse_side_cells(cells, side):
    broker_name = ' '.join(cells[0].xpath('.//text()')).strip()
    if not broker_name:
        return None

    hrefs = cells[0].xpath('.//a/@href')
    values = [' '.join(cell.xpath('.//text()')).strip() for cell in cells]
    buy = parse_number(values[1])
    sell = parse_number(values[2])
    net = parse_number(values[3])
    avg_price = parse_float(values[4])

    return {
        'Side': side,
        'BrokerName': broker_name,
        'BrokerId': extract_broker_id(hrefs[0] if hrefs else ''),
        'Buy': buy,
        'Sell': sell,
        'Net': net,
        'AvgPrice': avg_price,
    }


def fetch_stock_rows(session, stock, query_date):
    url, html_text = fetch_html(session, stock['Code'], query_date)
    return parse_branch_rows(html_text, stock['Code'], stock['Name'], query_date, url)


def date_has_data(session, query_date, probe_codes, throttle_min, throttle_max):
    for code in probe_codes:
        stock = {'Code': str(code), 'Name': ''}
        try:
            rows = fetch_stock_rows(session, stock, query_date)
        except Exception as exc:
            print(f'Probe failed for {query_date} {code}: {exc}')
            rows = []
        sleep_between_requests(throttle_min, throttle_max)
        if rows:
            return True
    return False


def write_rows(output_path, rows):
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    to_csv_storage(df, output_path, index=False, encoding='utf-8-sig')
    return len(df)


def download_date(session, stocks, query_date, force=False, throttle_min=2.0, throttle_max=4.0):
    output_path = output_path_for_date(query_date)
    if os.path.exists(output_path) and not force:
        print(f'Skip existing {output_path}')
        return 'skipped', output_path, 0

    all_rows = []
    total = len(stocks)
    for index, stock in enumerate(stocks, start=1):
        try:
            rows = fetch_stock_rows(session, stock, query_date)
            all_rows.extend(rows)
        except Exception as exc:
            print(f'Fetch failed {query_date} {stock["Code"]}: {exc}')

        if index % 50 == 0 or index == total:
            print(
                f'  {query_date} stocks {index}/{total}, '
                f'rows={len(all_rows)}'
            )

        sleep_between_requests(throttle_min, throttle_max)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    row_count = write_rows(output_path, all_rows)
    print(f'Wrote {row_count} rows to {output_path}')
    return 'written', output_path, row_count


def main():
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = (
        parse_iso_date(args.end_date)
        if args.end_date
        else start_date - timedelta(days=args.max_back_days)
    )
    stocks = load_listed_stocks(args.stock_codes)
    session = make_session()

    print(f'Loaded {len(stocks)} listed stocks.')
    available_dates = 0
    empty_probe_dates = 0

    for query_date in iter_weekdays_desc(start_date, end_date):
        print(f'Probe {query_date}')
        if not date_has_data(
            session,
            query_date,
            args.probe_codes,
            args.throttle_min,
            args.throttle_max,
        ):
            empty_probe_dates += 1
            print(f'No probe data for {query_date} (streak={empty_probe_dates}).')
            if available_dates > 0 and empty_probe_dates >= args.empty_probe_stop:
                print('Stop: consecutive empty probe date limit reached.')
                break
            continue

        empty_probe_dates = 0
        available_dates += 1
        print(f'Available date {query_date}')

        if args.probe_only:
            if args.max_dates and available_dates >= args.max_dates:
                break
            continue

        download_date(
            session,
            stocks,
            query_date,
            force=args.force,
            throttle_min=args.throttle_min,
            throttle_max=args.throttle_max,
        )

        if args.max_dates and available_dates >= args.max_dates:
            print('Stop: max available dates reached.')
            break

    print(f'Available dates found: {available_dates}')


if __name__ == '__main__':
    main()
