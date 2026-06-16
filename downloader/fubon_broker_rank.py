'''
fubon_broker_rank.py

Download Fubon eBroker DJ broker/branch buy-sell rank pages.

The page returns top buy/sell stock ranks for one broker branch over a selected
date range. It is not a complete all-stock exchange official report.
'''
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import re
import threading
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

import pandas as pd
import requests
from lxml import html


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'broker', 'fubon')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
BROKER_JS_URL = 'https://fubon-ebrokerdj.fbs.com.tw/z/js/zbrokerjs.djjs'
RANK_URL = 'https://fubon-ebrokerdj.fbs.com.tw/z/zg/zgb/zgb0.djhtm'
THREAD_LOCAL = threading.local()

LISTED_MARKET = '\u4e0a\u5e02'
COMMON_STOCK_TYPE = '\u80a1\u7968'
METRIC_CODES = {
    'volume': 'E',
    'amount': 'B',
}
SIDE_LABELS = {
    'buy': '\u8cb7\u8d85',
    'sell': '\u8ce3\u8d85',
}
OUTPUT_COLUMNS = [
    'Date',
    'Metric',
    'BrokerId',
    'BrokerName',
    'BranchId',
    'BranchName',
    'Side',
    'Rank',
    'Code',
    'Name',
    'Buy',
    'Sell',
    'Net',
    'SourceUrl',
    'FetchedAt',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Fubon eBroker DJ broker branch stock ranks.'
    )
    parser.add_argument(
        '--date',
        default=None,
        help='Single date in YYYY-MM-DD format. Default: today when no date range is set.',
    )
    parser.add_argument(
        '--start-date',
        default=None,
        help='Start date for backward download in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date for backward download in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--max-dates',
        type=int,
        default=None,
        help='Maximum dates to attempt when using start-date/end-date.',
    )
    parser.add_argument(
        '--include-weekends',
        action='store_true',
        help='Include Saturday/Sunday date requests. Default: skip weekends.',
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['volume'],
        choices=sorted(METRIC_CODES),
        help='Metrics to download. Default: volume.',
    )
    parser.add_argument(
        '--broker-ids',
        nargs='+',
        default=None,
        help='Optional broker company ids to limit the download.',
    )
    parser.add_argument(
        '--branch-ids',
        nargs='+',
        default=None,
        help='Optional branch ids to limit the download.',
    )
    parser.add_argument(
        '--include-non-listed',
        action='store_true',
        help='Keep ETFs, OTC, and non-listed rows instead of only listed stocks.',
    )
    parser.add_argument(
        '--throttle-min',
        type=float,
        default=0.05,
        help='Minimum seconds to sleep between requests. Default: 0.05.',
    )
    parser.add_argument(
        '--throttle-max',
        type=float,
        default=0.2,
        help='Maximum seconds to sleep between requests. Default: 0.2.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel branch requests per date. Default: 1.',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print date-level progress and failures.',
    )
    parser.add_argument(
        '--stop-after-empty-dates',
        type=int,
        default=None,
        help='Stop after this many consecutive requested dates write 0 rows.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path. Default: data/broker/fubon/fubon_broker_branch_rank_<date>.csv.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output CSV if it already exists.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def get_query_dates(args):
    if args.start_date or args.end_date:
        start_date = parse_iso_date(args.start_date or date.today().isoformat())
        if args.end_date:
            end_date = parse_iso_date(args.end_date)
        elif args.max_dates:
            end_date = date(1900, 1, 1)
        else:
            end_date = start_date - timedelta(days=30)
        current = start_date
        dates = []
        while current >= end_date:
            if args.include_weekends or current.weekday() < 5:
                dates.append(current)
                if args.max_dates and len(dates) >= args.max_dates:
                    break
            current -= timedelta(days=1)
        return dates

    return [parse_iso_date(args.date or date.today().isoformat())]


def format_param_date(value):
    return f'{value.year}-{value.month}-{value.day}'


def format_file_date(value):
    return value.strftime('%Y%m%d')


def get_output_path(args, query_date):
    if args.output:
        return os.path.abspath(args.output)

    metrics = '-'.join(args.metrics)
    broker_part = 'all' if args.broker_ids is None else '-'.join(args.broker_ids)
    branch_part = 'all' if args.branch_ids is None else '-'.join(args.branch_ids)
    filename = (
        f'fubon_broker_branch_rank_{format_file_date(query_date)}_'
        f'{metrics}_broker_{broker_part}_branch_{branch_part}.csv'
    )
    return os.path.join(OUTPUT_DIR, filename)


def make_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    })
    return session


def get_thread_session():
    session = getattr(THREAD_LOCAL, 'session', None)
    if session is None:
        session = make_session()
        THREAD_LOCAL.session = session
    return session


def sleep_between_requests(throttle_min, throttle_max):
    time.sleep(random.uniform(throttle_min, throttle_max))


def load_listed_stock_lookup():
    df = pd.read_csv(STOCK_METADATA_PATH, dtype=str).fillna('')
    mask = (
        (df['Market'] == LISTED_MARKET)
        & (df['Type'] == COMMON_STOCK_TYPE)
        & df['Code'].str.match(r'^\d{4}$')
    )
    return dict(zip(df.loc[mask, 'Code'], df.loc[mask, 'Name']))


def load_broker_branches(session):
    response = session.get(BROKER_JS_URL, timeout=30)
    response.raise_for_status()
    response.encoding = 'big5'
    match = re.search(r"var g_BrokerList = '([^']+)'", response.text)
    if not match:
        raise ValueError('Unable to locate g_BrokerList in Fubon broker JS.')

    branches = []
    for group in match.group(1).split(';'):
        if not group.strip():
            continue
        parts = group.split('!')
        if not parts or ',' not in parts[0]:
            continue
        broker_id, broker_name = parts[0].split(',', 1)
        for part in parts[1:]:
            if ',' not in part:
                continue
            branch_id, branch_name = part.split(',', 1)
            branches.append({
                'BrokerId': broker_id,
                'BrokerName': broker_name,
                'BranchId': branch_id,
                'BranchName': branch_name,
            })
    return branches


def filter_branches(branches, broker_ids=None, branch_ids=None):
    broker_set = set(broker_ids) if broker_ids else None
    branch_set = set(branch_ids) if branch_ids else None
    filtered = []
    for branch in branches:
        if broker_set and branch['BrokerId'] not in broker_set:
            continue
        if branch_set and branch['BranchId'] not in branch_set:
            continue
        filtered.append(branch)
    return filtered


def build_url(branch, metric, query_date):
    params = {
        'a': branch['BrokerId'],
        'b': branch['BranchId'],
        'c': METRIC_CODES[metric],
        'e': format_param_date(query_date),
        'f': format_param_date(query_date),
    }
    return f'{RANK_URL}?{urlencode(params)}'


def parse_number(value):
    text = str(value or '').strip().replace(',', '')
    if not text:
        return 0
    return int(float(text))


def parse_stock_cell(cell):
    hrefs = cell.xpath('.//a/@href')
    if hrefs:
        match = re.search(r"Link2Stk\('([^']+)'\)", hrefs[0])
        if match:
            code = match.group(1)
            text = ''.join(cell.xpath('.//a//text()')).strip()
            return code, text.replace(code, '', 1).strip()

    cell_html = html.tostring(cell, encoding='unicode')
    match = re.search(r"GenLink2stk\('AS([^']+)','([^']+)'\)", cell_html)
    if match:
        return match.group(1), match.group(2)

    text = ' '.join(cell.xpath('.//text()')).strip()
    match = re.match(r'([0-9A-Z]+)(.*)', text)
    if match:
        return match.group(1), match.group(2).strip()

    return '', text


def parse_rank_table(table, side, branch, metric, query_date, source_url, listed_lookup, include_non_listed):
    rows = []
    fetched_at = datetime.now().isoformat(timespec='seconds')
    rank = 0

    for tr in table.xpath('.//tr'):
        cells = tr.xpath('./td')
        if len(cells) != 4:
            continue

        code, name = parse_stock_cell(cells[0])
        if not code or code in ('\u5238\u5546\u540d\u7a31', SIDE_LABELS['buy'], SIDE_LABELS['sell']):
            continue

        is_listed = code in listed_lookup
        if not include_non_listed and not is_listed:
            continue

        rank += 1
        rows.append({
            'Date': query_date.isoformat(),
            'Metric': metric,
            'BrokerId': branch['BrokerId'],
            'BrokerName': branch['BrokerName'],
            'BranchId': branch['BranchId'],
            'BranchName': branch['BranchName'],
            'Side': side,
            'Rank': rank,
            'Code': code,
            'Name': listed_lookup.get(code, name),
            'Buy': parse_number(' '.join(cells[1].xpath('.//text()')).strip()),
            'Sell': parse_number(' '.join(cells[2].xpath('.//text()')).strip()),
            'Net': parse_number(' '.join(cells[3].xpath('.//text()')).strip()),
            'SourceUrl': source_url,
            'FetchedAt': fetched_at,
        })

    return rows


def fetch_branch_metric(session, branch, metric, query_date, listed_lookup, include_non_listed):
    url = build_url(branch, metric, query_date)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = 'big5'
    doc = html.fromstring(response.text)
    tables = doc.xpath('//table')
    rank_tables = []

    for table in tables:
        text = ' '.join(value.strip() for value in table.xpath('.//text()') if value.strip())
        if SIDE_LABELS['buy'] in text and '\u5dee\u984d' in text:
            rank_tables.append(('buy', table))
        elif SIDE_LABELS['sell'] in text and '\u5dee\u984d' in text:
            rank_tables.append(('sell', table))

    rows = []
    for side, table in rank_tables[-2:]:
        rows.extend(parse_rank_table(
            table,
            side,
            branch,
            metric,
            query_date,
            url,
            listed_lookup,
            include_non_listed,
        ))
    return rows


def fetch_branch_metric_job(branch, metric, query_date, listed_lookup, include_non_listed, throttle_min, throttle_max):
    sleep_between_requests(throttle_min, throttle_max)
    return fetch_branch_metric(
        get_thread_session(),
        branch,
        metric,
        query_date,
        listed_lookup,
        include_non_listed,
    )


def download(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = make_session()
    listed_lookup = load_listed_stock_lookup()
    branches = filter_branches(
        load_broker_branches(session),
        broker_ids=args.broker_ids,
        branch_ids=args.branch_ids,
    )
    if not branches:
        raise ValueError('No Fubon broker branches matched the requested filters.')

    output_paths = []
    query_dates = get_query_dates(args)
    total_jobs = len(query_dates) * len(branches) * len(args.metrics)
    job_index = 0
    empty_dates = 0
    print(
        f'Loaded {len(branches)} branches, {len(listed_lookup)} listed stocks, '
        f'{len(query_dates)} dates.'
    )

    for query_date in query_dates:
        output_path = get_output_path(args, query_date)
        if os.path.exists(output_path) and not args.force:
            print(f'Skip existing {output_path}')
            output_paths.append(output_path)
            job_index += len(branches) * len(args.metrics)
            continue

        all_rows = []
        jobs = [(branch, metric) for branch in branches for metric in args.metrics]
        print(f'Start {query_date}: {len(jobs)} branch/metric jobs')
        if args.workers <= 1:
            for branch, metric in jobs:
                job_index += 1
                if not args.quiet:
                    print(
                        f'Fetch {job_index}/{total_jobs} {query_date} '
                        f'{branch["BrokerId"]}/{branch["BranchId"]} {metric}'
                    )
                try:
                    all_rows.extend(fetch_branch_metric_job(
                        branch,
                        metric,
                        query_date,
                        listed_lookup,
                        args.include_non_listed,
                        args.throttle_min,
                        args.throttle_max,
                    ))
                except Exception as exc:
                    print(
                        f'Fetch failed {query_date} '
                        f'{branch["BrokerId"]}/{branch["BranchId"]} '
                        f'{metric}: {exc}'
                    )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        fetch_branch_metric_job,
                        branch,
                        metric,
                        query_date,
                        listed_lookup,
                        args.include_non_listed,
                        args.throttle_min,
                        args.throttle_max,
                    ): (branch, metric)
                    for branch, metric in jobs
                }
                for future in as_completed(futures):
                    branch, metric = futures[future]
                    job_index += 1
                    if not args.quiet:
                        print(
                            f'Fetch {job_index}/{total_jobs} {query_date} '
                            f'{branch["BrokerId"]}/{branch["BranchId"]} {metric}'
                        )
                    try:
                        all_rows.extend(future.result())
                    except Exception as exc:
                        print(
                            f'Fetch failed {query_date} '
                            f'{branch["BrokerId"]}/{branch["BranchId"]} '
                            f'{metric}: {exc}'
                        )

        df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f'Wrote {len(df)} rows to {output_path}')
        output_paths.append(output_path)
        if args.stop_after_empty_dates:
            if len(df) == 0:
                empty_dates += 1
                print(f'Empty date streak: {empty_dates}/{args.stop_after_empty_dates}')
                if empty_dates >= args.stop_after_empty_dates:
                    print('Stop after consecutive empty dates.')
                    break
            else:
                empty_dates = 0

    return output_paths


def main():
    args = parse_args()
    download(args)


if __name__ == '__main__':
    main()
