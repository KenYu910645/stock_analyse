'''
wantgoo_broker_rank.py

Download WantGoo broker buy/sell rank data shown on:
https://www.wantgoo.com/stock/major-investors/broker-buy-sell-rank

This is short-window rank data, not complete per-stock broker branch history.
'''
import argparse
import json
import os
import random
import time
from datetime import date
from urllib.parse import urlencode

import pandas as pd
from playwright.sync_api import sync_playwright

from column_schema import to_csv_storage


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'broker')
BASE_URL = 'https://www.wantgoo.com'
PAGE_PATH = '/stock/major-investors/broker-buy-sell-rank'
BRANCHES_PATH = '/stock/major-investors/branches-data'
ALIVE_PATH = '/investrue/all-alive'
RANK_PATH = '/stock/major-investors/broker-buy-sell-rank-data'

DEFAULT_DURINGS = [20]
DEFAULT_ORDER_BY = ['count']
DEFAULT_MAJOR_IDS = ['9800']
REQUEST_TIMEOUT_MS = 30000
PAGE_SETTLE_MS = 1500
THROTTLE_MIN_SECONDS = 0.8
THROTTLE_MAX_SECONDS = 1.8

OUTPUT_COLUMNS = [
    'FetchDate',
    'DataDate',
    'DuringDays',
    'OrderBy',
    'BrokerId',
    'BrokerName',
    'BranchId',
    'BranchName',
    'Side',
    'Rank',
    'StockNo',
    'StockName',
    'Market',
    'Type',
    'BuyQuantities',
    'SellQuantities',
    'NetQuantities',
    'Amount',
    'AvgPrice',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download WantGoo broker buy/sell rank CSV data.'
    )
    parser.add_argument(
        '--durings',
        nargs='+',
        type=int,
        default=DEFAULT_DURINGS,
        choices=[1, 5, 10, 20],
        help='Window sizes to download. Default: 20.',
    )
    parser.add_argument(
        '--order-by',
        nargs='+',
        default=DEFAULT_ORDER_BY,
        choices=['count', 'amount'],
        help='Ranking sort keys to download. Default: count.',
    )
    parser.add_argument(
        '--major-ids',
        nargs='+',
        default=DEFAULT_MAJOR_IDS,
        help=(
            'Broker ids to download. Use "all" for all broker main ids. '
            'Default: 9800, matching the pasted WantGoo URL.'
        ),
    )
    parser.add_argument(
        '--branch-ids',
        nargs='+',
        default=[''],
        help='Branch ids to download. Empty/default means all branches for each broker.',
    )
    parser.add_argument(
        '--include-non-listed',
        action='store_true',
        help='Keep OTC, ETFs, indexes, and other non-listed-stock rows.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path. Default: data/broker/wantgoo_broker_rank_<date>.csv.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output CSV if it already exists.',
    )
    return parser.parse_args()


def get_output_path(args):
    if args.output:
        return os.path.abspath(args.output)

    fetch_date = date.today().strftime('%Y%m%d')
    durations = '-'.join(str(value) for value in args.durings)
    order_by = '-'.join(args.order_by)
    major_part = 'all' if args.major_ids == ['all'] else '-'.join(args.major_ids)
    filename = (
        f'wantgoo_broker_rank_{fetch_date}_'
        f'during_{durations}_order_{order_by}_major_{major_part}.csv'
    )
    return os.path.join(OUTPUT_DIR, filename)


def create_context(playwright):
    browser = playwright.chromium.launch(
        headless=True,
        args=['--disable-blink-features=AutomationControlled'],
    )
    context = browser.new_context(
        user_agent=(
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/125.0.0.0 Safari/537.36'
        ),
        locale='zh-TW',
        viewport={'width': 1365, 'height': 768},
        extra_http_headers={
            'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'sec-ch-ua': (
                '"Google Chrome";v="125", "Chromium";v="125", '
                '"Not.A/Brand";v="24"'
            ),
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        },
    )
    page = context.new_page()
    page.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return browser, page


def sleep_between_requests():
    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))


def load_initial_page(page):
    query = urlencode({
        'during': 20,
        'majorId': DEFAULT_MAJOR_IDS[0],
        'orderBy': DEFAULT_ORDER_BY[0],
    })
    page.goto(
        f'{BASE_URL}{PAGE_PATH}?{query}',
        wait_until='load',
        timeout=REQUEST_TIMEOUT_MS,
    )
    page.wait_for_timeout(PAGE_SETTLE_MS)


def page_json(page, path, params=None, referer=None):
    url = f'{BASE_URL}{path}'
    if params:
        url = f'{url}?{urlencode(params)}'

    goto_kwargs = {
        'wait_until': 'load',
        'timeout': REQUEST_TIMEOUT_MS,
    }
    if referer:
        goto_kwargs['referer'] = referer

    response = page.goto(url, **goto_kwargs)
    if response is None:
        raise RuntimeError(f'No response for {url}')
    if response.status != 200:
        body = page.locator('body').inner_text(timeout=5000)
        raise RuntimeError(f'WantGoo returned HTTP {response.status} for {url}: {body[:200]}')

    text = page.locator('body').inner_text(timeout=5000)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'WantGoo returned non-JSON for {url}: {text[:200]}') from exc


def load_branches(page):
    payload = page_json(page, BRANCHES_PATH)
    branches = payload.get('data', [])
    if not branches:
        raise ValueError('WantGoo branches-data returned no broker records.')
    return branches, payload.get('date')


def load_alive_map(page):
    payload = page_json(page, ALIVE_PATH)
    alive_map = {}
    for item in payload:
        stock_id = str(item.get('id', '')).strip()
        if not stock_id:
            continue
        alive_map[stock_id] = {
            'StockName': item.get('name', ''),
            'Market': item.get('market', ''),
            'Type': item.get('type', ''),
        }
    return alive_map


def get_major_ids(branches, requested_major_ids):
    if requested_major_ids == ['all']:
        return [str(item['id']) for item in branches]
    return [str(value) for value in requested_major_ids]


def get_broker_lookup(branches):
    return {str(item['id']): item for item in branches}


def get_branch_name(broker, branch_id):
    if not branch_id:
        return 'All Branches'

    for branch in broker.get('branches', []):
        if str(branch.get('id')) == str(branch_id):
            return branch.get('name', '')
    return ''


def normalize_rank_rows(
    rows,
    fetch_date,
    data_date,
    during,
    order_by,
    broker,
    branch_id,
    alive_map,
    include_non_listed,
):
    normalized = []
    broker_id = str(broker.get('id', ''))
    broker_name = broker.get('name', '')
    branch_name = get_branch_name(broker, branch_id)

    enriched_rows = []
    for raw in rows:
        stock_no = str(raw.get('stockNo', '')).strip()
        stock_meta = alive_map.get(stock_no, {})
        market = stock_meta.get('Market', '')
        stock_type = stock_meta.get('Type', '')
        if not include_non_listed and not (market == 'Listed' and stock_type == 'Stock'):
            continue

        buy_qty = int(raw.get('buyQuantities') or 0)
        sell_qty = int(raw.get('sellQuantities') or 0)
        amount = float(raw.get('amount') or 0)
        enriched_rows.append({
            'StockNo': stock_no,
            'StockName': stock_meta.get('StockName', ''),
            'Market': market,
            'Type': stock_type,
            'BuyQuantities': buy_qty,
            'SellQuantities': sell_qty,
            'NetQuantities': buy_qty - sell_qty,
            'Amount': amount,
            'AvgPrice': raw.get('avgPrice'),
        })

    buy_rows = [row for row in enriched_rows if row['Amount'] > 0]
    sell_rows = [row for row in enriched_rows if row['Amount'] < 0]

    for side, side_rows in (('buy', buy_rows), ('sell', sell_rows)):
        for rank, row in enumerate(side_rows, start=1):
            normalized.append({
                'FetchDate': fetch_date,
                'DataDate': data_date,
                'DuringDays': during,
                'OrderBy': order_by,
                'BrokerId': broker_id,
                'BrokerName': broker_name,
                'BranchId': branch_id,
                'BranchName': branch_name,
                'Side': side,
                'Rank': rank,
                **row,
            })

    return normalized


def fetch_rank(page, during, order_by, major_id, branch_id):
    params = {
        'during': during,
        'majorId': major_id,
        'orderBy': order_by,
    }
    if branch_id:
        params['branchId'] = branch_id

    page_url = f'{BASE_URL}{PAGE_PATH}?{urlencode(params)}'
    expected_url_part = f'{RANK_PATH}?{urlencode(params)}'

    try:
        with page.expect_response(
            lambda response: expected_url_part in response.url,
            timeout=REQUEST_TIMEOUT_MS,
        ) as response_info:
            page.goto(page_url, wait_until='load', timeout=REQUEST_TIMEOUT_MS)

        response = response_info.value
        if response.status != 200:
            body = response.text()[:200]
            raise RuntimeError(
                f'WantGoo returned HTTP {response.status} for '
                f'{response.url}: {body}'
            )
        return response.json()
    except AttributeError:
        captured = []

        def on_response(response):
            if expected_url_part in response.url:
                captured.append(response)

        page.on('response', on_response)
        page.goto(page_url, wait_until='load', timeout=REQUEST_TIMEOUT_MS)
        page.wait_for_timeout(PAGE_SETTLE_MS)

        if not captured:
            raise RuntimeError(f'WantGoo rank API response was not observed: {page_url}')

        response = captured[-1]
        if response.status != 200:
            body = response.text()[:200]
            raise RuntimeError(
                f'WantGoo returned HTTP {response.status} for '
                f'{response.url}: {body}'
            )
        return response.json()


def download(args):
    output_path = get_output_path(args)
    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f'Output exists: {output_path}. Use --force to overwrite.')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fetch_date = date.today().isoformat()

    with sync_playwright() as playwright:
        browser, page = create_context(playwright)
        try:
            load_initial_page(page)
            branches, data_date = load_branches(page)
            alive_map = load_alive_map(page)
            broker_lookup = get_broker_lookup(branches)
            major_ids = get_major_ids(branches, args.major_ids)

            all_rows = []
            total_jobs = (
                len(major_ids)
                * len(args.branch_ids)
                * len(args.durings)
                * len(args.order_by)
            )
            job_index = 0

            for major_id in major_ids:
                broker = broker_lookup.get(str(major_id))
                if broker is None:
                    print(f'Skip unknown broker id: {major_id}')
                    continue

                for branch_id in args.branch_ids:
                    for during in args.durings:
                        for order_by in args.order_by:
                            job_index += 1
                            label = (
                                f'{job_index}/{total_jobs} '
                                f'major={major_id} branch={branch_id or "all"} '
                                f'during={during} order={order_by}'
                            )
                            print(f'Fetch {label}')
                            rows = fetch_rank(page, during, order_by, major_id, branch_id)
                            all_rows.extend(normalize_rank_rows(
                                rows,
                                fetch_date,
                                data_date,
                                during,
                                order_by,
                                broker,
                                branch_id,
                                alive_map,
                                args.include_non_listed,
                            ))
                            sleep_between_requests()
        finally:
            browser.close()

    df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    to_csv_storage(df, output_path, index=False, encoding='utf-8-sig')
    print(f'Wrote {len(df)} rows to {output_path}')
    return output_path


def main():
    args = parse_args()
    download(args)


if __name__ == '__main__':
    main()
