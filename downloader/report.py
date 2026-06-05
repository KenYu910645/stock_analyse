'''
report.py

Download MOPSFIN three-statement reports for one stock into a normalized CSV.
'''
import argparse
import html
import os
import random
import re
import time
from datetime import date

import pandas as pd
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'report')

MOPSFIN_BASE_URL = 'https://mopsfin.twse.com.tw'
MOPSFIN_REPORT_URL = f'{MOPSFIN_BASE_URL}/compare/report'

DEFAULT_STOCK_CODE = '2330'
DEFAULT_STOCK_NAME = '\u53f0\u7a4d\u96fb'
DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = date.today().year
DEFAULT_METADATA_PATH = os.path.join(DATA_DIR, 'stock_metadata.csv')
TWSE_MARKET = '\u4e0a\u5e02'

STATEMENTS = {
    'BalanceSheet': '\u8cc7\u7522\u8ca0\u50b5\u8868',
    'IncomeStatement': '\u7d9c\u5408\u640d\u76ca\u8868',
    'CashflowStatement': '\u73fe\u91d1\u6d41\u91cf\u8868',
}

OUTPUT_COLUMNS = [
    'Code',
    'Name',
    'Year',
    'Quarter',
    'Statement',
    'Account',
    'Value',
    'Unit',
]

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
THROTTLE_MIN_SECONDS = 1.0
THROTTLE_MAX_SECONDS = 2.0

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/125.0 Safari/537.36'
    ),
    'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8',
}

AJAX_HEADERS = {
    **HEADERS,
    'Accept': 'text/html, */*; q=0.01',
    'Origin': MOPSFIN_BASE_URL,
    'Referer': f'{MOPSFIN_BASE_URL}/',
    'X-Requested-With': 'XMLHttpRequest',
}


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Download MOPSFIN balance sheet, income statement, and cash flow.'
    )
    parser.add_argument(
        '--stock',
        default=DEFAULT_STOCK_CODE,
        help=f'Stock code. Default: {DEFAULT_STOCK_CODE}.',
    )
    parser.add_argument(
        '--name',
        default=DEFAULT_STOCK_NAME,
        help=f'Stock name. Default: {DEFAULT_STOCK_NAME}.',
    )
    parser.add_argument(
        '--all-stocks',
        action='store_true',
        help='Download reports for all TWSE listed stocks in stock_metadata.csv.',
    )
    parser.add_argument(
        '--metadata',
        default=DEFAULT_METADATA_PATH,
        help=f'Stock metadata CSV. Default: {DEFAULT_METADATA_PATH}.',
    )
    parser.add_argument(
        '--max-stocks',
        type=int,
        default=None,
        help='Optional cap for testing all-stock mode.',
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=DEFAULT_START_YEAR,
        help=f'Start year. Default: {DEFAULT_START_YEAR}.',
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=DEFAULT_END_YEAR,
        help=f'End year. Default: {DEFAULT_END_YEAR}.',
    )
    parser.add_argument(
        '--end-quarter',
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help='Last quarter to download in end-year. Default: 4.',
    )
    parser.add_argument(
        '--reports',
        nargs='+',
        choices=sorted(STATEMENTS.keys()),
        default=list(STATEMENTS.keys()),
        help='Report types to download.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the output CSV if it already exists.',
    )
    return parser.parse_args()


def validate_args(args):
    '''
    Validate command line arguments.
    '''
    if args.start_year > args.end_year:
        raise ValueError('start-year must be earlier than or equal to end-year.')
    if args.max_stocks is not None and args.max_stocks <= 0:
        raise ValueError('max-stocks must be greater than zero.')


def get_output_path(stock_code):
    '''
    Return the normalized per-stock report CSV path.
    '''
    return os.path.join(OUTPUT_DIR, f'{stock_code}.csv')


def load_twse_stock_catalog(metadata_path, max_stocks=None):
    '''
    Load TWSE listed stock codes and names from stock_metadata.csv.
    '''
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Stock metadata CSV does not exist: {metadata_path}')

    df = pd.read_csv(metadata_path, dtype={'Code': str})
    required_columns = {'Code', 'Name', 'Market'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f'Metadata CSV is missing columns: {sorted(missing_columns)}')

    df = df[df['Market'].eq(TWSE_MARKET)].copy()
    df = df[['Code', 'Name']].dropna().drop_duplicates('Code')
    df = df.sort_values('Code').reset_index(drop=True)

    if max_stocks is not None:
        df = df.head(max_stocks)

    return list(df.itertuples(index=False, name=None))


def sleep_between_requests():
    '''
    Sleep for a randomized polite throttle interval.
    '''
    time.sleep(random.uniform(THROTTLE_MIN_SECONDS, THROTTLE_MAX_SECONDS))


def create_session():
    '''
    Create a browser-like MOPSFIN session.
    '''
    session = requests.Session()
    response = session.get(
        f'{MOPSFIN_BASE_URL}/',
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return session


def strip_html_tags(value):
    '''
    Remove HTML tags and normalize whitespace.
    '''
    text = re.sub(r'<br\s*/?>', ' ', value, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = text.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def extract_tables(response_text):
    '''
    Return all HTML table blocks from a MOPSFIN report response.
    '''
    return re.findall(r'<table[\s\S]*?</table>', response_text, flags=re.IGNORECASE)


def extract_cells(row_html):
    '''
    Return text from th/td cells in a table row.
    '''
    cells = re.findall(r'<t[hd][^>]*>([\s\S]*?)</t[hd]>', row_html, flags=re.IGNORECASE)
    return [strip_html_tags(cell) for cell in cells]


def extract_table_rows(table_html):
    '''
    Return cell text rows from one HTML table.
    '''
    rows = re.findall(r'<tr[^>]*>([\s\S]*?)</tr>', table_html, flags=re.IGNORECASE)
    return [extract_cells(row) for row in rows]


def parse_number(value):
    '''
    Convert comma-formatted MOPSFIN values to numbers where possible.
    '''
    text = str(value).strip()
    if not text or text in {'-', '--'}:
        return pd.NA

    is_parenthesized_negative = text.startswith('(') and text.endswith(')')
    normalized = text.strip('()').replace(',', '')

    try:
        number = float(normalized)
    except ValueError:
        return text

    if is_parenthesized_negative:
        number = -number

    if number.is_integer():
        return int(number)

    return number


def parse_report_html(response_text, stock_code, stock_name, year, quarter, statement):
    '''
    Parse one MOPSFIN report response into normalized row dictionaries.
    '''
    tables = extract_tables(response_text)
    if len(tables) < 2:
        raise ValueError('MOPSFIN response did not include the expected report tables.')

    account_rows = extract_table_rows(tables[0])[2:]
    value_rows = extract_table_rows(tables[1])[2:]

    accounts = [row[0] for row in account_rows if row]
    values = [row[0] for row in value_rows if row]

    if len(accounts) != len(values):
        raise ValueError(
            f'Account/value row mismatch for {statement} {year}Q{quarter}: '
            f'{len(accounts)} accounts, {len(values)} values.'
        )

    rows = []
    for account, value in zip(accounts, values):
        rows.append({
            'Code': stock_code,
            'Name': stock_name,
            'Year': year,
            'Quarter': quarter,
            'Statement': statement,
            'Account': account,
            'Value': parse_number(value),
            'Unit': '\u65b0\u53f0\u5e63\u4edf\u5143',
        })

    return rows


def fetch_report(session, stock_code, stock_name, year, quarter, statement):
    '''
    Fetch one stock/year/quarter/statement report with retries.
    '''
    data = {
        'compareItem': statement,
        'quarter': 'false',
        'ylabel': '',
        'ys': f'{year}{quarter}',
        'revenue': 'false',
        'bcodeAvg': 'false',
        'companyAvg': 'false',
        'qnumber': '',
        'companyId': f'{stock_code} {stock_name}',
    }
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                MOPSFIN_REPORT_URL,
                data=data,
                headers=AJAX_HEADERS,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = 'utf-8'

            if 'THE PAGE CANNOT BE ACCESSED' in response.text:
                raise ValueError('MOPSFIN returned a security block page.')

            rows = parse_report_html(
                response.text,
                stock_code,
                stock_name,
                year,
                quarter,
                statement,
            )

            if not rows:
                raise ValueError('Parsed zero report rows.')

            return rows
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            print(
                f'Fetch failed for {stock_code} {year}Q{quarter} {statement} '
                f'(attempt {attempt}/{MAX_RETRIES}): {exc}'
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def iter_year_quarters(start_year, end_year, end_quarter=4):
    '''
    Yield all year/quarter pairs in the requested range.
    '''
    for year in range(start_year, end_year + 1):
        last_quarter = end_quarter if year == end_year else 4
        for quarter in range(1, last_quarter + 1):
            yield year, quarter


def download_reports(
    stock_code,
    stock_name,
    start_year,
    end_year,
    end_quarter,
    statements,
):
    '''
    Download selected MOPSFIN reports into normalized rows.
    '''
    session = create_session()
    all_rows = []
    failures = []

    for year, quarter in iter_year_quarters(start_year, end_year, end_quarter=end_quarter):
        for statement in statements:
            print(f'Fetching {stock_code} {year}Q{quarter} {statement}.')

            try:
                rows = fetch_report(
                    session,
                    stock_code,
                    stock_name,
                    year,
                    quarter,
                    statement,
                )
            except Exception as exc:
                failures.append({
                    'Code': stock_code,
                    'Year': year,
                    'Quarter': quarter,
                    'Statement': statement,
                    'Error': str(exc),
                })
                print(f'Failed {stock_code} {year}Q{quarter} {statement}: {exc}')
            else:
                all_rows.extend(rows)
                print(f'Downloaded {len(rows)} rows.')
            finally:
                sleep_between_requests()

    return all_rows, failures


def download_reports_with_session(
    session,
    stock_code,
    stock_name,
    start_year,
    end_year,
    end_quarter,
    statements,
):
    '''
    Download selected MOPSFIN reports using an existing session.
    '''
    all_rows = []
    failures = []

    for year, quarter in iter_year_quarters(start_year, end_year, end_quarter=end_quarter):
        for statement in statements:
            print(f'Fetching {stock_code} {year}Q{quarter} {statement}.')

            try:
                rows = fetch_report(
                    session,
                    stock_code,
                    stock_name,
                    year,
                    quarter,
                    statement,
                )
            except Exception as exc:
                failures.append({
                    'Code': stock_code,
                    'Year': year,
                    'Quarter': quarter,
                    'Statement': statement,
                    'Error': str(exc),
                })
                print(f'Failed {stock_code} {year}Q{quarter} {statement}: {exc}')
            else:
                all_rows.extend(rows)
                print(f'Downloaded {len(rows)} rows.')
            finally:
                sleep_between_requests()

    return all_rows, failures


def write_failures(failures, stock_code):
    '''
    Save failed report requests for debugging.
    '''
    if not failures:
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    failure_path = os.path.join(OUTPUT_DIR, f'{stock_code}_failures.csv')
    pd.DataFrame(failures).to_csv(failure_path, index=False, encoding='utf-8-sig')
    return failure_path


def write_stock_report(output_path, rows):
    '''
    Save normalized rows for one stock.
    '''
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.sort_values(['Year', 'Quarter', 'Statement', 'Account']).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return len(df)


def run_single_stock(args):
    '''
    Download reports for one stock.
    '''
    output_path = get_output_path(args.stock)

    if os.path.exists(output_path) and not args.force:
        print(f'Output already exists: {output_path}')
        print('Use --force to overwrite it.')
        return

    rows, failures = download_reports(
        args.stock,
        args.name,
            args.start_year,
            args.end_year,
            args.end_quarter,
            args.reports,
        )

    if not rows:
        raise ValueError('No MOPSFIN report rows were downloaded.')

    rows_saved = write_stock_report(output_path, rows)
    failure_path = write_failures(failures, args.stock)

    print('Report download summary:')
    print(f'rows_saved={rows_saved}')
    print(f"statements={','.join(args.reports)}")
    print(f'output_path={output_path}')
    print(f'failures={len(failures)}')
    if failure_path:
        print(f'failure_path={failure_path}')


def run_all_stocks(args):
    '''
    Download reports for all listed stocks in the metadata catalog.
    '''
    stocks = load_twse_stock_catalog(args.metadata, max_stocks=args.max_stocks)
    session = create_session()
    completed = 0
    skipped = 0
    failed = 0

    print(f'Preparing to download reports for {len(stocks)} stocks.')

    for index, (stock_code, stock_name) in enumerate(stocks, start=1):
        output_path = get_output_path(stock_code)

        if os.path.exists(output_path) and not args.force:
            skipped += 1
            print(f'[{index}/{len(stocks)}] {stock_code} already exists; skipped.')
            continue

        print(f'[{index}/{len(stocks)}] Downloading {stock_code} {stock_name}.')
        rows, failures = download_reports_with_session(
            session,
            stock_code,
            stock_name,
            args.start_year,
            args.end_year,
            args.end_quarter,
            args.reports,
        )

        write_failures(failures, stock_code)

        if rows:
            rows_saved = write_stock_report(output_path, rows)
            completed += 1
            print(f'Saved {rows_saved} rows to {output_path}.')
        else:
            failed += 1
            print(f'No rows downloaded for {stock_code}; output not written.')

    print('All-stock report download summary:')
    print(f'total_stocks={len(stocks)}')
    print(f'completed={completed}')
    print(f'skipped={skipped}')
    print(f'failed={failed}')


def main():
    '''
    Download MOPSFIN three-statement reports for one stock.
    '''
    args = parse_args()
    validate_args(args)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.all_stocks:
        run_all_stocks(args)
    else:
        run_single_stock(args)


if __name__ == '__main__':
    main()
