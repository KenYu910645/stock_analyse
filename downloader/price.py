'''
price.py

Download/cache stock price data and optionally render interactive charts.
'''
import argparse
import glob
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import twstock
import twstock.stock as twstock_stock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from column_schema import read_csv_canonical, to_csv_storage  # noqa: E402
#######################
### Global variable ###
#######################
# Starting date for the data fetch
START_YEAR = 2020
START_MONTH = 5
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'price')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'data_viz', 'price_charts')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata.csv')
ERROR_LOG_PATH = f'{LOG_DIR}/stock_download_errors.csv'
TAIEX_CODE = 'TAIEX'
TWSE_STOCK_DAY_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY'
TWSE_MI_INDEX_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
TWSE_STOCK_DAY_MIN_DATE = date(2010, 1, 4)
TWSE_MI_INDEX_STOCK_MIN_DATE = date(2004, 2, 11)
TRADING_DAYS_PATH = os.path.join(PROJECT_ROOT, 'data', 'trading_days.csv')

# twstock metadata values.  Use unicode escapes to avoid source encoding issues.
COMMON_STOCK_TYPE = '\u80a1\u7968'
ETF_TYPE = 'ETF'
INDEX_TYPE = 'INDEX'
TWSE_MARKET = '\u4e0a\u5e02'
TPEX_MARKET = '\u4e0a\u6ac3'

PRICE_COLUMNS = [
    'Date',
    'Capacity',
    'Turnover',
    'Open',
    'High',
    'Low',
    'Close',
    'Change',
    'Transaction',
]


@dataclass
class DownloadSettings:
    '''Runtime defaults shared by the price downloader entry points.'''

    throttle_min_seconds: float = 0.2
    throttle_max_seconds: float = 0.8
    max_retries: int = 3
    retry_backoff_seconds: float = 10.0
    is_plot: bool = False


DOWNLOAD_SETTINGS = DownloadSettings()


def _patch_twstock_extra_columns():
    '''
    twstock 1.4.0 expects 9 TWSE/TPEX data columns.  TWSE currently returns an
    extra trailing note column, so trim any new trailing columns before twstock
    builds its fixed Data tuple.
    '''
    for fetcher_name in ('TWSEFetcher', 'TPEXFetcher'):
        fetcher = getattr(twstock_stock, fetcher_name, None)
        if fetcher is None or getattr(fetcher, '_stock_analyse_patched', False):
            continue

        original_make_datatuple = fetcher._make_datatuple

        def make_datatuple(self, data, original_make_datatuple=original_make_datatuple):
            return original_make_datatuple(self, list(data[:9]))

        fetcher._make_datatuple = make_datatuple
        fetcher._stock_analyse_patched = True


def find_latest_cached_csv(stock_tar):
    '''
    Return the newest cached CSV for a stock, if one exists.
    '''
    cached_files = get_stock_existing_paths(stock_tar)
    return cached_files[-1] if cached_files else None


def safe_filename_part(value):
    '''
    Return a filesystem-safe filename component.
    '''
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', str(value or '')).strip()
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned.strip(' .') or 'Unknown'


def get_metadata_name_map(metadata_path=METADATA_PATH):
    '''
    Return metadata code to short-name mapping.
    '''
    if not os.path.exists(metadata_path):
        return {}

    df_metadata = read_csv_canonical(metadata_path, dtype={'Code': str}).fillna('')
    if 'Code' not in df_metadata.columns or 'Name' not in df_metadata.columns:
        return {}

    return {
        str(row['Code']).strip(): str(row['Name']).strip()
        for _, row in df_metadata.iterrows()
        if str(row['Code']).strip()
    }


def get_stock_output_path(stock_tar):
    '''
    Return the expected stock output CSV path.
    '''
    name = get_metadata_name_map().get(str(stock_tar).strip(), '')
    suffix = safe_filename_part(name) if name else str(stock_tar).strip()
    return os.path.join(DATA_DIR, f'{stock_tar}_{suffix}.csv')


def parse_roc_date(value):
    '''
    Convert a TWSE ROC date like 109/01/02 into an ISO date string.
    '''
    year_text, month_text, day_text = value.split('/')
    year = int(year_text) + 1911
    return f'{year:04d}-{int(month_text):02d}-{int(day_text):02d}'


def parse_twse_number(value):
    '''
    Convert TWSE formatted numbers into float values.
    '''
    if value in (None, '', '--', '---', '----'):
        return None

    text = strip_html(value).replace(',', '').replace('+', '').strip()
    text = text.lstrip('Xx')
    if text in ('', '--', '---', '----'):
        return None

    return float(text)


def parse_metadata_start(value):
    '''
    Parse a metadata Start value into a date.
    '''
    text = str(value or '').strip()
    if not text:
        return TWSE_STOCK_DAY_MIN_DATE

    for fmt in ('%Y-%m-%d', '%Y-%m'):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date()
        except ValueError:
            pass

    raise ValueError(f'Invalid metadata Start date: {value!r}')


def format_year_month(value):
    '''
    Format a date-like object as YYYYMM.
    '''
    return f'{value.year:04d}{value.month:02d}'


def normalize_price_dates(df_stock):
    '''
    Normalize Date values for sorting and duplicate removal.
    '''
    df_stock = df_stock.copy()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
    df_stock = df_stock.dropna(subset=['Date'])
    return df_stock


def clean_stock_day_stat(stat):
    '''
    Return a compact status string from TWSE STOCK_DAY responses.
    '''
    return str(stat or '').strip()


def strip_html(value):
    '''
    Strip small TWSE HTML fragments from a field value.
    '''
    return re.sub(r'<[^>]+>', '', str(value or '')).strip()


def parse_twse_signed_number(sign_value, number_value):
    '''
    Parse TWSE sign and amount fields into a signed float.
    '''
    number = parse_twse_number(number_value)
    if number is None:
        return None

    sign = strip_html(sign_value)
    if '-' in sign:
        return -abs(number)
    return number


def fetch_twse_stock_month(session, stock_tar, year, month):
    '''
    Fetch one month of TWSE STOCK_DAY data for a listed stock or ETF.
    '''
    response = session.get(
        TWSE_STOCK_DAY_URL,
        params={
            'date': f'{year}{month:02d}01',
            'stockNo': stock_tar,
            'response': 'json',
        },
        headers={
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'stock_analyse/1.0'
            ),
        },
        timeout=30,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:120].replace('\n', ' ')
        raise ValueError(
            f'TWSE STOCK_DAY returned non-JSON for '
            f'{stock_tar} {year}-{month:02d}: {snippet}'
        ) from exc

    stat = clean_stock_day_stat(payload.get('stat'))
    if stat != 'OK':
        if payload.get('total') == 0 or '無符合' in stat:
            return []
        raise ValueError(
            f'TWSE STOCK_DAY request failed for '
            f'{stock_tar} {year}-{month:02d}: {stat}'
        )

    rows = []
    for raw_row in payload.get('data', []):
        # TWSE currently includes a trailing note column.  Keep the stable
        # project schema and ignore source-only notes.
        rows.append({
            'Date': parse_roc_date(str(raw_row[0]).strip()),
            'Capacity': parse_twse_number(raw_row[1]),
            'Turnover': parse_twse_number(raw_row[2]),
            'Open': parse_twse_number(raw_row[3]),
            'High': parse_twse_number(raw_row[4]),
            'Low': parse_twse_number(raw_row[5]),
            'Close': parse_twse_number(raw_row[6]),
            'Change': parse_twse_number(raw_row[7]),
            'Transaction': parse_twse_number(raw_row[8]),
        })

    return rows


def find_mi_index_price_tables(payload):
    '''
    Return MI_INDEX tables that contain index close values.
    '''
    tables = []
    for table in payload.get('tables', []):
        fields = table.get('fields') or []
        if '指數' in fields and '收盤指數' in fields:
            tables.append(table)
    return tables


def fetch_mi_index_day(session, query_date):
    '''
    Fetch index close values for one TWSE trading day.
    '''
    response = session.get(
        TWSE_MI_INDEX_URL,
        params={
            'date': query_date.strftime('%Y%m%d'),
            'type': 'ALLBUT0999',
            'response': 'json',
        },
        headers={
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'stock_analyse/1.0'
            ),
        },
        timeout=30,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:120].replace('\n', ' ')
        raise ValueError(
            f'TWSE MI_INDEX returned non-JSON for '
            f'{query_date.isoformat()}: {snippet}'
        ) from exc

    rows_by_name = {}
    for table in find_mi_index_price_tables(payload):
        fields = table.get('fields') or []
        name_index = fields.index('指數')
        close_index = fields.index('收盤指數')
        sign_index = fields.index('漲跌(+/-)') if '漲跌(+/-)' in fields else None
        change_index = fields.index('漲跌點數') if '漲跌點數' in fields else None

        for raw_row in table.get('data', []):
            name = strip_html(raw_row[name_index])
            close = parse_twse_number(raw_row[close_index])
            if not name or close is None:
                continue

            change = None
            if sign_index is not None and change_index is not None:
                change = parse_twse_signed_number(
                    raw_row[sign_index],
                    raw_row[change_index],
                )

            rows_by_name[name] = {
                'Date': query_date.isoformat(),
                'Capacity': 0,
                'Turnover': 0,
                # MI_INDEX daily index tables expose close-level data only.
                # Fill OHLC with close to preserve the project CSV schema.
                'Open': close,
                'High': close,
                'Low': close,
                'Close': close,
                'Change': change,
                'Transaction': 0,
            }

    return rows_by_name


def fetch_mi_index_day_with_retries(session, query_date):
    '''
    Fetch one MI_INDEX trading day with retry backoff.
    '''
    max_retries = DOWNLOAD_SETTINGS.max_retries
    retry_backoff = DOWNLOAD_SETTINGS.retry_backoff_seconds
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fetch_mi_index_day(session, query_date)
        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            print(
                f'Fetch failed for MI_INDEX {query_date.isoformat()} '
                f'(attempt {attempt}/{max_retries}): {exc}'
            )
            print(f'Retrying after {retry_backoff} seconds.')
            time.sleep(retry_backoff)

    raise last_error


def find_mi_stock_price_table(payload):
    '''
    Return the MI_INDEX table with per-security OHLC rows.
    '''
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if len(fields) >= 11 and fields[0] == '證券代號':
            return table
    raise ValueError('TWSE MI_INDEX payload did not include a stock price table.')


def fetch_mi_stock_day(session, query_date, target_codes):
    '''
    Fetch all target stock/ETF OHLC rows for one trading day.
    '''
    response = session.get(
        TWSE_MI_INDEX_URL,
        params={
            'date': query_date.strftime('%Y%m%d'),
            'type': 'ALLBUT0999',
            'response': 'json',
        },
        headers={
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'stock_analyse/1.0'
            ),
        },
        timeout=30,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:120].replace('\n', ' ')
        raise ValueError(
            f'TWSE MI_INDEX returned non-JSON for '
            f'{query_date.isoformat()}: {snippet}'
        ) from exc

    if payload.get('stat') != 'OK':
        return {}

    table = find_mi_stock_price_table(payload)
    rows_by_code = {}
    for raw in table.get('data') or []:
        if len(raw) < 11:
            continue

        code = str(raw[0]).strip()
        if code not in target_codes:
            continue

        rows_by_code[code] = {
            'Date': query_date.isoformat(),
            'Capacity': parse_twse_number(raw[2]),
            'Turnover': parse_twse_number(raw[4]),
            'Open': parse_twse_number(raw[5]),
            'High': parse_twse_number(raw[6]),
            'Low': parse_twse_number(raw[7]),
            'Close': parse_twse_number(raw[8]),
            'Change': parse_twse_signed_number(raw[9], raw[10]),
            'Transaction': parse_twse_number(raw[3]),
        }

    return rows_by_code


def fetch_mi_stock_day_with_retries(session, query_date, target_codes):
    '''
    Fetch one MI_INDEX stock table with retry backoff.
    '''
    max_retries = DOWNLOAD_SETTINGS.max_retries
    retry_backoff = DOWNLOAD_SETTINGS.retry_backoff_seconds
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fetch_mi_stock_day(session, query_date, target_codes)
        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            print(
                f'Fetch failed for MI_INDEX prices {query_date.isoformat()} '
                f'(attempt {attempt}/{max_retries}): {exc}'
            )
            print(f'Retrying after {retry_backoff} seconds.')
            time.sleep(retry_backoff)

    raise last_error


def load_trading_days(start_date, end_date=None):
    '''
    Load canonical trading days for a date range.
    '''
    if end_date is None:
        end_date = datetime.now().date()

    df_days = pd.read_csv(TRADING_DAYS_PATH, dtype=str)
    if 'date' not in df_days.columns:
        raise ValueError(f'{TRADING_DAYS_PATH} is missing date column.')

    days = pd.to_datetime(df_days['date'], errors='coerce').dropna()
    days = days[(days.dt.date >= start_date) & (days.dt.date <= end_date)]
    return [day.date() for day in days]


def build_stock_catalog():
    '''
    Return metadata for TWSE-listed common stock codes known by twstock.

    twstock.codes also includes warrants, ETFs, ETNs, TDRs, and other
    instruments.  Filtering to TWSE common stock keeps the download list
    focused on listed stocks.
    '''
    rows = []

    for code, info in twstock.codes.items():
        if not (
            code.isdigit()
            and len(code) == 4
            and info.type == COMMON_STOCK_TYPE
            and info.market == TWSE_MARKET
        ):
            continue

        rows.append({
            'Code': code,
            'Name': info.name,
            'Type': info.type,
            'Market': info.market,
            'Group': info.group,
            'ISIN': info.ISIN,
            'Start': info.start,
            'CFI': info.CFI,
        })

    return pd.DataFrame(rows).sort_values('Code').reset_index(drop=True)


def load_metadata_catalog(metadata_path=METADATA_PATH):
    '''
    Load the project metadata catalog with stock codes preserved as strings.
    '''
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Metadata CSV does not exist: {metadata_path}')

    df_metadata = read_csv_canonical(metadata_path, dtype={'Code': str}).fillna('')
    required_columns = {'Code', 'Name', 'Type', 'Market', 'Start'}
    missing_columns = required_columns - set(df_metadata.columns)
    if missing_columns:
        raise ValueError(
            f'Metadata CSV is missing columns: {sorted(missing_columns)}'
        )

    df_metadata['Code'] = df_metadata['Code'].astype(str).str.strip()
    return df_metadata[df_metadata['Code'] != ''].reset_index(drop=True)


def filter_metadata_price_catalog(
    df_metadata,
    include_etf=True,
    include_index=False,
    codes=None,
    max_instruments=None,
):
    '''
    Return metadata rows eligible for price download.
    '''
    mask = df_metadata['Market'].eq(TWSE_MARKET) & df_metadata['Type'].isin(
        [COMMON_STOCK_TYPE]
    )

    if include_etf:
        mask = mask | (
            df_metadata['Market'].eq(TWSE_MARKET)
            & df_metadata['Type'].eq(ETF_TYPE)
        )

    if include_index:
        mask = mask | df_metadata['Type'].eq(INDEX_TYPE)

    catalog = df_metadata.loc[mask].copy()

    if codes:
        code_set = {str(code).strip() for code in codes if str(code).strip()}
        catalog = catalog[catalog['Code'].isin(code_set)]

    catalog = catalog.sort_values(['Type', 'Code']).reset_index(drop=True)

    if max_instruments is not None:
        catalog = catalog.head(max_instruments)

    return catalog


def get_stock_existing_paths(stock_tar):
    '''
    Return existing per-stock CSV paths for stock_tar.
    '''
    code = str(stock_tar).strip()
    patterns = [
        os.path.join(DATA_DIR, f'{code}_*.csv'),
        os.path.join(DATA_DIR, f'{code}.csv'),
    ]
    paths = sorted({
        path
        for pattern in patterns
        for path in glob.glob(pattern)
        if not os.path.basename(path).startswith('twse_price_')
    })
    return paths


def read_existing_stock_csvs(stock_tar):
    '''
    Read all existing per-stock CSVs for stock_tar.
    '''
    frames = []
    for csv_path in get_stock_existing_paths(stock_tar):
        try:
            frames.append(read_cached_stock_csv(csv_path))
        except Exception as exc:
            print(f'Could not read existing cache {csv_path}: {exc}')

    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def combine_price_frames(frames):
    '''
    Combine price frames idempotently by Date.
    '''
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df_stock = pd.concat(valid_frames, ignore_index=True)
    df_stock = ensure_price_columns(df_stock)
    df_stock = normalize_price_dates(df_stock)
    df_stock = df_stock.sort_values('Date')
    df_stock = df_stock.drop_duplicates(subset=['Date'], keep='last')
    return ensure_price_columns(df_stock.reset_index(drop=True))


def write_price_csv(stock_tar, df_stock):
    '''
    Write a per-stock CSV using the CODE_公司簡稱 filename convention.
    '''
    if df_stock.empty:
        raise ValueError(f'No rows available for {stock_tar}.')

    os.makedirs(DATA_DIR, exist_ok=True)
    df_stock = normalize_price_dates(df_stock)
    output_path = get_stock_output_path(stock_tar)
    df_stock = ensure_price_columns(df_stock)
    to_csv_storage(df_stock, output_path, index=False, encoding='utf-8-sig')
    print(f'Data merged and saved to {output_path}.')
    return output_path


def reaches_month(df_stock, target_date):
    '''
    Return True when cached rows cover target_date's month or earlier.
    '''
    if df_stock.empty:
        return False

    earliest = normalize_price_dates(df_stock)['Date'].min().date()
    return (earliest.year, earliest.month) <= (target_date.year, target_date.month)


def build_index_name_map(index_catalog):
    '''
    Return MI_INDEX names mapped to metadata codes.
    '''
    name_to_code = {}
    aliases = {
        '營造建材類': ['建材營造類指數'],
        '化學工業類': ['化學類指數'],
        'EDRIN': ['電子類反向指數'],
        'EDRL2': ['電子類兩倍槓桿指數'],
    }
    for _, row in index_catalog.iterrows():
        code = str(row['Code']).strip()
        names = {
            str(row.get('Name', '')).strip(),
            code,
        }
        for base_name in list(names):
            if ' ' in base_name:
                names.add(base_name.replace(' ', ''))
            if base_name.endswith('類') and not base_name.endswith('類指數'):
                names.add(f'{base_name}指數')
        for alias in aliases.get(code, []) + aliases.get(str(row.get('Name', '')).strip(), []):
            names.add(alias)
        if code == TAIEX_CODE:
            names.add('發行量加權股價指數')

        for name in names:
            if name:
                name_to_code[name] = code

    return name_to_code


def index_download_start(start_value):
    '''
    Return the earliest index date covered by the local trading calendar.
    '''
    metadata_start = parse_metadata_start(start_value)
    return max(metadata_start, date(1990, 1, 4))


def mi_index_download_start(start_value):
    '''
    Return the earliest date covered by MI_INDEX and the local calendar.
    '''
    metadata_start = parse_metadata_start(start_value)
    return max(metadata_start, TWSE_MI_INDEX_STOCK_MIN_DATE)


def write_merged_code_rows(code, rows):
    '''
    Merge fetched row dictionaries with existing cache and write a CSV.
    '''
    fetched_df = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    existing_df = read_existing_stock_csvs(code)
    merged_df = combine_price_frames([existing_df, fetched_df])
    write_price_csv(code, merged_df)


def download_metadata_stocks_bulk(stock_catalog, force=False):
    '''
    Download metadata stock/ETF histories from daily TWSE MI_INDEX data.
    '''
    stats = {
        'total': len(stock_catalog),
        'skipped': 0,
        'downloaded': 0,
        'fallback': 0,
        'failed': 0,
    }

    if stock_catalog.empty:
        return stats

    active_rows = []
    for _, row in stock_catalog.iterrows():
        code = str(row['Code']).strip()
        start_date = mi_index_download_start(row.get('Start', ''))
        existing_df = read_existing_stock_csvs(code)
        if not force and reaches_month(existing_df, start_date):
            print(
                f'{code} existing cache already reaches '
                f'{format_year_month(start_date)}. Skipping data fetch.'
            )
            stats['skipped'] += 1
            continue
        active_rows.append(row)

    if not active_rows:
        return stats

    active_catalog = pd.DataFrame(active_rows)
    code_start = {
        str(row['Code']).strip(): mi_index_download_start(row.get('Start', ''))
        for _, row in active_catalog.iterrows()
    }
    rows_by_code = {code: [] for code in code_start}
    target_codes = set(code_start)
    fetch_start = min(code_start.values())
    trading_days = load_trading_days(fetch_start)
    consecutive_failures = 0
    max_consecutive_failures = 20

    print(
        f'Fetching MI_INDEX security prices for {len(trading_days)} trading '
        f'days from {fetch_start.isoformat()} for {len(target_codes)} codes.'
    )
    with requests.Session() as session:
        for index, trading_day in enumerate(trading_days, start=1):
            print(
                f'[{index}/{len(trading_days)}] Fetching MI_INDEX prices '
                f'{trading_day.isoformat()}.'
            )
            try:
                day_rows = fetch_mi_stock_day_with_retries(
                    session,
                    trading_day,
                    target_codes,
                )
            except Exception as exc:
                consecutive_failures += 1
                print(
                    f'Failed MI_INDEX prices {trading_day.isoformat()}: {exc}'
                )
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        'Stopping bulk MI_INDEX price fetch after '
                        f'{consecutive_failures} consecutive failures.'
                    )
                    break
                continue

            consecutive_failures = 0
            for code, row_data in day_rows.items():
                if trading_day < code_start[code]:
                    continue
                rows_by_code[code].append(row_data)
            sleep_between_downloads()

    for code, rows in rows_by_code.items():
        try:
            write_merged_code_rows(code, rows)
            stats['downloaded'] += 1
        except Exception as exc:
            stats['failed'] += 1
            print(f'Error writing price data for {code}: {exc}')

    return stats


def download_metadata_indices(index_catalog, force=False, min_start_date=None):
    '''
    Download metadata index close histories from daily TWSE MI_INDEX data.
    '''
    if index_catalog.empty:
        return {
            'total': 0,
            'skipped': 0,
            'downloaded': 0,
            'fallback': 0,
            'failed': 0,
        }

    stats = {
        'total': len(index_catalog),
        'skipped': 0,
        'downloaded': 0,
        'fallback': 0,
        'failed': 0,
    }
    rows_by_code = {str(row['Code']).strip(): [] for _, row in index_catalog.iterrows()}
    active_rows = []
    resume_start_by_code = {}

    for _, row in index_catalog.iterrows():
        code = str(row['Code']).strip()
        start_date = index_download_start(row.get('Start', ''))
        if min_start_date is not None:
            start_date = max(start_date, min_start_date)
        existing_df = read_existing_stock_csvs(code)
        if not force and not existing_df.empty:
            latest = normalize_price_dates(existing_df)['Date'].max().date()
            if latest >= datetime.now().date():
                print(
                    f'{code} existing index cache is current through '
                    f'{latest.isoformat()}. Skipping data fetch.'
                )
                stats['skipped'] += 1
                continue
            start_date = latest + timedelta(days=1)
            if min_start_date is not None:
                start_date = max(start_date, min_start_date)

        resume_start_by_code[code] = start_date
        active_rows.append(row)

    if not active_rows:
        return stats

    active_catalog = pd.DataFrame(active_rows)
    start_dates = [
        resume_start_by_code[str(row['Code']).strip()]
        for _, row in active_catalog.iterrows()
    ]
    fetch_start = min(start_dates)
    trading_days = load_trading_days(fetch_start)
    name_to_code = build_index_name_map(active_catalog)
    code_start = {
        str(row['Code']).strip(): resume_start_by_code[str(row['Code']).strip()]
        for _, row in active_catalog.iterrows()
    }

    print(
        f'Fetching MI_INDEX for {len(trading_days)} trading days '
        f'from {fetch_start.isoformat()}.'
    )
    consecutive_failures = 0
    max_consecutive_failures = 20
    with requests.Session() as session:
        for index, trading_day in enumerate(trading_days, start=1):
            print(
                f'[{index}/{len(trading_days)}] Fetching MI_INDEX '
                f'{trading_day.isoformat()}.'
            )
            try:
                day_rows = fetch_mi_index_day_with_retries(session, trading_day)
            except Exception as exc:
                consecutive_failures += 1
                print(f'Failed MI_INDEX {trading_day.isoformat()}: {exc}')
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        'Stopping index MI_INDEX fetch after '
                        f'{consecutive_failures} consecutive failures.'
                    )
                    break
                sleep_between_downloads()
                continue

            consecutive_failures = 0
            for index_name, row_data in day_rows.items():
                code = name_to_code.get(index_name)
                if not code or trading_day < code_start[code]:
                    continue
                rows_by_code[code].append(row_data)
            sleep_between_downloads()

    for _, row in active_catalog.iterrows():
        code = str(row['Code']).strip()
        try:
            fetched_df = pd.DataFrame(rows_by_code[code], columns=PRICE_COLUMNS)
            existing_df = read_existing_stock_csvs(code)
            merged_df = combine_price_frames([existing_df, fetched_df])
            write_price_csv(code, merged_df)
            stats['downloaded'] += 1
        except Exception as exc:
            stats['failed'] += 1
            print(f'Error writing index data for {code}: {exc}')

    return stats


def write_stock_metadata(catalog_df):
    '''
    Save stock metadata to a separate catalog CSV.
    '''
    os.makedirs(DATA_DIR, exist_ok=True)
    to_csv_storage(catalog_df, METADATA_PATH, index=False, encoding='utf-8-sig')
    print(f'Stock metadata saved to {METADATA_PATH}.')


def get_stock_name_map(catalog_df):
    '''
    Return a stock code to stock name mapping for logging.
    '''
    return dict(zip(catalog_df['Code'], catalog_df['Name']))


def sleep_between_downloads():
    '''
    Sleep for a randomized throttle interval between network download attempts.
    '''
    min_seconds = DOWNLOAD_SETTINGS.throttle_min_seconds
    max_seconds = DOWNLOAD_SETTINGS.throttle_max_seconds

    if max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds

    sleep_seconds = random.uniform(min_seconds, max_seconds)
    print(f'Throttling for {sleep_seconds:.1f} seconds.')
    time.sleep(sleep_seconds)


def log_download_errors(error_rows):
    '''
    Persist failed stock downloads to a CSV log.
    '''
    if not error_rows:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    df_errors = pd.DataFrame(
        error_rows,
        columns=['Code', 'Name', 'Error', 'Timestamp'],
    )

    write_header = not os.path.exists(ERROR_LOG_PATH)
    df_errors.to_csv(
        ERROR_LOG_PATH,
        mode='a',
        header=write_header,
        index=False,
        encoding='utf-8-sig',
    )
    print(f'Download errors saved to {ERROR_LOG_PATH}.')


def fetch_stock_data(stock_tar):
    '''
    Fetch daily trading data for a given stock from START_YEAR/START_MONTH to
    the present.
    '''
    # Create a Stock object without the eager 31-day fetch.  The eager fetch
    # can fail before fetch_from gets a chance to run on some twstock versions.
    _patch_twstock_extra_columns()
    stock = twstock.Stock(stock_tar, initial_fetch=False)
    target_price = stock.fetch_from(START_YEAR, START_MONTH)

    return pd.DataFrame(columns=PRICE_COLUMNS, data=target_price)


def fetch_stock_data_with_retries(stock_tar):
    '''
    Fetch stock data with retry backoff and polite throttling.
    '''
    max_retries = DOWNLOAD_SETTINGS.max_retries
    retry_backoff = DOWNLOAD_SETTINGS.retry_backoff_seconds
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fetch_stock_data(stock_tar)
        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            print(
                f'Fetch failed for {stock_tar} '
                f'(attempt {attempt}/{max_retries}): {exc}'
            )
            print(f'Retrying after {retry_backoff} seconds.')
            time.sleep(retry_backoff)
        finally:
            sleep_between_downloads()

    raise last_error


def read_cached_stock_csv(csv_path):
    '''
    Read a cached stock CSV.
    '''
    return read_csv_canonical(csv_path, parse_dates=['Date'])


def ensure_price_columns(df_stock):
    '''
    Ensure the downloaded DataFrame has the expected daily price columns.
    '''
    if 'Date' not in df_stock.columns:
        raise ValueError('Downloaded data is missing Date column.')

    missing_cols = [col for col in PRICE_COLUMNS if col not in df_stock.columns]
    if missing_cols:
        raise ValueError(f'Downloaded data is missing columns: {missing_cols}')

    return df_stock[PRICE_COLUMNS]


def load_or_download_stock(stock_tar, is_plot=False):
    '''
    Load the current cached CSV for stock_tar, or download/cache current data.
    Older cache is used only as a fallback if the network fetch fails.
    '''
    os.makedirs(DATA_DIR, exist_ok=True)

    start_time = f'{START_YEAR}{str(START_MONTH).zfill(2)}'
    end_time = datetime.now().strftime('%Y%m')
    fn_out = get_stock_output_path(stock_tar)
    cached_fn = find_latest_cached_csv(stock_tar)

    if os.path.exists(fn_out):
        print(f'Stock data {fn_out} already exists. Skipping data fetch.')
        df_stock = read_cached_stock_csv(fn_out)
        source_csv_path = fn_out
        result = 'skipped'
    else:
        try:
            df_stock = fetch_stock_data_with_retries(stock_tar)
            df_stock = ensure_price_columns(df_stock)
        except Exception:
            if cached_fn and os.path.exists(cached_fn):
                print(
                    f'Fetch failed. Using older cached stock data {cached_fn}.'
                )
                df_stock = read_cached_stock_csv(cached_fn)
                source_csv_path = cached_fn
                result = 'fallback'
            else:
                raise
        else:
            to_csv_storage(df_stock, fn_out, index=False, encoding='utf-8-sig')
            print(f'Data fetched and saved to {fn_out}.')
            source_csv_path = fn_out
            result = 'downloaded'

    df_stock = ensure_price_columns(df_stock)

    if is_plot:
        from stock_viz import visualize_stock_csv

        plot_path = f'{PLOT_DIR}/{stock_tar}_{start_time}_to_{end_time}.html'
        visualize_stock_csv(source_csv_path, plot_path)

    return result


def download_metadata_prices(
    metadata_path=METADATA_PATH,
    include_etf=True,
    include_index=False,
    only_index=False,
    min_start_date=None,
    codes=None,
    max_instruments=None,
    force=False,
):
    '''
    Download price CSVs for instruments listed in metadata.csv.
    '''
    df_metadata = load_metadata_catalog(metadata_path)
    catalog = filter_metadata_price_catalog(
        df_metadata,
        include_etf=include_etf,
        include_index=include_index,
        codes=codes,
        max_instruments=max_instruments,
    )
    index_catalog = catalog[catalog['Type'].eq(INDEX_TYPE)].copy()
    stock_catalog = catalog[~catalog['Type'].eq(INDEX_TYPE)].copy()
    if only_index:
        stock_catalog = stock_catalog.iloc[0:0].copy()
    stock_name_by_code = get_stock_name_map(stock_catalog)

    stats = {
        'total': len(catalog),
        'skipped': 0,
        'downloaded': 0,
        'fallback': 0,
        'failed': 0,
    }
    errors = []

    print(f'Preparing to download/load {len(catalog)} metadata instruments.')
    try:
        stock_stats = download_metadata_stocks_bulk(stock_catalog, force=force)
        for key in ('skipped', 'downloaded', 'fallback', 'failed'):
            stats[key] += stock_stats[key]
    except Exception as exc:
        stats['failed'] += len(stock_catalog)
        print(f'Bulk metadata price download failed: {exc}')
        for stock_tar, name in stock_name_by_code.items():
            errors.append({
                'Code': stock_tar,
                'Name': name,
                'Error': str(exc),
                'Timestamp': datetime.now().isoformat(timespec='seconds'),
            })

    if include_index and not index_catalog.empty:
        index_stats = download_metadata_indices(
            index_catalog,
            force=force,
            min_start_date=min_start_date,
        )
        for key in ('skipped', 'downloaded', 'fallback', 'failed'):
            stats[key] += index_stats[key]

    log_download_errors(errors)

    print(
        'Metadata download summary: '
        f"total={stats['total']}, "
        f"skipped_current_cache={stats['skipped']}, "
        f"downloaded={stats['downloaded']}, "
        f"cached_fallback={stats['fallback']}, "
        f"failed={stats['failed']}"
    )

    return stats


def download_all_stocks(stock_list, stock_name_by_code):
    '''
    Download all stocks, continue on failures, and print a final summary.
    '''
    stats = {
        'total': len(stock_list),
        'skipped': 0,
        'downloaded': 0,
        'fallback': 0,
        'failed': 0,
    }
    errors = []

    for index, stock_tar in enumerate(stock_list, start=1):
        print(f'[{index}/{len(stock_list)}] Processing stock {stock_tar}.')

        try:
            result = load_or_download_stock(
                stock_tar,
                is_plot=DOWNLOAD_SETTINGS.is_plot,
            )
            stats[result] += 1
        except Exception as exc:
            stats['failed'] += 1
            print(f'Error fetching data for stock index {stock_tar}: {exc}')
            errors.append({
                'Code': stock_tar,
                'Name': stock_name_by_code.get(stock_tar, ''),
                'Error': str(exc),
                'Timestamp': datetime.now().isoformat(timespec='seconds'),
            })

    log_download_errors(errors)

    print(
        'Download summary: '
        f"total={stats['total']}, "
        f"skipped_current_cache={stats['skipped']}, "
        f"downloaded={stats['downloaded']}, "
        f"cached_fallback={stats['fallback']}, "
        f"failed={stats['failed']}"
    )

    return stats


def parse_args():
    '''
    Parse command-line options.
    '''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--from-metadata',
        action='store_true',
        help='Download prices for instruments listed in data/metadata.csv.',
    )
    parser.add_argument(
        '--metadata',
        default=METADATA_PATH,
        help=f'Metadata CSV path. Default: {METADATA_PATH}.',
    )
    parser.add_argument(
        '--include-etf',
        action='store_true',
        default=True,
        help='Include TWSE-listed ETFs from metadata.csv.',
    )
    parser.add_argument(
        '--exclude-etf',
        action='store_false',
        dest='include_etf',
        help='Exclude ETFs from metadata.csv downloads.',
    )
    parser.add_argument(
        '--include-index',
        action='store_true',
        help='Attempt supported index downloads from metadata.csv.',
    )
    parser.add_argument(
        '--only-index',
        action='store_true',
        help='Only process index rows from metadata.csv.',
    )
    parser.add_argument(
        '--codes',
        nargs='*',
        help='Optional list of metadata codes to download.',
    )
    parser.add_argument(
        '--max-instruments',
        type=int,
        help='Limit the number of instruments processed.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Fetch even when an existing CSV already covers the start month.',
    )
    parser.add_argument(
        '--throttle-min',
        type=float,
        help='Override minimum seconds to sleep between network requests.',
    )
    parser.add_argument(
        '--throttle-max',
        type=float,
        help='Override maximum seconds to sleep between network requests.',
    )
    parser.add_argument(
        '--min-start-date',
        help='Clamp metadata backfill start dates to this YYYY-MM-DD date.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.throttle_min is not None:
        DOWNLOAD_SETTINGS.throttle_min_seconds = args.throttle_min
    if args.throttle_max is not None:
        DOWNLOAD_SETTINGS.throttle_max_seconds = args.throttle_max
    if args.only_index:
        args.include_index = True
    min_start_date = (
        datetime.strptime(args.min_start_date, '%Y-%m-%d').date()
        if args.min_start_date
        else None
    )

    if args.from_metadata:
        download_metadata_prices(
            metadata_path=args.metadata,
            include_etf=args.include_etf,
            include_index=args.include_index,
            only_index=args.only_index,
            min_start_date=min_start_date,
            codes=args.codes,
            max_instruments=args.max_instruments,
            force=args.force,
        )
    else:
        catalog_df = build_stock_catalog()
        write_stock_metadata(catalog_df)

        stock_list = catalog_df['Code'].tolist()
        stock_name_by_code = get_stock_name_map(catalog_df)

        print(f'Preparing to download/load {len(stock_list)} stocks.')
        download_all_stocks(stock_list, stock_name_by_code)
