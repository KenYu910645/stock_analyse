'''
stockDownload.py

Download/cache stock price data and optionally render interactive charts.
'''
import glob
import os
import random
import time
from datetime import datetime

import pandas as pd
import twstock
import twstock.stock as twstock_stock

from config import cfg
from stock_viz import visualize_stock_csv


#######################
### Global variable ###
#######################
# Starting date for the data fetch
START_YEAR = 2020
START_MONTH = 5
DATA_DIR = './data'
LOG_DIR = './log'
PLOT_DIR = './plot'
METADATA_PATH = f'{DATA_DIR}/stock_metadata.csv'
ERROR_LOG_PATH = f'{LOG_DIR}/stock_download_errors.csv'

# twstock metadata values.  Use unicode escapes to avoid source encoding issues.
COMMON_STOCK_TYPE = '\u80a1\u7968'
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


_patch_twstock_extra_columns()


def get_config_value(name, default):
    '''
    Return a config value with a default for backward compatibility.
    '''
    return getattr(cfg, name, default)


def find_latest_cached_csv(stock_tar, start_time):
    '''
    Return the newest cached CSV for a stock, if one exists.
    '''
    pattern = f'{DATA_DIR}/{stock_tar}_{start_time}_to_*.csv'
    cached_files = sorted(glob.glob(pattern))
    return cached_files[-1] if cached_files else None


def get_stock_output_path(stock_tar, start_time, end_time):
    '''
    Return the expected stock output CSV path.
    '''
    return f'{DATA_DIR}/{stock_tar}_{start_time}_to_{end_time}.csv'


def build_stock_catalog():
    '''
    Return metadata for all TWSE/TPEX common stock codes known by twstock.

    twstock.codes also includes warrants, ETFs, ETNs, TDRs, and other
    instruments.  Filtering to common stock keeps the download list focused.
    '''
    rows = []

    for code, info in twstock.codes.items():
        if not (
            code.isdigit()
            and len(code) == 4
            and info.type == COMMON_STOCK_TYPE
            and info.market in (TWSE_MARKET, TPEX_MARKET)
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


def get_all_stock_codes():
    '''
    Return all TWSE/TPEX common stock codes known by twstock.
    '''
    return build_stock_catalog()['Code'].tolist()


def write_stock_metadata(catalog_df):
    '''
    Save stock metadata to a separate catalog CSV.
    '''
    os.makedirs(DATA_DIR, exist_ok=True)
    catalog_df.to_csv(METADATA_PATH, index=False, encoding='utf-8-sig')
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
    min_seconds = get_config_value('throttle_min_seconds', 1)
    max_seconds = get_config_value('throttle_max_seconds', 3)

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
    stock = twstock.Stock(stock_tar, initial_fetch=False)
    target_price = stock.fetch_from(START_YEAR, START_MONTH)

    return pd.DataFrame(columns=PRICE_COLUMNS, data=target_price)


def fetch_stock_data_with_retries(stock_tar):
    '''
    Fetch stock data with retry backoff and polite throttling.
    '''
    max_retries = get_config_value('max_retries', 3)
    retry_backoff = get_config_value('retry_backoff_seconds', 10)
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
    return pd.read_csv(csv_path, parse_dates=['Date'])


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
    fn_out = get_stock_output_path(stock_tar, start_time, end_time)
    cached_fn = find_latest_cached_csv(stock_tar, start_time)

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
            df_stock.to_csv(fn_out, index=False, encoding='utf-8-sig')
            print(f'Data fetched and saved to {fn_out}.')
            source_csv_path = fn_out
            result = 'downloaded'

    df_stock = ensure_price_columns(df_stock)

    if is_plot:
        plot_path = f'{PLOT_DIR}/{stock_tar}_{start_time}_to_{end_time}.html'
        visualize_stock_csv(source_csv_path, plot_path)

    return result


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
            result = load_or_download_stock(stock_tar, is_plot=cfg.is_plot)
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


if __name__ == '__main__':
    catalog_df = build_stock_catalog()
    write_stock_metadata(catalog_df)

    stock_list = (
        catalog_df['Code'].tolist()
        if cfg.download_all_stocks
        else cfg.stock_list
    )
    stock_name_by_code = get_stock_name_map(catalog_df)

    print(f'Preparing to download/load {len(stock_list)} stocks.')
    download_all_stocks(stock_list, stock_name_by_code)
