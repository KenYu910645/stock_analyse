'''
Incrementally update the CSV "database" for TWSE listed-stock datasets.

The script checks local CSVs first, downloads source data, and appends only
new rows.  It prints a plain-text summary and intentionally does not emit or
write JSON manifests.
'''
import argparse
import io
import glob
import os
import re
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from downloader import ex_right_dividend
from downloader import events as events_downloader
from downloader import institutional_investors
from downloader import margin_trading
from downloader import price
from downloader import report
from downloader import tdcc_shareholding
from downloader import trading_days
from column_schema import (
    canonical_name,
    csv_columns_canonical,
    normalize_date_text,
    read_csv_canonical,
    to_canonical_columns,
    to_csv_storage,
)
from downloader.daily.context import UpdateContext
from downloader.daily.registry import TaskSpec, run_task_specs
from downloader.daily.status import StatusCollector
from tools import apply_forward_adjustments_to_price as price_adjustments


def configure_console_encoding():
    """Keep CLI output readable without mutating global streams on import."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass


DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
TRADING_DAYS_PATH = os.path.join(DATA_DIR, 'trading_days.csv')
STATUS_COLLECTOR = StatusCollector()
TWSE_OPENAPI_BASE_URL = 'https://openapi.twse.com.tw/v1'
TWSE_MI_INDEX_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
TWSE_DAY_TRADING_URL = 'https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U'
TWSE_T86_URL = 'https://www.twse.com.tw/rwd/zh/fund/T86'
IR_ENGAGE_CONFERENCE_URL = 'https://irengage.taiwanindex.com.tw/conferenceList'

COMMON_STOCK_TYPE = '\u80a1\u7968'
TWSE_MARKET = '\u4e0a\u5e02'
COL_REPORT_DATE = '\u51fa\u8868\u65e5\u671f'
COL_CODE = '\u516c\u53f8\u4ee3\u865f'
COL_SHORT_NAME = '\u516c\u53f8\u7c21\u7a31'
COL_YEAR = '\u5e74\u5ea6'
COL_SEASON = '\u5b63\u5225'
COL_MONTH = '\u8cc7\u6599\u5e74\u6708'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 stock_analyse/1.0',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
}

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

ADJUSTED_PRICE_COLUMNS = [
    'open_adj',
    'close_adj',
    'high_adj',
    'low_adj',
    'AdjFactor',
]

PRICE_OUTPUT_COLUMNS = PRICE_COLUMNS + ADJUSTED_PRICE_COLUMNS

EXCLUDED_DATA_DIRS = {
    'codis_weather',
}

SUPPORTED_DATA_DIRS = {
    'broker',
    'company',
    'day_trading',
    'dividend',
    'yield_pe_pb',
    'events',
    'financial',
    'insiders',
    'institutional',
    'investor_conference',
    'margin',
    'price',
    'report',
    'revenue',
    'sbl',
    'shareholder_meeting',
    'shareholding',
}

DISALLOWED_DATA_TOKENS = (
    'by_stock',
    'latest_asof',
    'manifest',
    'raw',
    'dashboard',
    'debug',
    'failure',
)

DAILY_BACKFILL_LOOKBACK_DAYS = 14
MARGIN_REPAIR_LOOKBACK_DAYS = 2

VALUATION_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'Close',
    'DividendYield',
    'DividendYear',
    'PEratio',
    'PBratio',
    'FiscalYearQuarter',
]

DAY_TRADING_COLUMNS = [
    'Date',
    'Code',
    'Name',
    'SuspensionNote',
    'DayTradingVolume',
    'DayTradingBuyAmount',
    'DayTradingSellAmount',
]

DAY_TRADING_DEPRECATED_COLUMNS = [
    'MarketVolumeRatio',
    'MarketBuyAmountRatio',
    'MarketSellAmountRatio',
    'TotalVolume',
    'TotalBuyAmount',
    'TotalSellAmount',
]

DAY_TRADING_FEATURE_COLUMNS = [
    'DayTradingVolumeRatio',
    'DayTradingBuyAmountRatio',
    'DayTradingSellAmountRatio',
    'DayTradingTurnover',
    'DayTradingTurnoverRatio',
    'DayTradingAvgBuyPrice',
    'DayTradingAvgSellPrice',
    'DayTradingAvgSpreadRate',
    'DayTradingAmountImbalanceRatio',
    'IntradayRangeRate',
    'OpenCloseReturn',
    'DayTradingVolumeRatio20DayZScore',
    'DayTradingTurnover20DayZScore',
]

DAY_TRADING_OUTPUT_COLUMNS = DAY_TRADING_COLUMNS + DAY_TRADING_FEATURE_COLUMNS

MARGIN_FEATURE_COLUMNS = [
    'MarginFinancingUsageRate',
    'MarginBalance20DayChangeRate',
    'MarginMarketValue',
    'MarginMarketValueTo20DayAvgTurnover',
    'ShortMarginBalanceRatio',
]

MARGIN_OUTPUT_COLUMNS = margin_trading.OUTPUT_COLUMNS

OPENAPI_SNAPSHOTS = [
    ('financial_income_securities_futures', '/opendata/t187ap06_L_bd', 'financial'),
    ('financial_income_general', '/opendata/t187ap06_L_ci', 'financial'),
    ('financial_income_holding', '/opendata/t187ap06_L_fh', 'financial'),
    ('financial_income_insurance', '/opendata/t187ap06_L_ins', 'financial'),
    ('financial_income_mixed', '/opendata/t187ap06_L_mim', 'financial'),
    ('financial_income_banking', '/opendata/t187ap06_L_basi', 'financial'),
    ('financial_balance_securities_futures', '/opendata/t187ap07_L_bd', 'financial'),
    ('financial_balance_general', '/opendata/t187ap07_L_ci', 'financial'),
    ('financial_balance_holding', '/opendata/t187ap07_L_fh', 'financial'),
    ('financial_balance_insurance', '/opendata/t187ap07_L_ins', 'financial'),
    ('financial_balance_mixed', '/opendata/t187ap07_L_mim', 'financial'),
    ('financial_balance_banking', '/opendata/t187ap07_L_basi', 'financial'),
    ('monthly_revenue', '/opendata/t187ap05_L', 'revenue'),
    ('ex_dividend_forecast_openapi', '/exchangeReport/TWT48U_ALL', 'dividend'),
    ('dividend_distribution', '/opendata/t187ap45_L', 'dividend'),
    ('company_basic', '/opendata/t187ap03_L', 'company'),
    ('sbl_available', '/SBL/TWT96U', 'sbl'),
    ('material_events', '/opendata/t187ap04_L', 'events'),
    ('director_shareholding', '/opendata/t187ap11_L', 'insiders'),
    ('insider_transfer_pre', '/opendata/t187ap12_L', 'insiders'),
    ('insider_transfer_untransferred', '/opendata/t187ap13_L', 'insiders'),
    ('shareholder_meeting', '/opendata/t187ap38_L', 'shareholder_meeting'),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Incrementally update TWSE listed-stock CSV datasets.'
    )
    parser.add_argument(
        '--date',
        default=None,
        help='Preferred trading date in YYYY-MM-DD. Default: today with 7-day backoff.',
    )
    parser.add_argument(
        '--skip-broker',
        action='store_true',
        help='Skip captcha-protected TWSE BSR broker branch data.',
    )
    parser.add_argument(
        '--broker-max-stocks',
        type=int,
        default=None,
        help='Limit TWSE BSR broker metadata stocks for testing.',
    )
    parser.add_argument(
        '--broker-max-attempts',
        type=int,
        default=8,
        help='Captcha retry attempts per stock for TWSE BSR broker downloads.',
    )
    parser.add_argument(
        '--broker-throttle-min',
        type=float,
        default=0.2,
        help='Minimum seconds to sleep between TWSE BSR broker stock requests.',
    )
    parser.add_argument(
        '--broker-throttle-max',
        type=float,
        default=0.8,
        help='Maximum seconds to sleep between TWSE BSR broker stock requests.',
    )
    parser.add_argument(
        '--skip-snapshots',
        action='store_true',
        help='Skip OpenAPI/MOPS-style latest snapshot datasets.',
    )
    parser.add_argument(
        '--skip-daily',
        action='store_true',
        help='Skip daily market datasets.',
    )
    parser.add_argument(
        '--skip-repairs',
        action='store_true',
        help='Skip pre-update repair tasks for known-bad trailing price/margin rows.',
    )
    parser.add_argument(
        '--market-closed',
        action='store_true',
        help=(
            'Run non-price updates for a weekday market closure. Price updates '
            'are skipped, while shareholding and snapshot datasets still run.'
        ),
    )
    args = parser.parse_args()
    if args.broker_max_stocks is not None and args.broker_max_stocks < 1:
        parser.error('--broker-max-stocks must be positive')
    if args.broker_max_attempts < 1:
        parser.error('--broker-max-attempts must be positive')
    if (
        args.broker_throttle_min < 0
        or args.broker_throttle_max < 0
        or args.broker_throttle_min > args.broker_throttle_max
    ):
        parser.error('--broker-throttle-min/--broker-throttle-max must be non-negative and min <= max')
    return args


def parse_iso_date(value):
    return datetime.strptime(value, '%Y-%m-%d').date()


def format_yyyymmdd(value):
    return value.strftime('%Y%m%d')


def clean_html(value):
    return re.sub(r'<[^>]+>', '', str(value or '')).strip()


def clean_number(value):
    text = clean_html(value).replace(',', '').strip()
    return '' if text in ('', '--', '---', '-') else text


def parse_number(value):
    text = clean_number(value)
    return pd.NA if text == '' else pd.to_numeric(text, errors='coerce')


def parse_int(value):
    number = parse_number(value)
    if pd.isna(number):
        return pd.NA
    return int(number)


def parse_signed_change(sign_value, change_value):
    change = parse_number(change_value)
    if pd.isna(change):
        return pd.NA
    sign_text = clean_html(sign_value)
    if '-' in sign_text or 'color:green' in str(sign_value):
        return -abs(float(change))
    return float(change)


def normalize_source_date(value):
    text = normalize_date_text(value)
    text = str(text or '').strip()
    timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}:\d{2}', text)
    if timestamp_match:
        text = timestamp_match.group(1)
    text = text.replace('/', '').replace('-', '')
    if not text:
        return ''
    if len(text) == 7 and text.isdigit():
        return f'{int(text[:3]) + 1911:04d}{text[3:5]}{text[5:7]}'
    if len(text) == 8 and text.isdigit():
        return text
    return text


def yyyymmdd_to_iso(value):
    return normalize_date_text(value)


def normalize_roc_year_text(value):
    text = normalize_cell(value)
    if re.fullmatch(r'\d{2,3}', text):
        return str(int(text) + 1911)
    return normalize_date_text(text)


def normalize_date_range_text(value):
    text = normalize_cell(value)
    if not text:
        return text
    parts = re.split(r'\s*[~～]\s*', text)
    if len(parts) == 2:
        return '~'.join(normalize_date_text(part) for part in parts)
    return normalize_date_text(text)


def normalize_dividend_openapi_dates(dataset_name, df):
    result = df.copy()
    if dataset_name == 'dividend_distribution':
        normalizers = {
            COL_REPORT_DATE: normalize_date_text,
            '\u80a1\u5229\u5e74\u5ea6': normalize_roc_year_text,
            '\u80a1\u5229\u6240\u5c6c\u671f\u9593': normalize_date_range_text,
            '\u8463\u4e8b\u6703\uff08\u64ec\u8b70\uff09\u80a1\u5229\u5206\u6d3e\u65e5': normalize_date_text,
            '\u80a1\u6771\u6703\u65e5\u671f': normalize_date_text,
        }
    elif dataset_name == 'ex_dividend_forecast_openapi':
        normalizers = {'Date': normalize_date_text}
    else:
        return result
    for column, normalizer in normalizers.items():
        if column in result.columns:
            result[column] = result[column].map(normalizer)
    return result


def parse_normalized_yyyymmdd(value):
    text = normalize_source_date(value)
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return datetime.strptime(text, '%Y%m%d').date()
    except ValueError:
        return None


def ensure_stock_metadata():
    if os.path.exists(STOCK_METADATA_PATH):
        return
    catalog = price.build_stock_catalog()
    os.makedirs(DATA_DIR, exist_ok=True)
    to_csv_storage(catalog, STOCK_METADATA_PATH, index=False, encoding='utf-8-sig')


def load_listed_common_stocks():
    ensure_stock_metadata()
    df = read_csv_canonical(STOCK_METADATA_PATH, dtype=str).fillna('')
    df['Code'] = df['Code'].astype(str).str.strip()
    mask = (
        (df['Market'] == TWSE_MARKET)
        & (df['Type'] == COMMON_STOCK_TYPE)
        & df['Code'].str.match(r'^\d{4}$')
    )
    listed = df.loc[mask, ['Code', 'Name']].drop_duplicates('Code')
    return listed.sort_values('Code').reset_index(drop=True)


def load_listed_common_stock_names():
    listed = load_listed_common_stocks()
    return dict(zip(listed['Code'], listed['Name']))


def dataset_registry():
    return {
        name: {'path': os.path.join(DATA_DIR, name)}
        for name in sorted(SUPPORTED_DATA_DIRS)
    }


def scan_data_registry():
    registry = dataset_registry()
    if not os.path.isdir(DATA_DIR):
        return registry
    for entry in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(path):
            continue
        if entry in EXCLUDED_DATA_DIRS:
            status(entry, 'unsupported', path=path, note='excluded by updater scope')
        elif entry not in registry:
            status(entry, 'unsupported', path=path, note='no updater registry entry')
    return registry


def has_disallowed_data_token(path):
    parts = os.path.normpath(path).split(os.sep)
    return any(part in DISALLOWED_DATA_TOKENS for part in parts)


def ensure_managed_output_path(path, allow_create=True):
    if has_disallowed_data_token(path):
        raise ValueError(f'Unsupported data output path: {path}')
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        if not allow_create:
            return False
        if has_disallowed_data_token(parent):
            raise ValueError(f'Unsupported data output directory: {parent}')
        os.makedirs(parent, exist_ok=True)
    return True


def normalize_cell(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text.endswith('.0'):
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return str(int(number))
    return text


DATE_KEY_COLUMNS = {'Date', 'date', 'ex_date', 'DataDate', COL_REPORT_DATE, '\u8cc7\u6599\u65e5\u671f'}


def normalize_key_value(column, value):
    text = normalize_cell(value)
    if column in DATE_KEY_COLUMNS:
        return normalize_source_date(text)
    return text


def row_key(row, key_columns):
    values = []
    for column in key_columns:
        actual_column = column if column in row.index else canonical_name(column)
        values.append(normalize_key_value(column, row.get(actual_column, '')))
    return tuple(values)


def resolve_key_column(df, column):
    if column in df.columns:
        return column
    canonical = canonical_name(column)
    if canonical in df.columns:
        return canonical
    return column


def normalized_key_frame(df, key_columns):
    normalized = pd.DataFrame(index=df.index)
    for column in key_columns:
        actual_column = resolve_key_column(df, column)
        normalized[column] = df[actual_column].map(lambda value, col=column: normalize_key_value(col, value))
    return normalized.astype(str)


def rows_match(existing_row, new_row, columns, ignored_columns=None):
    ignored = set(ignored_columns or [])
    for column in columns:
        if column in ignored:
            continue
        if normalize_cell(existing_row.get(column, '')) != normalize_cell(new_row.get(column, '')):
            return False
    return True


def align_columns_for_write(existing_df, new_df, fallback_columns=None):
    columns = list(existing_df.columns) if not existing_df.empty else list(fallback_columns or new_df.columns)
    for column in new_df.columns:
        if column not in columns:
            columns.append(column)
    for frame in (existing_df, new_df):
        for column in columns:
            if column not in frame.columns:
                frame[column] = ''
    return columns


def append_or_refresh_rows(
    path,
    df,
    key_columns,
    fallback_columns=None,
    fetched_at_column='FetchedAt',
    refresh_fetched_at=False,
    replace_existing_on_change=False,
):
    if df.empty:
        return {'created': 0, 'appended': 0, 'refreshed': 0, 'unchanged': 0}

    ensure_managed_output_path(path, allow_create=True)
    if os.path.exists(path):
        existing = read_csv_canonical(path, dtype=str).fillna('')
        created = 0
    else:
        existing = pd.DataFrame()
        created = 1

    incoming = to_canonical_columns(df.copy()).fillna('')
    columns = align_columns_for_write(existing, incoming, fallback_columns=fallback_columns)
    existing = existing[columns] if not existing.empty else pd.DataFrame(columns=columns)
    incoming = incoming[columns]

    key_to_indices = {}
    for idx, row in existing.iterrows():
        key_to_indices.setdefault(row_key(row, key_columns), []).append(idx)

    append_rows = []
    refreshed = 0
    unchanged = 0
    for _, new_row in incoming.iterrows():
        key = row_key(new_row, key_columns)
        indices = key_to_indices.get(key, [])
        if not indices:
            append_rows.append(new_row.to_dict())
            continue

        last_idx = indices[-1]
        existing_row = existing.loc[last_idx]
        if rows_match(
            existing_row,
            new_row,
            columns,
            ignored_columns={fetched_at_column} if refresh_fetched_at else set(),
        ):
            if (
                refresh_fetched_at
                and fetched_at_column in columns
                and normalize_cell(existing.at[last_idx, fetched_at_column])
                != normalize_cell(new_row.get(fetched_at_column, ''))
            ):
                existing.at[last_idx, fetched_at_column] = new_row.get(fetched_at_column, '')
                refreshed += 1
            else:
                unchanged += 1
        elif replace_existing_on_change:
            for column in columns:
                existing.at[last_idx, column] = new_row.get(column, '')
            refreshed += 1
        else:
            append_rows.append(new_row.to_dict())

    if append_rows:
        existing = pd.concat([existing, pd.DataFrame(append_rows, columns=columns)], ignore_index=True)

    if created or refreshed:
        to_csv_storage(existing, path, index=False, encoding='utf-8-sig')
    elif append_rows:
        to_csv_storage(
            pd.DataFrame(append_rows, columns=columns),
            path,
            mode='a',
            header=False,
            index=False,
            encoding='utf-8-sig',
        )

    return {
        'created': created,
        'appended': len(append_rows),
        'refreshed': refreshed,
        'unchanged': unchanged,
    }


def load_trading_days():
    if not os.path.exists(TRADING_DAYS_PATH):
        return []
    df = pd.read_csv(TRADING_DAYS_PATH, dtype=str)
    date_column = 'date' if 'date' in df.columns else df.columns[0]
    dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
    return [value.date() for value in dates.sort_values().drop_duplicates()]


def refresh_trading_days(end_date):
    existing = pd.DataFrame(columns=['date'])
    if os.path.exists(TRADING_DAYS_PATH):
        existing = pd.read_csv(TRADING_DAYS_PATH, dtype=str).fillna('')
    existing_dates = pd.to_datetime(existing.get('date', pd.Series(dtype=str)), errors='coerce').dropna()
    if existing_dates.empty:
        start_date = trading_days.TWSE_ONLINE_START_DATE
    else:
        start_date = max(existing_dates.max().date() - timedelta(days=31), trading_days.TWSE_ONLINE_START_DATE)

    try:
        refreshed = trading_days.download_trading_days(start_date, end_date)
    except Exception as exc:
        cached_dates = load_trading_days()
        if not cached_dates:
            raise RuntimeError(
                f'Trading-day refresh failed and no usable cached calendar exists: {exc}'
            ) from exc
        status(
            'trading_days',
            'warning',
            len(cached_dates),
            TRADING_DAYS_PATH,
            note=(
                f'refresh_failed={exc}; using cached calendar through '
                f'{cached_dates[-1].isoformat()}'
            ),
        )
        return False

    combined = pd.concat([existing[['date']] if 'date' in existing.columns else existing, refreshed], ignore_index=True)
    combined = combined.dropna(subset=['date']).drop_duplicates().sort_values('date').reset_index(drop=True)
    ensure_managed_output_path(TRADING_DAYS_PATH, allow_create=True)
    combined.to_csv(TRADING_DAYS_PATH, index=False, encoding='utf-8-sig')
    status('trading_days', 'updated', len(refreshed), TRADING_DAYS_PATH, note=f'range={combined["date"].iloc[0]}..{combined["date"].iloc[-1]}')
    return True


def latest_trading_day_on_or_before(target_date):
    trading_day_values = load_trading_days()
    eligible = [value for value in trading_day_values if value <= target_date]
    if eligible:
        return eligible[-1]
    raise RuntimeError(
        f'No canonical trading day exists on or before {target_date.isoformat()}.'
    )


def missing_trading_dates_after(latest_value, target_date):
    latest = parse_normalized_yyyymmdd(latest_value) if latest_value else None
    trading_day_values = load_trading_days()
    if not trading_day_values:
        raise RuntimeError('Canonical trading-day calendar is empty.')
    return [
        value for value in trading_day_values
        if value <= target_date and (latest is None or value > latest)
    ]


def recent_trading_dates(target_date, lookback_days=DAILY_BACKFILL_LOOKBACK_DAYS):
    trading_day_values = load_trading_days()
    if not trading_day_values:
        raise RuntimeError('Canonical trading-day calendar is empty.')
    eligible = [value for value in trading_day_values if value <= target_date]
    return eligible[-lookback_days:]


def repair_incomplete_price_tails():
    price_dir = os.path.join(DATA_DIR, 'price')
    if not os.path.isdir(price_dir):
        return {}
    repaired = {}
    for path in sorted(glob.glob(os.path.join(price_dir, '*.csv'))):
        if os.path.basename(path).startswith('twse_price_'):
            continue
        try:
            df = read_csv_canonical(path, dtype=str).fillna('')
        except Exception as exc:
            status('price_repair', 'failed', path=path, note=str(exc))
            continue
        if df.empty or not set(ADJUSTED_PRICE_COLUMNS).issubset(df.columns):
            continue
        remove_count = 0
        while len(df) > remove_count:
            row = df.iloc[len(df) - remove_count - 1]
            if any(str(row[column]).strip() == '' for column in ADJUSTED_PRICE_COLUMNS):
                remove_count += 1
            else:
                break
        if not remove_count:
            continue
        trimmed = df.iloc[:-remove_count].copy()
        to_csv_storage(trimmed, path, index=False, encoding='utf-8-sig')
        code = os.path.basename(path).split('_', 1)[0]
        repaired[code] = path
    status('price_repair', 'updated' if repaired else 'up_to_date', len(repaired), price_dir, note='deleted_trailing_blank_adjusted_rows')
    return repaired


def repair_margin_order_tails(target_date=None):
    margin_dir = os.path.join(DATA_DIR, 'margin')
    if not os.path.isdir(margin_dir):
        return {}
    recent_dates = recent_trading_dates(target_date or date.today())
    recent_cutoff = recent_dates[0].strftime('%Y%m%d') if recent_dates else ''
    repaired = {}
    for path in sorted(glob.glob(os.path.join(margin_dir, '*.csv'))):
        try:
            df = read_csv_canonical(path, dtype=str).fillna('')
        except Exception as exc:
            status('margin_repair', 'failed', path=path, note=str(exc))
            continue
        if df.empty or 'Date' not in df.columns:
            continue
        normalized = df['Date'].astype(str).map(normalize_source_date).tolist()
        cut_index = None
        for index in range(1, len(normalized)):
            if normalized[index] < normalized[index - 1] and normalized[index - 1] >= recent_cutoff:
                cut_index = index - 1
                break
        if cut_index is None:
            continue
        trimmed = df.iloc[:cut_index].copy()
        to_csv_storage(trimmed, path, index=False, encoding='utf-8-sig')
        code = os.path.basename(path).split('_', 1)[0]
        repaired[code] = path
    status('margin_repair', 'updated' if repaired else 'up_to_date', len(repaired), margin_dir, note='deleted_out_of_order_suffixes')
    return repaired


def status(dataset, action, rows=0, path=None, note=''):
    return STATUS_COLLECTOR.emit(dataset, action, rows, path, note)


def get_csv_columns(path, fallback_columns):
    return csv_columns_canonical(path, fallback_columns)


def append_dataframe(path, df, fallback_columns=None):
    if df.empty:
        return 0
    df = to_canonical_columns(df.copy())
    ensure_managed_output_path(path, allow_create=True)
    columns = get_csv_columns(path, fallback_columns or df.columns.tolist())
    for column in columns:
        if column not in df.columns:
            df[column] = ''
    extra_columns = [column for column in df.columns if column not in columns]
    if extra_columns and not os.path.exists(path):
        columns.extend(extra_columns)
    df = df[columns]
    write_header = not os.path.exists(path)
    to_csv_storage(
        df,
        path,
        mode='a',
        header=write_header,
        index=False,
        encoding='utf-8-sig',
    )
    return len(df)


def read_existing_keys(path, key_columns):
    if not os.path.exists(path):
        return set()
    try:
        df = read_csv_canonical(path, dtype=str, usecols=key_columns).fillna('')
    except ValueError:
        return set()
    return build_key_set(df, key_columns)


def build_key_set(df, key_columns):
    if df.empty:
        return set()
    key_df = normalized_key_frame(df, key_columns)
    return set(tuple(row) for row in key_df.itertuples(index=False, name=None))


def append_new_by_keys(path, df, key_columns, fallback_columns=None):
    if df.empty:
        return 0
    df = df.copy()
    key_df = normalized_key_frame(df, key_columns)
    duplicate_mask = key_df.duplicated(keep='last')
    if duplicate_mask.any():
        df = df.loc[~duplicate_mask].copy()
        key_df = key_df.loc[~duplicate_mask].copy()
    existing = read_existing_keys(path, key_columns)
    if existing:
        key_series = key_df.apply(tuple, axis=1)
        df = df.loc[~key_series.isin(existing)].copy()
    return append_dataframe(path, df, fallback_columns=fallback_columns)


def append_or_fill_blank_rows(path, df, key_columns, fill_columns, fallback_columns=None):
    if df.empty:
        return {'appended': 0, 'filled': 0}

    ensure_managed_output_path(path, allow_create=True)
    incoming = to_canonical_columns(df.copy()).fillna('')
    if os.path.exists(path):
        existing = read_csv_canonical(path, dtype=str).fillna('')
        created = False
    else:
        existing = pd.DataFrame()
        created = True

    columns = align_columns_for_write(existing, incoming, fallback_columns=fallback_columns)
    existing = existing[columns] if not existing.empty else pd.DataFrame(columns=columns)
    incoming = incoming[columns]

    key_to_index = {}
    if not existing.empty:
        for idx, row in existing.iterrows():
            key_to_index[row_key(row, key_columns)] = idx

    append_rows = []
    filled = 0
    for _, new_row in incoming.iterrows():
        key = row_key(new_row, key_columns)
        existing_idx = key_to_index.get(key)
        if existing_idx is None:
            append_rows.append(new_row.to_dict())
            continue

        row_filled = False
        for column in fill_columns:
            if column not in columns:
                continue
            if normalize_cell(existing.at[existing_idx, column]) == '' and normalize_cell(new_row[column]) != '':
                existing.at[existing_idx, column] = new_row[column]
                row_filled = True
        if row_filled:
            filled += 1

    if append_rows:
        existing = pd.concat([existing, pd.DataFrame(append_rows, columns=columns)], ignore_index=True)

    if created or filled:
        to_csv_storage(existing, path, index=False, encoding='utf-8-sig')
    elif append_rows:
        to_csv_storage(
            pd.DataFrame(append_rows, columns=columns),
            path,
            mode='a',
            header=False,
            index=False,
            encoding='utf-8-sig',
        )

    return {'appended': len(append_rows), 'filled': filled}


def latest_date_in_csv(path, date_column='Date'):
    if not os.path.exists(path):
        return ''
    try:
        df = read_csv_canonical(path, dtype=str).dropna(how='all')
    except ValueError:
        return ''
    if df.empty:
        return ''
    canonical_column = canonical_name(date_column)
    if date_column in df.columns:
        column = date_column
    elif canonical_column in df.columns:
        column = canonical_column
    else:
        return ''
    normalized = df[column].dropna().map(normalize_source_date)
    normalized = normalized[normalized.astype(bool)]
    return normalized.max() if not normalized.empty else ''


def latest_date_in_directory(directory, date_column='Date'):
    if not os.path.isdir(directory):
        return ''
    latest = ''
    for path in glob.glob(os.path.join(directory, '*.csv')):
        value = latest_date_in_csv(path, date_column)
        if value and (not latest or value > latest):
            latest = value
    return latest


def find_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_main_history_file(pattern):
    excluded_tokens = (
        'manifest',
        'progress',
        'errors',
        'missing',
        'recovered',
        'sample',
        'latest_asof',
    )
    files = [
        path for path in glob.glob(pattern)
        if not any(token in os.path.basename(path) for token in excluded_tokens)
    ]
    if not files:
        return None
    return max(files, key=os.path.getsize)


def fetch_json(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.json()


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype='Float64')
    return pd.to_numeric(
        df[column].astype('string').str.replace(',', '', regex=False),
        errors='coerce',
    )


def safe_ratio(numerator, denominator):
    result = numerator / denominator.where(denominator != 0)
    return result.replace([float('inf'), float('-inf')], pd.NA)


def stock_keyed_output_path(output_dir, code, name):
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', '_', str(name)).strip('._ ')
    filename = f'{code}_{safe_name}.csv' if safe_name else f'{code}.csv'
    return os.path.join(output_dir, filename)


def update_stock_keyed_by_stock(
    df,
    output_dir,
    code_column,
    key_columns,
    fallback_columns=None,
    name_column=None,
    code_to_name=None,
    refresh_fetched_at=False,
    replace_existing_on_change=False,
):
    if df.empty or code_column not in df.columns:
        return 0
    ensure_managed_output_path(os.path.join(output_dir, '.managed'), allow_create=True)
    total = 0
    refreshed = 0
    code_to_name = code_to_name or {}
    for code, stock_df in df.groupby(code_column, sort=True):
        code = str(code).strip()
        name = code_to_name.get(code, '')
        if not name and name_column and name_column in stock_df.columns:
            names = stock_df[name_column].astype(str).str.strip()
            names = names[names != '']
            name = names.iloc[-1] if not names.empty else ''
        path = stock_keyed_output_path(output_dir, code, name)
        if refresh_fetched_at:
            result = append_or_refresh_rows(
                path,
                stock_df,
                key_columns,
                fallback_columns=fallback_columns,
                refresh_fetched_at=True,
                replace_existing_on_change=replace_existing_on_change,
            )
            total += result['appended']
            refreshed += result['refreshed']
        else:
            total += append_new_by_keys(path, stock_df, key_columns, fallback_columns=fallback_columns)
    return {'appended': total, 'refreshed': refreshed} if refresh_fetched_at else total


from downloader.daily.tasks.broker_twse import (
    broker_twse_by_broker_dir,
    broker_twse_by_date_dir,
    broker_twse_by_stock_dir,
    broker_twse_dir,
    build_twse_broker_outputs,
    clear_generated_csv_dir,
    download_twse_broker_batch,
    sync_twse_broker_date_by_stock,
    update_twse_broker,
)
from downloader.daily.tasks.day_trading import (
    add_day_trading_features,
    fetch_day_trading_rows,
    find_day_trading_stock_table,
    price_context_for_day_trading,
    refresh_day_trading_features,
    refresh_day_trading_features_for_path,
    update_day_trading,
)
from downloader.daily.tasks.dividend import update_dividend
from downloader.daily.tasks.institutional import update_institutional
from downloader.daily.tasks.investor_conference import (
    parse_irengage_conference_rows,
    update_investor_conference,
)
from downloader.daily.tasks.margin import (
    add_margin_features,
    price_context_for_margin,
    refresh_margin_features,
    refresh_margin_features_for_path,
    update_margin,
)
from downloader.daily.tasks.openapi_snapshots import (
    filter_listed_openapi_df,
    normalize_material_events_openapi,
    openapi_per_stock_output_dir,
    source_key_columns,
    update_company_basic_by_stock,
    update_dividend_openapi_by_stock,
    update_openapi_per_stock,
    update_openapi_snapshot,
)
from downloader.daily.tasks.price import (
    append_price_per_stock,
    fetch_price_rows,
    find_stock_price_table,
    refresh_adjusted_price_columns,
    stock_price_path,
    update_price,
)
from downloader.daily.tasks.reports import (
    latest_completed_report_period,
    latest_financial_period_by_code,
    latest_report_period_in_csv,
    next_quarter,
    previous_quarter,
    report_periods_to_check,
    update_report_latest_periods,
)
from downloader.daily.tasks.shareholding import update_shareholding
from downloader.daily.tasks.valuation import (
    add_price_close_to_valuation,
    fetch_valuation_snapshot,
    price_close_lookup_for_valuation,
    update_valuation,
    update_valuation_by_stock,
)

def daily_task_specs(query_date, listed_codes, market_closed=False):
    tasks = []
    if market_closed:
        tasks.append(
            TaskSpec(
                'price_by_stock',
                lambda: status(
                    'price_by_stock',
                    'skipped',
                    path=os.path.join(DATA_DIR, 'price'),
                    note='market_closed; price update skipped',
                ),
                path=os.path.join(DATA_DIR, 'price'),
            )
        )
    else:
        tasks.append(
            TaskSpec(
                'price_by_stock',
                lambda: update_price(query_date, listed_codes),
                path=os.path.join(DATA_DIR, 'price'),
            )
        )
    tasks.extend([
        TaskSpec(
            'institutional_per_stock',
            lambda: update_institutional(query_date, listed_codes),
            path=os.path.join(DATA_DIR, 'institutional'),
        ),
        TaskSpec(
            'margin_per_stock',
            lambda: update_margin(query_date),
            path=os.path.join(DATA_DIR, 'margin'),
        ),
        TaskSpec(
            'shareholding_per_stock',
            update_shareholding,
            path=os.path.join(DATA_DIR, 'shareholding'),
        ),
        TaskSpec(
            'ex_right_dividend_per_stock',
            lambda: update_dividend(query_date),
            path=os.path.join(DATA_DIR, 'dividend', 'ex_right_dividend'),
        ),
        TaskSpec(
            'valuation_per_stock',
            lambda: update_valuation(listed_codes),
            path=os.path.join(DATA_DIR, 'yield_pe_pb'),
        ),
        TaskSpec(
            'day_trading_per_stock',
            lambda: update_day_trading(query_date, listed_codes),
            path=os.path.join(DATA_DIR, 'day_trading'),
        ),
    ])
    return tasks


def run_daily_updates(query_date, listed_codes=None, market_closed=False):
    listed_codes = listed_codes or set()
    return run_task_specs(
        daily_task_specs(query_date, listed_codes, market_closed=market_closed),
        status,
    )


def run_snapshot_updates(listed):
    listed_codes = set(listed['Code'])
    tasks = [
        TaskSpec(
            dataset_name,
            lambda dataset_name=dataset_name, source_path=source_path, output_dir=output_dir: update_openapi_snapshot(
                dataset_name,
                source_path,
                output_dir,
                listed_codes,
            ),
            path=os.path.join(DATA_DIR, output_dir),
        )
        for dataset_name, source_path, output_dir in OPENAPI_SNAPSHOTS
    ]
    tasks.extend([
        TaskSpec(
            'investor_conference',
            lambda: update_investor_conference(listed_codes),
            path=os.path.join(DATA_DIR, 'investor_conference'),
        ),
        TaskSpec(
            'report_latest_periods',
            lambda: update_report_latest_periods(listed),
            path=os.path.join(DATA_DIR, 'report'),
        ),
    ])
    return run_task_specs(tasks, status)


def run_preflight_tasks(target_date, skip_repairs=False):
    tasks = [
        TaskSpec(
            'trading_days',
            lambda: refresh_trading_days(target_date),
            path=TRADING_DAYS_PATH,
            fail_fast=True,
        ),
    ]
    if skip_repairs:
        tasks.extend([
            TaskSpec(
                'price_repair',
                lambda: status(
                    'price_repair',
                    'skipped',
                    path=os.path.join(DATA_DIR, 'price'),
                    note='skip_repairs',
                ),
                path=os.path.join(DATA_DIR, 'price'),
            ),
            TaskSpec(
                'margin_repair',
                lambda: status(
                    'margin_repair',
                    'skipped',
                    path=os.path.join(DATA_DIR, 'margin'),
                    note='skip_repairs',
                ),
                path=os.path.join(DATA_DIR, 'margin'),
            ),
        ])
    else:
        tasks.extend([
            TaskSpec(
                'price_repair',
                repair_incomplete_price_tails,
                path=os.path.join(DATA_DIR, 'price'),
            ),
            TaskSpec(
                'margin_repair',
                lambda: repair_margin_order_tails(target_date),
                path=os.path.join(DATA_DIR, 'margin'),
            ),
        ])
    return run_task_specs(tasks, status)


def run_broker_update(args, query_date):
    if args.skip_broker:
        return status('broker_twse', 'skipped', path=str(broker_twse_dir()))
    if args.skip_daily:
        return status(
            'broker_twse',
            'skipped',
            path=str(broker_twse_dir()),
            note='skip_daily; broker is a daily market dataset',
        )
    if args.market_closed:
        return status(
            'broker_twse',
            'skipped',
            path=str(broker_twse_dir()),
            note='market_closed; broker update skipped',
        )
    return update_twse_broker(
        query_date,
        max_stocks=args.broker_max_stocks,
        max_attempts=args.broker_max_attempts,
        throttle_min=args.broker_throttle_min,
        throttle_max=args.broker_throttle_max,
    )


def main():
    configure_console_encoding()
    STATUS_COLLECTOR.clear()
    args = parse_args()
    preferred_date = parse_iso_date(args.date) if args.date else None
    target_date = preferred_date or date.today()

    run_preflight_tasks(target_date, skip_repairs=args.skip_repairs)

    scan_data_registry()
    listed = load_listed_common_stocks()
    query_date = latest_trading_day_on_or_before(target_date)
    context = UpdateContext(
        listed=listed,
        query_date=query_date,
    )
    status('source_trading_date', 'resolved', note=query_date.isoformat())

    if not args.skip_daily:
        run_daily_updates(
            context.query_date,
            context.listed_codes,
            market_closed=args.market_closed,
        )
    if not args.skip_snapshots:
        run_snapshot_updates(context.listed)
    run_task_specs(
        [TaskSpec('broker_twse', lambda: run_broker_update(args, context.query_date), path=str(broker_twse_dir()))],
        status,
    )
    return 1 if STATUS_COLLECTOR.has_failures() else 0


if __name__ == '__main__':
    raise SystemExit(main())
