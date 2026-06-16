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
from lxml import html


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from downloader import ex_right_dividend
from downloader import institutional_investors
from downloader import margin_trading
from downloader import price
from downloader import tdcc_shareholding


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass


DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
STOCK_METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')
TWSE_OPENAPI_BASE_URL = 'https://openapi.twse.com.tw/v1'
TWSE_MI_INDEX_URL = 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX'
TWSE_DAY_TRADING_URL = 'https://www.twse.com.tw/rwd/zh/dayTrading/TWTB4U'
TWSE_T86_URL = 'https://www.twse.com.tw/rwd/zh/fund/T86'
IR_ENGAGE_CONFERENCE_URL = 'https://irengage.taiwanindex.com.tw/conferenceList'

COMMON_STOCK_TYPE = '\u80a1\u7968'
TWSE_MARKET = '\u4e0a\u5e02'
COL_REPORT_DATE = '\u51fa\u8868\u65e5\u671f'
COL_CODE = '\u516c\u53f8\u4ee3\u865f'
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
    'MarketVolumeRatio',
    'MarketBuyAmountRatio',
    'MarketSellAmountRatio',
    'TotalVolume',
    'TotalBuyAmount',
    'TotalSellAmount',
]

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
        help='Kept for compatibility. Broker branch data is not updated here.',
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
    return parser.parse_args()


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
    text = str(value or '').strip().replace('/', '').replace('-', '')
    if not text:
        return ''
    if len(text) == 7 and text.isdigit():
        return f'{int(text[:3]) + 1911:04d}{text[3:5]}{text[5:7]}'
    if len(text) == 8 and text.isdigit():
        return text
    return text


def yyyymmdd_to_iso(value):
    text = normalize_source_date(value)
    if len(text) == 8 and text.isdigit():
        return f'{text[:4]}-{text[4:6]}-{text[6:8]}'
    return text


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
    catalog.to_csv(STOCK_METADATA_PATH, index=False, encoding='utf-8-sig')


def load_listed_common_stocks():
    ensure_stock_metadata()
    df = pd.read_csv(STOCK_METADATA_PATH, dtype=str).fillna('')
    df['Code'] = df['Code'].astype(str).str.strip()
    mask = (
        (df['Market'] == TWSE_MARKET)
        & (df['Type'] == COMMON_STOCK_TYPE)
        & df['Code'].str.match(r'^\d{4}$')
    )
    listed = df.loc[mask, ['Code', 'Name']].drop_duplicates('Code')
    return listed.sort_values('Code').reset_index(drop=True)


def status(dataset, action, rows=0, path=None, note=''):
    parts = [f'{dataset}: {action}', f'rows={rows}']
    if path:
        parts.append(f'path={path}')
    if note:
        parts.append(note)
    print(' | '.join(parts), flush=True)


def safe_read_csv_rows(path):
    if not os.path.exists(path):
        return 0
    return len(pd.read_csv(path, dtype=str))


def get_csv_columns(path, fallback_columns):
    if os.path.exists(path):
        return pd.read_csv(path, nrows=0).columns.tolist()
    return list(fallback_columns)


def append_dataframe(path, df, fallback_columns=None):
    if df.empty:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = get_csv_columns(path, fallback_columns or df.columns.tolist())
    for column in columns:
        if column not in df.columns:
            df[column] = ''
    extra_columns = [column for column in df.columns if column not in columns]
    if extra_columns and not os.path.exists(path):
        columns.extend(extra_columns)
    df = df[columns]
    write_header = not os.path.exists(path)
    df.to_csv(
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
        df = pd.read_csv(path, dtype=str, usecols=key_columns).fillna('')
    except ValueError:
        return set()
    return build_key_set(df, key_columns)


def build_key_set(df, key_columns):
    if df.empty:
        return set()
    key_df = df[key_columns].astype(str).copy()
    if 'Date' in key_df.columns:
        key_df['Date'] = key_df['Date'].map(normalize_source_date)
    if 'ex_date' in key_df.columns:
        key_df['ex_date'] = key_df['ex_date'].map(normalize_source_date)
    return set(tuple(row) for row in key_df.itertuples(index=False, name=None))


def append_new_by_keys(path, df, key_columns, fallback_columns=None):
    existing = read_existing_keys(path, key_columns)
    if existing:
        key_df = df[key_columns].astype(str).copy()
        if 'Date' in key_df.columns:
            key_df['Date'] = key_df['Date'].map(normalize_source_date)
        if 'ex_date' in key_df.columns:
            key_df['ex_date'] = key_df['ex_date'].map(normalize_source_date)
        key_series = key_df.apply(tuple, axis=1)
        df = df.loc[~key_series.isin(existing)].copy()
    return append_dataframe(path, df, fallback_columns=fallback_columns)


def latest_date_in_csv(path, date_column='Date'):
    if not os.path.exists(path):
        return ''
    try:
        df = pd.read_csv(path, dtype=str, usecols=[date_column]).dropna()
    except ValueError:
        return ''
    if df.empty:
        return ''
    normalized = df[date_column].map(normalize_source_date)
    normalized = normalized[normalized.astype(bool)]
    return normalized.max() if not normalized.empty else ''


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


def find_stock_price_table(payload):
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if len(fields) >= 9 and fields[0] == '\u8b49\u5238\u4ee3\u865f':
            return table
    raise ValueError('TWSE MI_INDEX payload did not include a stock price table.')


def fetch_price_rows(query_date, listed_codes):
    payload = fetch_json(
        TWSE_MI_INDEX_URL,
        {
            'date': format_yyyymmdd(query_date),
            'type': 'ALLBUT0999',
            'response': 'json',
        },
    )
    if payload.get('stat') != 'OK':
        return pd.DataFrame(), ''

    table = find_stock_price_table(payload)
    row_date = payload.get('date') or format_yyyymmdd(query_date)
    source_date = parse_normalized_yyyymmdd(row_date)
    if source_date is None:
        raise ValueError(f'TWSE MI_INDEX returned an invalid source date: {row_date!r}')
    today = date.today()
    if source_date > query_date or source_date > today:
        raise ValueError(
            'TWSE MI_INDEX returned a future source date: '
            f'{source_date.isoformat()} for query {query_date.isoformat()} '
            f'with today={today.isoformat()}'
        )
    rows = []
    for raw in table.get('data') or []:
        if len(raw) < 16:
            continue
        code = str(raw[0]).strip()
        if code not in listed_codes:
            continue
        rows.append({
            'Date': yyyymmdd_to_iso(row_date),
            'Code': code,
            'Name': str(raw[1]).strip(),
            'Capacity': parse_int(raw[2]),
            'Transaction': parse_int(raw[3]),
            'Turnover': parse_int(raw[4]),
            'Open': parse_number(raw[5]),
            'High': parse_number(raw[6]),
            'Low': parse_number(raw[7]),
            'Close': parse_number(raw[8]),
            'Change': parse_signed_change(raw[9], raw[10]),
        })
    return pd.DataFrame(rows), normalize_source_date(row_date)


def resolve_trading_date(preferred_date, listed_codes):
    start = preferred_date or date.today()
    if preferred_date is not None:
        try:
            df, source_date = fetch_price_rows(preferred_date, listed_codes)
        except Exception:
            return preferred_date, pd.DataFrame(), ''
        return preferred_date, df, source_date

    for offset in range(8):
        query_date = start - timedelta(days=offset)
        try:
            df, source_date = fetch_price_rows(query_date, listed_codes)
        except Exception:
            continue
        if not df.empty:
            return query_date, df, source_date
    return start, pd.DataFrame(), ''


def stock_price_path(code):
    return find_latest_file(os.path.join(DATA_DIR, 'price', f'{code}_*_to_*.csv'))


def append_price_per_stock(price_df):
    rows_written = 0
    for code, stock_df in price_df.groupby('Code', sort=True):
        path = stock_price_path(code)
        if path is None:
            start_month = stock_df['Date'].iloc[0].replace('-', '')[:6]
            path = os.path.join(DATA_DIR, 'price', f'{code}_{start_month}_to_{start_month}.csv')

        output_df = stock_df[PRICE_COLUMNS].copy()
        latest = latest_date_in_csv(path)
        output_df['_norm_date'] = output_df['Date'].map(normalize_source_date)
        output_df = output_df[output_df['_norm_date'] > latest].drop(columns=['_norm_date'])
        rows_written += append_dataframe(path, output_df, fallback_columns=PRICE_COLUMNS)
    return rows_written


def update_price(price_df, source_date):
    if price_df.empty:
        status('price', 'no_source_data')
        return
    daily_path = os.path.join(DATA_DIR, 'price', f'twse_price_{source_date}.csv')
    daily_written = append_new_by_keys(
        daily_path,
        price_df,
        ['Date', 'Code'],
        fallback_columns=['Date', 'Code', 'Name'] + PRICE_COLUMNS[1:],
    )
    per_stock_written = append_price_per_stock(price_df)
    status('price_daily_file', 'updated' if daily_written else 'up_to_date', daily_written, daily_path)
    status('price_by_stock', 'updated' if per_stock_written else 'up_to_date', per_stock_written)


def update_institutional(query_date, listed_codes):
    payload = fetch_json(
        TWSE_T86_URL,
        {
            'date': format_yyyymmdd(query_date),
            'selectType': 'ALLBUT0999',
            'response': 'json',
        },
    )
    if payload.get('stat') != 'OK':
        status('institutional', 'no_source_data', note=str(payload.get('stat')))
        return
    row_date = payload.get('date') or format_yyyymmdd(query_date)
    rows = []
    for raw in payload.get('data') or []:
        values = raw.get('value', raw) if isinstance(raw, dict) else raw
        if len(values) < 19:
            continue
        code = str(values[0]).strip()
        if code not in listed_codes:
            continue
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
    if not rows:
        status('institutional', 'no_listed_rows')
        return
    df = institutional_investors.normalize_dataframe(rows)
    path = find_main_history_file(os.path.join(DATA_DIR, 'institutional', 'twse_institutional_investors_*_to_*.csv'))
    if path is None:
        path = institutional_investors.get_output_path(query_date, query_date)
    written = append_new_by_keys(path, df, ['Date', 'Code'], fallback_columns=institutional_investors.OUTPUT_COLUMNS)
    status('institutional_history', 'updated' if written else 'up_to_date', written, path)
    update_by_stock(df, os.path.join(DATA_DIR, 'institutional', 'by_stock'))


def update_margin(query_date):
    manifest = margin_trading.download_range(query_date, query_date, rebuild_only=False)
    day_path = manifest['output_csv']
    if manifest['total_rows'] == 0:
        status('margin', 'no_source_data')
        return
    day_df = pd.read_csv(day_path, dtype=str).fillna('')
    history_path = find_main_history_file(os.path.join(DATA_DIR, 'margin', 'twse_margin_stocks_*_to_*.csv'))
    if history_path is None:
        history_path = day_path
    written = append_new_by_keys(history_path, day_df, ['Date', 'Code'], fallback_columns=margin_trading.OUTPUT_COLUMNS)
    status('margin_history', 'updated' if written else 'up_to_date', written, history_path)


def update_shareholding():
    session = tdcc_shareholding.make_session()
    response = session.get(tdcc_shareholding.TDCC_OPEN_DATA_URL, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8-sig'
    preview = pd.read_csv(io.StringIO(response.text), dtype=str, nrows=1)
    latest_date = str(preview.iloc[0, 0]).strip()
    listed_path = os.path.join(
        tdcc_shareholding.LISTED_DIR,
        f'tdcc_shareholding_listed_{latest_date}.csv',
    )
    if os.path.exists(listed_path):
        status('shareholding', 'up_to_date', safe_read_csv_rows(listed_path), listed_path, note=f'source_date={latest_date}')
        return
    latest_date, _raw_path, listed_path, _raw_count, listed_count = (
        tdcc_shareholding.download_latest_open_data(session)
    )
    status('shareholding', 'updated', listed_count, listed_path, note=f'source_date={latest_date}')


def update_dividend(query_date):
    history_path = find_main_history_file(os.path.join(DATA_DIR, 'dividend', 'twse_ex_right_dividend_merged_*_to_*.csv'))
    latest = latest_date_in_csv(history_path, 'ex_date') if history_path else ''
    start = datetime.strptime(latest, '%Y%m%d').date() + timedelta(days=1) if latest else query_date
    if start > query_date:
        status('ex_right_dividend_history', 'up_to_date', path=history_path)
        return
    df = ex_right_dividend.download('merged', start, query_date, include_details=True)
    if df.empty:
        status('ex_right_dividend_history', 'no_source_data', path=history_path)
        return
    if history_path is None:
        history_path = ex_right_dividend.get_output_path('merged', start, query_date)
    written = append_new_by_keys(history_path, df, ['ex_date', 'stock_id'], fallback_columns=ex_right_dividend.FINAL_COLUMNS)
    status('ex_right_dividend_history', 'updated' if written else 'up_to_date', written, history_path)


def find_day_trading_stock_table(payload):
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if fields and fields[0] == '\u8b49\u5238\u4ee3\u865f':
            return table
    return None


def fetch_day_trading_rows(query_date, listed_codes):
    payload = fetch_json(TWSE_DAY_TRADING_URL, {'date': format_yyyymmdd(query_date), 'response': 'json'})
    if payload.get('stat') != 'OK':
        return pd.DataFrame()
    table = find_day_trading_stock_table(payload)
    if table is None:
        return pd.DataFrame()
    rows = []
    row_date = yyyymmdd_to_iso(payload.get('date') or format_yyyymmdd(query_date))
    for raw in table.get('data') or []:
        if len(raw) < 6:
            continue
        code = str(raw[0]).strip()
        if code not in listed_codes:
            continue
        row = {
            'Date': row_date,
            'Code': code,
            'Name': str(raw[1]).strip(),
            'SuspensionNote': '',
            'DayTradingVolume': '',
            'DayTradingBuyAmount': '',
            'DayTradingSellAmount': '',
            'MarketVolumeRatio': '',
            'MarketBuyAmountRatio': '',
            'MarketSellAmountRatio': '',
            'TotalVolume': '',
            'TotalBuyAmount': '',
            'TotalSellAmount': '',
        }
        if len(raw) == 6:
            row.update({
                'SuspensionNote': str(raw[2]).strip(),
                'DayTradingVolume': clean_number(raw[3]),
                'DayTradingBuyAmount': clean_number(raw[4]),
                'DayTradingSellAmount': clean_number(raw[5]),
            })
        elif len(raw) >= 11:
            row.update({
                'DayTradingVolume': clean_number(raw[2]),
                'DayTradingBuyAmount': clean_number(raw[3]),
                'DayTradingSellAmount': clean_number(raw[4]),
                'MarketVolumeRatio': clean_number(raw[5]),
                'MarketBuyAmountRatio': clean_number(raw[6]),
                'MarketSellAmountRatio': clean_number(raw[7]),
                'TotalVolume': clean_number(raw[8]),
                'TotalBuyAmount': clean_number(raw[9]),
                'TotalSellAmount': clean_number(raw[10]),
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=DAY_TRADING_COLUMNS)


def update_day_trading(query_date, listed_codes):
    df = fetch_day_trading_rows(query_date, listed_codes)
    if df.empty:
        status('day_trading_history', 'no_source_data')
        return
    path = find_main_history_file(os.path.join(DATA_DIR, 'day_trading', 'twse_day_trading_history_*_to_*.csv'))
    if path is None:
        path = os.path.join(DATA_DIR, 'day_trading', f'twse_day_trading_history_{format_yyyymmdd(query_date)}_to_{format_yyyymmdd(query_date)}.csv')
    written = append_new_by_keys(path, df, ['Date', 'Code'], fallback_columns=DAY_TRADING_COLUMNS)
    status('day_trading_history', 'updated' if written else 'up_to_date', written, path)
    update_by_stock(df, os.path.join(DATA_DIR, 'day_trading', 'by_stock'), DAY_TRADING_COLUMNS)


def fetch_valuation_snapshot(listed_codes):
    rows = fetch_json(f'{TWSE_OPENAPI_BASE_URL}/exchangeReport/BWIBBU_ALL')
    out = []
    for row in rows:
        code = str(row.get('Code', '')).strip()
        if code not in listed_codes:
            continue
        out.append({
            'Date': yyyymmdd_to_iso(row.get('Date')),
            'Code': code,
            'Name': row.get('Name', ''),
            'Close': '',
            'DividendYield': clean_number(row.get('DividendYield', '')),
            'DividendYear': '',
            'PEratio': clean_number(row.get('PEratio', '')),
            'PBratio': clean_number(row.get('PBratio', '')),
            'FiscalYearQuarter': '',
        })
    return pd.DataFrame(out, columns=VALUATION_COLUMNS)


def update_valuation(listed_codes):
    df = fetch_valuation_snapshot(listed_codes)
    if df.empty:
        status('valuation_history', 'no_source_data')
        return
    path = find_main_history_file(os.path.join(DATA_DIR, 'dividend_pe_pb', 'twse_valuation_history_*_to_*.csv'))
    if path is None:
        source_date = normalize_source_date(df['Date'].iloc[0])
        path = os.path.join(DATA_DIR, 'dividend_pe_pb', f'twse_valuation_history_{source_date}_to_{source_date}.csv')
    written = append_new_by_keys(path, df, ['Date', 'Code'], fallback_columns=VALUATION_COLUMNS)
    status('valuation_history', 'updated' if written else 'up_to_date', written, path)
    update_by_stock(df, os.path.join(DATA_DIR, 'dividend_pe_pb', 'by_stock'), VALUATION_COLUMNS)


def update_by_stock(df, output_dir, fallback_columns=None):
    if df.empty or 'Code' not in df.columns:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    for code, stock_df in df.groupby('Code', sort=True):
        names = stock_df['Name'].astype(str).str.strip() if 'Name' in stock_df.columns else pd.Series(dtype=str)
        names = names[names != '']
        name = names.iloc[-1] if not names.empty else ''
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', '_', name).strip('._ ')
        filename = f'{code}_{safe_name}.csv' if safe_name else f'{code}.csv'
        path = os.path.join(output_dir, filename)
        total += append_new_by_keys(path, stock_df, ['Date', 'Code'], fallback_columns=fallback_columns)
    status(f'{os.path.basename(os.path.dirname(output_dir))}_by_stock', 'updated' if total else 'up_to_date', total, output_dir)
    return total


def source_key_columns(dataset_name, df):
    if dataset_name.startswith('financial_') and {COL_YEAR, COL_SEASON, COL_CODE}.issubset(df.columns):
        return [COL_YEAR, COL_SEASON, COL_CODE]
    if dataset_name == 'monthly_revenue' and {COL_MONTH, COL_CODE}.issubset(df.columns):
        return [COL_MONTH, COL_CODE]
    if COL_REPORT_DATE in df.columns and COL_CODE in df.columns:
        return [COL_REPORT_DATE, COL_CODE]
    if 'Date' in df.columns and 'Code' in df.columns:
        return ['Date', 'Code']
    if 'Code' in df.columns:
        return ['Code']
    return df.columns.tolist()


def filter_listed_openapi_df(df, listed_codes):
    if df.empty:
        return df
    for column in ('Code', COL_CODE, '\u8b49\u5238\u4ee3\u865f', 'TWSECode'):
        if column in df.columns:
            return df[df[column].astype(str).str.strip().isin(listed_codes)].copy()
    return df


def update_openapi_snapshot(dataset_name, source_path, output_dir, listed_codes):
    rows = fetch_json(f'{TWSE_OPENAPI_BASE_URL}{source_path}')
    if not isinstance(rows, list):
        status(dataset_name, 'unexpected_source_payload')
        return
    if dataset_name == 'sbl_available':
        df = pd.DataFrame([
            {
                'Date': date.today().isoformat(),
                'Code': str(row.get('TWSECode', '')).strip(),
                'AvailableVolume': clean_number(row.get('TWSEAvailableVolume', '')),
            }
            for row in rows
            if str(row.get('TWSECode', '')).strip() in listed_codes
        ])
    else:
        df = filter_listed_openapi_df(pd.DataFrame(rows), listed_codes)
    if df.empty:
        status(dataset_name, 'no_source_data')
        return
    df.insert(0, 'FetchedAt', datetime.now().isoformat(timespec='seconds'))
    df.insert(1, 'SourcePath', source_path)
    path = os.path.join(DATA_DIR, output_dir, f'twse_{dataset_name}_history.csv')
    key_columns = source_key_columns(dataset_name, df)
    written = append_new_by_keys(path, df, key_columns)
    status(dataset_name, 'updated' if written else 'up_to_date', written, path)


def parse_irengage_conference_rows(html_text):
    document = html.fromstring(html_text)
    rows = []
    for table in document.xpath('//table'):
        for tr in table.xpath('.//tr'):
            cells = [' '.join(cell.xpath('.//text()')).strip() for cell in tr.xpath('./th|./td')]
            if len(cells) >= 5 and re.match(r'^\d{4}/\d{2}/\d{2}$', cells[0]):
                rows.append({
                    'Date': cells[0].replace('/', '-'),
                    'Time': cells[1],
                    'Company': cells[2],
                    'Location': cells[3],
                    'Message': cells[4],
                    'Download': cells[5] if len(cells) > 5 else '',
                })
    return pd.DataFrame(rows)


def update_investor_conference(listed_codes):
    response = requests.get(IR_ENGAGE_CONFERENCE_URL, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8'
    df = parse_irengage_conference_rows(response.text)
    if df.empty:
        status('investor_conference', 'no_source_data')
        return
    df['Code'] = df['Company'].astype(str).str.extract(r'(\d{4})', expand=False)
    df = df[df['Code'].isin(listed_codes)].copy()
    df.insert(0, 'FetchedAt', datetime.now().isoformat(timespec='seconds'))
    df.insert(1, 'SourcePath', IR_ENGAGE_CONFERENCE_URL)
    path = os.path.join(DATA_DIR, 'investor_conference', 'twse_investor_conference_irengage_history.csv')
    written = append_new_by_keys(path, df, ['Date', 'Time', 'Code', 'Message'])
    status('investor_conference', 'updated' if written else 'up_to_date', written, path)


def run_daily_updates(query_date, price_df, source_date, listed_codes):
    update_price(price_df, source_date)
    update_institutional(query_date, listed_codes)
    update_margin(query_date)
    update_shareholding()
    update_dividend(query_date)
    update_valuation(listed_codes)
    update_day_trading(query_date, listed_codes)


def run_snapshot_updates(listed_codes):
    for dataset_name, source_path, output_dir in OPENAPI_SNAPSHOTS:
        try:
            update_openapi_snapshot(dataset_name, source_path, output_dir, listed_codes)
        except Exception as exc:
            status(dataset_name, 'failed', note=str(exc))
    try:
        update_investor_conference(listed_codes)
    except Exception as exc:
        status('investor_conference', 'failed', note=str(exc))


def main():
    args = parse_args()
    listed = load_listed_common_stocks()
    listed_codes = set(listed['Code'])
    preferred_date = parse_iso_date(args.date) if args.date else None

    query_date, price_df, source_date = resolve_trading_date(preferred_date, listed_codes)
    if source_date:
        status('source_trading_date', 'resolved', note=yyyymmdd_to_iso(source_date))
    else:
        status('source_trading_date', 'not_found')

    if not args.skip_daily and source_date:
        run_daily_updates(query_date, price_df, source_date, listed_codes)
    if not args.skip_snapshots:
        run_snapshot_updates(listed_codes)
    if args.skip_broker:
        status('broker', 'skipped')
    else:
        status('broker', 'skipped', note='official/paid source pending; not updated by this script')


if __name__ == '__main__':
    main()

