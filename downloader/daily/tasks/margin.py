from __future__ import annotations

import glob
import os

import pandas as pd
import requests

from downloader.daily.tasks._runtime import core


__all__ = ['price_context_for_margin', 'add_margin_features', 'refresh_margin_features_for_path', 'refresh_margin_features', 'update_margin']


def price_context_for_margin(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['Date', 'Close', 'Turnover20DayAverage'])
    try:
        price_df = core.read_csv_canonical(path, dtype=str, usecols=['Date', 'Close', 'Turnover']).fillna('')
    except Exception:
        return pd.DataFrame(columns=['Date', 'Close', 'Turnover20DayAverage'])
    if price_df.empty:
        return pd.DataFrame(columns=['Date', 'Close', 'Turnover20DayAverage'])
    price_df['Date'] = price_df['Date'].map(core.yyyymmdd_to_iso)
    price_df['_date_sort'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df = price_df.dropna(subset=['_date_sort']).sort_values('_date_sort')
    price_df['Close'] = core.numeric_series(price_df, 'Close')
    price_df['Turnover'] = core.numeric_series(price_df, 'Turnover')
    price_df['Turnover20DayAverage'] = price_df['Turnover'].rolling(20, min_periods=20).mean()
    return price_df[['Date', 'Close', 'Turnover20DayAverage']]


def add_margin_features(df, price_df):
    result = df.copy().fillna('')
    if result.empty:
        for column in core.MARGIN_FEATURE_COLUMNS:
            result[column] = ''
        return result

    result['Date'] = result['Date'].map(core.yyyymmdd_to_iso)
    result['_date_sort'] = pd.to_datetime(result['Date'], errors='coerce')
    result = result.sort_values('_date_sort').drop(columns=['_date_sort']).reset_index(drop=True)

    margin_balance = core.numeric_series(result, 'MarginCurrentBalance')
    margin_limit = core.numeric_series(result, 'MarginNextDayLimit')
    short_balance = core.numeric_series(result, 'ShortCurrentBalance')

    result['MarginFinancingUsageRate'] = core.safe_ratio(margin_balance, margin_limit)
    result['MarginBalance20DayChangeRate'] = core.safe_ratio(
        margin_balance - margin_balance.shift(20),
        margin_balance.shift(20),
    )
    result['ShortMarginBalanceRatio'] = core.safe_ratio(short_balance, margin_balance)

    if not price_df.empty:
        merged = result[['Date']].merge(price_df, on='Date', how='left')
        close = pd.to_numeric(merged['Close'], errors='coerce')
        turnover_avg = pd.to_numeric(merged['Turnover20DayAverage'], errors='coerce')
        margin_market_value = margin_balance * close * 1000
        result['MarginMarketValue'] = margin_market_value
        result['MarginMarketValueTo20DayAvgTurnover'] = core.safe_ratio(margin_market_value, turnover_avg)
    else:
        result['MarginMarketValue'] = pd.NA
        result['MarginMarketValueTo20DayAvgTurnover'] = pd.NA

    for column in core.MARGIN_FEATURE_COLUMNS:
        result[column] = result[column].map(lambda value: '' if pd.isna(value) else value)
    return result


def refresh_margin_features_for_path(path, code=None):
    if not os.path.exists(path):
        return False
    try:
        margin_df = core.read_csv_canonical(path, dtype=str).fillna('')
    except Exception as exc:
        core.status('margin_features', 'failed', path=path, note=str(exc))
        return False
    if margin_df.empty or 'Date' not in margin_df.columns:
        return False
    if code is None:
        code = os.path.basename(path).split('_', 1)[0]
    price_path = core.stock_price_path(code)
    enriched = core.add_margin_features(margin_df, core.price_context_for_margin(price_path))
    for column in core.MARGIN_OUTPUT_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = ''
    extra_columns = [column for column in enriched.columns if column not in core.MARGIN_OUTPUT_COLUMNS]
    output = enriched[core.MARGIN_OUTPUT_COLUMNS + extra_columns]
    old = margin_df.copy()
    if list(old.columns) != list(output.columns):
        core.to_csv_storage(output, path, index=False, encoding='utf-8-sig')
        return True
    for column in output.columns:
        if column not in old.columns:
            old[column] = ''
    old = old[output.columns].fillna('')
    comparable = output.fillna('').astype(str)
    if old.astype(str).equals(comparable):
        return False
    core.to_csv_storage(output, path, index=False, encoding='utf-8-sig')
    return True


def refresh_margin_features(paths=None):
    margin_dir = os.path.join(core.DATA_DIR, 'margin')
    if paths is None:
        paths = sorted(glob.glob(os.path.join(margin_dir, '*.csv')))
    refreshed = 0
    for path in sorted(set(paths)):
        code = os.path.basename(path).split('_', 1)[0]
        if core.refresh_margin_features_for_path(path, code):
            refreshed += 1
    core.status('margin_features', 'updated' if refreshed else 'up_to_date', refreshed, margin_dir)
    return refreshed


def update_margin(target_date):
    code_to_name = core.load_listed_common_stock_names()
    listed_codes = set(code_to_name)
    allowed_dates = set(
        core.recent_trading_dates(
            target_date,
            lookback_days=core.MARGIN_REPAIR_LOOKBACK_DAYS,
        )
    )
    missing_by_code = {}
    dates_to_fetch = set()
    for code in sorted(listed_codes):
        path = core.stock_keyed_output_path(
            os.path.join(core.DATA_DIR, 'margin'),
            code,
            code_to_name.get(code, ''),
        )
        latest = core.latest_date_in_csv(path) if os.path.exists(path) else ''
        missing_dates = [
            value for value in core.missing_trading_dates_after(latest, target_date)
            if value in allowed_dates
        ]
        if missing_dates:
            missing_by_code[code] = missing_dates
            dates_to_fetch.update(missing_dates)

    touched_paths = set()
    if not dates_to_fetch:
        core.status('margin_per_stock', 'up_to_date', path=os.path.join(core.DATA_DIR, 'margin'))
        core.refresh_margin_features()
        return

    session = requests.Session()
    total_written = 0
    no_source_dates = []
    failed_dates = []
    for query_date in sorted(dates_to_fetch):
        try:
            payload = core.margin_trading.fetch_payload(session, query_date)
            rows = core.margin_trading.parse_payload_rows(payload, query_date)
        except Exception as exc:
            failed_dates.append(f'{query_date.isoformat()}:{exc}')
            core.status('margin', 'failed', note=f'{query_date.isoformat()}: {exc}')
            continue
        if not rows:
            no_source_dates.append(query_date.isoformat())
            continue
        needed_codes = {
            code for code, dates in missing_by_code.items()
            if query_date in dates
        }
        day_df = pd.DataFrame(rows, columns=core.margin_trading.OUTPUT_COLUMNS).fillna('')
        day_df = day_df[day_df['Code'].astype(str).str.strip().isin(needed_codes)].copy()
        if day_df.empty:
            continue
        written = core.update_stock_keyed_by_stock(
            day_df,
            os.path.join(core.DATA_DIR, 'margin'),
            'Code',
            ['Date', 'Code'],
            fallback_columns=core.MARGIN_OUTPUT_COLUMNS,
            name_column='Name',
            code_to_name=code_to_name,
        )
        total_written += written
        if written:
            for code in day_df['Code'].astype(str).str.strip().unique():
                touched_paths.add(
                    core.stock_keyed_output_path(
                        os.path.join(core.DATA_DIR, 'margin'),
                        code,
                        code_to_name.get(code, ''),
                    )
                )
    if touched_paths:
        core.refresh_margin_features(touched_paths)
    if total_written:
        core.status(
            'margin_per_stock',
            'updated',
            total_written,
            os.path.join(core.DATA_DIR, 'margin'),
            note=f'fetched_dates={len(dates_to_fetch)} failed_dates={len(failed_dates)}',
        )
    elif failed_dates:
        core.status('margin_per_stock', 'failed', len(failed_dates), os.path.join(core.DATA_DIR, 'margin'))
    elif no_source_dates:
        core.status('margin_per_stock', 'no_source_data', note=','.join(no_source_dates))
    else:
        core.status('margin_per_stock', 'up_to_date', path=os.path.join(core.DATA_DIR, 'margin'))
