from __future__ import annotations

import os

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['price_context_for_day_trading', 'add_day_trading_features', 'refresh_day_trading_features_for_path', 'refresh_day_trading_features', 'find_day_trading_stock_table', 'fetch_day_trading_rows', 'update_day_trading']


def price_context_for_day_trading(path):
    columns = ['Date', 'Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close']
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        price_df = core.read_csv_canonical(path, dtype=str, usecols=columns).fillna('')
    except Exception:
        return pd.DataFrame(columns=columns)
    if price_df.empty:
        return pd.DataFrame(columns=columns)
    price_df['Date'] = price_df['Date'].map(core.yyyymmdd_to_iso)
    price_df['_date_sort'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df = price_df.dropna(subset=['_date_sort']).sort_values('_date_sort')
    for column in columns:
        if column != 'Date':
            price_df[column] = core.numeric_series(price_df, column)
    return price_df[columns]


def add_day_trading_features(df, price_df):
    result = df.copy().fillna('')
    if result.empty:
        for column in core.DAY_TRADING_FEATURE_COLUMNS:
            result[column] = ''
        return result

    result['Date'] = result['Date'].map(core.yyyymmdd_to_iso)
    result['_date_sort'] = pd.to_datetime(result['Date'], errors='coerce')
    result = result.sort_values('_date_sort').drop(columns=['_date_sort']).reset_index(drop=True)

    day_volume = core.numeric_series(result, 'DayTradingVolume')
    buy_amount = core.numeric_series(result, 'DayTradingBuyAmount')
    sell_amount = core.numeric_series(result, 'DayTradingSellAmount')
    turnover = (buy_amount + sell_amount) / 2

    result['DayTradingTurnover'] = turnover
    result['DayTradingAvgBuyPrice'] = core.safe_ratio(buy_amount, day_volume)
    result['DayTradingAvgSellPrice'] = core.safe_ratio(sell_amount, day_volume)
    result['DayTradingAvgSpreadRate'] = core.safe_ratio(
        result['DayTradingAvgSellPrice'] - result['DayTradingAvgBuyPrice'],
        result['DayTradingAvgBuyPrice'],
    )

    if not price_df.empty:
        merged = result[['Date']].merge(price_df, on='Date', how='left')
        capacity = pd.to_numeric(merged['Capacity'], errors='coerce')
        price_turnover = pd.to_numeric(merged['Turnover'], errors='coerce')
        open_price = pd.to_numeric(merged['Open'], errors='coerce')
        high = pd.to_numeric(merged['High'], errors='coerce')
        low = pd.to_numeric(merged['Low'], errors='coerce')
        close = pd.to_numeric(merged['Close'], errors='coerce')

        result['DayTradingVolumeRatio'] = core.safe_ratio(day_volume, capacity)
        result['DayTradingBuyAmountRatio'] = core.safe_ratio(buy_amount, price_turnover)
        result['DayTradingSellAmountRatio'] = core.safe_ratio(sell_amount, price_turnover)
        result['DayTradingTurnoverRatio'] = core.safe_ratio(turnover, price_turnover)
        result['DayTradingAmountImbalanceRatio'] = core.safe_ratio(sell_amount - buy_amount, price_turnover)
        result['IntradayRangeRate'] = core.safe_ratio(high - low, close)
        result['OpenCloseReturn'] = core.safe_ratio(close - open_price, open_price)
    else:
        for column in [
            'DayTradingVolumeRatio',
            'DayTradingBuyAmountRatio',
            'DayTradingSellAmountRatio',
            'DayTradingTurnoverRatio',
            'DayTradingAmountImbalanceRatio',
            'IntradayRangeRate',
            'OpenCloseReturn',
        ]:
            result[column] = pd.NA

    volume_ratio = pd.to_numeric(result['DayTradingVolumeRatio'], errors='coerce')
    turnover_value = pd.to_numeric(result['DayTradingTurnover'], errors='coerce')
    volume_ratio_mean = volume_ratio.rolling(20, min_periods=20).mean()
    volume_ratio_std = volume_ratio.rolling(20, min_periods=20).std()
    turnover_mean = turnover_value.rolling(20, min_periods=20).mean()
    turnover_std = turnover_value.rolling(20, min_periods=20).std()
    result['DayTradingVolumeRatio20DayZScore'] = core.safe_ratio(
        volume_ratio - volume_ratio_mean,
        volume_ratio_std,
    )
    result['DayTradingTurnover20DayZScore'] = core.safe_ratio(
        turnover_value - turnover_mean,
        turnover_std,
    )

    for column in core.DAY_TRADING_FEATURE_COLUMNS:
        result[column] = result[column].map(lambda value: '' if pd.isna(value) else value)
    return result


def refresh_day_trading_features_for_path(path, code=None):
    if not os.path.exists(path):
        return False
    try:
        day_trading_df = core.read_csv_canonical(path, dtype=str).fillna('')
    except Exception as exc:
        core.status('day_trading_features', 'failed', path=path, note=str(exc))
        return False
    if day_trading_df.empty or 'Date' not in day_trading_df.columns:
        return False
    if code is None:
        code = os.path.basename(path).split('_', 1)[0]
    enriched = core.add_day_trading_features(
        day_trading_df,
        core.price_context_for_day_trading(core.stock_price_path(code)),
    )
    for column in core.DAY_TRADING_OUTPUT_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = ''
    extra_columns = [
        column for column in enriched.columns
        if column not in core.DAY_TRADING_OUTPUT_COLUMNS
        and column not in core.DAY_TRADING_DEPRECATED_COLUMNS
    ]
    output = enriched[core.DAY_TRADING_OUTPUT_COLUMNS + extra_columns]
    old = day_trading_df.copy()
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


def refresh_day_trading_features(paths=None):
    day_trading_dir = os.path.join(core.DATA_DIR, 'day_trading')
    if paths is None:
        paths = sorted(core.glob.glob(os.path.join(day_trading_dir, '*.csv')))
    refreshed = 0
    for path in sorted(set(paths)):
        if path.endswith('.tmp_header_migration'):
            continue
        code = os.path.basename(path).split('_', 1)[0]
        if core.refresh_day_trading_features_for_path(path, code):
            refreshed += 1
    core.status('day_trading_features', 'updated' if refreshed else 'up_to_date', refreshed, day_trading_dir)
    return refreshed


def find_day_trading_stock_table(payload):
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if fields and fields[0] == '\u8b49\u5238\u4ee3\u865f':
            return table
    return None


def fetch_day_trading_rows(query_date, listed_codes):
    payload = core.fetch_json(
        core.TWSE_DAY_TRADING_URL,
        {'date': core.format_yyyymmdd(query_date), 'response': 'json'},
    )
    if payload.get('stat') != 'OK':
        return pd.DataFrame()
    table = core.find_day_trading_stock_table(payload)
    if table is None:
        return pd.DataFrame()
    rows = []
    row_date = core.yyyymmdd_to_iso(payload.get('date') or core.format_yyyymmdd(query_date))
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
        }
        if len(raw) == 6:
            row.update({
                'SuspensionNote': str(raw[2]).strip(),
                'DayTradingVolume': core.clean_number(raw[3]),
                'DayTradingBuyAmount': core.clean_number(raw[4]),
                'DayTradingSellAmount': core.clean_number(raw[5]),
            })
        elif len(raw) >= 11:
            row.update({
                'DayTradingVolume': core.clean_number(raw[2]),
                'DayTradingBuyAmount': core.clean_number(raw[3]),
                'DayTradingSellAmount': core.clean_number(raw[4]),
            })
        rows.append(row)
    return pd.DataFrame(rows, columns=core.DAY_TRADING_COLUMNS)


def update_day_trading(query_date, listed_codes):
    df = core.fetch_day_trading_rows(query_date, listed_codes)
    if df.empty:
        core.status('day_trading_history', 'no_source_data')
        core.refresh_day_trading_features()
        return
    output_dir = os.path.join(core.DATA_DIR, 'day_trading')
    code_to_name = core.load_listed_common_stock_names()
    written = core.update_stock_keyed_by_stock(
        df,
        output_dir,
        'Code',
        ['Date', 'Code'],
        fallback_columns=core.DAY_TRADING_OUTPUT_COLUMNS,
        name_column='Name',
        code_to_name=code_to_name,
    )
    touched_paths = [
        core.stock_keyed_output_path(output_dir, code, code_to_name.get(code, ''))
        for code in df['Code'].astype(str).str.strip().unique()
    ]
    if written:
        core.refresh_day_trading_features(touched_paths)
    else:
        core.refresh_day_trading_features()
    core.status('day_trading_per_stock', 'updated' if written else 'up_to_date', written, output_dir)
