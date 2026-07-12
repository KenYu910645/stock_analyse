from __future__ import annotations

import os

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['fetch_valuation_snapshot', 'price_close_lookup_for_valuation', 'add_price_close_to_valuation', 'update_valuation', 'update_valuation_by_stock']


def fetch_valuation_snapshot(listed_codes):
    rows = core.fetch_json(f'{core.TWSE_OPENAPI_BASE_URL}/exchangeReport/BWIBBU_ALL')
    out = []
    for row in rows:
        code = str(row.get('Code', '')).strip()
        if code not in listed_codes:
            continue
        out.append({
            'Date': core.yyyymmdd_to_iso(row.get('Date')),
            'Code': code,
            'Name': row.get('Name', ''),
            'Close': '',
            'DividendYield': core.clean_number(row.get('DividendYield', '')),
            'DividendYear': '',
            'PEratio': core.clean_number(row.get('PEratio', '')),
            'PBratio': core.clean_number(row.get('PBratio', '')),
            'FiscalYearQuarter': '',
        })
    return pd.DataFrame(out, columns=core.VALUATION_COLUMNS)


def price_close_lookup_for_valuation(df):
    if df.empty or not {'Date', 'Code'}.issubset(df.columns):
        return {}

    dates_by_code = (
        df[['Date', 'Code']]
        .fillna('')
        .astype(str)
        .groupby('Code')['Date']
        .apply(lambda values: {value for value in values if value})
        .to_dict()
    )
    lookup = {}
    for code, dates in dates_by_code.items():
        path = core.stock_price_path(code)
        if path is None:
            continue
        try:
            price_df = core.read_csv_canonical(path, dtype=str, usecols=['Date', 'Close']).fillna('')
        except ValueError:
            continue
        if price_df.empty:
            continue
        price_df['Date'] = price_df['Date'].astype(str)
        price_df = price_df[price_df['Date'].isin(dates)]
        for _, row in price_df.iterrows():
            close = str(row.get('Close', '')).strip()
            if close:
                lookup[(str(code), str(row['Date']))] = close
    return lookup


def add_price_close_to_valuation(df):
    if df.empty:
        return df
    result = df.copy()
    close_lookup = core.price_close_lookup_for_valuation(result)
    if not close_lookup:
        return result
    for idx, row in result.iterrows():
        if str(row.get('Close', '')).strip():
            continue
        key = (str(row.get('Code', '')).strip(), str(row.get('Date', '')).strip())
        close = close_lookup.get(key, '')
        if close:
            result.at[idx, 'Close'] = close
    return result


def update_valuation(listed_codes):
    df = core.fetch_valuation_snapshot(listed_codes)
    if df.empty:
        core.status('valuation_per_stock', 'no_source_data')
        return
    df = core.add_price_close_to_valuation(df)
    output_dir = os.path.join(core.DATA_DIR, 'yield_pe_pb')
    code_to_name = core.load_listed_common_stock_names()
    result = core.update_valuation_by_stock(df, output_dir, code_to_name)
    rows_changed = result['appended'] + result['filled']
    core.status(
        'valuation_per_stock',
        'updated' if rows_changed else 'up_to_date',
        rows_changed,
        output_dir,
        note=f"appended={result['appended']} filled_close={result['filled']}",
    )


def update_valuation_by_stock(df, output_dir, code_to_name):
    if df.empty or 'Code' not in df.columns:
        return {'appended': 0, 'filled': 0}
    core.ensure_managed_output_path(os.path.join(output_dir, '.managed'), allow_create=True)
    result = {'appended': 0, 'filled': 0}
    for code, stock_df in df.groupby('Code', sort=True):
        code = str(code).strip()
        name = code_to_name.get(code, '')
        if not name:
            names = stock_df['Name'].astype(str).str.strip()
            names = names[names != '']
            name = names.iloc[-1] if not names.empty else ''
        path = core.stock_keyed_output_path(output_dir, code, name)
        stock_result = core.append_or_fill_blank_rows(
            path,
            stock_df,
            ['Date', 'Code'],
            fill_columns=['Close'],
            fallback_columns=core.VALUATION_COLUMNS,
        )
        result['appended'] += stock_result['appended']
        result['filled'] += stock_result['filled']
    return result
