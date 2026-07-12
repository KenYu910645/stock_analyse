from __future__ import annotations

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['find_stock_price_table', 'fetch_price_rows', 'stock_price_path', 'refresh_adjusted_price_columns', 'append_price_per_stock', 'update_price']


def find_stock_price_table(payload):
    for table in payload.get('tables') or []:
        fields = table.get('fields') or []
        if len(fields) >= 9 and fields[0] == '\u8b49\u5238\u4ee3\u865f':
            return table
    raise ValueError('TWSE MI_INDEX payload did not include a stock price table.')


def fetch_price_rows(query_date, listed_codes):
    payload = core.fetch_json(
        core.TWSE_MI_INDEX_URL,
        {
            'date': core.format_yyyymmdd(query_date),
            'type': 'ALLBUT0999',
            'response': 'json',
        },
    )
    if payload.get('stat') != 'OK':
        return pd.DataFrame(), ''

    table = core.find_stock_price_table(payload)
    row_date = payload.get('date') or core.format_yyyymmdd(query_date)
    source_date = core.parse_normalized_yyyymmdd(row_date)
    if source_date is None:
        raise ValueError(f'TWSE MI_INDEX returned an invalid source date: {row_date!r}')
    today = core.date.today()
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
            'Date': core.yyyymmdd_to_iso(row_date),
            'Code': code,
            'Name': str(raw[1]).strip(),
            'Capacity': core.parse_int(raw[2]),
            'Transaction': core.parse_int(raw[3]),
            'Turnover': core.parse_int(raw[4]),
            'Open': core.parse_number(raw[5]),
            'High': core.parse_number(raw[6]),
            'Low': core.parse_number(raw[7]),
            'Close': core.parse_number(raw[8]),
            'Change': core.parse_signed_change(raw[9], raw[10]),
        })
    return pd.DataFrame(rows), core.normalize_source_date(row_date)


def stock_price_path(code):
    return core.find_latest_file(core.os.path.join(core.DATA_DIR, 'price', f'{code}_*.csv'))


def refresh_adjusted_price_columns(path, code, metadata, previous_trading_day_by_date):
    if not core.os.path.exists(path):
        return False
    raw_df = core.read_csv_canonical(path, dtype=str).fillna('')
    raw_before = raw_df[[column for column in core.PRICE_COLUMNS if column in raw_df.columns]].copy()
    events = core.price_adjustments.load_adjustment_events(code, metadata)
    instrument_type = str(metadata.at[code, 'Type']).strip() if code in metadata.index else ''
    adjusted, _merged_events, _inferred_events = core.price_adjustments.add_adjusted_columns(
        raw_df.copy(),
        events,
        previous_trading_day_by_date,
        allow_price_inferred_events=instrument_type.upper() == 'ETF',
    )
    for column in core.ADJUSTED_PRICE_COLUMNS:
        raw_df[column] = adjusted[column].astype(str)
    for column in core.PRICE_OUTPUT_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = ''
    if not raw_before.equals(raw_df[[column for column in raw_before.columns]]):
        raise ValueError(f'Raw price columns changed while adjusting {path}')
    core.to_csv_storage(raw_df[core.PRICE_OUTPUT_COLUMNS], path, index=False, encoding='utf-8-sig')
    return True


def append_price_per_stock(price_df, touched_paths):
    rows_written = 0
    for code, stock_df in price_df.groupby('Code', sort=True):
        path = core.stock_price_path(code)
        if path is None:
            path = core.price.get_stock_output_path(code)

        output_df = stock_df[core.PRICE_COLUMNS].copy()
        latest = core.latest_date_in_csv(path)
        output_df['_norm_date'] = output_df['Date'].map(core.normalize_source_date)
        output_df = output_df[output_df['_norm_date'] > latest].drop(columns=['_norm_date'])
        written = core.append_dataframe(path, output_df, fallback_columns=core.PRICE_OUTPUT_COLUMNS)
        if written:
            touched_paths[str(code)] = path
        rows_written += written
    return rows_written


def update_price(target_date, listed_codes):
    missing_by_code = {}
    dates_to_fetch = set()
    for code in sorted(listed_codes):
        path = core.stock_price_path(code)
        latest = core.latest_date_in_csv(path) if path else ''
        if path is None:
            missing_dates = [target_date]
        else:
            missing_dates = core.missing_trading_dates_after(latest, target_date)
        if missing_dates:
            missing_by_code[code] = missing_dates
            dates_to_fetch.update(missing_dates)

    if not dates_to_fetch:
        core.status('price_by_stock', 'up_to_date', path=core.os.path.join(core.DATA_DIR, 'price'))
        return

    metadata = core.price_adjustments.load_metadata()
    previous_trading_day_by_date = core.price_adjustments.load_previous_trading_day_map()
    touched_paths = {}
    per_stock_written = 0
    no_source_dates = []
    for query_date in sorted(dates_to_fetch):
        try:
            price_df, _source_date = core.fetch_price_rows(query_date, listed_codes)
        except Exception as exc:
            core.status('price', 'failed', note=f'{query_date}: {exc}')
            continue
        if price_df.empty:
            no_source_dates.append(query_date.isoformat())
            continue
        price_df = price_df[price_df['Code'].isin([
            code for code, dates in missing_by_code.items() if query_date in dates
        ])].copy()
        if price_df.empty:
            continue
        per_stock_written += core.append_price_per_stock(price_df, touched_paths)

    adjusted_files = 0
    for code, path in touched_paths.items():
        core.refresh_adjusted_price_columns(path, code, metadata, previous_trading_day_by_date)
        adjusted_files += 1

    if per_stock_written:
        core.status(
            'price_by_stock',
            'updated',
            per_stock_written,
            core.os.path.join(core.DATA_DIR, 'price'),
            note=f'fetched_dates={len(dates_to_fetch)} adjusted_files={adjusted_files}',
        )
    elif no_source_dates:
        core.status('price_by_stock', 'no_source_data', note=','.join(no_source_dates))
    else:
        core.status('price_by_stock', 'up_to_date', path=core.os.path.join(core.DATA_DIR, 'price'))
