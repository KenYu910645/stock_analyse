from __future__ import annotations

import os

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['source_key_columns', 'filter_listed_openapi_df', 'update_company_basic_by_stock', 'update_dividend_openapi_by_stock', 'openapi_per_stock_output_dir', 'update_openapi_per_stock', 'normalize_material_events_openapi', 'update_openapi_snapshot']


def source_key_columns(dataset_name, df):
    if dataset_name.startswith('financial_') and {
        core.COL_YEAR,
        core.COL_SEASON,
        core.COL_CODE,
    }.issubset(df.columns):
        return [core.COL_YEAR, core.COL_SEASON, core.COL_CODE]
    if dataset_name == 'monthly_revenue' and {core.COL_MONTH, core.COL_CODE}.issubset(df.columns):
        return [core.COL_MONTH, core.COL_CODE]
    if dataset_name == 'material_events':
        columns = ['Date', 'Time', 'Code', 'Subject']
        if set(columns).issubset(df.columns):
            return columns
        columns = [core.COL_REPORT_DATE, core.COL_CODE, '\u767c\u8a00\u65e5\u671f', '\u767c\u8a00\u6642\u9593']
        if set(columns).issubset(df.columns):
            return columns
    if dataset_name == 'shareholder_meeting':
        columns = [core.COL_CODE, '\u516c\u544a\u65e5\u671f', '\u516c\u544a\u6642\u9593', '\u7a2e\u985e']
        if set(columns).issubset(df.columns):
            return columns
    if dataset_name == 'director_shareholding':
        columns = [
            core.COL_MONTH,
            core.COL_CODE,
            '\u8077\u7a31',
            '\u59d3\u540d',
            '\u76ee\u524d\u6301\u80a1',
            '\u8a2d\u8cea\u80a1\u6578',
            '\u5167\u90e8\u4eba\u95dc\u4fc2\u4eba\u76ee\u524d\u6301\u80a1\u5408\u8a08',
            '\u5167\u90e8\u4eba\u95dc\u4fc2\u4eba\u8a2d\u8cea\u80a1\u6578',
        ]
        if set(columns).issubset(df.columns):
            return columns
    if dataset_name in ('insider_transfer_pre', 'insider_transfer_untransferred'):
        preferred = [column for column in df.columns if column not in {'FetchedAt', 'SourcePath'}]
        if core.COL_REPORT_DATE in preferred and core.COL_CODE in preferred:
            return preferred
    if core.COL_REPORT_DATE in df.columns and core.COL_CODE in df.columns:
        return [core.COL_REPORT_DATE, core.COL_CODE]
    if 'Date' in df.columns and 'Code' in df.columns:
        return ['Date', 'Code']
    if 'Code' in df.columns:
        return ['Code']
    return df.columns.tolist()


def filter_listed_openapi_df(df, listed_codes):
    if df.empty:
        return df
    for column in ('Code', core.COL_CODE, '\u8b49\u5238\u4ee3\u865f', 'TWSECode'):
        if column in df.columns:
            return df[df[column].astype(str).str.strip().isin(listed_codes)].copy()
    return df


def update_company_basic_by_stock(df):
    total = 0
    refreshed = 0
    output_dir = os.path.join(core.DATA_DIR, 'company')
    core.ensure_managed_output_path(os.path.join(output_dir, '.managed'), allow_create=True)
    for code, stock_df in df.groupby(core.COL_CODE, sort=True):
        stock_df = stock_df.copy()
        short_names = stock_df.get(core.COL_SHORT_NAME, pd.Series(dtype=str)).astype(str).str.strip()
        short_names = short_names[short_names != '']
        short_name = short_names.iloc[-1] if not short_names.empty else ''
        path = core.stock_keyed_output_path(output_dir, code, short_name)
        result = core.append_or_refresh_rows(
            path,
            stock_df,
            [core.COL_REPORT_DATE, core.COL_CODE],
            refresh_fetched_at=True,
            replace_existing_on_change=True,
        )
        total += result['appended']
        refreshed += result['refreshed']
    if total:
        action = 'updated'
        rows = total
    elif refreshed:
        action = 'fetched_at_refreshed'
        rows = refreshed
    else:
        action = 'up_to_date'
        rows = 0
    core.status('company_basic_by_stock', action, rows, output_dir)


def update_dividend_openapi_by_stock(dataset_name, df):
    df = core.normalize_dividend_openapi_dates(dataset_name, df)
    code_to_name = core.load_listed_common_stock_names()
    if dataset_name == 'dividend_distribution':
        output_dir = os.path.join(core.DATA_DIR, 'dividend', 'dividend_distribution')
        result = core.update_stock_keyed_by_stock(
            df,
            output_dir,
            core.COL_CODE,
            [core.COL_REPORT_DATE, core.COL_CODE],
            name_column=core.COL_SHORT_NAME if core.COL_SHORT_NAME in df.columns else '\u516c\u53f8\u540d\u7a31',
            code_to_name=code_to_name,
            refresh_fetched_at=True,
            replace_existing_on_change=True,
        )
    elif dataset_name == 'ex_dividend_forecast_openapi':
        output_dir = os.path.join(core.DATA_DIR, 'dividend', 'ex_dividend_forecast')
        result = core.update_stock_keyed_by_stock(
            df,
            output_dir,
            'Code',
            ['Date', 'Code'],
            name_column='Name',
            code_to_name=code_to_name,
            refresh_fetched_at=True,
        )
    else:
        return False
    if result['appended']:
        core.status(f'{dataset_name}_per_stock', 'updated', result['appended'], output_dir)
    elif result['refreshed']:
        core.status(f'{dataset_name}_per_stock', 'fetched_at_refreshed', result['refreshed'], output_dir)
    else:
        core.status(f'{dataset_name}_per_stock', 'up_to_date', 0, output_dir)
    return True


def openapi_per_stock_output_dir(dataset_name, output_dir):
    if dataset_name.startswith('financial_'):
        return os.path.join(core.DATA_DIR, output_dir, dataset_name)
    if dataset_name in (
        'director_shareholding',
        'insider_transfer_pre',
        'insider_transfer_untransferred',
    ):
        return os.path.join(core.DATA_DIR, output_dir, dataset_name)
    if dataset_name in (
        'monthly_revenue',
        'sbl_available',
        'material_events',
        'shareholder_meeting',
    ):
        return os.path.join(core.DATA_DIR, output_dir)
    return ''


def update_openapi_per_stock(dataset_name, output_dir, df):
    target_dir = core.openapi_per_stock_output_dir(dataset_name, output_dir)
    if not target_dir:
        return False
    if dataset_name == 'material_events':
        df = core.normalize_material_events_openapi(df)
    key_columns = core.source_key_columns(dataset_name, df)
    code_column = core.COL_CODE if core.COL_CODE in df.columns else 'Code'
    name_column = core.COL_SHORT_NAME if core.COL_SHORT_NAME in df.columns else None
    if name_column is None:
        name_column = '\u516c\u53f8\u540d\u7a31' if '\u516c\u53f8\u540d\u7a31' in df.columns else 'Name'
    result = core.update_stock_keyed_by_stock(
        df,
        target_dir,
        code_column,
        key_columns,
        name_column=name_column,
        code_to_name=core.load_listed_common_stock_names(),
        refresh_fetched_at=True,
        replace_existing_on_change=True,
    )
    if result['appended']:
        core.status(f'{dataset_name}_per_stock', 'updated', result['appended'], target_dir)
    elif result['refreshed']:
        core.status(f'{dataset_name}_per_stock', 'fetched_at_refreshed', result['refreshed'], target_dir)
    else:
        core.status(f'{dataset_name}_per_stock', 'up_to_date', 0, target_dir)
    return True


def normalize_material_events_openapi(df):
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'Date': core.normalize_date_text(row.get(core.COL_REPORT_DATE, '')),
            'Time': str(row.get('\u767c\u8a00\u6642\u9593', '') or '').strip(),
            'Code': str(row.get(core.COL_CODE, '') or '').strip(),
            'Name': str(row.get(core.COL_SHORT_NAME, '') or '').strip(),
            'Subject': str(row.get('\u4e3b\u65e8', '') or '').strip(),
            'FactDate': core.normalize_date_text(row.get('\u4e8b\u5be6\u767c\u751f\u65e5', '')),
            'Clause': str(row.get('\u7b26\u5408\u689d\u6b3e', '') or '').strip(),
            'Description': str(row.get('\u8aaa\u660e', '') or '').strip(),
            'Spokesperson': '',
            'SpokespersonTitle': '',
            'SpokespersonPhone': '',
            'Source': 'TWSE OpenAPI',
            'SourcePath': str(row.get('SourcePath', '') or '').strip(),
            'SourceMarket': core.TWSE_MARKET,
            'DetailSeqNo': '',
            'DetailSpokeDate': core.normalize_date_text(row.get('\u767c\u8a00\u65e5\u671f', '')),
            'DetailSpokeTime': str(row.get('\u767c\u8a00\u6642\u9593', '') or '').strip(),
            'FetchedAt': str(row.get('FetchedAt', '') or '').strip(),
        })
    return pd.DataFrame(rows, columns=core.events_downloader.OUTPUT_COLUMNS)


def update_openapi_snapshot(dataset_name, source_path, output_dir, listed_codes):
    rows = core.fetch_json(f'{core.TWSE_OPENAPI_BASE_URL}{source_path}')
    if not isinstance(rows, list):
        core.status(dataset_name, 'unexpected_source_payload')
        return
    if dataset_name == 'sbl_available':
        df = pd.DataFrame([
            {
                'Date': core.date.today().isoformat(),
                'Code': str(row.get('TWSECode', '')).strip(),
                'AvailableVolume': core.clean_number(row.get('TWSEAvailableVolume', '')),
            }
            for row in rows
            if str(row.get('TWSECode', '')).strip() in listed_codes
        ])
    else:
        df = core.filter_listed_openapi_df(pd.DataFrame(rows), listed_codes)
    if df.empty:
        core.status(dataset_name, 'no_source_data')
        return
    df.insert(0, 'FetchedAt', core.datetime.now().isoformat(timespec='seconds'))
    df.insert(1, 'SourcePath', source_path)
    if dataset_name == 'company_basic':
        core.update_company_basic_by_stock(df)
        return
    if dataset_name in ('dividend_distribution', 'ex_dividend_forecast_openapi'):
        core.update_dividend_openapi_by_stock(dataset_name, df)
        return
    if core.update_openapi_per_stock(dataset_name, output_dir, df):
        return
    raise ValueError(
        f'Unsupported OpenAPI snapshot dataset {dataset_name!r}; '
        'register an explicit canonical per-stock writer before enabling it.'
    )
