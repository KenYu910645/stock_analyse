from __future__ import annotations

import glob
import os
from datetime import date

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['previous_quarter', 'next_quarter', 'latest_completed_report_period', 'latest_report_period_in_csv', 'latest_financial_period_by_code', 'report_periods_to_check', 'update_report_latest_periods']


def previous_quarter(year, quarter):
    if quarter > 1:
        return year, quarter - 1
    return year - 1, 4


def next_quarter(year, quarter):
    if quarter < 4:
        return year, quarter + 1
    return year + 1, 1


def latest_completed_report_period(today=None):
    today = today or date.today()
    current_quarter = ((today.month - 1) // 3) + 1
    return core.previous_quarter(today.year, current_quarter)


def latest_report_period_in_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = core.read_csv_canonical(path, dtype=str, usecols=['Year', 'Quarter']).fillna('')
    except ValueError:
        return None
    if df.empty:
        return None
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce')
    df = df.dropna(subset=['Year', 'Quarter'])
    if df.empty:
        return None
    row = df.sort_values(['Year', 'Quarter']).iloc[-1]
    return int(row['Year']), int(row['Quarter'])


def latest_financial_period_by_code():
    periods = {}
    financial_dir = os.path.join(core.DATA_DIR, 'financial')
    if not os.path.isdir(financial_dir):
        return periods
    for path in glob.glob(os.path.join(financial_dir, '*', '*.csv')):
        code = os.path.basename(path).split('_', 1)[0]
        try:
            df = pd.read_csv(path, dtype=str, usecols=['\u5e74\u5ea6', '\u5b63\u5225']).fillna('')
        except ValueError:
            continue
        if df.empty:
            continue
        df['\u5e74\u5ea6'] = pd.to_numeric(df['\u5e74\u5ea6'], errors='coerce')
        df['\u5b63\u5225'] = pd.to_numeric(df['\u5b63\u5225'], errors='coerce')
        df = df.dropna(subset=['\u5e74\u5ea6', '\u5b63\u5225'])
        if df.empty:
            continue
        row = df.sort_values(['\u5e74\u5ea6', '\u5b63\u5225']).iloc[-1]
        period = int(row['\u5e74\u5ea6']) + 1911, int(row['\u5b63\u5225'])
        if code not in periods or period > periods[code]:
            periods[code] = period
    return periods


def report_periods_to_check(path, latest_available=None, today=None):
    latest_available = latest_available or core.latest_completed_report_period(today)
    latest_existing = core.latest_report_period_in_csv(path)
    if latest_existing is None:
        return [latest_available]
    if latest_existing >= latest_available:
        return []
    periods = [latest_available]
    next_period = core.next_quarter(*latest_existing)
    while next_period <= latest_available:
        periods.append(next_period)
        next_period = core.next_quarter(*next_period)
    return sorted(set(periods))[-2:]


def update_report_latest_periods(listed):
    session = core.report.create_session()
    total_appended = 0
    total_failures = 0
    total_no_source = 0
    total_skipped_current = 0
    checked_periods = set()
    output_dir = os.path.join(core.DATA_DIR, 'report')
    core.ensure_managed_output_path(os.path.join(output_dir, '.managed'), allow_create=True)
    latest_period_by_code = core.latest_financial_period_by_code()
    fallback_latest_period = core.latest_completed_report_period()

    for _, stock in listed.iterrows():
        code = str(stock['Code']).strip()
        name = str(stock['Name']).strip()
        path = core.report.get_output_path(code, name)
        latest_available = latest_period_by_code.get(code, fallback_latest_period)
        periods = core.report_periods_to_check(path, latest_available=latest_available)
        if not periods:
            total_skipped_current += 1
            continue
        for year, quarter in periods:
            checked_periods.add(f'{year}Q{quarter}')
            rows = []
            filing_period = latest_period_by_code.get(code)
            filing_absence_evidenced = filing_period is not None and filing_period < (year, quarter)
            for statement in core.report.STATEMENTS:
                try:
                    rows.extend(core.report.fetch_report(session, code, name, year, quarter, statement))
                except Exception as exc:
                    if str(exc) == 'Parsed zero report rows.' and filing_absence_evidenced:
                        total_no_source += 1
                        core.status(
                            'report',
                            'no_source_data',
                            path=path,
                            note=f'{code} {year}Q{quarter} {statement}: filing not evidenced',
                        )
                    else:
                        total_failures += 1
                        core.status(
                            'report',
                            'failed',
                            path=path,
                            note=f'{code} {year}Q{quarter} {statement}: {exc}',
                        )
                finally:
                    core.report.sleep_between_requests()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=core.report.OUTPUT_COLUMNS)
            result = core.append_or_refresh_rows(
                path,
                df,
                ['Code', 'Year', 'Quarter', 'Statement', 'Account'],
                fallback_columns=core.report.OUTPUT_COLUMNS,
                refresh_fetched_at=False,
            )
            total_appended += result['appended']

    if total_appended:
        note = f'periods={",".join(sorted(checked_periods))} current_files={total_skipped_current}'
        if total_no_source:
            note += f' no_source_data={total_no_source}'
        core.status('report_latest_periods', 'updated', total_appended, output_dir, note=note)
    elif total_failures:
        core.status('report_latest_periods', 'failed', total_failures, output_dir)
    elif total_no_source:
        core.status('report_latest_periods', 'no_source_data', total_no_source, output_dir)
    else:
        core.status(
            'report_latest_periods',
            'up_to_date',
            path=output_dir,
            note=f'current_files={total_skipped_current}',
        )
