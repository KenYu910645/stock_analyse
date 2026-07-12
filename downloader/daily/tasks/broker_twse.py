from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from downloader.daily.tasks._runtime import core


__all__ = ['broker_twse_dir', 'broker_twse_by_stock_dir', 'broker_twse_by_broker_dir', 'broker_twse_by_date_dir', 'assert_path_within', 'clear_generated_csv_dir', 'sync_twse_broker_date_by_stock', 'download_twse_broker_batch', 'build_twse_broker_outputs', 'update_twse_broker']


def broker_twse_dir():
    return Path(core.DATA_DIR) / 'broker' / 'twse'


def broker_twse_by_stock_dir():
    return core.broker_twse_dir() / 'by_stock'


def broker_twse_by_broker_dir():
    return core.broker_twse_dir() / 'by_broker'


def broker_twse_by_date_dir(query_date):
    return core.broker_twse_dir() / 'by_date' / query_date.isoformat()


def assert_path_within(path, parent):
    resolved_path = Path(path).resolve()
    resolved_parent = Path(parent).resolve()
    if resolved_path != resolved_parent and resolved_parent not in resolved_path.parents:
        raise ValueError(f'Refusing to write outside {resolved_parent}: {resolved_path}')


def clear_generated_csv_dir(path, allowed_parent):
    path = Path(path)
    allowed_parent = Path(allowed_parent)
    allowed_parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    assert_path_within(path, allowed_parent)

    removed = 0
    for csv_path in path.glob('*.csv'):
        if not csv_path.is_file():
            continue
        assert_path_within(csv_path, path)
        csv_path.unlink()
        removed += 1
    return removed


def sync_twse_broker_date_by_stock(query_date):
    date_token = core.format_yyyymmdd(query_date)
    source_dir = core.broker_twse_by_stock_dir()
    date_dir = core.broker_twse_by_date_dir(query_date)
    target_dir = date_dir / 'by_stock'

    removed = core.clear_generated_csv_dir(target_dir, date_dir)
    copied = 0
    for source_path in sorted(source_dir.glob(f'*_bsr_twse_{date_token}_*.csv')):
        if not source_path.is_file():
            continue
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied += 1
    return {'path': target_dir, 'removed': removed, 'copied': copied}


def download_twse_broker_batch(args):
    from downloader import broker as broker_downloader

    return broker_downloader.run_metadata_batch_summary(args, print_summary=False)


def build_twse_broker_outputs(by_stock_dir, by_broker_dir, summary_json):
    from tools import build_twse_broker_by_broker

    build_args = argparse.Namespace(
        twse_dir=core.broker_twse_dir(),
        by_stock_dir=Path(by_stock_dir),
        by_broker_dir=Path(by_broker_dir),
        metadata=Path(core.STOCK_METADATA_PATH),
        migrate_root_raw=False,
        keep_existing=False,
        summary_json=Path(summary_json),
    )
    return build_twse_broker_by_broker.build_outputs(build_args)


def update_twse_broker(
    query_date,
    max_stocks=None,
    max_attempts=8,
    throttle_min=0.2,
    throttle_max=0.8,
):
    twse_dir = core.broker_twse_dir()
    by_stock_dir = core.broker_twse_by_stock_dir()
    by_broker_dir = core.broker_twse_by_broker_dir()
    date_dir = core.broker_twse_by_date_dir(query_date)
    date_by_broker_dir = date_dir / 'by_broker'
    log_dir = Path(core.PROJECT_ROOT) / 'logs' / 'broker'
    date_text = query_date.isoformat()
    date_token = core.format_yyyymmdd(query_date)

    try:
        download_args = argparse.Namespace(
            metadata=Path(core.STOCK_METADATA_PATH),
            codes=None,
            max_stocks=max_stocks,
            output_dir=by_stock_dir,
            raw_dir=log_dir / 'twse_captcha',
            log_dir=log_dir,
            date=date_text,
            max_attempts=max_attempts,
            throttle_min=throttle_min,
            throttle_max=throttle_max,
            force=False,
            update_metadata=True,
            quiet=True,
        )
        download_result = core.download_twse_broker_batch(download_args)
        date_sync = core.sync_twse_broker_date_by_stock(query_date)
        cumulative_summary = core.build_twse_broker_outputs(
            by_stock_dir,
            by_broker_dir,
            log_dir / f'twse_by_broker_build_all_{date_token}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )
        date_summary = core.build_twse_broker_outputs(
            date_sync['path'],
            date_by_broker_dir,
            log_dir / f'twse_by_broker_build_{date_token}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )
    except Exception as exc:
        core.status('broker_twse', 'failed', path=str(twse_dir), note=str(exc))
        return

    download_summary = download_result.get('summary', {})
    status_counts = download_summary.get('status_counts', {})
    failed = int(status_counts.get('failed', 0) or 0)
    success = int(status_counts.get('success', 0) or 0)
    existing = int(status_counts.get('skipped_existing', 0) or 0)
    no_data = int(status_counts.get('no_data', 0) or 0)
    selected = int(download_summary.get('selected_stocks', 0) or 0)
    exit_code = int(download_result.get('exit_code', 0) or 0)

    date_stats = date_summary.get('stats', {})
    cumulative_stats = cumulative_summary.get('stats', {})
    rows = int(date_stats.get('records', 0) or 0)
    date_raw = int(date_sync.get('copied', 0) or 0)

    if failed or exit_code not in (0, 2):
        action = 'failed'
    elif success or rows:
        action = 'updated'
    else:
        action = 'up_to_date'

    note = (
        f'date={date_text} selected={selected} success={success} existing={existing} '
        f'no_data={no_data} failed={failed} date_raw={date_raw} '
        f'date_brokers={int(date_stats.get("output_files", 0) or 0)} '
        f'cumulative_brokers={int(cumulative_stats.get("output_files", 0) or 0)}'
    )
    core.status('broker_twse', action, rows, path=str(date_dir), note=note)
