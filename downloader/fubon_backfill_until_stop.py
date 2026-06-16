'''
Run Fubon broker rank backfill in small resumable chunks.

This wrapper is intended for long-running background downloads. It repeatedly
invokes fubon_broker_rank.py for earlier dates and stops only after a long
streak of empty weekday files.
'''
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'broker', 'fubon')
DOWNLOADER = os.path.join(PROJECT_ROOT, 'downloader', 'fubon_broker_rank.py')


def parse_args():
    parser = argparse.ArgumentParser(description='Backfill Fubon broker ranks until old data ends.')
    parser.add_argument('--start-date', required=True, help='Start date in YYYY-MM-DD.')
    parser.add_argument('--end-date', default='2000-01-01', help='Hard lower bound in YYYY-MM-DD.')
    parser.add_argument('--chunk-dates', type=int, default=10, help='Weekday dates per child run.')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--empty-stop', type=int, default=30)
    parser.add_argument('--sleep-seconds', type=float, default=5)
    parser.add_argument('--log', default=os.path.join(OUTPUT_DIR, 'backfill.log'))
    return parser.parse_args()


def log(message, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stamp = datetime.now().isoformat(timespec='seconds')
    line = f'[{stamp}] {message}'
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write(line + '\n')


def parse_file_date(path):
    match = re.search(r'fubon_broker_branch_rank_(\d{8})_', os.path.basename(path))
    if not match:
        return None
    return datetime.strptime(match.group(1), '%Y%m%d').date()


def count_csv_rows(path):
    try:
        return sum(len(chunk) for chunk in pd.read_csv(path, usecols=['Date'], chunksize=100000))
    except pd.errors.EmptyDataError:
        return 0
    except Exception:
        with open(path, 'r', encoding='utf-8-sig', errors='ignore') as file:
            return max(sum(1 for _ in file) - 1, 0)


def output_path_for(day):
    return os.path.join(
        OUTPUT_DIR,
        f'fubon_broker_branch_rank_{day.strftime("%Y%m%d")}_volume_broker_all_branch_all.csv',
    )


def previous_weekday(day):
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def earliest_downloaded_date():
    paths = glob.glob(os.path.join(OUTPUT_DIR, 'fubon_broker_branch_rank_*_volume_broker_all_branch_all.csv'))
    dates = [parse_file_date(path) for path in paths]
    dates = [value for value in dates if value is not None]
    return min(dates) if dates else None


def recent_empty_streak(start_day, limit):
    streak = 0
    day = start_day
    while streak < limit:
        path = output_path_for(day)
        if not os.path.exists(path):
            break
        if count_csv_rows(path) == 0:
            streak += 1
            day = previous_weekday(day)
            continue
        break
    return streak


def chunk_dates(start_day, count):
    dates = []
    day = start_day
    while len(dates) < count:
        if day.weekday() < 5:
            dates.append(day)
        day -= timedelta(days=1)
    return dates


def main():
    args = parse_args()
    current = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    earliest = earliest_downloaded_date()
    if earliest and earliest <= current:
        current = previous_weekday(earliest)

    log(f'Start backfill at {current}, end bound {end_date}', args.log)
    empty_streak = 0

    while current >= end_date:
        requested_dates = chunk_dates(current, args.chunk_dates)
        cmd = [
            sys.executable,
            DOWNLOADER,
            '--start-date',
            current.isoformat(),
            '--max-dates',
            str(args.chunk_dates),
            '--metrics',
            'volume',
            '--workers',
            str(args.workers),
            '--throttle-min',
            '0.02',
            '--throttle-max',
            '0.08',
            '--quiet',
        ]
        log('Run: ' + ' '.join(cmd), args.log)
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        with open(args.log, 'a', encoding='utf-8') as file:
            file.write(completed.stdout)
        log(f'Child exit code: {completed.returncode}', args.log)

        for day in requested_dates:
            path = output_path_for(day)
            if os.path.exists(path) and count_csv_rows(path) == 0:
                empty_streak += 1
            else:
                empty_streak = 0

        earliest = earliest_downloaded_date()
        if not earliest:
            raise RuntimeError('No downloaded files found after child run.')

        log(f'Earliest file: {earliest}; empty streak: {empty_streak}/{args.empty_stop}', args.log)
        if empty_streak >= args.empty_stop:
            log('Stop: reached consecutive empty-date threshold.', args.log)
            break

        current = previous_weekday(earliest)
        time.sleep(args.sleep_seconds)

    log('Backfill runner finished.', args.log)


if __name__ == '__main__':
    main()
