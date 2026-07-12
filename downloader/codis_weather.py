'''
codis_weather.py

Download historical hourly weather observations from CWA CODiS.
'''
import argparse
import csv
import os
import random
import time
from datetime import date, datetime, timedelta

import requests

from column_schema import storage_fieldnames, storage_name, storage_record


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'codis_weather', 'hourly')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

CODIS_BASE_URL = 'https://codis.cwa.gov.tw'
CODIS_STATION_PAGE = f'{CODIS_BASE_URL}/StationData'
CODIS_STATION_API = f'{CODIS_BASE_URL}/api/station'

DEFAULT_START_DATE = '1995-01-01'
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
THROTTLE_MIN_SECONDS = 0.2
THROTTLE_MAX_SECONDS = 0.8

REPORT_TYPE = 'report_date'

STATION_CHAINS = {
    'taipei': [
        {
            'station_id': '466920',
            'station_name': 'Taipei',
            'stn_type': 'cwb',
            'start_date': '1896-01-01',
            'end_date': None,
        },
    ],
    'new_taipei_banqiao': [
        {
            'station_id': '466880',
            'station_name': 'Banqiao',
            'stn_type': 'cwb',
            'start_date': '1972-03-01',
            'end_date': '2022-12-31',
        },
        {
            'station_id': '466881',
            'station_name': 'NewTaipei',
            'stn_type': 'cwb',
            'start_date': '2023-01-01',
            'end_date': None,
        },
    ],
    'taoyuan': [
        {
            'station_id': 'C0C480',
            'station_name': 'Taoyuan',
            'stn_type': 'auto_C0',
            'start_date': '1995-01-01',
            'end_date': '2025-12-31',
        },
        {
            'station_id': 'C2C480',
            'station_name': 'Taoyuan',
            'stn_type': 'agr',
            'start_date': '2026-03-16',
            'end_date': None,
        },
    ],
    'hsinchu': [
        {
            'station_id': '467571',
            'station_name': 'Hsinchu',
            'stn_type': 'cwb',
            'start_date': '1991-07-01',
            'end_date': None,
        },
    ],
    'taichung': [
        {
            'station_id': '467490',
            'station_name': 'Taichung',
            'stn_type': 'cwb',
            'start_date': '1896-01-01',
            'end_date': None,
        },
    ],
    'tainan': [
        {
            'station_id': '467410',
            'station_name': 'Tainan',
            'stn_type': 'cwb',
            'start_date': '1897-01-01',
            'end_date': None,
        },
    ],
    'kaohsiung': [
        {
            'station_id': '467440',
            'station_name': 'Kaohsiung',
            'stn_type': 'cwb',
            'start_date': '1931-01-01',
            'end_date': '2022-01-24',
        },
        {
            'station_id': '467441',
            'station_name': 'Kaohsiung',
            'stn_type': 'cwb',
            'start_date': '2022-01-24',
            'end_date': None,
        },
    ],
}

OBSERVATION_COLUMNS = [
    'StationPressure.Instantaneous',
    'SeaLevelPressure.Instantaneous',
    'AirTemperature.Instantaneous',
    'DewPointTemperature.Instantaneous',
    'RelativeHumidity.Instantaneous',
    'WindSpeed.Mean',
    'WindSpeed.TenMinutelyMaximum',
    'WindDirection.Mean',
    'WindDirection.TenMinutelyMaximum',
    'PeakGust.Maximum',
    'PeakGust.Direction',
    'Precipitation.Accumulation',
    'PrecipitationDuration.Total',
    'SunshineDuration.Total',
    'GlobalSolarRadiation.Accumulation',
    'Visibility.Instantaneous',
    'Visibility.AutoMean',
    'UVIndex.Accumulation',
    'TotalCloudAmount.Instantaneous',
    'TotalCloudAmount.SatRetrieved',
    'SoilTemperatureAt0cm.Instantaneous',
    'SoilTemperatureAt5cm.Instantaneous',
    'SoilTemperatureAt10cm.Instantaneous',
    'SoilTemperatureAt20cm.Instantaneous',
    'SoilTemperatureAt30cm.Instantaneous',
    'SoilTemperatureAt50cm.Instantaneous',
    'SoilTemperatureAt100cm.Instantaneous',
]

OUTPUT_COLUMNS = [
    'city',
    'station_id',
    'station_name',
    'stn_type',
    'DataTime',
    *OBSERVATION_COLUMNS,
]

LOG_COLUMNS = [
    'city',
    'query_date',
    'station_id',
    'station_name',
    'stn_type',
    'status',
    'message',
]


def parse_args():
    '''
    Parse command line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Download CODiS hourly weather observations.'
    )
    parser.add_argument(
        '--start-date',
        default=DEFAULT_START_DATE,
        help='Start date in YYYY-MM-DD format. Default: 1995-01-01.',
    )
    parser.add_argument(
        '--end-date',
        default=date.today().isoformat(),
        help='End date in YYYY-MM-DD format. Default: today.',
    )
    parser.add_argument(
        '--cities',
        default='all',
        help=(
            'Comma-separated city keys to download. Use "all" for every '
            f'default city. Available: {", ".join(sorted(STATION_CHAINS))}.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=DATA_DIR,
        help=f'Output directory. Default: {DATA_DIR}.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing city CSV files instead of resuming.',
    )
    parser.add_argument(
        '--sleep-min',
        type=float,
        default=THROTTLE_MIN_SECONDS,
        help='Minimum sleep seconds between requests.',
    )
    parser.add_argument(
        '--sleep-max',
        type=float,
        default=THROTTLE_MAX_SECONDS,
        help='Maximum sleep seconds between requests.',
    )
    return parser.parse_args()


def parse_iso_date(value):
    '''
    Return a date from YYYY-MM-DD text.
    '''
    return datetime.strptime(value, '%Y-%m-%d').date()


def iter_dates(start_date, end_date):
    '''
    Yield every calendar date in the requested range.
    '''
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def get_city_keys(value):
    '''
    Return requested city keys.
    '''
    if value == 'all':
        return list(STATION_CHAINS)

    city_keys = [item.strip() for item in value.split(',') if item.strip()]
    unknown = sorted(set(city_keys) - set(STATION_CHAINS))
    if unknown:
        raise ValueError(f'Unknown city keys: {", ".join(unknown)}')

    return city_keys


def get_segment_for_date(city_key, query_date):
    '''
    Return the station segment active on query_date.
    '''
    for segment in STATION_CHAINS[city_key]:
        start_date = parse_iso_date(segment['start_date'])
        end_text = segment.get('end_date')
        end_date = parse_iso_date(end_text) if end_text else date.max
        if start_date <= query_date <= end_date:
            return segment

    return None


def get_output_path(output_dir, city_key, start_date, end_date):
    '''
    Return the per-city CSV path.
    '''
    start_text = start_date.strftime('%Y%m%d')
    end_text = end_date.strftime('%Y%m%d')
    return os.path.join(output_dir, f'{city_key}_hourly_{start_text}_to_{end_text}.csv')


def get_log_path(start_date, end_date):
    '''
    Return the failed/empty day log path.
    '''
    start_text = start_date.strftime('%Y%m%d')
    end_text = end_date.strftime('%Y%m%d')
    return os.path.join(LOG_DIR, f'codis_weather_log_{start_text}_to_{end_text}.csv')


def get_downloaded_dates(output_path):
    '''
    Return dates already present in an existing city CSV.
    '''
    if not os.path.exists(output_path):
        return set()

    downloaded_dates = set()
    with open(output_path, newline='', encoding='utf-8-sig') as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            data_time = row.get('DataTime') or row.get(storage_name('DataTime'))
            if data_time:
                downloaded_dates.add(data_time[:10])

    return downloaded_dates


def open_output_writer(output_path, force):
    '''
    Open a CSV writer, preserving existing data unless force is set.
    '''
    exists = os.path.exists(output_path)
    mode = 'w' if force or not exists else 'a'
    file_obj = open(output_path, mode, newline='', encoding='utf-8-sig')
    writer = csv.DictWriter(file_obj, fieldnames=storage_fieldnames(OUTPUT_COLUMNS))
    if mode == 'w':
        writer.writeheader()

    return file_obj, writer


def flatten_dict(value, prefix=''):
    '''
    Flatten nested CODiS observation dictionaries.
    '''
    flattened = {}
    for key, item in value.items():
        full_key = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_dict(item, full_key))
        else:
            flattened[full_key] = item

    return flattened


def build_payload(query_date, segment):
    '''
    Build one CODiS hourly report request payload.
    '''
    start = f'{query_date.isoformat()}T00:00:00'
    end = f'{query_date.isoformat()}T23:59:59'
    return {
        'date': start,
        'type': REPORT_TYPE,
        'stn_ID': segment['station_id'],
        'stn_type': segment['stn_type'],
        'start': start,
        'end': end,
    }


def create_session():
    '''
    Create a CODiS session.
    '''
    session = requests.Session()
    response = session.get(CODIS_STATION_PAGE, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return session


def fetch_day(session, query_date, segment):
    '''
    Fetch one station/day of CODiS hourly observations.
    '''
    payload = build_payload(query_date, segment)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                CODIS_STATION_API,
                data=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            json_payload = response.json()
            if json_payload.get('code') != 200:
                raise ValueError(
                    f"CODiS code={json_payload.get('code')}: "
                    f"{json_payload.get('message')}"
                )

            data = json_payload.get('data') or []
            if not data:
                return []

            return data[0].get('dts') or []
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            print(
                f'Fetch failed for {segment["station_id"]} {query_date} '
                f'(attempt {attempt}/{MAX_RETRIES}): {exc}'
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    raise last_error


def normalize_rows(city_key, segment, rows):
    '''
    Normalize CODiS rows to the output CSV schema.
    '''
    normalized = []
    for raw_row in rows:
        flattened = flatten_dict(raw_row)
        row = {
            'city': city_key,
            'station_id': segment['station_id'],
            'station_name': segment['station_name'],
            'stn_type': segment['stn_type'],
            'DataTime': flattened.get('DataTime'),
        }
        for column in OBSERVATION_COLUMNS:
            row[column] = flattened.get(column)
        normalized.append(row)

    return normalized


def write_log_rows(log_rows, start_date, end_date):
    '''
    Save failed/empty/no-station days for audit and reruns.
    '''
    if not log_rows:
        return None

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = get_log_path(start_date, end_date)
    with open(log_path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=storage_fieldnames(LOG_COLUMNS))
        writer.writeheader()
        writer.writerows(storage_record(row) for row in log_rows)

    return log_path


def sleep_between_requests(min_seconds, max_seconds):
    '''
    Sleep a randomized polite interval.
    '''
    if max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds
    time.sleep(random.uniform(min_seconds, max_seconds))


def download_city(session, city_key, start_date, end_date, args):
    '''
    Download all requested dates for one city.
    '''
    output_path = get_output_path(args.output_dir, city_key, start_date, end_date)
    downloaded_dates = set() if args.force else get_downloaded_dates(output_path)
    file_obj, writer = open_output_writer(output_path, args.force)
    total_dates = (end_date - start_date).days + 1
    stats = {
        'city': city_key,
        'dates_checked': 0,
        'dates_skipped_existing': 0,
        'dates_downloaded': 0,
        'empty_days': 0,
        'failed_days': 0,
        'no_station_days': 0,
        'rows_saved': 0,
        'output_path': output_path,
    }
    log_rows = []

    try:
        for index, query_date in enumerate(iter_dates(start_date, end_date), start=1):
            query_date_text = query_date.isoformat()
            stats['dates_checked'] += 1

            if query_date_text in downloaded_dates:
                stats['dates_skipped_existing'] += 1
                continue

            segment = get_segment_for_date(city_key, query_date)
            if segment is None:
                stats['no_station_days'] += 1
                log_rows.append({
                    'city': city_key,
                    'query_date': query_date_text,
                    'station_id': '',
                    'station_name': '',
                    'stn_type': '',
                    'status': 'no_station_segment',
                    'message': 'No configured station segment for this date.',
                })
                continue

            print(
                f'[{city_key} {index}/{total_dates}] '
                f'{query_date_text} {segment["station_id"]}'
            )
            try:
                rows = fetch_day(session, query_date, segment)
            except Exception as exc:
                stats['failed_days'] += 1
                log_rows.append({
                    'city': city_key,
                    'query_date': query_date_text,
                    'station_id': segment['station_id'],
                    'station_name': segment['station_name'],
                    'stn_type': segment['stn_type'],
                    'status': 'failed',
                    'message': str(exc),
                })
            else:
                if not rows:
                    stats['empty_days'] += 1
                    log_rows.append({
                        'city': city_key,
                        'query_date': query_date_text,
                        'station_id': segment['station_id'],
                        'station_name': segment['station_name'],
                        'stn_type': segment['stn_type'],
                        'status': 'empty',
                        'message': 'CODiS returned zero hourly rows.',
                    })
                else:
                    normalized_rows = normalize_rows(city_key, segment, rows)
                    writer.writerows(storage_record(row) for row in normalized_rows)
                    file_obj.flush()
                    downloaded_dates.add(query_date_text)
                    stats['dates_downloaded'] += 1
                    stats['rows_saved'] += len(normalized_rows)
            finally:
                sleep_between_requests(args.sleep_min, args.sleep_max)
    finally:
        file_obj.close()

    return stats, log_rows


def validate_date_range(start_date, end_date):
    '''
    Validate command line date inputs.
    '''
    if start_date > end_date:
        raise ValueError('start-date must be earlier than or equal to end-date.')


def main():
    '''
    Download CODiS hourly weather data.
    '''
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    validate_date_range(start_date, end_date)
    city_keys = get_city_keys(args.cities)

    os.makedirs(args.output_dir, exist_ok=True)
    session = create_session()
    all_log_rows = []
    all_stats = []

    for city_key in city_keys:
        stats, log_rows = download_city(session, city_key, start_date, end_date, args)
        all_stats.append(stats)
        all_log_rows.extend(log_rows)

    log_path = write_log_rows(all_log_rows, start_date, end_date)

    print('Download summary:')
    for stats in all_stats:
        print(
            f"{stats['city']}: "
            f"downloaded_days={stats['dates_downloaded']}, "
            f"rows_saved={stats['rows_saved']}, "
            f"skipped_existing={stats['dates_skipped_existing']}, "
            f"empty_days={stats['empty_days']}, "
            f"failed_days={stats['failed_days']}, "
            f"no_station_days={stats['no_station_days']}, "
            f"output_path={stats['output_path']}"
        )
    if log_path:
        print(f'log_path={log_path}')


if __name__ == '__main__':
    main()
