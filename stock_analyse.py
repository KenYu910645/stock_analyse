'''
stock_analyse.py

Analyze cached stock CSV files and write a daily movement report.
'''
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from column_schema import read_csv_canonical


DATA_DIR = Path('./data')
PRICE_DIR = DATA_DIR / 'price'
METADATA_PATH = DATA_DIR / 'metadata.csv'
RESULT_PATH = Path('./statistic.txt')
RESULT_CSV_PATH = Path('./statistic.csv')
RESULT_XLSX_PATH = Path('./statistic.xlsx')
CSV_PATTERN = '*.csv'
REQUIRED_COLUMNS = ['Date', 'Close', 'Change', 'Turnover']


def get_stock_code(csv_path):
    '''
    Extract stock code from filenames like 2308_台達電.csv.
    '''
    return csv_path.stem.split('_')[0]


def load_metadata():
    '''
    Return stock metadata indexed by stock code.
    '''
    if not METADATA_PATH.exists():
        return pd.DataFrame()

    metadata_df = read_csv_canonical(METADATA_PATH, dtype={'Code': str})
    if 'Code' not in metadata_df.columns:
        return pd.DataFrame()

    return metadata_df.set_index('Code', drop=False)


def get_metadata_value(metadata_df, stock_code, column, default=''):
    '''
    Return a metadata value for a stock when available.
    '''
    if metadata_df.empty or stock_code not in metadata_df.index:
        return default
    if column not in metadata_df.columns:
        return default

    value = metadata_df.at[stock_code, column]
    if pd.isna(value):
        return default

    return str(value)


def read_stock_csv(csv_path):
    '''
    Read and validate one stock CSV.
    '''
    if csv_path.stat().st_size <= 100:
        raise ValueError('file too small or empty')

    df_stock = read_csv_canonical(csv_path)
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df_stock.columns
    ]
    if missing_columns:
        raise ValueError(f'missing columns: {missing_columns}')

    df_stock = df_stock.copy()
    df_stock['Close'] = pd.to_numeric(df_stock['Close'], errors='coerce')
    df_stock['Change'] = pd.to_numeric(df_stock['Change'], errors='coerce')
    df_stock['Turnover'] = pd.to_numeric(df_stock['Turnover'], errors='coerce')
    df_stock = df_stock.dropna(subset=['Close', 'Change', 'Turnover'])

    previous_close = df_stock['Close'] - df_stock['Change']
    df_stock = df_stock[previous_close != 0].copy()
    df_stock['Change Percent'] = (
        df_stock['Change'] / (df_stock['Close'] - df_stock['Change']) * 100
    )
    df_stock = df_stock.dropna(subset=['Change Percent'])

    if df_stock.empty:
        raise ValueError('no valid price rows')

    return df_stock


def get_average_turnover(df_stock):
    '''
    Return the average daily trading turnover.
    '''
    return float(df_stock['Turnover'].mean())


def count_movement_categories(df_stock):
    '''
    Count daily movement categories.
    '''
    change_percent = df_stock['Change Percent']

    return {
        'ge_2pct': int((change_percent >= 2).sum()),
        'le_neg_2pct': int((change_percent <= -2).sum()),
        'rise_lt_2pct': int(((change_percent > 0) & (change_percent < 2)).sum()),
        'fall_gt_neg_2pct': int(((change_percent < 0) & (change_percent > -2)).sum()),
        'equal': int((change_percent == 0).sum()),
        'total_days': int(len(df_stock)),
    }


def analyze_stock(csv_path, metadata_df):
    '''
    Analyze one stock CSV and return one report row.
    '''
    stock_code = get_stock_code(csv_path)
    df_stock = read_stock_csv(csv_path)
    counts = count_movement_categories(df_stock)

    row = {
        'Code': stock_code,
        'Name': get_metadata_value(metadata_df, stock_code, 'Name'),
        'Group': get_metadata_value(metadata_df, stock_code, 'Group', 'Unknown'),
        'avg_turnover': get_average_turnover(df_stock),
        'File': csv_path.name,
    }
    row.update(counts)
    return row


def get_latest_csv_by_stock(metadata_df):
    '''
    Return the newest cached CSV path for each listed common-stock code.
    '''
    required_columns = {'Code', 'Type', 'Market'}
    missing_columns = required_columns.difference(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            f'{METADATA_PATH} missing required columns: {sorted(missing_columns)}'
        )
    listed_codes = set(
        metadata_df.loc[
            metadata_df['Type'].eq('股票')
            & metadata_df['Market'].eq('上市')
            & metadata_df['Code'].astype(str).str.fullmatch(r'\d{4}'),
            'Code',
        ].astype(str)
    )
    latest_csv_by_stock = {}

    for csv_path in sorted(PRICE_DIR.glob(CSV_PATTERN)):
        stock_code = get_stock_code(csv_path)
        if stock_code not in listed_codes:
            continue
        current_path = latest_csv_by_stock.get(stock_code)
        if current_path is None or csv_path.name > current_path.name:
            latest_csv_by_stock[stock_code] = csv_path

    return latest_csv_by_stock


def format_turnover(value):
    '''
    Format average turnover as a rounded whole number with separators.
    '''
    return f'{value:,.0f}'


def format_probability(count, total_days):
    '''
    Format a movement bucket count as a probability over valid trading days.
    '''
    if total_days <= 0:
        return '0.00%'

    return f'{count / total_days * 100:.2f}%'


def get_probability(count, total_days):
    '''
    Return a movement bucket probability as a 0-100 percentage value.
    '''
    if total_days <= 0:
        return 0.0

    return round(count / total_days * 100, 2)


def build_result_dataframe(rows):
    '''
    Build the tabular result used by CSV and Excel outputs.
    '''
    result_rows = []
    for row in rows:
        total_days = row['total_days']
        result_rows.append({
            'Code': row['Code'],
            'Name': row['Name'],
            'Group': row['Group'],
            'Avg Turnover': round(row['avg_turnover']),
            'Days': total_days,
            '>= +2% (%)': get_probability(row['ge_2pct'], total_days),
            '<= -2% (%)': get_probability(row['le_neg_2pct'], total_days),
            '0~+2% (%)': get_probability(row['rise_lt_2pct'], total_days),
            '-2~0% (%)': get_probability(row['fall_gt_neg_2pct'], total_days),
            'Equal (%)': get_probability(row['equal'], total_days),
        })

    return pd.DataFrame(result_rows)


def clean_sheet_name(name, used_names):
    '''
    Return an Excel-safe worksheet name.
    '''
    invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
    sheet_name = str(name or 'Unknown')
    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, '_')
    sheet_name = sheet_name.strip() or 'Unknown'
    sheet_name = sheet_name[:31]

    base_name = sheet_name
    suffix = 1
    while sheet_name in used_names:
        suffix_text = f'_{suffix}'
        sheet_name = f'{base_name[:31 - len(suffix_text)]}{suffix_text}'
        suffix += 1

    used_names.add(sheet_name)
    return sheet_name


def write_result_files(rows):
    '''
    Write result.csv and result.xlsx.
    '''
    result_df = build_result_dataframe(rows)
    result_df.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8-sig')

    used_sheet_names = set()
    with pd.ExcelWriter(RESULT_XLSX_PATH, engine='openpyxl') as writer:
        for group in sorted(result_df['Group'].unique()):
            group_df = (
                result_df[result_df['Group'] == group]
                .sort_values('Avg Turnover', ascending=False)
            )
            sheet_name = clean_sheet_name(group, used_sheet_names)
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)


def format_stock_table(rows, include_group=True):
    '''
    Format per-stock statistics as a fixed-width text table.
    '''
    headers = [
        'Code',
        'Name',
    ]
    if include_group:
        headers.append('Group')

    headers.extend([
        'Avg Turnover',
        'Days',
        '>= +2%',
        '<= -2%',
        '0~+2%',
        '-2~0%',
        'Equal',
    ])
    table_rows = []

    for row in rows:
        values = [
            row['Code'],
            row['Name'],
        ]
        if include_group:
            values.append(row['Group'])

        values.extend([
            format_turnover(row['avg_turnover']),
            str(row['total_days']),
            format_probability(row['ge_2pct'], row['total_days']),
            format_probability(row['le_neg_2pct'], row['total_days']),
            format_probability(row['rise_lt_2pct'], row['total_days']),
            format_probability(row['fall_gt_neg_2pct'], row['total_days']),
            format_probability(row['equal'], row['total_days']),
        ])
        table_rows.append(values)

    widths = [
        max(len(str(item)) for item in [header] + [row[index] for row in table_rows])
        for index, header in enumerate(headers)
    ]

    def format_row(values):
        return '  '.join(
            str(value).ljust(widths[index])
            for index, value in enumerate(values)
        )

    lines = [format_row(headers), format_row(['-' * width for width in widths])]
    lines.extend(format_row(row) for row in table_rows)
    return lines


def format_grouped_stock_tables(rows):
    '''
    Format stock statistics as one turnover-sorted table per group.
    '''
    rows_by_group = defaultdict(list)
    for row in rows:
        rows_by_group[row['Group']].append(row)

    lines = []
    for group in sorted(rows_by_group):
        group_rows = sorted(
            rows_by_group[group],
            key=lambda row: row['avg_turnover'],
            reverse=True,
        )
        lines.extend([
            '',
            f'Group: {group}',
        ])
        lines.extend(format_stock_table(group_rows, include_group=False))

    return lines


def build_report(rows):
    '''
    Build the full report text.
    '''
    lines = [
        'Stock Daily Movement Report',
        '',
        'Section 1: Per-stock movement probabilities by group, sorted by average turnover',
    ]
    lines.extend(format_grouped_stock_tables(rows))

    return '\n'.join(lines) + '\n'


def main():
    '''
    Analyze cached CSVs, print report, and write result.txt.
    '''
    metadata_df = load_metadata()
    rows = []
    skipped_reasons = Counter()

    for csv_path in get_latest_csv_by_stock(metadata_df).values():
        try:
            rows.append(analyze_stock(csv_path, metadata_df))
        except Exception as exc:
            skipped_reasons[str(exc)] += 1

    rows.sort(key=lambda row: (row['Group'], -row['avg_turnover'], row['Code']))
    report = build_report(rows)
    write_result_files(rows)

    print(report)
    RESULT_PATH.write_text(report, encoding='utf-8')
    print(f'Result written to {RESULT_PATH}.')
    print(f'CSV written to {RESULT_CSV_PATH}.')
    print(f'Excel workbook written to {RESULT_XLSX_PATH}.')


if __name__ == '__main__':
    main()
