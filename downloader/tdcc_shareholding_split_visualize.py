'''
tdcc_shareholding_split_visualize.py

Split latest TDCC listed shareholding data into per-stock CSV files and render
one histogram-style bar chart per stock.
'''
import argparse
import glob
import html
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SHAREHOLDING_DIR = os.path.join(DATA_DIR, 'shareholding')
LISTED_DIR = os.path.join(SHAREHOLDING_DIR, 'listed')
BY_STOCK_DIR = os.path.join(LISTED_DIR, 'by_stock')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'output', 'shareholdingDistribution')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

DATE_COL = '資料日期'
CODE_COL = '證券代號'
LEVEL_COL = '持股分級'
HOLDERS_COL = '人數'
SHARES_COL = '股數'
RATIO_COL = '占集保庫存數比例%'
RANGE_COL = '持股/單位數分級'

HOLDING_LEVEL_LABELS = {
    1: '1-999',
    2: '1,000-5,000',
    3: '5,001-10,000',
    4: '10,001-15,000',
    5: '15,001-20,000',
    6: '20,001-30,000',
    7: '30,001-40,000',
    8: '40,001-50,000',
    9: '50,001-100,000',
    10: '100,001-200,000',
    11: '200,001-400,000',
    12: '400,001-600,000',
    13: '600,001-800,000',
    14: '800,001-1,000,000',
    15: '1,000,001以上',
    16: '差異數調整（說明4）',
    17: '合計',
}


def configure_fonts():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in (
        'Microsoft JhengHei',
        'Noto Sans CJK TC',
        'MingLiU',
        'Arial Unicode MS',
    ):
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            break
    plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split and visualize latest TDCC listed shareholding data.'
    )
    parser.add_argument(
        '--input',
        default=None,
        help='Input listed CSV. Default: newest data/shareholding/listed file.',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Only render the first N stock charts after splitting all CSVs.',
    )
    return parser.parse_args()


def find_latest_input():
    pattern = os.path.join(LISTED_DIR, 'tdcc_shareholding_listed_*.csv')
    paths = sorted(glob.glob(pattern))
    paths = [path for path in paths if os.path.basename(path) != 'by_stock']
    if not paths:
        raise FileNotFoundError(f'No listed TDCC shareholding CSV found: {pattern}')
    return paths[-1]


def load_stock_names():
    if not os.path.exists(METADATA_PATH):
        return {}
    metadata = pd.read_csv(METADATA_PATH, dtype={'Code': str})
    metadata['Code'] = metadata['Code'].astype(str).str.strip()
    return dict(zip(metadata['Code'], metadata['Name']))


def prepare_dataframe(path):
    df = pd.read_csv(path, dtype={DATE_COL: str, CODE_COL: str, LEVEL_COL: str})
    df[CODE_COL] = df[CODE_COL].astype(str).str.strip()
    for col in (HOLDERS_COL, SHARES_COL, RATIO_COL):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df[LEVEL_COL] = pd.to_numeric(df[LEVEL_COL], errors='coerce').astype('Int64')
    if RANGE_COL in df.columns:
        df[RANGE_COL] = df[LEVEL_COL].astype(int).map(HOLDING_LEVEL_LABELS)
    else:
        df.insert(
            df.columns.get_loc(LEVEL_COL) + 1,
            RANGE_COL,
            df[LEVEL_COL].astype(int).map(HOLDING_LEVEL_LABELS),
        )
    df = adjust_difference_rows(df)
    df = df.sort_values([CODE_COL, LEVEL_COL])
    return df


def adjust_difference_rows(df):
    '''
    Recompute level 16 as level 17 total minus levels 1-15.
    '''
    for _code, group in df.groupby(CODE_COL, sort=False):
        detail_mask = group.index[df.loc[group.index, LEVEL_COL].between(1, 15)]
        diff_index = group.index[df.loc[group.index, LEVEL_COL] == 16]
        total_index = group.index[df.loc[group.index, LEVEL_COL] == 17]
        if len(diff_index) != 1 or len(total_index) != 1:
            continue

        diff_idx = diff_index[0]
        total_idx = total_index[0]
        holders_diff = (
            df.at[total_idx, HOLDERS_COL] - df.loc[detail_mask, HOLDERS_COL].sum()
        )
        shares_diff = (
            df.at[total_idx, SHARES_COL] - df.loc[detail_mask, SHARES_COL].sum()
        )
        total_shares = df.at[total_idx, SHARES_COL]
        ratio_diff = (shares_diff / total_shares * 100) if total_shares else 0

        df.at[diff_idx, HOLDERS_COL] = holders_diff
        df.at[diff_idx, SHARES_COL] = shares_diff
        df.at[diff_idx, RATIO_COL] = round(ratio_diff, 2)

    df[HOLDERS_COL] = df[HOLDERS_COL].round().astype('Int64')
    df[SHARES_COL] = df[SHARES_COL].round().astype('Int64')
    return df


def split_csvs(df, date_text):
    output_dir = os.path.join(BY_STOCK_DIR, date_text)
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for code, stock_df in df.groupby(CODE_COL, sort=True):
        path = os.path.join(output_dir, f'{code}.csv')
        ordered_cols = [
            DATE_COL,
            CODE_COL,
            LEVEL_COL,
            RANGE_COL,
            HOLDERS_COL,
            SHARES_COL,
            RATIO_COL,
        ]
        stock_df[ordered_cols].to_csv(path, index=False, encoding='utf-8-sig')
        paths[code] = path
    return output_dir, paths


def plot_stock(stock_df, code, name, date_text, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plot_df = stock_df[stock_df[LEVEL_COL].between(1, 15)].copy()
    labels = plot_df[RANGE_COL]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(
        labels,
        plot_df[RATIO_COL],
        color='#2f80ed',
        edgecolor='#1b4f9c',
        linewidth=0.6,
    )
    ax.set_title(f'{code} {name} TDCC Shareholding Distribution {date_text}')
    ax.set_xlabel('Holding / Unit Range')
    ax.set_ylabel('Custody Inventory Ratio (%)')
    ax.set_ylim(0, max(5, plot_df[RATIO_COL].max() * 1.15))
    ax.grid(axis='y', alpha=0.25)
    ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2)
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()

    path = os.path.join(output_dir, f'{code}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_index(date_text, chart_paths, csv_paths, stock_names, output_dir):
    index_path = os.path.join(output_dir, 'index.html')
    rows = []
    for code in sorted(chart_paths):
        name = stock_names.get(code, '')
        chart_name = os.path.basename(chart_paths[code])
        csv_path = csv_paths.get(code, '')
        csv_rel = os.path.relpath(csv_path, output_dir).replace(os.sep, '/')
        rows.append(
            '<article>'
            f'<h2>{html.escape(code)} {html.escape(name)}</h2>'
            f'<a href="{html.escape(csv_rel)}">CSV</a>'
            f'<img src="{html.escape(chart_name)}" alt="{html.escape(code)} chart">'
            '</article>'
        )

    content = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TDCC Shareholding Histograms {html.escape(date_text)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #52606d; margin-bottom: 24px; }}
    article {{ break-inside: avoid; margin-bottom: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 20px; }}
    h2 {{ font-size: 18px; margin: 0 0 8px; }}
    a {{ display: inline-block; margin-bottom: 8px; color: #1d4ed8; }}
    img {{ display: block; width: min(100%, 960px); height: auto; }}
  </style>
</head>
<body>
  <h1>TDCC Shareholding Histograms</h1>
  <div class="meta">Data date: {html.escape(date_text)}. Stocks: {len(chart_paths)}.</div>
  {''.join(rows)}
</body>
</html>
'''
    with open(index_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(content)
    return index_path


def main():
    args = parse_args()
    configure_fonts()
    input_path = args.input or find_latest_input()
    df = prepare_dataframe(input_path)
    date_text = str(df[DATE_COL].max())
    stock_names = load_stock_names()

    csv_dir, csv_paths = split_csvs(df, date_text)

    chart_dir = os.path.join(PLOT_DIR, date_text)
    chart_paths = {}
    grouped = list(df.groupby(CODE_COL, sort=True))
    if args.top is not None:
        grouped = grouped[:args.top]

    for index, (code, stock_df) in enumerate(grouped, start=1):
        chart_paths[code] = plot_stock(
            stock_df=stock_df,
            code=code,
            name=stock_names.get(code, ''),
            date_text=date_text,
            output_dir=chart_dir,
        )
        if index % 100 == 0:
            print(f'Rendered {index}/{len(grouped)} charts.')

    index_path = build_index(date_text, chart_paths, csv_paths, stock_names, chart_dir)
    print(f'Input: {input_path}')
    print(f'Data date: {date_text}')
    print(f'Split CSV directory: {csv_dir}')
    print(f'Chart directory: {chart_dir}')
    print(f'Index: {index_path}')
    print(f'Stocks split: {len(csv_paths)}')
    print(f'Charts rendered: {len(chart_paths)}')


if __name__ == '__main__':
    main()
