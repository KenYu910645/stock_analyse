from __future__ import annotations

import os

from downloader.daily.tasks._runtime import core


__all__ = ['update_dividend']


def update_dividend(query_date):
    output_dir = os.path.join(core.DATA_DIR, 'dividend', 'ex_right_dividend')
    latest = core.latest_date_in_directory(output_dir, 'ex_date')
    history_path = core.find_main_history_file(
        os.path.join(core.DATA_DIR, 'dividend', 'twse_ex_right_dividend_merged_*_to_*.csv')
    )
    if not latest and history_path:
        latest = core.latest_date_in_csv(history_path, 'ex_date')
    start = core.datetime.strptime(latest, '%Y%m%d').date() + core.timedelta(days=1) if latest else query_date
    if start > query_date:
        core.status('ex_right_dividend_per_stock', 'up_to_date', path=output_dir)
        return
    df = core.ex_right_dividend.download('merged', start, query_date, include_details=True)
    if df.empty:
        core.status('ex_right_dividend_per_stock', 'no_source_data', path=output_dir)
        return
    written = core.update_stock_keyed_by_stock(
        df,
        output_dir,
        'stock_id',
        ['ex_date', 'stock_id'],
        fallback_columns=core.ex_right_dividend.FINAL_COLUMNS,
        name_column='stock_name',
        code_to_name=core.load_listed_common_stock_names(),
    )
    core.status('ex_right_dividend_per_stock', 'updated' if written else 'up_to_date', written, output_dir)
