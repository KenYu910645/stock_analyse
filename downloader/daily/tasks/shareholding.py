from __future__ import annotations

import os

import pandas as pd

from downloader.daily.tasks._runtime import core


__all__ = ['update_shareholding']


def update_shareholding():
    session = core.tdcc_shareholding.make_session()
    response = session.get(core.tdcc_shareholding.TDCC_OPEN_DATA_URL, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8-sig'
    preview = pd.read_csv(core.io.StringIO(response.text), dtype=str, nrows=1)
    latest_date = core.normalize_source_date(str(preview.iloc[0, 0]).strip())
    output_dir = os.path.join(core.DATA_DIR, 'shareholding')
    code_to_name = core.load_listed_common_stock_names()
    rows = pd.read_csv(core.io.StringIO(response.text), dtype=str).fillna('')
    rows['\u8cc7\u6599\u65e5\u671f'] = rows['\u8cc7\u6599\u65e5\u671f'].map(core.yyyymmdd_to_iso)
    rows['\u8b49\u5238\u4ee3\u865f'] = rows['\u8b49\u5238\u4ee3\u865f'].astype(str).str.strip()
    level_label_col = '\u6301\u80a1/\u55ae\u4f4d\u6578\u5206\u7d1a'
    if level_label_col not in rows.columns or rows[level_label_col].astype(str).str.strip().eq('').all():
        levels = pd.to_numeric(rows['\u6301\u80a1\u5206\u7d1a'], errors='coerce')
        labels = levels.map(core.tdcc_shareholding.HOLDING_LEVEL_LABELS).fillna('')
        insert_at = rows.columns.get_loc('\u6301\u80a1\u5206\u7d1a') + 1
        if level_label_col in rows.columns:
            rows[level_label_col] = labels
        else:
            rows.insert(insert_at, level_label_col, labels)
    rows = rows[rows['\u8b49\u5238\u4ee3\u865f'].isin(code_to_name)].copy()
    written = core.update_stock_keyed_by_stock(
        rows,
        output_dir,
        '\u8b49\u5238\u4ee3\u865f',
        ['\u8cc7\u6599\u65e5\u671f', '\u8b49\u5238\u4ee3\u865f', '\u6301\u80a1\u5206\u7d1a'],
        name_column='\u8b49\u5238\u540d\u7a31',
        code_to_name=code_to_name,
    )
    core.status(
        'shareholding_per_stock',
        'updated' if written else 'up_to_date',
        written,
        output_dir,
        note=f'source_date={latest_date}',
    )
