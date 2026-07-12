from __future__ import annotations

import os
import re
from datetime import datetime

import requests
from lxml import html

from downloader.daily.tasks._runtime import core


__all__ = ['parse_irengage_conference_rows', 'update_investor_conference']


def parse_irengage_conference_rows(html_text):
    document = html.fromstring(html_text)
    rows = []
    for table in document.xpath('//table'):
        for tr in table.xpath('.//tr'):
            cells = [' '.join(cell.xpath('.//text()')).strip() for cell in tr.xpath('./th|./td')]
            if len(cells) >= 5 and re.match(r'^\d{4}/\d{2}/\d{2}$', cells[0]):
                rows.append({
                    'Date': cells[0].replace('/', '-'),
                    'Time': cells[1],
                    'Company': cells[2],
                    'Location': cells[3],
                    'Message': cells[4],
                    'Download': cells[5] if len(cells) > 5 else '',
                })
    return core.pd.DataFrame(rows)


def update_investor_conference(listed_codes):
    response = requests.get(core.IR_ENGAGE_CONFERENCE_URL, headers=core.HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = 'utf-8'
    df = core.parse_irengage_conference_rows(response.text)
    if df.empty:
        core.status('investor_conference', 'no_source_data')
        return
    df['Code'] = df['Company'].astype(str).str.extract(r'(\d{4})', expand=False)
    df = df[df['Code'].isin(listed_codes)].copy()
    df.insert(0, 'FetchedAt', datetime.now().isoformat(timespec='seconds'))
    df.insert(1, 'SourcePath', core.IR_ENGAGE_CONFERENCE_URL)
    result = core.update_stock_keyed_by_stock(
        df,
        os.path.join(core.DATA_DIR, 'investor_conference'),
        'Code',
        ['Date', 'Time', 'Code', 'Message'],
        name_column='Company',
        code_to_name=core.load_listed_common_stock_names(),
        refresh_fetched_at=True,
    )
    if result['appended']:
        core.status(
            'investor_conference_per_stock',
            'updated',
            result['appended'],
            os.path.join(core.DATA_DIR, 'investor_conference'),
        )
    elif result['refreshed']:
        core.status(
            'investor_conference_per_stock',
            'fetched_at_refreshed',
            result['refreshed'],
            os.path.join(core.DATA_DIR, 'investor_conference'),
        )
    else:
        core.status(
            'investor_conference_per_stock',
            'up_to_date',
            0,
            os.path.join(core.DATA_DIR, 'investor_conference'),
        )
