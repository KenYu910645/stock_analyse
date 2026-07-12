from __future__ import annotations

import os

from downloader.daily.tasks._runtime import core


__all__ = ['update_institutional']


def update_institutional(query_date, listed_codes):
    payload = core.fetch_json(
        core.TWSE_T86_URL,
        {
            'date': core.format_yyyymmdd(query_date),
            'selectType': 'ALLBUT0999',
            'response': 'json',
        },
    )
    if payload.get('stat') != 'OK':
        core.status('institutional', 'no_source_data', note=str(payload.get('stat')))
        return
    row_date = core.yyyymmdd_to_iso(payload.get('date') or core.format_yyyymmdd(query_date))
    rows = []
    for raw in payload.get('data') or []:
        values = raw.get('value', raw) if isinstance(raw, dict) else raw
        if len(values) < 19:
            continue
        code = str(values[0]).strip()
        if code not in listed_codes:
            continue
        rows.append({
            'Date': row_date,
            'Code': code,
            'Name': str(values[1]).strip(),
            'ForeignBuyExDealer': values[2],
            'ForeignSellExDealer': values[3],
            'ForeignNetExDealer': values[4],
            'ForeignDealerBuy': values[5],
            'ForeignDealerSell': values[6],
            'ForeignDealerNet': values[7],
            'InvestmentTrustBuy': values[8],
            'InvestmentTrustSell': values[9],
            'InvestmentTrustNet': values[10],
            'DealerNet': values[11],
            'DealerSelfBuy': values[12],
            'DealerSelfSell': values[13],
            'DealerSelfNet': values[14],
            'DealerHedgeBuy': values[15],
            'DealerHedgeSell': values[16],
            'DealerHedgeNet': values[17],
            'InstitutionalNet': values[18],
        })
    if not rows:
        core.status('institutional', 'no_listed_rows')
        return
    df = core.institutional_investors.normalize_dataframe(rows)
    written = core.update_stock_keyed_by_stock(
        df,
        os.path.join(core.DATA_DIR, 'institutional'),
        'Code',
        ['Date', 'Code'],
        fallback_columns=core.institutional_investors.OUTPUT_COLUMNS,
        name_column='Name',
        code_to_name=core.load_listed_common_stock_names(),
    )
    core.status(
        'institutional_per_stock',
        'updated' if written else 'up_to_date',
        written,
        os.path.join(core.DATA_DIR, 'institutional'),
    )
