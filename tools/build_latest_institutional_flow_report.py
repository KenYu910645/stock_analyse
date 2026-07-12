"""Build a latest-date market institutional net-flow report."""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical


DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_flow"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_flow"

METADATA_TYPE = "類型"
METADATA_MARKET = "市場"
METADATA_GROUP = "產業群組"
LISTED_STOCK_TYPE = "股票"
LISTED_MARKET = "上市"

REQUIRED_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "ForeignNetExDealer",
    "ForeignDealerNet",
    "InvestmentTrustNet",
    "DealerNet",
    "InstitutionalNet",
]


@dataclass(frozen=True)
class Participant:
    key: str
    label: str
    color: str


PARTICIPANTS = [
    Participant("foreign", "外資", "#2563eb"),
    Participant("trust", "投信", "#d97706"),
    Participant("dealer", "自營商", "#7c3aed"),
    Participant("other", "其他市場參與者", "#64748b"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the latest all-listed-stock institutional net-flow report."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional ISO date. Defaults to the newest broadly covered institutional date.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.70,
        help="Minimum tail-date coverage ratio needed to choose the newest date directly.",
    )
    return parser.parse_args()


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def path_by_code(directory: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(directory.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def clean_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def format_lots(shares: float, digits: int = 0) -> str:
    lots = shares / 1000.0
    if digits:
        return f"{lots:,.{digits}f} 張"
    return f"{lots:,.0f} 張"


def format_percent(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value * 100:.2f}%"


def signed_class(value: float) -> str:
    if value > 0:
        return "pos"
    if value < 0:
        return "neg"
    return "flat"


def read_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    required = ["Code", "Name", METADATA_TYPE, METADATA_MARKET, METADATA_GROUP]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata_missing_columns:{missing}")
    metadata["Code"] = metadata["Code"].astype(str).str.strip()
    return metadata


def listed_common_stock_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    mask = metadata[METADATA_TYPE].eq(LISTED_STOCK_TYPE) & metadata[METADATA_MARKET].eq(LISTED_MARKET)
    result = metadata.loc[mask].copy()
    return result.drop_duplicates("Code", keep="last")


def read_last_date(path: Path) -> str:
    last_date = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = str(row.get("Date", "")).strip()
            if date:
                last_date = date
    return last_date


def choose_target_date(
    inst_paths: dict[str, Path],
    universe_codes: set[str],
    requested_date: str | None,
    min_coverage_ratio: float,
) -> tuple[str, dict[str, int], str]:
    counts: dict[str, int] = {}
    for code in sorted(universe_codes):
        path = inst_paths.get(code)
        if path is None:
            continue
        date = read_last_date(path)
        if not date:
            continue
        counts[date] = counts.get(date, 0) + 1

    if requested_date:
        return requested_date, counts, "指定日期"
    if not counts:
        raise ValueError("no_institutional_dates_found")

    universe_size = max(1, len(universe_codes))
    newest_date = max(counts)
    if counts[newest_date] / universe_size >= min_coverage_ratio:
        return newest_date, counts, "最新高覆蓋日期"

    best_count = max(counts.values())
    best_dates = [date for date, count in counts.items() if count == best_count]
    return max(best_dates), counts, "最高覆蓋日期"


def read_target_row(path: Path, target_date: str) -> dict[str, Any] | None:
    try:
        df = read_csv_canonical(
            path,
            dtype={"Code": str},
            usecols=REQUIRED_COLUMNS,
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
    except ValueError:
        df = read_csv_canonical(
            path,
            dtype={"Code": str},
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
    if df.empty or "Date" not in df.columns:
        return None
    rows = df[df["Date"].astype(str).eq(target_date)]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def build_stock_rows(
    metadata: pd.DataFrame,
    inst_paths: dict[str, Path],
    target_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    metadata_by_code = metadata.set_index("Code", drop=False)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for code in sorted(metadata_by_code.index.astype(str)):
        path = inst_paths.get(code)
        if path is None:
            skipped.append(f"{code}:missing_institutional_file")
            continue
        row = read_target_row(path, target_date)
        if row is None:
            skipped.append(f"{code}:missing_target_date")
            continue

        meta = metadata_by_code.loc[code]
        name = str(meta.get("Name") or row.get("Name") or "")
        group = str(meta.get(METADATA_GROUP) or "未分類")
        foreign_net = clean_number(row.get("ForeignNetExDealer")) + clean_number(row.get("ForeignDealerNet"))
        trust_net = clean_number(row.get("InvestmentTrustNet"))
        dealer_net = clean_number(row.get("DealerNet"))
        calculated_institutional_net = foreign_net + trust_net + dealer_net
        reported_institutional_net = clean_number(row.get("InstitutionalNet"))
        institutional_net = (
            reported_institutional_net
            if abs(reported_institutional_net - calculated_institutional_net) <= 0.5
            else calculated_institutional_net
        )

        records.append(
            {
                "Date": target_date,
                "Code": code,
                "Name": name,
                "IndustryGroup": group,
                "ForeignNetShares": foreign_net,
                "TrustNetShares": trust_net,
                "DealerNetShares": dealer_net,
                "OtherNetShares": -institutional_net,
                "InstitutionalNetShares": institutional_net,
                "AbsInstitutionalNetShares": abs(institutional_net),
            }
        )

    if not records:
        raise ValueError(f"no_rows_for_target_date:{target_date}")
    return pd.DataFrame(records), skipped


def market_totals(stock_rows: pd.DataFrame) -> dict[str, float]:
    return {
        "foreign": float(stock_rows["ForeignNetShares"].sum()),
        "trust": float(stock_rows["TrustNetShares"].sum()),
        "dealer": float(stock_rows["DealerNetShares"].sum()),
        "other": float(stock_rows["OtherNetShares"].sum()),
    }


def participant_by_key() -> dict[str, Participant]:
    return {participant.key: participant for participant in PARTICIPANTS}


def allocate_flows(totals: dict[str, float]) -> list[dict[str, Any]]:
    sellers = [(key, -value) for key, value in totals.items() if value < -0.5]
    buyers = [(key, value) for key, value in totals.items() if value > 0.5]
    total_buy = sum(value for _key, value in buyers)
    if total_buy <= 0:
        return []

    flows: list[dict[str, Any]] = []
    for seller_key, sell_amount in sellers:
        for buyer_key, buy_amount in buyers:
            amount = sell_amount * buy_amount / total_buy
            if amount <= 0.5:
                continue
            flows.append(
                {
                    "source": seller_key,
                    "target": buyer_key,
                    "shares": amount,
                    "lots": amount / 1000.0,
                }
            )
    return flows


def svg_sankey(totals: dict[str, float], flows: list[dict[str, Any]]) -> str:
    participants = participant_by_key()
    sellers = [(key, -value) for key, value in totals.items() if value < -0.5]
    buyers = [(key, value) for key, value in totals.items() if value > 0.5]
    width = 1120
    height = 600
    lane_top = 118
    lane_bottom = 478
    lane_height = lane_bottom - lane_top
    node_gap = 40
    min_node_height = 54
    node_width = 166
    left_x = 38
    right_x = width - left_x - node_width
    x0 = left_x + node_width
    x1 = right_x

    def fit_scale(items: list[tuple[str, float]]) -> float:
        if not items:
            return 1.0
        total_amount = sum(amount for _key, amount in items)
        if total_amount <= 0:
            return 1.0
        gaps = node_gap * max(0, len(items) - 1)
        high = max(0.0, (lane_height - gaps) / total_amount)
        low = 0.0
        for _ in range(64):
            mid = (low + high) / 2.0
            total_height = sum(max(min_node_height, amount * mid) for _key, amount in items) + gaps
            if total_height <= lane_height:
                low = mid
            else:
                high = mid
        return low

    link_scale = min(fit_scale(sellers), fit_scale(buyers))

    def node_layout(items: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        heights = {
            key: max(min_node_height, amount * link_scale)
            for key, amount in items
        }
        total_height = sum(heights.values()) + node_gap * max(0, len(items) - 1)
        y = lane_top + max(0.0, (lane_height - total_height) / 2.0)
        layout: dict[str, dict[str, float]] = {}
        for key, amount in items:
            height_value = heights[key]
            flow_height = amount * link_scale
            flow_y = y + (height_value - flow_height) / 2.0
            layout[key] = {
                "amount": amount,
                "y": y,
                "height": height_value,
                "center": y + height_value / 2.0,
                "flow_y": flow_y,
                "flow_height": flow_height,
            }
            y += height_value + node_gap
        return layout

    seller_layout = node_layout(sellers)
    buyer_layout = node_layout(buyers)

    source_ranges: dict[int, tuple[float, float]] = {}
    target_ranges: dict[int, tuple[float, float]] = {}
    for source_key, _amount in sellers:
        y = seller_layout[source_key]["flow_y"]
        source_flows = sorted(
            enumerate(flows),
            key=lambda item: buyer_layout[item[1]["target"]]["center"],
        )
        for index, flow in source_flows:
            if flow["source"] != source_key:
                continue
            thickness = float(flow["shares"]) * link_scale
            source_ranges[index] = (y, y + thickness)
            y += thickness

    for target_key, _amount in buyers:
        y = buyer_layout[target_key]["flow_y"]
        target_flows = sorted(
            enumerate(flows),
            key=lambda item: seller_layout[item[1]["source"]]["center"],
        )
        for index, flow in target_flows:
            if flow["target"] != target_key:
                continue
            thickness = float(flow["shares"]) * link_scale
            target_ranges[index] = (y, y + thickness)
            y += thickness

    gradient_parts = []
    link_parts = []
    sorted_flow_items = sorted(enumerate(flows), key=lambda item: item[1]["shares"], reverse=True)
    for order, (index, flow) in enumerate(sorted_flow_items):
        source = flow["source"]
        target = flow["target"]
        source_top, source_bottom = source_ranges[index]
        target_top, target_bottom = target_ranges[index]
        gradient_id = f"flowGradient{index}"
        gradient_parts.append(
            f'<linearGradient id="{gradient_id}" x1="0%" x2="100%" y1="0%" y2="0%">'
            f'<stop offset="0%" stop-color="{participants[source].color}" stop-opacity="0.72"/>'
            f'<stop offset="100%" stop-color="{participants[target].color}" stop-opacity="0.50"/>'
            f"</linearGradient>"
        )
        c0 = x0 + (x1 - x0) * 0.42
        c1 = x0 + (x1 - x0) * 0.58
        path = (
            f"M {x0:.1f} {source_top:.1f} "
            f"C {c0:.1f} {source_top:.1f}, {c1:.1f} {target_top:.1f}, {x1:.1f} {target_top:.1f} "
            f"L {x1:.1f} {target_bottom:.1f} "
            f"C {c1:.1f} {target_bottom:.1f}, {c0:.1f} {source_bottom:.1f}, {x0:.1f} {source_bottom:.1f} Z"
        )
        link_parts.append(
            f'<path class="flow-ribbon" d="{path}" fill="url(#{gradient_id})" '
            f'stroke="#f8fafc" stroke-width="1.2" data-source="{html.escape(source)}" '
            f'data-target="{html.escape(target)}" data-source-top="{source_top:.3f}" '
            f'data-source-bottom="{source_bottom:.3f}" data-target-top="{target_top:.3f}" '
            f'data-target-bottom="{target_bottom:.3f}" data-order="{order}"><title>'
            f'{html.escape(participants[source].label)} → {html.escape(participants[target].label)} '
            f'{html.escape(format_lots(float(flow["shares"])))}'
            f"</title></path>"
        )

    node_parts = []
    for side, items, x, layout, anchor in [
        ("賣超來源", sellers, left_x, seller_layout, "start"),
        ("買超去向", buyers, right_x, buyer_layout, "end"),
    ]:
        title_x = x if anchor == "start" else x + node_width
        node_parts.append(
            f'<text x="{title_x}" y="60" text-anchor="{anchor}" class="side-title">{html.escape(side)}</text>'
        )
        for key, amount in items:
            participant = participants[key]
            node = layout[key]
            y = node["y"]
            node_height = node["height"]
            text_anchor = "start" if anchor == "start" else "end"
            text_x = x + 16 if anchor == "start" else x + node_width - 16
            accent_x = x if anchor == "start" else x + node_width - 7
            node_parts.append(
                f'<rect class="node-shadow" x="{x}" y="{y:.1f}" width="{node_width}" height="{node_height:.1f}" rx="9"/>'
                f'<rect class="node-card" x="{x}" y="{y:.1f}" width="{node_width}" height="{node_height:.1f}" rx="9"/>'
                f'<rect x="{accent_x}" y="{y:.1f}" width="7" height="{node_height:.1f}" rx="4" '
                f'fill="{participant.color}" opacity="0.92"/>'
                f'<text x="{text_x}" y="{node["center"] - 5:.1f}" text-anchor="{text_anchor}" class="node-label">'
                f'{html.escape(participant.label)}</text>'
                f'<text x="{text_x}" y="{node["center"] + 17:.1f}" text-anchor="{text_anchor}" class="node-value">'
                f'{html.escape(format_lots(amount))}</text>'
            )

    balance = sum(totals.values())
    return f"""
<svg class="sankey" viewBox="0 0 {width} {height}" role="img" aria-label="全市場三大法人籌碼淨流向圖">
  <defs>{"".join(gradient_parts)}</defs>
  <style>
    .sankey {{ width: 100%; height: auto; display: block; }}
    .side-title {{ fill: #475569; font-size: 14px; font-weight: 700; }}
    .flow-ribbon {{ shape-rendering: geometricPrecision; }}
    .node-shadow {{ fill: #0f172a; opacity: 0.08; transform: translate(0, 3px); }}
    .node-card {{ fill: rgba(255, 255, 255, 0.95); stroke: #cbd5e1; stroke-width: 1; }}
    .node-label {{ fill: #172033; font-size: 15px; font-weight: 700; }}
    .node-value {{ fill: #475569; font-size: 13px; }}
    .center-note {{ fill: #64748b; font-size: 13px; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#f8fafc"/>
  {"".join(link_parts)}
  {"".join(node_parts)}
  <text x="{width / 2}" y="{height - 28}" text-anchor="middle" class="center-note">
    淨額平衡差：{html.escape(format_lots(balance, digits=2))}，其他市場參與者為三大法人淨額的反向殘差
  </text>
</svg>
"""


def svg_net_bars(totals: dict[str, float]) -> str:
    width = 1120
    height = 300
    chart_left = 190
    chart_right = 1040
    zero_x = (chart_left + chart_right) / 2
    max_abs = max([abs(value) for value in totals.values()] or [1.0])
    scale = (chart_right - chart_left) / 2 / max_abs
    rows = []
    for index, participant in enumerate(PARTICIPANTS):
        value = totals[participant.key]
        y = 58 + index * 55
        x = zero_x if value >= 0 else zero_x + value * scale
        bar_width = abs(value) * scale
        text_anchor = "start" if value >= 0 else "end"
        text_x = x + bar_width + 8 if value >= 0 else x - 8
        rows.append(
            f'<text x="24" y="{y + 18}" class="bar-label">{html.escape(participant.label)}</text>'
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="28" rx="5" '
            f'fill="{participant.color}" opacity="0.85"/>'
            f'<text x="{text_x:.1f}" y="{y + 19}" text-anchor="{text_anchor}" class="bar-value">'
            f'{html.escape(format_lots(value))}</text>'
        )
    return f"""
<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="三大法人與其他市場參與者淨買賣超長條圖">
  <style>
    .bar-chart {{ width: 100%; height: auto; display: block; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1; }}
    .bar-label {{ fill: #172033; font-size: 14px; font-weight: 700; }}
    .bar-value {{ fill: #334155; font-size: 13px; }}
    .axis-label {{ fill: #64748b; font-size: 12px; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#ffffff"/>
  <line x1="{zero_x}" y1="34" x2="{zero_x}" y2="{height - 28}" class="axis"/>
  <text x="{chart_left}" y="22" text-anchor="start" class="axis-label">淨賣超</text>
  <text x="{chart_right}" y="22" text-anchor="end" class="axis-label">淨買超</text>
  {"".join(rows)}
</svg>
"""


def metric_cards(totals: dict[str, float], stock_count: int, target_date: str) -> str:
    cards = [
        ("資料日期", target_date, "flat"),
        ("納入股票", f"{stock_count:,} 檔", "flat"),
    ]
    for participant in PARTICIPANTS:
        value = totals[participant.key]
        cards.append((f"{participant.label}淨買賣超", format_lots(value), signed_class(value)))
    return "\n".join(
        f'<div class="metric {css_class}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value, css_class in cards
    )


def flow_table_rows(flows: list[dict[str, Any]]) -> str:
    participants = participant_by_key()
    rows = []
    for index, flow in enumerate(sorted(flows, key=lambda item: item["shares"], reverse=True), start=1):
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(participants[flow['source']].label)}</td>"
            f"<td>{html.escape(participants[flow['target']].label)}</td>"
            f"<td>{html.escape(format_lots(float(flow['shares'])))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def stock_table_rows(df: pd.DataFrame, value_column: str, ascending: bool, limit: int = 20) -> str:
    rows = []
    subset = df.sort_values(value_column, ascending=ascending).head(limit)
    for rank, row in enumerate(subset.itertuples(index=False), start=1):
        value = float(getattr(row, value_column))
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(str(row.Code))}</td>"
            f"<td>{html.escape(str(row.Name))}</td>"
            f"<td>{html.escape(str(row.IndustryGroup))}</td>"
            f'<td class="{signed_class(value)}">{html.escape(format_lots(value))}</td>'
            f'<td class="{signed_class(float(row.ForeignNetShares))}">{html.escape(format_lots(float(row.ForeignNetShares)))}</td>'
            f'<td class="{signed_class(float(row.TrustNetShares))}">{html.escape(format_lots(float(row.TrustNetShares)))}</td>'
            f'<td class="{signed_class(float(row.DealerNetShares))}">{html.escape(format_lots(float(row.DealerNetShares)))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def participant_rank_rows(df: pd.DataFrame, participant: Participant, limit: int = 8) -> str:
    column_by_key = {
        "foreign": "ForeignNetShares",
        "trust": "TrustNetShares",
        "dealer": "DealerNetShares",
        "other": "OtherNetShares",
    }
    column = column_by_key[participant.key]
    buy_rows = df.sort_values(column, ascending=False).head(limit)
    sell_rows = df.sort_values(column, ascending=True).head(limit)

    def mini_rows(rows_df: pd.DataFrame) -> str:
        rows = []
        for row in rows_df.itertuples(index=False):
            value = float(getattr(row, column))
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.Code))}</td>"
                f"<td>{html.escape(str(row.Name))}</td>"
                f'<td class="{signed_class(value)}">{html.escape(format_lots(value))}</td>'
                "</tr>"
            )
        return "\n".join(rows)

    return f"""
<section class="participant-card">
  <h3><span class="swatch" style="background:{participant.color}"></span>{html.escape(participant.label)}</h3>
  <div class="rank-grid">
    <div>
      <h4>淨買超</h4>
      <table><thead><tr><th>代號</th><th>名稱</th><th>張數</th></tr></thead><tbody>{mini_rows(buy_rows)}</tbody></table>
    </div>
    <div>
      <h4>淨賣超</h4>
      <table><thead><tr><th>代號</th><th>名稱</th><th>張數</th></tr></thead><tbody>{mini_rows(sell_rows)}</tbody></table>
    </div>
  </div>
</section>
"""


def build_industry_rows(stock_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        stock_rows.groupby("IndustryGroup", as_index=False)[
            [
                "ForeignNetShares",
                "TrustNetShares",
                "DealerNetShares",
                "OtherNetShares",
                "InstitutionalNetShares",
                "AbsInstitutionalNetShares",
            ]
        ]
        .sum()
        .rename(
            columns={
                "IndustryGroup": "產業群組",
                "ForeignNetShares": "外資淨買賣超股數",
                "TrustNetShares": "投信淨買賣超股數",
                "DealerNetShares": "自營商淨買賣超股數",
                "OtherNetShares": "其他市場參與者推估淨買賣超股數",
                "InstitutionalNetShares": "三大法人合計淨買賣超股數",
                "AbsInstitutionalNetShares": "個股三大法人絕對流量加總股數",
            }
        )
    )
    counts = stock_rows.groupby("IndustryGroup")["Code"].count()
    grouped["產業內股票數"] = grouped["產業群組"].map(counts).fillna(0).astype(int)
    return grouped.sort_values("個股三大法人絕對流量加總股數", ascending=False)


def industry_table_rows(industry_rows: pd.DataFrame, limit: int = 20) -> str:
    rows = []
    for rank, (_index, row) in enumerate(industry_rows.head(limit).iterrows(), start=1):
        institutional = float(row["三大法人合計淨買賣超股數"])
        foreign = float(row["外資淨買賣超股數"])
        trust = float(row["投信淨買賣超股數"])
        dealer = float(row["自營商淨買賣超股數"])
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(str(row['產業群組']))}</td>"
            f"<td>{int(row['產業內股票數']):,}</td>"
            f'<td class="{signed_class(foreign)}">{html.escape(format_lots(foreign))}</td>'
            f'<td class="{signed_class(trust)}">{html.escape(format_lots(trust))}</td>'
            f'<td class="{signed_class(dealer)}">{html.escape(format_lots(dealer))}</td>'
            f'<td class="{signed_class(institutional)}">{html.escape(format_lots(institutional))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def write_csv_outputs(
    stock_rows: pd.DataFrame,
    industry_rows: pd.DataFrame,
    flow_rows: pd.DataFrame,
    target_date: str,
) -> tuple[Path, Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stock_path = OUTPUT_ROOT / f"latest_stock_flows_{target_date}.csv"
    industry_path = OUTPUT_ROOT / f"latest_industry_flows_{target_date}.csv"
    flow_path = OUTPUT_ROOT / f"latest_market_flow_links_{target_date}.csv"

    stock_output = stock_rows.rename(
        columns={
            "Date": "日期",
            "Code": "代號",
            "Name": "名稱",
            "IndustryGroup": "產業群組",
            "ForeignNetShares": "外資淨買賣超股數",
            "TrustNetShares": "投信淨買賣超股數",
            "DealerNetShares": "自營商淨買賣超股數",
            "OtherNetShares": "其他市場參與者推估淨買賣超股數",
            "InstitutionalNetShares": "三大法人合計淨買賣超股數",
        }
    )
    stock_output.to_csv(stock_path, index=False, encoding="utf-8-sig")
    industry_rows.to_csv(industry_path, index=False, encoding="utf-8-sig")
    flow_rows.to_csv(flow_path, index=False, encoding="utf-8-sig")
    return stock_path, industry_path, flow_path


def write_report(
    stock_rows: pd.DataFrame,
    industry_rows: pd.DataFrame,
    totals: dict[str, float],
    flows: list[dict[str, Any]],
    target_date: str,
    date_counts: dict[str, int],
    target_reason: str,
    skipped: list[str],
    stock_path: Path,
    industry_path: Path,
    flow_path: Path,
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "latest_market_flow.html"
    stock_count = len(stock_rows)
    latest_count = date_counts.get(target_date, 0)
    flow_rows = flow_table_rows(flows)
    participant_cards = "\n".join(participant_rank_rows(stock_rows, participant) for participant in PARTICIPANTS)
    top_buy_rows = stock_table_rows(stock_rows, "InstitutionalNetShares", ascending=False)
    top_sell_rows = stock_table_rows(stock_rows, "InstitutionalNetShares", ascending=True)
    industry_rows_html = industry_table_rows(industry_rows)
    date_count_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(date)}</td>"
        f"<td>{count:,}</td>"
        "</tr>"
        for date, count in sorted(date_counts.items(), reverse=True)[:8]
    )
    skipped_note = "；".join(skipped[:8])
    if len(skipped) > 8:
        skipped_note += f"；另有 {len(skipped) - 8:,} 筆略"

    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全市場三大法人籌碼淨流向</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #64748b;
  --line: #d7dee9;
  --pos: #b91c1c;
  --neg: #047857;
}}
body {{
  margin: 0;
  font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif;
  color: var(--ink);
  background: #eef2f7;
}}
main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
h2 {{ margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }}
h3 {{ margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }}
h4 {{ margin: 0 0 8px; font-size: 14px; color: var(--muted); letter-spacing: 0; }}
p {{ line-height: 1.7; margin: 8px 0; }}
a {{ color: #1d4ed8; text-decoration: none; }}
code {{ color: #334155; }}
.meta {{ color: var(--muted); font-size: 13px; margin-bottom: 16px; }}
.summary {{
  display: grid;
  grid-template-columns: repeat(6, minmax(140px, 1fr));
  gap: 10px;
  margin: 18px 0;
}}
.metric {{
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 12px;
  min-height: 74px;
}}
.metric .label {{ color: var(--muted); font-size: 12px; }}
.metric .value {{ font-size: 19px; font-weight: 700; margin-top: 8px; }}
.metric.pos .value, .pos {{ color: var(--pos); }}
.metric.neg .value, .neg {{ color: var(--neg); }}
.metric.flat .value, .flat {{ color: #334155; }}
.panel {{
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}}
.grid-2 {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}}
.participant-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(280px, 1fr));
  gap: 14px;
}}
.participant-card {{
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}}
.participant-card h3 {{ display: flex; align-items: center; gap: 8px; }}
.rank-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.swatch {{ width: 13px; height: 13px; border-radius: 3px; display: inline-block; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid var(--line); }}
th, td {{
  border-bottom: 1px solid #e2e8f0;
  padding: 8px 10px;
  text-align: right;
  font-size: 13px;
  vertical-align: top;
}}
th {{ background: #f1f5f9; color: #334155; font-weight: 700; }}
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
.participant-card td:nth-child(1), .participant-card th:nth-child(1),
.participant-card td:nth-child(2), .participant-card th:nth-child(2) {{ text-align: left; }}
.note {{ color: var(--muted); font-size: 13px; }}
.links {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0; }}
@media (max-width: 900px) {{
  main {{ padding: 14px; }}
  .summary {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
  .grid-2, .participant-grid, .rank-grid {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 24px; }}
}}
</style>
</head>
<body>
<main>
<h1>全市場三大法人籌碼淨流向</h1>
<div class="meta">資料日期 {html.escape(target_date)}，選日方式：{html.escape(target_reason)}，來源：<code>data/institutional/</code> 與 <code>data/metadata.csv</code></div>
<section class="summary">
{metric_cards(totals, stock_count, target_date)}
</section>
<section class="panel">
<h2>每日籌碼流向圖</h2>
<p>圖中的來源是當日淨賣超族群，去向是當日淨買超族群；線寬代表推估流向張數。這是用每日淨買賣超做出的淨額分配，不代表逐筆成交對手方。</p>
{svg_sankey(totals, flows)}
</section>
<section class="panel">
<h2>族群淨買賣超</h2>
{svg_net_bars(totals)}
</section>
<section class="panel">
<h2>推估流向明細</h2>
<table>
<thead><tr><th>排名</th><th>來源</th><th>去向</th><th>推估張數</th></tr></thead>
<tbody>
{flow_rows}
</tbody>
</table>
</section>
<section class="grid-2">
<div class="panel">
<h2>三大法人合計買超前 20</h2>
<table>
<thead><tr><th>排名</th><th>代號</th><th>名稱</th><th>產業群組</th><th>三大法人</th><th>外資</th><th>投信</th><th>自營商</th></tr></thead>
<tbody>{top_buy_rows}</tbody>
</table>
</div>
<div class="panel">
<h2>三大法人合計賣超前 20</h2>
<table>
<thead><tr><th>排名</th><th>代號</th><th>名稱</th><th>產業群組</th><th>三大法人</th><th>外資</th><th>投信</th><th>自營商</th></tr></thead>
<tbody>{top_sell_rows}</tbody>
</table>
</div>
</section>
<h2>各族群個股排行</h2>
<section class="participant-grid">
{participant_cards}
</section>
<section class="panel">
<h2>產業群組淨流向</h2>
<table>
<thead><tr><th>排名</th><th>產業群組</th><th>股票數</th><th>外資</th><th>投信</th><th>自營商</th><th>三大法人合計</th></tr></thead>
<tbody>{industry_rows_html}</tbody>
</table>
</section>
<section class="panel">
<h2>資料覆蓋</h2>
<p>本報告納入 {stock_count:,} 檔上市普通股；目標日期在法人檔尾出現 {latest_count:,} 檔。未納入清單多半是缺法人檔或該檔最新資料尚未到 {html.escape(target_date)}。</p>
<table>
<thead><tr><th>檔尾日期</th><th>檔案數</th></tr></thead>
<tbody>{date_count_rows}</tbody>
</table>
<p class="note">未納入樣本：{html.escape(skipped_note) if skipped_note else "無"}</p>
</section>
<section class="panel">
<h2>輸出檔案</h2>
<div class="links">
<a href="../../output/institutional_flow/{html.escape(stock_path.name)}">下載個股彙總表</a>
<a href="../../output/institutional_flow/{html.escape(industry_path.name)}">下載產業彙總表</a>
<a href="../../output/institutional_flow/{html.escape(flow_path.name)}">下載流向明細表</a>
</div>
<p class="note">其他市場參與者是用每檔股票三大法人合計淨買賣超的反向殘差估算，包含自然人、一般法人與其他未分類交易者；報告中不直接稱為散戶。</p>
</section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_report(date: str | None, min_coverage_ratio: float) -> dict[str, Any]:
    metadata_all = read_metadata()
    metadata = listed_common_stock_metadata(metadata_all)
    universe_codes = set(metadata["Code"].astype(str))
    inst_paths = path_by_code(INSTITUTIONAL_DIR)
    target_date, date_counts, target_reason = choose_target_date(
        inst_paths, universe_codes, date, min_coverage_ratio
    )
    stock_rows, skipped = build_stock_rows(metadata, inst_paths, target_date)
    totals = market_totals(stock_rows)
    flows = allocate_flows(totals)
    industry_rows = build_industry_rows(stock_rows)
    flow_rows = pd.DataFrame(
        [
            {
                "日期": target_date,
                "來源": participant_by_key()[flow["source"]].label,
                "去向": participant_by_key()[flow["target"]].label,
                "推估股數": flow["shares"],
                "推估張數": flow["lots"],
            }
            for flow in sorted(flows, key=lambda item: item["shares"], reverse=True)
        ]
    )
    stock_path, industry_path, flow_path = write_csv_outputs(
        stock_rows, industry_rows, flow_rows, target_date
    )
    report_path = write_report(
        stock_rows,
        industry_rows,
        totals,
        flows,
        target_date,
        date_counts,
        target_reason,
        skipped,
        stock_path,
        industry_path,
        flow_path,
    )
    return {
        "target_date": target_date,
        "stock_count": len(stock_rows),
        "skipped_count": len(skipped),
        "totals": totals,
        "flow_count": len(flows),
        "report_path": report_path,
        "stock_path": stock_path,
        "industry_path": industry_path,
        "flow_path": flow_path,
    }


def main() -> None:
    args = parse_args()
    result = build_report(args.date, args.min_coverage_ratio)
    print("Latest institutional flow report:")
    print(f"target_date={result['target_date']}")
    print(f"stock_count={result['stock_count']}")
    print(f"skipped_count={result['skipped_count']}")
    for participant in PARTICIPANTS:
        print(f"{participant.key}_net_shares={result['totals'][participant.key]:.0f}")
    print(f"flow_count={result['flow_count']}")
    print(f"report={result['report_path']}")
    print(f"stock_csv={result['stock_path']}")
    print(f"industry_csv={result['industry_path']}")
    print(f"flow_csv={result['flow_path']}")


if __name__ == "__main__":
    main()
