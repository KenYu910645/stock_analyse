from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "broker" / "by_broker"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
DEFAULT_VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "broker"

FIELD_DATE = "Date"
FIELD_BRANCH_NAME = "分點名稱"
FIELD_SIDE = "買賣別"
FIELD_CODE = "Code"
FIELD_STOCK_NAME = "Name"
FIELD_BUY = "買進"
FIELD_SELL = "賣出"
FIELD_NET = "買賣超"
FIELD_SOURCE_URL = "來源網址"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_URL_DATE_RE = re.compile(r"[?&]e=(\d{4})-(\d{1,2})-(\d{1,2})(?:&|$)")
MIN_REASONABLE_DATE = "2000-01-01"

BRANCH_SUMMARY_COLUMNS = [
    "分點名稱",
    "是否明顯外資分點",
    "外資判斷關鍵字",
    "資料筆數",
    "活躍日期數",
    "起始日期",
    "結束日期",
    "股票數",
    "買進合計",
    "賣出合計",
    "成交量指標",
    "買賣超合計",
    "買方占比",
    "淨買超占成交量比率",
]

TOP_STOCK_COLUMNS = [
    "分點名稱",
    "是否明顯外資分點",
    "外資判斷關鍵字",
    "排名",
    "股票代號",
    "股票名稱",
    "產業群組",
    "資料筆數",
    "買方筆數",
    "賣方筆數",
    "起始日期",
    "結束日期",
    "買進合計",
    "賣出合計",
    "成交量指標",
    "買賣超合計",
    "買方占比",
    "淨買超占成交量比率",
]

NET_STOCK_COLUMNS = [
    "分點名稱",
    "是否明顯外資分點",
    "外資判斷關鍵字",
    "方向",
    "排名",
    "股票代號",
    "股票名稱",
    "產業群組",
    "資料筆數",
    "買進合計",
    "賣出合計",
    "成交量指標",
    "買賣超合計",
    "起始日期",
    "結束日期",
]

FOREIGN_BRANCH_KEYWORDS = [
    "台灣摩根士丹利",
    "摩根士丹利",
    "摩根大通",
    "美商高盛",
    "新加坡商瑞銀",
    "瑞士信貸",
    "美林",
    "花旗環球",
    "香港上海匯豐",
    "港商野村",
    "法銀巴黎",
    "法國巴黎",
    "港商麥格理",
    "大和國泰",
    "港商法國興業",
    "法國興業",
    "里昂",
    "巴克萊",
    "德意志",
    "美商",
    "港商",
    "香港商",
    "新加坡商",
    "法商",
]


@dataclass
class StockStats:
    code: str
    name: str
    rows: int = 0
    buy_rows: int = 0
    sell_rows: int = 0
    buy: int = 0
    sell: int = 0
    net: int = 0
    first_date: str | None = None
    last_date: str | None = None

    def add(self, date: str, side: str, buy: int, sell: int, net: int) -> None:
        self.rows += 1
        if side == "buy":
            self.buy_rows += 1
        elif side == "sell":
            self.sell_rows += 1
        self.buy += buy
        self.sell += sell
        self.net += net
        if self.first_date is None or date < self.first_date:
            self.first_date = date
        if self.last_date is None or date > self.last_date:
            self.last_date = date

    @property
    def gross(self) -> int:
        return self.buy + self.sell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Fubon broker branch volume and frequent-stock reports."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=DEFAULT_VIZ_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--top-stocks-per-branch", type=int, default=20)
    parser.add_argument("--top-net-stocks-per-branch", type=int, default=10)
    return parser.parse_args()


def parse_int(value: str | None) -> int:
    if not value:
        return 0
    text = value.replace(",", "").strip()
    if not text or text == "-":
        return 0
    return int(float(text))


def parse_source_url_date(source_url: str | None) -> str | None:
    if not source_url:
        return None
    match = SOURCE_URL_DATE_RE.search(source_url)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_report_date(raw_date: str, source_url: str | None) -> tuple[str | None, str | None]:
    if DATE_RE.match(raw_date) and raw_date >= MIN_REASONABLE_DATE:
        return raw_date, None
    source_date = parse_source_url_date(source_url)
    if source_date:
        return source_date, "來源網址日期修正"
    if DATE_RE.match(raw_date):
        return None, "日期早於合理範圍"
    return None, "日期格式異常"


def ratio_text(numerator: int, denominator: int) -> str:
    if not denominator:
        return ""
    return f"{numerator / denominator:.6f}"


def pct_text(numerator: int, denominator: int) -> str:
    if not denominator:
        return ""
    return f"{numerator / denominator:.2%}"


def fmt_int(value: int | str) -> str:
    return f"{int(value):,}"


def fmt_pct_from_text(value: str) -> str:
    if not value:
        return ""
    return f"{float(value):.2%}"


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row.get("Code", ""): row for row in reader if row.get("Code")}


def foreign_keywords(branch_name: str) -> list[str]:
    return [keyword for keyword in FOREIGN_BRANCH_KEYWORDS if keyword in branch_name]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_ratio_fields(buy: int, sell: int, net: int) -> tuple[str, str]:
    gross = buy + sell
    return ratio_text(buy, gross), ratio_text(net, gross)


def build_reports(args: argparse.Namespace) -> dict[str, object]:
    source_dir = args.source_dir
    output_dir = args.output_dir
    viz_dir = args.viz_dir
    metadata = load_metadata(args.metadata)

    branch_files = sorted(source_dir.glob("*.csv"))
    if not branch_files:
        raise FileNotFoundError(f"No CSV files found under {source_dir}")

    branch_summary_rows: list[dict[str, object]] = []
    top_stock_rows: list[dict[str, object]] = []
    net_stock_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    corrected_date_rows: list[dict[str, object]] = []
    skipped_row_count = 0
    corrected_date_count = 0

    total_rows = 0
    for index, path in enumerate(branch_files, 1):
        branch_name = path.stem
        dates: set[str] = set()
        stock_stats: dict[str, StockStats] = {}
        branch_buy = 0
        branch_sell = 0
        branch_net = 0
        branch_rows = 0
        first_date: str | None = None
        last_date: str | None = None

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                FIELD_DATE,
                FIELD_BRANCH_NAME,
                FIELD_SIDE,
                FIELD_CODE,
                FIELD_STOCK_NAME,
                FIELD_BUY,
                FIELD_SELL,
                FIELD_NET,
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")

            for row_number, row in enumerate(reader, start=2):
                raw_date = (row.get(FIELD_DATE) or "").strip()
                source_url = (row.get(FIELD_SOURCE_URL) or "").strip()
                date, date_note = normalize_report_date(raw_date, source_url)
                code = (row.get(FIELD_CODE) or "").strip()
                if not date or not code:
                    skipped_row_count += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(PROJECT_ROOT)),
                                "列號": row_number,
                                "原因": date_note or "日期或股票代號異常",
                                "Date": row.get(FIELD_DATE, ""),
                                "Code": row.get(FIELD_CODE, ""),
                                "買進": row.get(FIELD_BUY, ""),
                                "賣出": row.get(FIELD_SELL, ""),
                                "買賣超": row.get(FIELD_NET, ""),
                            }
                        )
                    continue
                if date_note:
                    corrected_date_count += 1
                    if len(corrected_date_rows) < 1000:
                        corrected_date_rows.append(
                            {
                                "檔案": str(path.relative_to(PROJECT_ROOT)),
                                "列號": row_number,
                                "原始Date": raw_date,
                                "修正Date": date,
                                "原因": date_note,
                                "來源網址": source_url,
                            }
                        )

                stock_name = (row.get(FIELD_STOCK_NAME) or "").strip()
                side = (row.get(FIELD_SIDE) or "").strip()
                try:
                    buy = parse_int(row.get(FIELD_BUY))
                    sell = parse_int(row.get(FIELD_SELL))
                    net = parse_int(row.get(FIELD_NET))
                except ValueError:
                    skipped_row_count += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(PROJECT_ROOT)),
                                "列號": row_number,
                                "原因": "買進賣出欄位非數字",
                                "Date": row.get(FIELD_DATE, ""),
                                "Code": row.get(FIELD_CODE, ""),
                                "買進": row.get(FIELD_BUY, ""),
                                "賣出": row.get(FIELD_SELL, ""),
                                "買賣超": row.get(FIELD_NET, ""),
                            }
                        )
                    continue

                branch_rows += 1
                total_rows += 1
                dates.add(date)
                branch_buy += buy
                branch_sell += sell
                branch_net += net
                if first_date is None or date < first_date:
                    first_date = date
                if last_date is None or date > last_date:
                    last_date = date

                stock = stock_stats.get(code)
                if stock is None:
                    stock = StockStats(code=code, name=stock_name)
                    stock_stats[code] = stock
                elif not stock.name and stock_name:
                    stock.name = stock_name
                stock.add(date, side, buy, sell, net)

        keywords = foreign_keywords(branch_name)
        is_foreign = "是" if keywords else "否"
        keyword_text = "、".join(keywords)
        branch_gross = branch_buy + branch_sell
        buy_ratio, net_ratio = row_ratio_fields(branch_buy, branch_sell, branch_net)
        branch_summary_rows.append(
            {
                "分點名稱": branch_name,
                "是否明顯外資分點": is_foreign,
                "外資判斷關鍵字": keyword_text,
                "資料筆數": branch_rows,
                "活躍日期數": len(dates),
                "起始日期": first_date or "",
                "結束日期": last_date or "",
                "股票數": len(stock_stats),
                "買進合計": branch_buy,
                "賣出合計": branch_sell,
                "成交量指標": branch_gross,
                "買賣超合計": branch_net,
                "買方占比": buy_ratio,
                "淨買超占成交量比率": net_ratio,
            }
        )

        stocks_by_gross = sorted(
            stock_stats.values(), key=lambda item: (item.gross, abs(item.net), item.rows), reverse=True
        )
        for rank, stock in enumerate(stocks_by_gross[: args.top_stocks_per_branch], 1):
            group = metadata.get(stock.code, {}).get("產業群組", "")
            buy_ratio, net_ratio = row_ratio_fields(stock.buy, stock.sell, stock.net)
            top_stock_rows.append(
                {
                    "分點名稱": branch_name,
                    "是否明顯外資分點": is_foreign,
                    "外資判斷關鍵字": keyword_text,
                    "排名": rank,
                    "股票代號": stock.code,
                    "股票名稱": stock.name,
                    "產業群組": group,
                    "資料筆數": stock.rows,
                    "買方筆數": stock.buy_rows,
                    "賣方筆數": stock.sell_rows,
                    "起始日期": stock.first_date or "",
                    "結束日期": stock.last_date or "",
                    "買進合計": stock.buy,
                    "賣出合計": stock.sell,
                    "成交量指標": stock.gross,
                    "買賣超合計": stock.net,
                    "買方占比": buy_ratio,
                    "淨買超占成交量比率": net_ratio,
                }
            )

        net_buy = sorted(stock_stats.values(), key=lambda item: (item.net, item.gross), reverse=True)
        net_sell = sorted(stock_stats.values(), key=lambda item: (item.net, -item.gross))
        for direction, ranked_stocks in [("偏買超", net_buy), ("偏賣超", net_sell)]:
            for rank, stock in enumerate(ranked_stocks[: args.top_net_stocks_per_branch], 1):
                group = metadata.get(stock.code, {}).get("產業群組", "")
                net_stock_rows.append(
                    {
                        "分點名稱": branch_name,
                        "是否明顯外資分點": is_foreign,
                        "外資判斷關鍵字": keyword_text,
                        "方向": direction,
                        "排名": rank,
                        "股票代號": stock.code,
                        "股票名稱": stock.name,
                        "產業群組": group,
                        "資料筆數": stock.rows,
                        "買進合計": stock.buy,
                        "賣出合計": stock.sell,
                        "成交量指標": stock.gross,
                        "買賣超合計": stock.net,
                        "起始日期": stock.first_date or "",
                        "結束日期": stock.last_date or "",
                    }
                )

        if index % 50 == 0 or index == len(branch_files):
            print(f"processed {index}/{len(branch_files)} branch files", flush=True)

    branch_summary_rows.sort(key=lambda row: int(row["成交量指標"]), reverse=True)
    foreign_branch_rows = [row for row in branch_summary_rows if row["是否明顯外資分點"] == "是"]
    foreign_top_stock_rows = [row for row in top_stock_rows if row["是否明顯外資分點"] == "是"]
    foreign_net_stock_rows = [row for row in net_stock_rows if row["是否明顯外資分點"] == "是"]

    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    branch_summary_path = output_dir / "branch_volume_summary.csv"
    foreign_summary_path = output_dir / "foreign_branch_summary.csv"
    top_stock_path = output_dir / "branch_top_stocks_by_volume.csv"
    foreign_top_stock_path = output_dir / "foreign_branch_top_stocks_by_volume.csv"
    net_stock_path = output_dir / "branch_top_net_buy_sell.csv"
    foreign_net_stock_path = output_dir / "foreign_branch_top_net_buy_sell.csv"
    html_path = viz_dir / "by_broker_initial_report.html"
    metadata_path = output_dir / "broker_branch_report_metadata.json"

    write_csv(branch_summary_path, BRANCH_SUMMARY_COLUMNS, branch_summary_rows)
    write_csv(foreign_summary_path, BRANCH_SUMMARY_COLUMNS, foreign_branch_rows)
    write_csv(top_stock_path, TOP_STOCK_COLUMNS, top_stock_rows)
    write_csv(foreign_top_stock_path, TOP_STOCK_COLUMNS, foreign_top_stock_rows)
    write_csv(net_stock_path, NET_STOCK_COLUMNS, net_stock_rows)
    write_csv(foreign_net_stock_path, NET_STOCK_COLUMNS, foreign_net_stock_rows)
    if skipped_rows:
        write_csv(
            args.log_dir / "broker_branch_report_skipped_rows.csv",
            ["檔案", "列號", "原因", "Date", "Code", "買進", "賣出", "買賣超"],
            skipped_rows,
        )
    if corrected_date_rows:
        write_csv(
            args.log_dir / "broker_branch_report_corrected_dates.csv",
            ["檔案", "列號", "原始Date", "修正Date", "原因", "來源網址"],
            corrected_date_rows,
        )

    html_path.write_text(
        render_html(branch_summary_rows, foreign_branch_rows, top_stock_rows, foreign_top_stock_rows, foreign_net_stock_rows),
        encoding="utf-8",
    )

    report_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "branch_files": len(branch_files),
        "source_rows": total_rows,
        "branch_summary_rows": len(branch_summary_rows),
        "foreign_branch_rows": len(foreign_branch_rows),
        "top_stock_rows": len(top_stock_rows),
        "foreign_top_stock_rows": len(foreign_top_stock_rows),
        "net_stock_rows": len(net_stock_rows),
        "foreign_net_stock_rows": len(foreign_net_stock_rows),
        "skipped_rows": skipped_row_count,
        "skipped_rows_logged": len(skipped_rows),
        "corrected_dates": corrected_date_count,
        "corrected_dates_logged": len(corrected_date_rows),
        "outputs": [
            str(branch_summary_path),
            str(foreign_summary_path),
            str(top_stock_path),
            str(foreign_top_stock_path),
            str(net_stock_path),
            str(foreign_net_stock_path),
            str(html_path),
        ],
    }
    metadata_path.write_text(json.dumps(report_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_metadata


def html_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_bar_chart(title: str, rows: list[dict[str, object]], value_key: str, label_key: str, limit: int) -> str:
    visible = rows[:limit]
    max_value = max((int(row[value_key]) for row in visible), default=0)
    items = []
    for row in visible:
        value = int(row[value_key])
        width = 0 if max_value == 0 else max(2, value / max_value * 100)
        badge = "外資推定" if row.get("是否明顯外資分點") == "是" else ""
        items.append(
            f"""
            <div class="bar-row">
              <div class="bar-label"><span>{html_escape(row[label_key])}</span><em>{html_escape(badge)}</em></div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>
              <div class="bar-value">{fmt_int(value)}</div>
            </div>
            """
        )
    return f"""
    <section class="section">
      <h2>{html_escape(title)}</h2>
      <div class="bar-chart">{''.join(items)}</div>
    </section>
    """


def render_table(headers: list[str], rows: list[dict[str, object]], css_class: str = "") -> str:
    header_html = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if header in {"成交量指標", "買進合計", "賣出合計", "買賣超合計", "資料筆數", "活躍日期數", "股票數", "買方筆數", "賣方筆數"}:
                try:
                    value = fmt_int(value)
                except ValueError:
                    pass
            elif header in {"買方占比", "淨買超占成交量比率"}:
                value = fmt_pct_from_text(str(value)) if value != "" else ""
            cells.append(f"<td>{html_escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"""
    <div class="table-wrap {css_class}">
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def render_foreign_details(
    foreign_branch_rows: list[dict[str, object]],
    foreign_top_stock_rows: list[dict[str, object]],
    foreign_net_stock_rows: list[dict[str, object]],
) -> str:
    top_by_branch: dict[str, list[dict[str, object]]] = defaultdict(list)
    net_by_branch: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in foreign_top_stock_rows:
        if int(row.get("排名", 0)) <= 12:
            top_by_branch[str(row["分點名稱"])].append(row)
    for row in foreign_net_stock_rows:
        if int(row.get("排名", 0)) <= 5:
            net_by_branch[(str(row["分點名稱"]), str(row["方向"]))].append(row)

    details = []
    for branch in foreign_branch_rows:
        branch_name = str(branch["分點名稱"])
        top_table = render_table(
            ["排名", "股票代號", "股票名稱", "產業群組", "成交量指標", "買進合計", "賣出合計", "買賣超合計", "起始日期", "結束日期"],
            top_by_branch.get(branch_name, []),
        )
        buy_table = render_table(
            ["排名", "股票代號", "股票名稱", "產業群組", "買賣超合計", "成交量指標", "買進合計", "賣出合計"],
            net_by_branch.get((branch_name, "偏買超"), []),
        )
        sell_table = render_table(
            ["排名", "股票代號", "股票名稱", "產業群組", "買賣超合計", "成交量指標", "買進合計", "賣出合計"],
            net_by_branch.get((branch_name, "偏賣超"), []),
        )
        details.append(
            f"""
            <details>
              <summary>
                <span>{html_escape(branch_name)}</span>
                <small>成交量指標 {fmt_int(branch["成交量指標"])}，資料 {fmt_int(branch["資料筆數"])} 筆，關鍵字 {html_escape(branch["外資判斷關鍵字"])}</small>
              </summary>
              <h3>最常交易股票</h3>
              {top_table}
              <div class="split">
                <div>
                  <h3>長期偏買超</h3>
                  {buy_table}
                </div>
                <div>
                  <h3>長期偏賣超</h3>
                  {sell_table}
                </div>
              </div>
            </details>
            """
        )
    return "".join(details)


def render_observations(branch_rows: list[dict[str, object]], foreign_rows: list[dict[str, object]]) -> str:
    top_branch = branch_rows[0]
    top_foreign = foreign_rows[0] if foreign_rows else None
    foreign_volume = sum(int(row["成交量指標"]) for row in foreign_rows)
    total_volume = sum(int(row["成交量指標"]) for row in branch_rows)
    foreign_share = pct_text(foreign_volume, total_volume)
    foreign_text = (
        f"名稱規則推定的外資分點共有 {len(foreign_rows)} 個，合計成交量指標 {fmt_int(foreign_volume)}，占全部分點 {foreign_share}。"
        if foreign_rows
        else "這次名稱規則沒有抓到明顯外資分點。"
    )
    top_foreign_text = (
        f"外資推定分點中，成交量最高的是 {html_escape(top_foreign['分點名稱'])}，成交量指標 {fmt_int(top_foreign['成交量指標'])}。"
        if top_foreign
        else ""
    )
    return f"""
    <section class="section observation">
      <h2>初步觀察</h2>
      <p>目前 `by_broker` 共有 {len(branch_rows)} 個分點。成交量指標最高的是 {html_escape(top_branch['分點名稱'])}，合計 {fmt_int(top_branch['成交量指標'])}。</p>
      <p>{foreign_text} {top_foreign_text}</p>
      <p>這份報告的成交量指標是 Fubon broker rank row 裡的買進加賣出彙總；因為來源是排行榜資料，不是每個分點完整逐筆成交紀錄，所以適合拿來比較「在排行榜資料中出現的活躍度」，不應直接解讀為官方全市場成交量。</p>
    </section>
    """


def render_html(
    branch_rows: list[dict[str, object]],
    foreign_rows: list[dict[str, object]],
    top_stock_rows: list[dict[str, object]],
    foreign_top_stock_rows: list[dict[str, object]],
    foreign_net_stock_rows: list[dict[str, object]],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_top_stocks_preview = sorted(
        top_stock_rows,
        key=lambda row: (int(row["成交量指標"]), abs(int(row["買賣超合計"]))),
        reverse=True,
    )[:500]
    table_headers = [
        "分點名稱",
        "是否明顯外資分點",
        "外資判斷關鍵字",
        "資料筆數",
        "活躍日期數",
        "起始日期",
        "結束日期",
        "股票數",
        "買進合計",
        "賣出合計",
        "成交量指標",
        "買賣超合計",
        "買方占比",
    ]
    top_stock_headers = [
        "分點名稱",
        "排名",
        "股票代號",
        "股票名稱",
        "產業群組",
        "成交量指標",
        "買進合計",
        "賣出合計",
        "買賣超合計",
        "買方占比",
    ]
    all_branch_table = render_table(table_headers, branch_rows)
    foreign_table = render_table(table_headers, foreign_rows)
    top_stock_table = render_table(top_stock_headers, all_top_stocks_preview)
    foreign_details = render_foreign_details(foreign_rows, foreign_top_stock_rows, foreign_net_stock_rows)
    top_all_chart = render_bar_chart("全部分點成交量指標排行前 30", branch_rows, "成交量指標", "分點名稱", 30)
    top_foreign_chart = render_bar_chart("明顯外資分點成交量指標排行", foreign_rows, "成交量指標", "分點名稱", min(30, len(foreign_rows)))
    observations = render_observations(branch_rows, foreign_rows)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>分點成交量與常交易股票初報告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6975;
      --line: #d8dee6;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #9a3412;
      --soft: #e8f3f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      padding: 28px 36px 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 18px 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    p {{ margin: 6px 0; color: var(--muted); }}
    main {{ padding: 24px 36px 48px; }}
    .section {{
      margin-bottom: 22px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .pill {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }}
    .observation p {{ color: var(--ink); }}
    .bar-row {{
      display: grid;
      grid-template-columns: minmax(190px, 300px) 1fr 150px;
      align-items: center;
      gap: 12px;
      min-height: 30px;
      margin: 8px 0;
    }}
    .bar-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }}
    .bar-label span {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-label em {{
      flex: 0 0 auto;
      color: var(--accent-2);
      font-size: 12px;
      font-style: normal;
    }}
    .bar-track {{
      height: 14px;
      background: #edf0f3;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #14b8a6);
      border-radius: 999px;
    }}
    .bar-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .table-tools {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 12px;
    }}
    input[type="search"] {{
      width: min(420px, 100%);
      height: 36px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 960px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f1f4f7;
      color: #2f3b45;
      font-weight: 700;
    }}
    td:first-child, th:first-child,
    td:nth-child(2), th:nth-child(2),
    td:nth-child(3), th:nth-child(3),
    td:nth-child(4), th:nth-child(4),
    td:nth-child(5), th:nth-child(5) {{
      text-align: left;
    }}
    details {{
      margin: 12px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    summary {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 8px;
      cursor: pointer;
      padding: 12px 14px;
      background: var(--soft);
      font-weight: 700;
    }}
    summary small {{
      color: var(--muted);
      font-weight: 400;
    }}
    details > .table-wrap,
    details > h3,
    details > .split {{
      margin: 0 14px 14px;
    }}
    .split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .note {{
      padding: 12px 14px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      color: #7c2d12;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 6px; }}
      .bar-value {{ text-align: left; }}
      .split {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>分點成交量與常交易股票初報告</h1>
    <p>資料來源：`data/broker/by_broker/` 的 Fubon broker rank 分點歷史檔。</p>
    <div class="meta">
      <span class="pill">產生時間：{html_escape(generated_at)}</span>
      <span class="pill">分點數：{fmt_int(len(branch_rows))}</span>
      <span class="pill">外資推定分點：{fmt_int(len(foreign_rows))}</span>
      <span class="pill">每分點常交易股票：前 20 檔輸出至 CSV</span>
    </div>
  </header>
  <main>
    <div class="note">注意：這裡的成交量指標為排行榜資料列中的買進加賣出，保留 Fubon 原始單位，未換算為官方全市場分點成交量。</div>
    {observations}
    {top_all_chart}
    {top_foreign_chart}
    <section class="section">
      <h2>全部分點成交量摘要</h2>
      <div class="table-tools"><input type="search" data-filter-target="branch-table" placeholder="輸入分點名稱、外資關鍵字或日期篩選"></div>
      <div id="branch-table">{all_branch_table}</div>
    </section>
    <section class="section">
      <h2>明顯外資分點摘要</h2>
      {foreign_table}
    </section>
    <section class="section">
      <h2>全市場分點常交易股票預覽</h2>
      <p>此表只嵌入成交量指標最高的 500 組分點-股票；完整每個分點前 20 檔在 `output/broker/branch_top_stocks_by_volume.csv`。</p>
      <div class="table-tools"><input type="search" data-filter-target="stock-table" placeholder="輸入分點、股票代號、股票名稱或產業篩選"></div>
      <div id="stock-table">{top_stock_table}</div>
    </section>
    <section class="section">
      <h2>明顯外資分點細看</h2>
      <p>每個分點列出最常交易股票，以及長期偏買超、偏賣超股票。</p>
      {foreign_details}
    </section>
  </main>
  <script>
    for (const input of document.querySelectorAll("input[type='search'][data-filter-target]")) {{
      const target = document.getElementById(input.dataset.filterTarget);
      input.addEventListener("input", () => {{
        const keyword = input.value.trim().toLowerCase();
        for (const row of target.querySelectorAll("tbody tr")) {{
          row.style.display = row.innerText.toLowerCase().includes(keyword) ? "" : "none";
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    metadata = build_reports(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
