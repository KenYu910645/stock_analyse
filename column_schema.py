"""Column name translation helpers for CSV storage.

CSV files may use Chinese display headers on disk while Python code continues
to work with the existing canonical column names.
"""
from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


TRIVIAL_COLUMNS = {
    "Date",
    "date",
    "Code",
    "Name",
    "Year",
    "Quarter",
    "Time",
    "ISIN",
    "CFI",
    "C/P",
}


COLUMN_TRANSLATIONS = {
    "FetchedAt": "抓取時間",
    "SourcePath": "來源路徑",
    "SourceUrl": "來源網址",
    "Metric": "指標",
    "BrokerId": "券商代號",
    "BrokerName": "券商名稱",
    "BranchId": "分點代號",
    "BranchName": "分點名稱",
    "Side": "買賣別",
    "Rank": "排名",
    "Buy": "買進",
    "Sell": "賣出",
    "Net": "買賣超",
    "FetchDate": "抓取日期",
    "DataDate": "資料日期",
    "DuringDays": "統計天數",
    "OrderBy": "排序依據",
    "StockNo": "個股代號",
    "StockName": "個股名稱",
    "BuyQuantities": "買進數量",
    "SellQuantities": "賣出數量",
    "NetQuantities": "買賣超數量",
    "Amount": "金額",
    "AvgPrice": "平均價格",
    "Capacity": "成交股數",
    "Turnover": "成交金額",
    "Open": "開盤價",
    "High": "最高價",
    "Low": "最低價",
    "Close": "收盤價",
    "Change": "漲跌",
    "Transaction": "成交筆數",
    "TradeVolume": "成交股數(成交資訊)",
    "TradeValue": "成交金額(成交資訊)",
    "LastBestBidPrice": "最後揭示買價",
    "LastBestBidVolume": "最後揭示買量",
    "LastBestAskPrice": "最後揭示賣價",
    "LastBestAskVolume": "最後揭示賣量",
    "PERatio": "本益比(成交資訊)",
    "open_adj": "前復權開盤價",
    "close_adj": "前復權收盤價",
    "high_adj": "前復權最高價",
    "low_adj": "前復權最低價",
    "AdjFactor": "復權因子",
    "DividendYield": "殖利率",
    "DividendYear": "股利年度",
    "PEratio": "本益比",
    "PBratio": "股價淨值比",
    "FiscalYearQuarter": "財報年季",
    "ForeignBuyExDealer": "外陸資買進股數(不含外資自營商)",
    "ForeignSellExDealer": "外陸資賣出股數(不含外資自營商)",
    "ForeignNetExDealer": "外陸資買賣超股數(不含外資自營商)",
    "ForeignDealerBuy": "外資自營商買進股數",
    "ForeignDealerSell": "外資自營商賣出股數",
    "ForeignDealerNet": "外資自營商買賣超股數",
    "InvestmentTrustBuy": "投信買進股數",
    "InvestmentTrustSell": "投信賣出股數",
    "InvestmentTrustNet": "投信買賣超股數",
    "DealerNet": "自營商買賣超股數",
    "DealerSelfBuy": "自營商買進股數(自行買賣)",
    "DealerSelfSell": "自營商賣出股數(自行買賣)",
    "DealerSelfNet": "自營商買賣超股數(自行買賣)",
    "DealerHedgeBuy": "自營商買進股數(避險)",
    "DealerHedgeSell": "自營商賣出股數(避險)",
    "DealerHedgeNet": "自營商買賣超股數(避險)",
    "InstitutionalNet": "三大法人買賣超股數",
    "Statement": "報表",
    "Account": "科目",
    "Value": "數值",
    "Unit": "單位",
    "MarginPurchase": "融資買進",
    "MarginSale": "融資賣出",
    "MarginCashRepayment": "融資現金償還",
    "MarginPreviousBalance": "融資前日餘額",
    "MarginCurrentBalance": "融資今日餘額",
    "MarginNextDayLimit": "融資限額",
    "MarginFinancingUsageRate": "融資使用率",
    "MarginBalance20DayChangeRate": "融資餘額20日變化率",
    "MarginMarketValue": "融資市值",
    "MarginMarketValueTo20DayAvgTurnover": "融資市值20日均成交值比",
    "ShortPurchase": "融券買進",
    "ShortSale": "融券賣出",
    "ShortStockRepayment": "融券現券償還",
    "ShortPreviousBalance": "融券前日餘額",
    "ShortCurrentBalance": "融券今日餘額",
    "ShortNextDayLimit": "融券限額",
    "ShortMarginBalanceRatio": "券資比",
    "Offsetting": "資券相抵",
    "Note": "註記",
    "SuspensionNote": "暫停現股賣出後現款買進註記",
    "DayTradingVolume": "當日沖銷成交股數",
    "DayTradingBuyAmount": "當日沖銷買進成交金額",
    "DayTradingSellAmount": "當日沖銷賣出成交金額",
    "MarketVolumeRatio": "市場成交股數占比",
    "MarketBuyAmountRatio": "市場買進成交金額占比",
    "MarketSellAmountRatio": "市場賣出成交金額占比",
    "TotalVolume": "總成交股數",
    "TotalBuyAmount": "總買進成交金額",
    "TotalSellAmount": "總賣出成交金額",
    "DayTradingVolumeRatio": "當沖成交股數占比",
    "DayTradingBuyAmountRatio": "當沖買進成交金額占比",
    "DayTradingSellAmountRatio": "當沖賣出成交金額占比",
    "DayTradingTurnover": "當沖成交值",
    "DayTradingTurnoverRatio": "當沖成交值占比",
    "DayTradingAvgBuyPrice": "當沖平均買進價格",
    "DayTradingAvgSellPrice": "當沖平均賣出價格",
    "DayTradingAvgSpreadRate": "當沖平均價差率",
    "DayTradingAmountImbalanceRatio": "當沖買賣金額差率",
    "IntradayRangeRate": "日內振幅",
    "OpenCloseReturn": "開收報酬率",
    "DayTradingVolumeRatio20DayZScore": "當沖成交股數占比20日ZScore",
    "DayTradingTurnover20DayZScore": "當沖成交值20日ZScore",
    "ex_date": "除權息日期",
    "stock_id": "股票代號",
    "stock_name": "股票名稱",
    "previous_close": "除權息前收盤價",
    "ex_reference_price": "除權息參考價",
    "opening_reference_price": "開盤參考價",
    "opening_auction_base": "開盤競價基準",
    "limit_up_price": "漲停價格",
    "limit_down_price": "跌停價格",
    "cash_dividend": "現金股利",
    "dividend_value": "權值息值",
    "stock_dividend_rate": "股票股利率",
    "cash_capital_increase_price": "現金增資認購價",
    "cash_capital_increase_rate": "現金增資配股率",
    "right_or_dividend": "權息別",
    "deducted_dividend_reference_price": "減除股利參考價",
    "detail_key": "明細鍵",
    "AvailableVolume": "可借券賣出股數",
    "Exdividend": "除權息",
    "StockDividendRatio": "股票股利比率",
    "SubscriptionRatio": "認購比率",
    "SubscriptionPricePerShare": "每股認購價格",
    "CashDividend": "現金股利(預告)",
    "SharesOffered": "公開申購股數",
    "SharesEmpOwner": "員工認購股數",
    "SharesholderOwner": "原股東認購股數",
    "StockHoldingRatio": "持股比率",
    "Company": "公司",
    "Location": "地點",
    "Message": "訊息內容",
    "Download": "下載",
    "city": "城市",
    "station_id": "測站代號",
    "station_name": "測站名稱",
    "stn_type": "測站類型",
    "note": "備註",
    "DataTime": "觀測時間",
    "StationPressure.Instantaneous": "測站氣壓_瞬時",
    "SeaLevelPressure.Instantaneous": "海平面氣壓_瞬時",
    "AirTemperature.Instantaneous": "氣溫_瞬時",
    "DewPointTemperature.Instantaneous": "露點溫度_瞬時",
    "RelativeHumidity.Instantaneous": "相對濕度_瞬時",
    "WindSpeed.Mean": "風速_平均",
    "WindSpeed.TenMinutelyMaximum": "風速_十分鐘最大",
    "WindDirection.Mean": "風向_平均",
    "WindDirection.TenMinutelyMaximum": "風向_十分鐘最大",
    "PeakGust.Maximum": "最大陣風",
    "PeakGust.Direction": "最大陣風風向",
    "Precipitation.Accumulation": "降水量_累積",
    "PrecipitationDuration.Total": "降水時數_總計",
    "SunshineDuration.Total": "日照時數_總計",
    "GlobalSolarRadiation.Accumulation": "全天空日射量_累積",
    "Visibility.Instantaneous": "能見度_瞬時",
    "Visibility.AutoMean": "能見度_自動平均",
    "UVIndex.Accumulation": "紫外線指數_累積",
    "TotalCloudAmount.Instantaneous": "總雲量_瞬時",
    "TotalCloudAmount.SatRetrieved": "總雲量_衛星反演",
    "SoilTemperatureAt0cm.Instantaneous": "地溫0公分_瞬時",
    "SoilTemperatureAt5cm.Instantaneous": "地溫5公分_瞬時",
    "SoilTemperatureAt10cm.Instantaneous": "地溫10公分_瞬時",
    "SoilTemperatureAt20cm.Instantaneous": "地溫20公分_瞬時",
    "SoilTemperatureAt30cm.Instantaneous": "地溫30公分_瞬時",
    "SoilTemperatureAt50cm.Instantaneous": "地溫50公分_瞬時",
    "SoilTemperatureAt100cm.Instantaneous": "地溫100公分_瞬時",
    "market_turnover": "市場成交金額",
    "stock_count": "股票數量",
    "precip_09_14_mm": "09至14時降水量毫米",
    "precip_hour_count": "降水小時計數",
    "precip_valid_hours": "有效降水小時數",
    "complete_precip_hours": "完整降水小時數",
    "log_market_turnover": "市場成交金額對數",
    "is_rainy": "是否下雨",
    "is_heavy_rain_10mm": "是否大雨10毫米",
    "http": "HTTP狀態",
    "code": "狀態碼",
    "count": "筆數",
    "message": "狀態訊息",
    "Group": "產業群組",
    "Start": "起始日",
    "CompanyName": "公司全名",
    "Type": "類型",
    "Market": "市場",
    "Board": "板別",
    "has_price": "有價格資料",
    "has_adj_price": "有復權價格資料",
    "has_institutional": "有法人買賣資料",
    "has_margin": "有融資融券資料",
    "has_day_trading": "有當沖資料",
    "has_yield_pe_pb": "有殖利率本益比股淨比資料",
    "has_report": "有財報資料",
    "has_broker": "有分點資料",
    "available_dataset_count": "可用資料集數",
    "contract": "期貨契約",
    "Contract": "契約",
    "contract month(Week)": "期貨契約月份(週別)",
    "Contract Month(Week)": "契約月份(週別)",
    "Settlement Month": "交割月份",
    "open": "期貨開盤價",
    "high": "期貨最高價",
    "low": "期貨最低價",
    "last": "最後成交價",
    "%": "漲跌幅%",
    "volume": "期貨成交量",
    "settlement_price": "期貨結算價",
    "open_interest": "期貨未平倉量",
    "best_bid": "期貨最佳買價",
    "best_ask": "期貨最佳賣價",
    "historical_high": "期貨歷史最高價",
    "historical_low": "期貨歷史最低價",
    "Item": "項目",
    "Futures Trading Volume (Long)": "期貨交易口數_多方",
    "Options Trading Volume (Long)": "選擇權交易口數_多方",
    "Futures Trading Value (Long)(Thousands)": "期貨交易契約金額_多方_千元",
    "Options Trading Value (Long)(Thousands)": "選擇權交易契約金額_多方_千元",
    "Futures Trading Volume (Short)": "期貨交易口數_空方",
    "Options Trading Volume (Short)": "選擇權交易口數_空方",
    "Futures Trading Value (Short)(Thousands)": "期貨交易契約金額_空方_千元",
    "Options Trading Value (Short)(Thousands)": "選擇權交易契約金額_空方_千元",
    "Futures Trading Volume (Net)": "期貨交易口數_淨額",
    "Options Trading Volume (Net)": "選擇權交易口數_淨額",
    "Futures Trading Value (Net)(Thousands)": "期貨交易契約金額_淨額_千元",
    "Options Trading Value (Net)(Thousands)": "選擇權交易契約金額_淨額_千元",
    "Futures Open Interest (Long)": "期貨未平倉口數_多方",
    "Options Open Interest (Long)": "選擇權未平倉口數_多方",
    "Futures Contract Value of Open Interest (Long)(Thousands)": "期貨未平倉契約金額_多方_千元",
    "Options Contract Value of Open Interest (Long)(Thousands)": "選擇權未平倉契約金額_多方_千元",
    "Futures Open Interest (Short)": "期貨未平倉口數_空方",
    "Options Open Interest (Short)": "選擇權未平倉口數_空方",
    "Futures Contract Value of Open Interest (Short)(Thousands)": "期貨未平倉契約金額_空方_千元",
    "Options Contract Value of Open Interest (Short)(Thousands)": "選擇權未平倉契約金額_空方_千元",
    "Futures Open Interest (Net)": "期貨未平倉口數_淨額",
    "Options Open Interest (Net)": "選擇權未平倉口數_淨額",
    "Futures Contract Value of Open Interest (Net)(Thousands)": "期貨未平倉契約金額_淨額_千元",
    "Options Contract Value of Open Interest (Net)(Thousands)": "選擇權未平倉契約金額_淨額_千元",
    "Trading Volume (Long)": "交易口數_多方",
    "Trading Value (Long)(Millions)": "交易契約金額_多方_百萬元",
    "Trading Volume (Short)": "交易口數_空方",
    "Trading Value (Short)(Millions)": "交易契約金額_空方_百萬元",
    "Trading Volume (Net)": "交易口數_淨額",
    "Trading Value (Net)(Millions)": "交易契約金額_淨額_百萬元",
    "Open Interest (Long)": "未平倉口數_多方",
    "Contract Value of Open Interest (Long)(Millions)": "未平倉契約金額_多方_百萬元",
    "Open Interest (Short)": "未平倉口數_空方",
    "Contract Value of Open Interest (Short)(Millions)": "未平倉契約金額_空方_百萬元",
    "Open Interest (Net)": "未平倉口數_淨額",
    "Contract Value of Open Interest (Net)(Millions)": "未平倉契約金額_淨額_百萬元",
    "Type of Traders": "交易人類別",
    "Top 5_Buy": "前五大買方",
    "Top 5_Sell": "前五大賣方",
    "Top 10_Buy": "前十大買方",
    "Top 10_Sell": "前十大賣方",
    "OI of Market": "全市場未平倉量",
    "Strike Price": "履約價",
    "Call/Put": "買賣權",
    "Volume": "成交量",
    "Settlement Price": "結算價",
    "OI": "選擇權未平倉量",
    "Best Bid": "最佳買價",
    "Best Ask": "最佳賣價",
    "Historical High": "歷史最高價",
    "Historical Low": "歷史最低價",
    "Put Volume": "賣權成交量",
    "Call Volume": "買權成交量",
    "Total Volume": "總成交量",
    "Put/Call Volume Ratio%": "賣買權成交量比率%",
    "Put OI": "賣權未平倉量",
    "Call OI": "買權未平倉量",
    "Put/Call OI Ratio%": "賣買權未平倉比率%",
}


CHINESE_TO_CANONICAL = {value: key for key, value in COLUMN_TRANSLATIONS.items()}

DATE_COLUMNS = {
    "Date",
    "date",
    "ex_date",
    "DataDate",
    "FetchDate",
    "FetchedForDate",
    "資料日期",
    "交易日期",
    "抓取日期",
    "除權息日期",
    "出表日期",
    "起始日",
}


def storage_name(column: str) -> str:
    return COLUMN_TRANSLATIONS.get(column, column)


def canonical_name(column: str) -> str:
    return CHINESE_TO_CANONICAL.get(column, column)


def storage_columns(columns: Iterable[str]) -> list[str]:
    return [storage_name(column) for column in columns]


def canonical_columns(columns: Iterable[str]) -> list[str]:
    return [canonical_name(column) for column in columns]


def to_storage_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_TRANSLATIONS)


def to_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=CHINESE_TO_CANONICAL)


def normalize_date_text(value):
    if pd.isna(value):
        return value
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()

    text = str(value).strip()
    if not text:
        return text
    if text.endswith(".0"):
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            if number.is_integer():
                text = str(int(number))

    text = text.replace("/", "-")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    if re.match(r"^\d{4}-\d{2}$", text):
        return text

    match = re.match(r"^(\d{2,3})-(\d{1,2})-(\d{1,2})$", text)
    if match:
        year, month, day = match.groups()
        try:
            return date(int(year) + 1911, int(month), int(day)).isoformat()
        except ValueError:
            return text

    if re.match(r"^\d{8}$", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date().isoformat()
        except ValueError:
            return text
    if re.match(r"^\d{7}$", text):
        try:
            return date(int(text[:3]) + 1911, int(text[3:5]), int(text[5:7])).isoformat()
        except ValueError:
            return text
    if re.match(r"^\d{6}$", text):
        try:
            return datetime.strptime(text, "%Y%m").date().strftime("%Y-%m")
        except ValueError:
            return text
    if re.match(r"^\d{5}$", text):
        try:
            return date(int(text[:3]) + 1911, int(text[3:5]), 1).strftime("%Y-%m")
        except ValueError:
            return text
    return text


def normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        canonical = canonical_name(column)
        if canonical in DATE_COLUMNS or column in DATE_COLUMNS:
            result[column] = result[column].map(normalize_date_text)
    return result


def _csv_header(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file_obj:
        return next(csv.reader(file_obj), [])


def _map_read_columns(path: str | Path, columns):
    if columns is None or callable(columns):
        return columns
    header = set(_csv_header(path))

    def map_one(column):
        if column in header:
            return column
        translated = storage_name(column)
        if translated in header:
            return translated
        return column

    return [map_one(column) for column in columns]


def _map_read_dict_keys(path: str | Path, values: dict):
    header = set(_csv_header(path))
    mapped = {}
    for column, value in values.items():
        if column in header:
            mapped[column] = value
            continue
        translated = storage_name(column)
        mapped[translated if translated in header else column] = value
    return mapped


def read_csv_canonical(path: str | Path, *args, **kwargs) -> pd.DataFrame:
    if "usecols" in kwargs:
        kwargs = dict(kwargs)
        kwargs["usecols"] = _map_read_columns(path, kwargs["usecols"])
    if "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
        kwargs = dict(kwargs)
        kwargs["dtype"] = _map_read_dict_keys(path, kwargs["dtype"])
    if "parse_dates" in kwargs and isinstance(kwargs["parse_dates"], list):
        kwargs = dict(kwargs)
        kwargs["parse_dates"] = _map_read_columns(path, kwargs["parse_dates"])
    result = pd.read_csv(path, *args, **kwargs)
    if isinstance(result, pd.DataFrame):
        return to_canonical_columns(result)
    return (to_canonical_columns(chunk) for chunk in result)


def to_csv_storage(df: pd.DataFrame, path: str | Path, *args, **kwargs) -> None:
    to_storage_columns(normalize_date_columns(df)).to_csv(path, *args, **kwargs)


def csv_columns_canonical(path: str | Path, fallback_columns=None) -> list[str]:
    if not Path(path).exists():
        return list(fallback_columns or [])
    return canonical_columns(_csv_header(path))


def storage_fieldnames(fieldnames: Iterable[str]) -> list[str]:
    return storage_columns(fieldnames)


def storage_record(record: dict) -> dict:
    normalized = {}
    for column, value in record.items():
        if column in DATE_COLUMNS or storage_name(column) in DATE_COLUMNS:
            value = normalize_date_text(value)
        normalized[storage_name(column)] = value
    return normalized
