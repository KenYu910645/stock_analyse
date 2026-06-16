# Project Instructions

This AGENTS.md need to be up-to-date for EVERY change you made to the codebase
You MUST read this AGENTS.md before doing ANY task.

## Scope

- This project is for Taiwan's stock & ETF & index analysis.
- For price CSV downloads and analysis, the default universe is TWSE 上市 common stocks only.
- Treat 上櫃/TPEX/OTC stocks, warrants, ETNs, TDRs, other than explicit benchmarks, and other non-common-stock instruments as out of scope unless the user explicitly asks for them.
- Use `data/metadata.csv` as the stock catalog when filtering the universe. Preserve stock codes as strings so leading zeros are not lost.
- Use Gregorian calendar dates in generated data and metadata. Do not output ROC/Minguo-year dates.
- The Fubon Neo realtime market-data collector lives in `inDayTradeBook/`; keep that feature's setup, credentials, database code, and tests scoped there.

## Project Tree

```text
stock_analyse/
  AGENTS.md                                      AI-facing project rules and repository map.
  .gitignore                                     Git ignore rules for local/generated files and local SDK artifacts.
  config.py                                      Shared legacy config values for stock lists, download throttles, and model settings.
  backtesting.py                                 General strategy backtesting script.
  stock_analyse.py                               Main listed-stock screening/analysis script using price CSVs and metadata.
  stock_price_updater.py                         Updates the local stock holding workbook with latest prices.
  stock_viz.py                                   Creates interactive stock price visualizations.
  stock_correlation_analysis.py                  Builds price/return correlation reports from adjusted prices.
  stock_regime_analysis.py                       Generates adjusted-price regime trajectory HTML reports.
  cointegration_pair_analysis.py                 Finds candidate cointegrated stock pairs.
  pair_trading_backtest.py                       Backtests pair-trading candidates against TAIEX.
  visualize_pair_trading_candidates.py           Creates static pair-trading candidate visualizations.
  visualize_pair_trading_candidates_interactive.py Creates interactive pair-trading candidate visualizations.
  splitStockBasic.py                             Splits TWSE valuation history into per-stock CSVs.
  train.py                                       Legacy neural-network training entry point.
  fcn.py                                         Legacy model/network helper code.
  chatGPT_recommend.ins                          Historical prompt/instruction notes.
  Stock_history.xlsx                             Local portfolio workbook used by `stock_price_updater.py`.
  result.* / statistic.*                         Generated legacy analysis outputs; avoid writing new root outputs here.
  misc.xlsx                                      Local spreadsheet artifact; inspect before relying on it.

  data/                                          Local CSV datasets and metadata.
    metadata.csv                                 Stock, ETF, and index catalog used for universe filters.
    trading_days.csv                             Canonical TWSE trading dates from official online FMTQIK history.
    price/                                       Unadjusted OHLCV per-stock price CSVs.
    adj_price/                                   Adjusted OHLCV per-stock price CSVs for analysis.
    single_day/                                  Daily combined snapshot outputs.
    institutional/                               Institutional investor flow data.
    margin/                                      Margin trading balance data.
    shareholding/                                TDCC shareholding distribution data.
    dividend/                                    Ex-right/dividend event data.
    dividend_pe_pb/                              Daily valuation/dividend yield/PER/PBR data.
    day_trading/                                 Day-trading statistics.
    broker/                                      Broker branch buy/sell rank data.
    report/                                      Financial statement/report downloads.
    revenue/ financial/ company/ events/         TWSE/MOPS/OpenAPI company and event datasets.
    codis/ insiders/ investor_conference/ sbl/ shareholder_meeting/
                                                   Other source-specific datasets.

  downloader/                                    Batch and daily data downloaders.
    trading_days.py                              Downloads official TWSE trading dates to `data/trading_days.csv`.
    price.py                                     Downloads TWSE listed stock/ETF price CSVs from metadata start dates and TAIEX benchmark data.
    download_single_day_all_data.py              Daily updater for price, institutional, margin, shareholding, dividend, valuation, and snapshot datasets.
    download_valuation_day_trading_history.py    Backfills TWSE valuation and day-trading history.
    dividend_pe_pb.py                            Downloads daily valuation/dividend-yield/PER/PBR history.
    ex_right_dividend.py                         Downloads and normalizes TWSE ex-right/dividend data.
    institutional_investors.py                   Downloads institutional investor trading data.
    margin_trading.py                            Downloads TWSE margin trading balances.
    tdcc_shareholding.py                         Downloads TDCC shareholding distribution open data.
    tdcc_shareholding_split_visualize.py         Splits and charts TDCC shareholding data by stock.
    report.py                                    Downloads MOPS financial reports.
    split_history_by_stock.py                    Splits long-form history files into per-stock files.
    fubon_broker_rank.py                         Downloads Fubon eBroker DJ broker branch rank pages.
    histock_broker_daily.py                      Downloads HiStock broker daily data.
    wantgoo_broker_rank.py                       Downloads WantGoo broker rank data.
    fubon_backfill_until_stop.py                 Runs repeated Fubon broker backfills until no more data.
    codis_weather.py                             Downloads CODiS weather data for external context.

  alpha_model/                                   Factor research pipeline package.
    main.py                                      CLI entry point for alpha-factor evaluation.
    config.py                                    Default and YAML-loaded alpha-model configuration.
    data.py                                      Loads and normalizes price data and stock universe.
    preprocessing.py                             Prepares factor values for evaluation.
    labels.py                                    Computes future-return labels.
    reporting.py                                 Writes alpha-model tables and plots.
    config/momentum.yaml                         Momentum factor pipeline configuration.
    factors/momentum.py                          Momentum factor implementations.
    metrics/                                     IC, quantile, stability, turnover, and performance metrics.

  strategies/                                    Backtest strategy implementations.
    trade_cost.py                                Taiwan stock transaction fee and tax cost helper.
    buy_and_hold.py                              Buy-and-hold strategy.
    macd.py                                      MACD strategy.
    naive.py                                     Simple baseline strategy.
    optimal.py                                   Lookback/benchmark strategy helper.
    pair_trading.py                              Pair-trading strategy logic.

  inDayTradeBook/                                Fubon Neo realtime collector project.
    README.md                                    Setup and usage guide for realtime collection.
    docker-compose.yml                           TimescaleDB service definition.
    fubon.py                                     Small Fubon SDK entry/test script.
    monitor_list.txt                             Realtime subscription stock list.
    requirements.txt                             Realtime collector Python dependencies.
    realtime/                                    Config, parser, logger, DB writer, and collector runtime.
    scripts/                                     DB init, task scheduler, market-session, query, and API trial scripts.
    sql/                                         TimescaleDB table, hypertable, index, and compression migrations.
    tests/                                       Realtime parser, collector, and monitor-list tests.
    vendor/                                      Local ignored Fubon Neo SDK wheel/archive; do not modify casually.

  tests/                                         Root test suite for analysis/modeling modules.
  tools/                                         Ad-hoc helper tools; inspect before reuse.
  output/                                        Generated reports, charts, and matrices.
  logs/                                          Runtime and downloader logs.
  runs/                                          Experiment run artifacts.
```


## Data Layout

- Keep project data under `data/`, outputs under `output/`, logs under `logs/`, and experiment runs under `runs/`.
- Use `data/trading_days.csv` as the canonical trading-day calendar instead of guessing weekdays or holidays.
- `data/trading_days.csv` is sourced from TWSE FMTQIK online history, which starts at `1990-01-04`.
- Price CSV files live in `data/price/`. The current raw price schema is:
  `Date,Capacity,Turnover,Open,High,Low,Close,Change,Transaction`.
- Adjusted-price CSV files live in `data/adj_price/` and are the preferred input for correlation, regime, pair-trading, and alpha-model analysis when adjusted returns matter.
- Prefer per-stock time-series CSVs for cleaned analysis datasets. The existing convention is `STOCK_YYYYMM_to_YYYYMM.csv`, for example `2330_202005_to_202605.csv`.
- Daily all-market snapshots are acceptable for raw or source-style data, but do not make them the main analysis format unless the downstream script already expects that layout.
- When adding new daily-updated datasets, make updates idempotent: key by date plus stock code or source-specific unique keys, replace or skip existing keys, and avoid duplicate rows.
- Write CSVs as UTF-8 with BOM (`encoding="utf-8-sig"`) when the file may contain Chinese text or is meant to open cleanly in Excel.
- Normalize output dates to Gregorian `YYYY-MM-DD` when a full date is available. If the source only provides year-month, use Gregorian `YYYY-MM`.
- Do not commit secrets, credentials, certificate passwords, local `.env` files, or broker login material. `inDayTradeBook/.env.example` is the template; real credentials stay local.

## Downloaders
- The downloaders script name must identical to the dataset name. for example: price.py will output to data/price/.
- Put new download scripts in `downloader/` unless they belong specifically to `inDayTradeBook/`.
- Reuse existing downloader patterns: `argparse` CLIs, `PROJECT_ROOT`, explicit output paths, polite throttling, retry/backoff, clear status prints, and failure logs.
- TWSE API dates should be normalized to ISO `YYYY-MM-DD` in cleaned outputs. Convert ROC dates explicitly.
- TWSE `MI_INDEX` full-market security history currently starts at `2004-02-11`; older metadata `Start` dates can only be backfilled when another source supports them.
- Filter downloaded rows to listed common stocks by default. Add an explicit opt-in flag if a downloader can include OTC, ETFs, or other instruments.
- Keep source schemas explicit with column constants. Avoid silent column drift; validate required columns before writing or analyzing.
- For large historical downloads, support resume/idempotency through completed-date tracking, existing-key checks, or cached-file reuse.

## Analysis And Modeling

- Prefer adjusted prices from `data/adj_price/` for return, correlation, regime, pair-trading, and alpha-factor work.
- Keep unadjusted prices from `data/price/` for raw OHLCV inspection, daily update ingestion, and compatibility with existing scripts.
- Avoid lookahead bias: factors must use only information available before the label/return horizon.
- Preserve the existing normalized analysis columns where used: `date`, `stock_id`, `open`, `high`, `low`, `close`, `volume`, `turnover`, `change`, `transactions`.
- Put generated reports, charts, manifests, matrices, and summaries in `output/` rather than the repo root.
- For reusable analysis logic, prefer package modules such as `alpha_model/` or `strategies/` over one-off root scripts when the code will be called by tests or other scripts.

## Realtime Collector

- `inDayTradeBook/` is a separate Fubon Neo realtime collector for TimescaleDB.
- Keep realtime DB schema changes in `inDayTradeBook/sql/` migrations.
- Keep realtime parser, config, DB writer, and collector logic in `inDayTradeBook/realtime/`.
- Use `monitor_list.txt` for realtime subscription symbols; one leading 4-digit code per line is parsed, with comments/descriptions allowed after it.
- Respect Fubon subscription limits when modifying monitor-list logic. The current workflow assumes channels like `trades` and `books`.
- Do not enable Timescale compression by default while schema/debugging behavior is still changing; leave it opt-in.

## Code Style

- Python is the primary language. Follow existing style in the touched file rather than introducing broad formatting churn.
- Prefer `pathlib.Path` in new code, but work with existing `os.path` code locally when that file already uses it.
- Keep stock IDs as strings in pandas reads and joins: use `dtype={"Code": str}`, `dtype={"stock_id": str}`, or equivalent.
- Use structured parsing for JSON/CSV/HTML tables where practical. Avoid brittle string slicing when a parser is already available.
- Keep comments short and useful, especially around TWSE/Fubon quirks, date conversion, schema assumptions, and anti-lookahead logic.
- Do not rewrite unrelated generated data files as part of code changes.

## Testing And Verification

- For root analysis/modeling changes, run the relevant tests under `tests/`, or at least the targeted test file.
- For realtime collector changes, run `python -m pytest inDayTradeBook/tests` from the repo root.
- For downloader changes that hit the network, prefer unit-level parser/schema tests when possible. If a live request is necessary, keep it narrow and mention it.
- Before finishing changes that affect CSV outputs, verify headers, date parsing, stock-code dtype, duplicate-key behavior, and output path.
- If tests cannot be run because of missing dependencies, credentials, network access, or market availability, report that clearly.
