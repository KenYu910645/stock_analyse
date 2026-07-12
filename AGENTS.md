# Project Instructions

This AGENTS.md need to be up-to-date for EVERY change you made to the codebase
You MUST read this AGENTS.md before doing ANY task.

## Scope

- This project is for Taiwan's stock & ETF & index analysis.
- For price CSV downloads and analysis, the default universe is TWSE 上市 common stocks only.
- Treat 上櫃/TPEX/OTC stocks, warrants, ETNs, TDRs, other than explicit benchmarks, and other non-common-stock instruments as out of scope unless the user explicitly asks for them.
- Use `data/metadata.csv` as the stock catalog when filtering the universe. Preserve stock codes as strings so leading zeros are not lost.
- For Chinese company-name references, use `公司簡稱`; do not use the full legal `公司名稱` unless the user explicitly asks for it.
- Use Gregorian calendar dates in generated data and metadata. Do not output ROC/Minguo-year dates.

## Project Tree

```text
stock_analyse/
  AGENTS.md                                      AI-facing project rules and repository map.
  .gitattributes                                 Cross-platform text/binary and line-ending rules.
  .gitignore                                     Git ignore rules for local/generated files and local SDK artifacts.
  pyproject.toml                                 Pytest and Ruff configuration for reproducible validation.
  requirements.txt                              Core Python runtime dependencies.
  requirements-{broker,browser,dev}.txt          Optional OCR, browser-automation, and development dependency stacks.
  column_schema.py                               CSV column translation helpers for Chinese storage headers and canonical internal names.
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
  chatGPT_recommend.ins                          Historical prompt/instruction notes.
  Stock_history.xlsx                             Local portfolio workbook used by `stock_price_updater.py`.
  result.* / statistic.*                         Generated legacy analysis outputs; avoid writing new root outputs here.
  misc.xlsx                                      Local spreadsheet artifact; inspect before relying on it.

  data/                                          Local CSV datasets and metadata.
    metadata.csv                                 Stock, ETF, and index catalog used for universe filters.
    trading_days.csv                             Canonical TWSE trading dates from official online FMTQIK history.
    price/                                       Per-stock price CSVs with raw OHLCV plus forward-adjusted OHLC and `AdjFactor`.
    institutional/                               Institutional investor flow data as per-stock CSVs.
    margin/                                      Margin trading balance data as per-stock CSVs, including derived leverage indicators joined from price data.
    shareholding/                                TDCC shareholding distribution data as per-stock CSVs.
    dividend/                                    Ex-right/dividend event data; source-of-truth files live in per-dataset stock CSV folders.
    events/                                      MOPS historical material events as per-stock CSVs, with event detail keys for resumable enrichment.
    yield_pe_pb/                                 Daily valuation yield/PER/PBR data as per-stock CSVs.
    day_trading/                                 Day-trading statistics; canonical data is flat per-stock CSVs plus `day_trading.logs`.
    broker/                                      Broker branch buy/sell rank data.
      twse/                                     TWSE BSR broker trading data.
        by_stock/                              Raw per-stock TWSE BSR CSVs from `downloader/broker.py`; filenames start with stock code for metadata coverage checks.
        by_broker/                             TWSE BSR rows regrouped by normalized broker name as one CSV per broker.
        by_date/YYYY-MM-DD/                    Date-scoped TWSE BSR batch folders containing that date's raw `by_stock/` files and converted `by_broker/` files.
      by_broker/                                Fubon broker rank history regrouped by branch name as `分點名稱.csv`.
    report/                                      Financial statement/report downloads as one CSV per stock.
    revenue/ financial/ company/ events/         TWSE/MOPS/OpenAPI company and event datasets; `company/` stores per-stock CSVs named with `公司簡稱`.
    codis_weather/ insiders/ investor_conference/ sbl/ shareholder_meeting/
                                                   Other source-specific datasets.
    taifex/                                      TAIFEX futures/options daily, put/call ratio, institutional, and large-trader OI CSVs.

  downloader/                                    Batch and daily data downloaders.
    metadata.py                                  Builds the TWSE catalog and refreshes local dataset-availability flags; `--availability-only` is offline.
    trading_days.py                              Downloads official TWSE trading dates to `data/trading_days.csv`; default CLI refreshes incrementally from the existing calendar.
    price.py                                     Downloads TWSE listed stock/ETF price CSVs from metadata start dates and TAIEX benchmark data; owns its runtime throttle/retry defaults.
    update_all_data.py                           Thin CLI/import surface for the registry-driven daily updater covering canonical per-stock datasets, latest report periods, adjusted price columns, and TWSE BSR broker branch data.
    daily/                                      Daily-updater orchestration helpers.
      context.py                               Shared immutable listed-universe and resolved-query-date context.
      registry.py                              TaskSpec runner that isolates independent updater task failures and keeps later tasks running.
      status.py                                Structured status collector that preserves the legacy stdout status-line format.
      tasks/                                   Dataset-specific daily updater implementations imported by `update_all_data.py`.
        _runtime.py                           Explicit lazy compatibility proxy used by tasks; no task-global namespace injection.
        price.py                              Daily TWSE price fetch, per-stock append, and adjusted-price refresh.
        institutional.py                      Daily institutional investor flow update.
        margin.py                             Daily margin update plus margin feature refresh.
        shareholding.py                       TDCC shareholding snapshot update.
        dividend.py                           Ex-right/dividend update.
        day_trading.py                        Day-trading update plus derived feature refresh.
        valuation.py                          Yield/PER/PBR valuation update and close-price fill.
        openapi_snapshots.py                  TWSE OpenAPI snapshot routing for company, financial, revenue, events, insiders, SBL, and shareholder-meeting datasets.
        investor_conference.py                Investor conference page parser and per-stock writer.
        reports.py                            MOPS financial-report latest-period updater.
        broker_twse.py                        TWSE BSR broker raw download, date-batch sync, and by-broker rebuild.
    download_valuation_day_trading_history.py    Backfills TWSE valuation and day-trading history.
    yield_pe_pb.py                               Downloads daily valuation yield/PER/PBR history directly to per-stock CSVs.
    ex_right_dividend.py                         Downloads and normalizes TWSE ex-right/dividend data.
    institutional_investors.py                   Downloads institutional investor trading data.
    margin_trading.py                            Downloads TWSE margin trading balances.
    events.py                                    Downloads MOPS historical material events to flat per-stock CSVs under `data/events/`.
    tdcc_shareholding.py                         Downloads TDCC shareholding distribution open data.
    report.py                                    Downloads MOPS financial reports.
    fubon_broker_rank.py                         Downloads Fubon eBroker DJ broker branch rank pages.
    broker.py                                    Downloads captcha-protected TWSE BSR per-stock broker trading CSVs to `data/broker/twse/by_stock/` and updates metadata broker availability.
    histock_broker_daily.py                      Downloads HiStock broker daily data.
    wantgoo_broker_rank.py                       Downloads WantGoo broker rank data.
    fubon_backfill_until_stop.py                 Runs repeated Fubon broker backfills until no more data.
    codis_weather.py                             Downloads CODiS weather data for external context.
    taifex.py                                    Downloads TAIFEX futures/options daily, put/call ratio, institutional, and large-trader OI data.

  alpha_model/                                   Factor research pipeline package.
    main.py                                      CLI entry point for alpha-factor evaluation.
    config.py                                    Default and YAML-loaded alpha-model configuration.
    data.py                                      Loads and normalizes price data and stock universe.
    preprocessing.py                             Prepares factor values for evaluation.
    labels.py                                    Computes future-return labels.
    reporting.py                                 Writes alpha-model tables and plots.
    config/momentum.yaml                         Momentum factor pipeline configuration.
    config/trust_flow.yaml                       Investment-trust institutional-flow factor IC pipeline configuration using next-open labels.
    factors/institutional_flow.py                Investment-trust flow factor implementations from `data/institutional/` joined to price turnover.
    factors/momentum.py                          Momentum factor implementations.
    metrics/                                     IC, quantile, stability, turnover, and performance metrics.

  strategies/                                    Backtest strategy implementations.
    trade_cost.py                                Taiwan stock transaction fee and tax cost helper.
    buy_and_hold.py                              Buy-and-hold strategy.
    macd.py                                      MACD strategy.
    naive.py                                     Simple baseline strategy.
    optimal.py                                   Lookback/benchmark strategy helper.
    pair_trading.py                              Pair-trading strategy logic.

  tests/                                         Root test suite for analysis/modeling modules.
  viz/                                           Dataset visualization scripts that mirror selected `data/` folders into `data_viz/`, including WebGL price HTML.
  tools/                                         Tracked ad-hoc and research CLIs; generated outputs remain ignored.
    analyze_2330_margin_patterns.py             Explores 2330 price and margin history with hypothesis tests and an HTML pattern report.
    run_day_trading_spike_study.py              Tests whether day-trading spikes coincide with sharp same-day price moves and studies forward returns.
    rebuild_day_trading_by_stock_full.py        Rebuilds canonical flat per-stock day-trading CSVs and `day_trading.logs`.
    repair_institutional_gaps.py                Finds date-level institutional-investor coverage anomalies, verifies missing rows against TWSE T86, and repairs per-stock CSVs.
    build_institutional_participation_report.py Computes all-stock historical participant volume share, net-flow ratio, and direction-purity reports from price and institutional CSVs, writing `data_viz/institutional_participation/` WebGL K-line pages with stacked foreign/trust/dealer/other volume bars.
    build_foreign_participation_distribution_report.py Analyzes latest foreign-investor participation concentration and writes `data_viz/institutional_participation/foreign_distribution.html` plus CSV summaries, including the ranking needed to reach 80% cumulative foreign volume and industry/category summaries joined from `data/metadata.csv`.
    build_participant_distribution_reports.py   Builds matching concentration reports for 投信, 自營商, and other/non-institutional participation using the institutional-participation summaries.
    build_historical_participant_distribution_reports.py Builds full-history concentration reports for 外資/投信/自營商/其他 using only overlapping price and institutional dates, overwriting the participant distribution HTML/CSV reports with full-history versions.
    build_participant_decision_pnl_report.py    Estimates future adjusted-price payoff for 外資/投信/自營商/其他 daily net-buy decisions and writes `participant_decision_pnl.html` plus CSV summaries.
    backtest_institutional_flow_strategy.py     Backtests recent five-year long-only institutional-flow signal baskets for 外資/投信/自營商/反做其他 using 1/3/5-day signals and 1/5/10/20/30/60-day fixed holding periods, with summary grouped bar charts comparing participants under identical parameters plus 投信 dynamic-exit and configurable consecutive-net-buy quota stop-loss reports.
    backtest_trust_top50_cumulative_sell_stop_strategy.py Selects the full-history best 50 stocks for 投信 1-day buy signals using 60-day payoff, then reruns the 投信 net-buy/sell quota stop-loss strategy only on that universe with an interactive canvas trade-timeline report.
    build_latest_institutional_flow_report.py   Builds the latest-date all-listed-stock 三大法人淨流向 Sankey-style HTML report under `data_viz/institutional_flow/`, with supporting CSV summaries under `output/institutional_flow/`.
    build_stock_chip_inventory_report.py        Builds per-stock and all-listed-stock rolling-window flow-implied chip inventory reports for 外資/投信/自營商/其他, using full-history warmup, minimum initial inventory inference, FIFO/FILO/average-cost distributions with dynamic price-spread bins, current-price cost markers, market-level method-comparison histograms, and FIFO-only market rankings.
    build_yield_pe_pb_research_report.py       Builds yield/PER/PBR valuation research reports, including next-open forward-return IC, quantile-return summaries, latest factor snapshots, and per-stock price WebGL pages with valuation overlays plus price/valuation percentile butterfly charts under `data_viz/yield_pe_pb_research/`.
    repair_yield_pe_pb_missing_dates.py        Repairs `data/yield_pe_pb/` whole-market missing dates that TWSE `BWIBBU_d` officially has, using the quality-audit CSV and merging fixed rows back into canonical per-stock CSVs.
    build_broker_branch_report.py              Builds Fubon `data/broker/by_broker/` branch volume, foreign-branch heuristic, frequent-stock CSV summaries, and an initial HTML report under `data_viz/broker/`.
    build_broker_volume_share_report.py        Estimates Fubon broker-branch volume share by matching `data/broker/by_broker/` buy/sell lots to listed-common-stock `data/price/` official volume, writing branch/date/stock-day coverage CSVs under `output/broker/` and an HTML report under `data_viz/broker/`.
    build_foreign_broker_performance_report.py Builds Fubon foreign-branch net-buy/net-sell decision-payoff CSV summaries and an HTML report under `data_viz/broker/`, joining `data/broker/by_broker/` to adjusted `data/price/` rows.
    build_all_broker_performance_report.py     Builds Fubon all-branch net-buy/net-sell decision-performance CSV summaries plus branch-category and Taiwan city/county HTML reports under `data_viz/broker/`, filtering out extreme low-event branches by default, merging small heuristic categories into `其他分點`, adding cached same-day churn metrics from `broker_trading_style_churn_metrics.csv` when available, showing 1/5/20-day Top 50 branch ranking tables, and using current TWSE broker-branch addresses plus branch-name location fallbacks for city grouping while excluding foreign, headquarters/main-code, inactive, and still-unmapped branches.
    backtest_top_broker_following_strategy.py   Selects the Fubon all-branch 20D performance Top 10 branches, validates each branch independently with IC/RankIC, and builds independent next-open follow-the-branch long/short strategy reports under `data_viz/broker/` with supporting CSVs under `output/broker/`.
    build_broker_vs_institutional_strategy_comparison.py Compares all valid Fubon broker-branch long-only buy-following results against three-major-institution institutional-flow backtest metrics after reusing the all-broker small-branch filter, excluding stopped/old branches, head-office/main-code branches, and low-activity branches by default, recomputing broker basket-style metrics with next-open fixed holding periods and writing ranked 1/5/10/20/60-day comparison CSV/HTML artifacts plus excluded-branch details under `output/broker/` and `data_viz/broker/`.
    build_broker_trading_style_report.py       Classifies Fubon broker branches into `當沖傾向`, `隔日衝`, `波段分點`, and `長線傾向` using existing all-branch decision-performance outputs plus raw same-day buy/sell churn metrics, writing CSV summaries under `output/broker/` and an HTML report under `data_viz/broker/`.
    build_twse_broker_by_broker.py             Moves TWSE BSR root raw CSVs into `data/broker/twse/by_stock/` when requested, parses CP950 BSR rows, and writes one UTF-8-BOM CSV per broker under `data/broker/twse/by_broker/`.
    backfill_margin_history.py                  Backfills TWSE margin history into canonical per-stock margin CSVs without data/raw caches.
    repair_price_health.py                      Scans suspicious price rows, verifies official TWSE rows, repairs confirmed mismatches, and recomputes adjusted prices.
    run_events_detail_enrichment_batches.py     Runs repeated resumable MOPS event detail-enrichment batches, optional until-complete terminal runs, status-only progress checks, and batch-level progress logs.
    run_margin_event_study.py                   Runs cross-sectional margin-financing event studies and writes tables plus HTML charts.
    run_margin_plateau_study.py                 Tests whether margin-financing surges are followed by flat 20-day average adjusted-price action.
    run_day_trading_heat_rank.py                Ranks listed stocks by daily day-trading heat, writes daily Top 20 CSVs, and builds an HTML heat-rank report.
    run_day_trading_return_rank.py              Ranks daily positive and negative aggregate day-trading spread returns, writes daily Top 20 CSVs, and builds an HTML return-rank report.
    split_csvs_by_stock.py                       Splits stock-keyed CSV datasets into per-stock CSVs using metadata filters.
    validate_2330_margin_regimes.py             Validates 2330 margin surge/high/low regimes with daily forward metrics, contiguous interval summaries, and an HTML report.
    validate_all_stock_margin_regimes.py        Generates per-stock margin-regime validation HTML reports and cross-stock CSV summaries using stock-specific thresholds.
    validate_market_margin_regimes.py           Aggregates listed-common-stock margin balances and validates market-level margin regimes against all-stock average or TAIEX price modes.
    rank_current_margin_states.py               Ranks stocks by target-date margin regime states using per-stock thresholds and writes current-state CSV/HTML reports.
    backtest_margin_contrarian_strategy.py      Backtests margin contrarian long/short variants across listed common stocks with optional 0050 buy-and-hold benchmark reporting.
    visualize_2330_margin_extremes.py           Builds an interactive WebGL/canvas 2330 price and margin overlay marking margin surge/drop and high/low margin-balance zones.
  data_viz/                                      Generated visualization artifacts, including HTML reports, PNG charts, and chart indexes.
  output/                                        Generated non-visual analysis tables, model outputs, and intermediate reports.
  logs/                                          Runtime and downloader logs.
  runs/                                          Experiment run artifacts.
```


## Data Layout

- Keep project data under `data/`, visualization artifacts under `data_viz/`, non-visual analysis outputs under `output/`, logs under `logs/`, and experiment runs under `runs/`.
- Put generated HTML charts, PNG charts, visualization indexes, and visualization report bundles in `data_viz/`. Keep source datasets and downloader outputs out of `data_viz/`.
- Broker branch visual reports live under `data_viz/broker/`, with supporting CSV summaries under `output/broker/`. Treat Fubon broker-rank branch `成交量` as rank-list gross volume (`買進 + 賣出` from ranked rows), not complete full-market branch volume. Treat broker decision-performance reports as decision payoff from ranked net buy/sell signals joined to adjusted prices, not true realized inventory P&L.
- TWSE BSR raw per-stock broker CSVs live under `data/broker/twse/by_stock/` and are downloaded by `downloader/broker.py`. Filenames should start with the stock code, and `data/metadata.csv` column `有分點資料` / canonical `has_broker` should be `1` only when a non-empty TWSE BSR CSV exists for that code under `data/broker/twse/by_stock/`.
- TWSE BSR converted broker files live under `data/broker/twse/by_broker/`; build them with `tools/build_twse_broker_by_broker.py`, which writes one CSV per normalized broker name with `Date`, `Code`, `Name`, broker, price, buy shares, sell shares, source sequence, and source filename.
- TWSE BSR date-scoped batches live under `data/broker/twse/by_date/YYYY-MM-DD/`, with `by_stock/` mirroring raw files for that date and `by_broker/` holding broker-regrouped files for that date only.
- Use `data/trading_days.csv` as the canonical trading-day calendar instead of guessing weekdays or holidays.
- `data/trading_days.csv` is sourced from TWSE FMTQIK online history, which starts at `1990-01-04`.
- Run `python downloader/trading_days.py` before automation preflight checks that depend on `data/trading_days.csv`; the default command refreshes only the recent overlap from the existing file, while explicit `--start-date` can rebuild a wider range.
- Price CSV files live in `data/price/`. The current schema is:
  `Date,Capacity,Turnover,Open,High,Low,Close,Change,Transaction,open_adj,close_adj,high_adj,low_adj,AdjFactor`.
- `data/price/` contains only catalog-keyed per-security CSVs. Do not write `twse_price_YYYY-MM-DD.csv` aggregates or retain files whose leading code is absent from `data/metadata.csv`.
- `data/price/` adjusted columns use 前復權 factors derived from `data/dividend/ex_right_dividend/` when applicable. Use `tools/apply_forward_adjustments_to_price.py` after price or dividend updates.
- Use `tools/repair_price_health.py` for full-stock price health scans. It should only overwrite rows after official TWSE data confirms a different OHLCV row, and it must recompute `open_adj`, `close_adj`, `high_adj`, `low_adj`, and `AdjFactor` for touched price files.
- Prefer per-stock time-series CSVs for cleaned analysis datasets. Price CSVs use `CODE_公司簡稱.csv`, for example `2330_台積電.csv`.
- In `data/day_trading/`, use flat per-stock CSVs as the canonical complete dataset, with `day_trading.logs` holding the manifest, missing-date log, and skipped-code log. Rebuild it with `tools/rebuild_day_trading_by_stock_full.py`, which normalizes dates to Gregorian `YYYY-MM-DD` and filters to TWSE listed common stocks from `data/metadata.csv`.
- `data/day_trading/` keeps only TWSE day-trading raw fields that the source currently provides (`Date`, `Code`, `Name`, suspension note, day-trading volume, buy amount, and sell amount) plus derived analysis fields calculated from unadjusted `data/price/` rows: `當沖成交股數占比`, `當沖買進成交金額占比`, `當沖賣出成交金額占比`, `當沖成交值`, `當沖成交值占比`, `當沖平均買進價格`, `當沖平均賣出價格`, `當沖平均價差率`, `當沖買賣金額差率`, `日內振幅`, `開收報酬率`, `當沖成交股數占比20日ZScore`, and `當沖成交值20日ZScore`.
- Use per-stock CSVs as the source of truth for stock-keyed datasets. Do not keep parallel root-level aggregate CSVs when per-stock files can represent the same data.
- CSV files in `data/` should use Chinese display column names where practical. Keep trivial identifier/time columns such as `Date`, `date`, `Code`, `Name`, `Year`, `Quarter`, and `Time` in English.
- Python code should use canonical internal names through `column_schema.py`: read storage CSVs with `read_csv_canonical(...)`, write data CSVs with `to_csv_storage(...)`, and use `storage_fieldnames(...)` / `storage_record(...)` for `csv.DictWriter` outputs.
- Keep `column_schema.COLUMN_TRANSLATIONS` bijective. TAIFEX futures `open_interest` stores as `期貨未平倉量`, while options `OI` stores as `選擇權未平倉量`; never map two canonical fields to one storage header.
- When adding or renaming data columns, update `column_schema.COLUMN_TRANSLATIONS` first, then update downloader/writer code and migrate existing CSV headers. Avoid duplicate translated names within the same CSV schema.
- Flat single-schema per-stock folders: `data/yield_pe_pb/`, `data/events/`, `data/institutional/`, `data/investor_conference/`, `data/margin/`, `data/report/`, `data/revenue/`, `data/sbl/`, `data/shareholder_meeting/`, and `data/shareholding/`. Use `CODE_公司簡稱.csv` filenames when a metadata name is available.
- `data/events/` stores MOPS historical material events from `downloader/events.py` as one CSV per stock, with schema `Date,Time,Code,Name,Subject,FactDate,Clause,Description,Spokesperson,SpokespersonTitle,SpokespersonPhone,Source,SourcePath,SourceMarket,DetailSeqNo,DetailSpokeDate,DetailSpokeTime,FetchedAt`. The MOPS old-site historical query path currently returns data from `2011-01-01` onward; older metadata start dates should not be treated as guaranteed MOPS event coverage.
- `data/shareholding/` stores TDCC shareholding distribution rows keyed by data date, exact security code, and holding level. Keep only exact TWSE listed common-stock codes from `data/metadata.csv`; do not fold preferred-share suffix codes into common-stock files.
- `downloader/update_all_data.py` should fill `data/shareholding/` TDCC bucket labels from `持股分級` when the Open Data snapshot omits `持股/單位數分級`, so visualizations keep readable bucket names.
- Shareholding snapshots must be processed idempotently for every listed code; one file already carrying the latest source date is not sufficient evidence to skip the market-wide snapshot.
- Flat multi-schema per-stock folders: `data/dividend/{dividend_distribution,ex_dividend_forecast,ex_right_dividend}/`, `data/financial/<dataset_name>/`, and `data/insiders/{director_shareholding,insider_transfer_pre,insider_transfer_untransferred}/`.
- Only `data/dividend/ex_right_dividend/` should have generated visualization output. Do not generate `data_viz/dividend/` pages for `dividend_distribution/` or `ex_dividend_forecast/`. Dividend visualizations should use Chinese titles/labels, quarterly x-axis labels, and visible numeric labels on each plotted data point.
- `data_viz/margin/` visualizations should reuse the price WebGL/canvas viewer where possible, showing price OHLC and selectable margin/short leverage indicators in the same interactive chart with Chinese titles and labels.
- `data_viz/day_trading/`, `data_viz/institutional/`, `data_viz/dividend/ex_right_dividend/`, and `data_viz/yield_pe_pb/` visualizations should reuse the same price WebGL/canvas overlay viewer used by `data_viz/margin/`, pairing price OHLC with selectable dataset-specific indicators instead of maintaining separate duplicated chart code.
- `viz/generate_dataset_viz.py` must enumerate only flat, catalog-keyed source CSVs and return a non-zero exit code when any renderer fails; a written failure placeholder is not a successful run.
- `data_viz/institutional_participation/` reports should also reuse `viz.generate_dataset_viz.write_price_webgl_page(...)` so prices are drawn as WebGL K-lines; only the volume bars should differ by using stacked `外資`/`投信`/`自營商`/`其他` volume segments.
- The `data_viz/institutional_participation/index.html` stock table should list stocks whose dominant group is `外資` first, sorted by dominant participation descending, followed by the other dominant groups also sorted by dominant participation descending.
- Use `tools/build_foreign_participation_distribution_report.py` to turn the latest institutional-participation summaries into a foreign-investor concentration report at `data_viz/institutional_participation/foreign_distribution.html`, with reusable CSV summaries under `output/institutional_participation/`, including `foreign_distribution_top_80pct_stocks.csv`, broad-category summaries, and metadata `產業群組` summaries.
- Use `tools/build_participant_distribution_reports.py` when matching concentration reports are needed for 投信, 自營商, or 其他. It writes `trust_distribution.html`, `dealer_distribution.html`, and `other_distribution.html` plus matching CSV summaries under `output/institutional_participation/`.
- Use `tools/build_historical_participant_distribution_reports.py` when the participant concentration reports should use all available history instead of a latest-day snapshot. The historical path must restrict each stock to dates where both `data/price/` and `data/institutional/` have rows, so older price-only dates are not misclassified as `其他`.
- Use `tools/build_participant_decision_pnl_report.py` when estimating whether 外資/投信/自營商/其他 net-buy decisions were followed by favorable adjusted-price moves. Treat the output as decision payoff, not true inventory accounting P&L; `其他` is the inverse of the three major institutions' net flow.
- Use `tools/backtest_institutional_flow_strategy.py` for artifact-first institutional-flow strategy backtests. The first version is long-only, uses盤後 1/3/5-day net-flow strength signals, buys at the next adjusted open, exits after 1/5/10/20/30/60 fixed holding days, and writes four participant reports plus a summary with grouped same-parameter comparison charts under `data_viz/institutional_flow_backtest/`. It also writes 投信 dynamic-exit variants, including strong-sell-rank exit and a net-buy/sell quota exit with a next-open stop-loss rule. The quota-entry rule is configurable; the current full-market quota report uses 投信連續 3 個交易日淨買超 plus a minimum total buy-value threshold, initial quota is the consecutive-entry window's total 投信 net-buy shares, later 投信 net-buy increases quota, net-sell decreases quota, and quota <= 0 exits at the next open.
- Use `tools/backtest_trust_top50_cumulative_sell_stop_strategy.py` when limiting the 投信 net-buy/sell quota stop-loss strategy to the full-history best 50 stocks. Treat the universe selection as in-sample unless a separate walk-forward selection is added. Its HTML report should include the canvas-based trade timeline showing daily buys, exits, active positions, and hover transaction details.
- Use `tools/build_latest_institutional_flow_report.py` for latest-date all-listed-stock 三大法人淨流向 reports. It should write standalone HTML under `data_viz/institutional_flow/`, supporting CSVs under `output/institutional_flow/`, and label `其他市場參與者` as a residual net-flow estimate rather than a direct trade counterparty or pure retail flow. The Sankey-style flow chart should render source and target flow segments as stacked, non-overlapping ribbons aligned to their node edges.
- Use `tools/build_stock_chip_inventory_report.py` when estimating single-stock or all-listed-stock rolling-window institutional chip inventory and cost distribution. Treat the report as flow-implied inventory, not official ownership; compare FIFO, FILO/LIFO, and average-cost methods for cost and P&L sensitivity. The current model warms up from all overlapping price/institutional history, infers the minimum initial inventory required to avoid negative lots from historical net-sell drawdowns, estimates that initial inventory cost from the warmup-start average trade price, and marks current price on cost-distribution charts. Cost-distribution bins should be derived from each stock/method price spread and should render at least 20 bins instead of using a fixed 50 TWD width. The market index should use method-comparison histograms for the four-group summary, while stock rankings and all-stock tables should use FIFO as the primary baseline. Batch mode writes the market index under `data_viz/institutional_flow_inventory/` and supporting CSVs under `output/institutional_flow_inventory/`.
- Do not recreate generic `by_stock/` wrappers, `latest_asof` snapshots, root-level aggregate CSVs, downloader manifests, raw caches, dashboard files, or failure/debug CSVs under `data/` for consolidated datasets. Explicit source-specific layouts such as `data/broker/twse/by_stock/` are allowed when documented here. Put diagnostics under `logs/`.
- `downloader/update_all_data.py` must append or refresh canonical dataset CSVs only: it may create catalogued per-stock files, refresh `FetchedAt` for unchanged snapshot rows, rewrite only price adjusted columns, and refresh documented TWSE BSR broker outputs under `data/broker/twse/{by_stock,by_broker,by_date}/`, but it must not create undocumented `by_stock/` wrappers, raw caches, manifests, debug files, or CODiS/weather outputs.
- Daily updater orchestration should go through `downloader/daily/registry.py` `TaskSpec`s and `downloader/daily/status.py` structured status collection. Independent dataset failures should be reported as `failed` statuses while later independent tasks continue; the CLI should return non-zero when any collected status is failed.
- Daily task modules must reference compatibility dependencies explicitly through `downloader/daily/tasks/_runtime.py`; do not restore broad copying of `update_all_data` globals into task modules.
- Trading-day refresh may continue in an explicit warning/degraded mode only when a valid cached calendar exists. A missing, empty, or unusable calendar is fatal before any updater writes, and date resolution must never guess that a weekday is a trading session.
- `downloader/update_all_data.py` should refresh or replace OpenAPI snapshot rows by dataset-specific natural keys instead of appending repeated identical snapshots just because `FetchedAt` changed.
- `downloader/update_all_data.py` should classify zero-row MOPS report fetches as `no_source_data` / 未申報 only when filing-period evidence shows the company has not filed; a filed-period zero-row parse is a failure.
- `downloader/update_all_data.py` should fill `data/yield_pe_pb/` `Close` values from same-date unadjusted `data/price/` rows when the TWSE valuation snapshot omits close prices; do not infer valuation fiscal fields from price data.
- `downloader/update_all_data.py` must recompute `data/margin/` derived leverage columns after margin/price updates and also when margin files are otherwise up to date, so stale or missing derived values are repaired.
- On weekday TWSE market closures, run `downloader/update_all_data.py --market-closed` instead of skipping the updater entirely. This flag skips `data/price/` updates while still running non-price daily and snapshot sources such as TDCC shareholding, revenue, financial/report, company, events, and other OpenAPI/MOPS-style datasets.
- `downloader/update_all_data.py` may delete and refetch only known-bad trailing rows: price rows with blank adjusted columns at the file tail, and recent margin rows from the first date-order inversion suffix. Do not use this as a general cleanup rule or historical margin backfill.
- Daily all-market snapshots are acceptable for raw or source-style data, but do not make them the main analysis format unless the downstream script already expects that layout.
- `data/broker/by_broker/` stores Fubon broker rank rows regrouped from `data/broker/fubon/` into one CSV per broker branch name (`分點名稱.csv`). Keep these files as Fubon-only branch histories; exclude WantGoo and HiStock rows unless a future workflow explicitly requests a mixed-source layout.
- Use `downloader/broker.py --all-metadata` for captcha-protected TWSE BSR all-stock broker CSV attempts. It reads listed common stocks from `data/metadata.csv`, writes progress logs under `logs/broker/`, writes raw CSV outputs under `data/broker/twse/by_stock/`, and refreshes the metadata `有分點資料` flag from files actually present.
- `downloader/update_all_data.py` runs the TWSE BSR broker workflow unless `--skip-broker`, `--skip-daily`, or `--market-closed` is used. The workflow downloads missing date-labeled raw stock files, mirrors that date's raw files into `data/broker/twse/by_date/YYYY-MM-DD/by_stock/`, rebuilds cumulative `data/broker/twse/by_broker/`, and writes date-only broker files under `data/broker/twse/by_date/YYYY-MM-DD/by_broker/`. Run it in an environment with PaddleOCR available for captcha recognition.
- TAIFEX derivatives data lives under `data/taifex/` and is separate from stock-keyed TWSE datasets. Keep TAIFEX futures/options daily history, put/call ratio, institutional positions, and large-trader OI as source-style CSVs with ISO `Date` values.
- Refresh identity fields with `python downloader/metadata.py`; use `python downloader/metadata.py --availability-only` to recompute catalog flags from current canonical files without network access.
- When adding new daily-updated datasets, make updates idempotent: key by date plus stock code or source-specific unique keys, replace or skip existing keys, and avoid duplicate rows.
- Write CSVs as UTF-8 with BOM (`encoding="utf-8-sig"`) when the file may contain Chinese text or is meant to open cleanly in Excel.
- Normalize all CSV date columns under `data/` to Gregorian `YYYY-MM-DD` when a full date is available. If the source only provides year-month, use Gregorian `YYYY-MM`.
- Use `column_schema.normalize_date_text(...)` / `normalize_date_columns(...)` or `to_csv_storage(...)` for CSV writers so new downloader output keeps ISO date storage.
- `data/margin/` includes derived leverage columns: `融資使用率`, `融資餘額20日變化率`, `融資市值`, `融資市值20日均成交值比`, and `券資比`. These are calculated from margin balances plus unadjusted `Close` and `Turnover` from `data/price/`; do not use adjusted prices for these fields.
- Do not commit secrets, credentials, certificate passwords, local `.env` files, or broker login material.

## Downloaders
- The downloaders script name must identical to the dataset name. for example: price.py will output to data/price/.
- Put new download scripts in `downloader/`.
- Reuse existing downloader patterns: `argparse` CLIs, `PROJECT_ROOT`, explicit output paths, polite throttling, retry/backoff, clear status prints, and failure logs.
- TWSE API dates should be normalized to ISO `YYYY-MM-DD` in cleaned outputs. Convert ROC dates explicitly.
- TWSE `MI_INDEX` full-market security history currently starts at `2004-02-11`; older metadata `Start` dates can only be backfilled when another source supports them.
- Filter downloaded rows to listed common stocks by default. Add an explicit opt-in flag if a downloader can include OTC, ETFs, or other instruments.
- Keep source schemas explicit with column constants. Avoid silent column drift; validate required columns before writing or analyzing.
- For large historical downloads, support resume/idempotency through completed-date tracking, existing-key checks, or cached-file reuse.
- Use `downloader/events.py` for MOPS historical material-event backfills. Prefer monthly all-listed-market list queries plus metadata filtering over per-stock monthly queries; use `--enrich-existing-details` to resume filling detail fields in existing `data/events/` CSVs without refetching list pages. Detail failures are logged to `logs/events/events_detail_failures.csv` and skipped on later normal enrichment runs until `--retry-known-detail-failures` is used. Per-row detail enrichment is much slower and should be run with conservative throttling and resume support.
- MOPS detail enrichment should skip malformed existing rows with non-ISO or blank `Date` values and report the skip count, rather than aborting the whole resumable run.
- Use `tools/run_events_detail_enrichment_batches.py` for long MOPS detail-enrichment runs that need repeated bounded chunks, batch-level progress CSVs, status-only progress checks, optional `--run-until-complete` terminal execution, and automatic stopping after zero-progress batches.
- Use `downloader/taifex.py` for TAIFEX derivatives downloads. It should use official TAIFEX annual ZIPs where available, official date-range CSV endpoints for current-year or non-annual datasets, polite throttling, and `logs/taifex/` manifests.

## Analysis And Modeling

- Prefer adjusted columns from `data/price/` for return, correlation, regime, pair-trading, and alpha-factor work.
- Keep unadjusted prices from `data/price/` for raw OHLCV inspection, daily update ingestion, and compatibility with existing scripts.
- Avoid lookahead bias: factors must use only information available before the label/return horizon.
- Use `alpha_model/config/trust_flow.yaml` for 投信買賣 alpha-factor IC validation. It evaluates 投信淨買金額/成交值 rolling factors, 投信買進純度, 3 日連續買超長度, and 3 日賣壓; labels use next adjusted-open to later adjusted-open returns so the盤後法人 signal is not treated as same-day tradable.
- Alpha-model loading fails closed when the metadata catalog is missing, unreadable, malformed, or empty. Broad loading requires the explicit `--allow-unfiltered-universe` CLI flag or `data.allow_unfiltered_universe: true` configuration.
- `trust_buy_streak_3d` is the trailing consecutive positive-buy run length capped at three and requires a complete three-observation institutional window. Missing institutional rows remain missing rather than becoming zero signals, and coverage reports retain zero-valid-row factor/date combinations.
- Preserve the existing normalized analysis columns where used: `date`, `stock_id`, `open`, `high`, `low`, `close`, `volume`, `turnover`, `change`, `transactions`.
- Put generated visualization reports, charts, visualization manifests, and chart indexes in `data_viz/` rather than the repo root.
- Put non-visual analysis tables, matrices, and model artifacts in `output/` unless they are part of a visualization report bundle.
- Keep reusable dataset visualization scripts under `viz/`. Legacy root-level visualization entry points may remain as compatibility wrappers, but new dataset visualization work should live in `viz/`; price visualizations may use WebGL canvas rendering for full-history interactive OHLC pages with cursor OHLC/volume readouts and one-trading-day keyboard navigation.
- Price-backed event or factor reports should reuse `viz.generate_dataset_viz.write_price_webgl_page(...)` with selectable overlay metrics and optional `highlight_rules` instead of static SVG when zoom/pan inspection matters.
- WebGL price and price-backed K-line visualizations should render a visible minimum-height candle body when open equals close, mark ex-right/dividend dates from `data/dividend/ex_right_dividend/`, label cash-capital-increase ex-right events as `現金增資` instead of generic `除權息`, and use the ex-right/dividend reference price rather than the previous close for cursor daily percent change on those event dates.
- Raw WebGL K-line cursor percent change should use the price CSV `Change` column to infer the official reference price when available, falling back to the previous row close only when `Change` is missing; this avoids false jumps after trading suspensions or special reference-price days.
- Never use scientific notation in visualization labels, axis ticks, tooltips, or visible annotations. Use full numbers with separators or compact suffixes such as `1.23K`, `2.56M`, or `1.2B`.
- When generating user-facing explanation documents, Markdown reports, or HTML reports, use Chinese for visible headings, labels, table headers, state names, and explanatory body text unless the user explicitly asks for another language. Before reporting completion, verify the generated document or report does not contain leftover English UI labels or headings.
- Shareholding visualizations under `data_viz/shareholding/` should use Chinese titles, visible bar-value labels, and abbreviated x-axis bucket labels such as `1-999`, `1k-5k`, `10k-15k`, `50k-100k`, and `1M+`.
- Shareholding visualizations under `data_viz/shareholding/` should include a three-type pie chart below the existing histograms: `散戶`, `大型散戶`, and `1%以上股東`. The three slices must be mutually exclusive and sum to 100%; market-value and 1% capital thresholds should be snapped to the nearest TDCC holding-level boundary when an exact split falls inside a histogram bucket. Pie labels should be drawn as non-overlapping callouts from the pie instead of a separate legend, and should show the close price plus calculated holding-position market-value thresholds (`持有部位市值`).
- For reusable analysis logic, prefer package modules such as `alpha_model/` or `strategies/` over one-off root scripts when the code will be called by tests or other scripts.

## Code Style

- Python is the primary language. Follow existing style in the touched file rather than introducing broad formatting churn.
- Prefer `pathlib.Path` in new code, but work with existing `os.path` code locally when that file already uses it.
- Keep stock IDs as strings in pandas reads and joins: use `dtype={"Code": str}`, `dtype={"stock_id": str}`, or equivalent.
- Use structured parsing for JSON/CSV/HTML tables where practical. Avoid brittle string slicing when a parser is already available.
- Keep comments short and useful, especially around TWSE/Fubon quirks, date conversion, schema assumptions, and anti-lookahead logic.
- Do not rewrite unrelated generated data files as part of code changes.

## Testing And Verification

- Install core dependencies from `requirements.txt`; use `requirements-dev.txt` for pytest, Ruff, and Vulture. OCR broker runs use `requirements-broker.txt`; Playwright workflows use `requirements-browser.txt` followed by `python -m playwright install chromium`.
- Run `python -m ruff check .` and the relevant pytest suite before committing. Use Vulture as a review aid, but do not delete standalone CLIs or dynamically discovered strategy classes solely because static incoming references are absent.
- For root analysis/modeling changes, run the relevant tests under `tests/`, or at least the targeted test file.
- For downloader changes that hit the network, prefer unit-level parser/schema tests when possible. If a live request is necessary, keep it narrow and mention it.
- Before finishing changes that affect CSV outputs, verify headers, date parsing, stock-code dtype, duplicate-key behavior, and output path.
- Whenever a task modifies a CSV data folder or visualization output folder, randomly sample 10 output files from each modified folder and verify they match the expected schema/content/format. If fewer than 10 outputs exist, verify all of them. Report the sampled paths and what was checked.
- In general, after producing generated outputs, randomly inspect representative output artifacts before reporting completion so the delivered files are confirmed to match the requested result rather than only checking that commands succeeded.
- If tests cannot be run because of missing dependencies, credentials, network access, or market availability, report that clearly.
