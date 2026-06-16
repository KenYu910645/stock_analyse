# inDayTradeBook

Fubon Neo realtime market-data collector for TimescaleDB.

## Setup

```powershell
cd inDayTradeBook
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your Fubon credentials and certificate path. Do not commit
`.env`.

Leave `FUBON_WS_URL` empty for production. For Fubon’s test environment, set it
to `wss://neoapitest.fbs.com.tw/TASP/XCPXWS`.

## Start TimescaleDB

```powershell
docker compose up -d timescaledb
python scripts/init_db.py
```

Compression is intentionally disabled by default for early debugging. Enable it
after the schema is stable:

```powershell
python scripts/init_db.py --with-compression
```

## Run Collector

```powershell
python -m realtime.main
```

The default realtime symbols are `2330,2308,3105`, and the default channels are
`trades,books`.

## Daily Monitor-List Recording

Put one stock per line in `monitor_list.txt`. Lines may include comments or
descriptions after the 4-digit code; only the leading code is used.

Check the parsed symbols and subscription count:

```powershell
python scripts/load_monitor_list.py --format summary
```

Current list:

```text
59 stocks x 2 channels = 118 WebSocket subscriptions
```

Fubon allows 200 subscriptions per WebSocket connection, so this fits in the
current one-connection collector.

Install the weekday 09:00 Windows Task Scheduler job:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1
```

The task name is `InDayTradeBookRealtime`. It runs Monday-Friday at 09:00 and
stops the collector at 13:30. It loads credentials from
`C:\CAFubon\credential.txt`, leaves the certificate password unset so the SDK
receives Python `None`, starts TimescaleDB, initializes the DB schema, and writes
dated logs under `logs/`.

Manual dry run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_market_session.ps1 -DryRun
```

Manual run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_market_session.ps1
```

Check recording status:

```powershell
python scripts/check_recording_status.py
```

## Sample Queries

```powershell
python scripts/query_sample.py --symbol 2330 latest-trade
python scripts/query_sample.py --symbol 2330 latest-orderbook
python scripts/query_sample.py --symbol 2330 spread
python scripts/query_sample.py --symbol 2330 imbalance
python scripts/query_sample.py event-counts
```

Raw events are stored first in `raw_market_events`; parsed trades and order book
snapshots are best-effort normalized afterward.

## Historical API Trial

Use this to verify Fubon login and HTTP market-data access without waiting for
the realtime market to be open:

```powershell
python scripts/trial_historical_candles.py --symbol 2330 --timeframe 1
```

For minute K data, Fubon returns the recent five days and ignores `from`/`to`.

To verify the historical 52-week stats endpoint:

```powershell
python scripts/trial_historical_stats.py --symbol 2330
```

## Tests

From the repo root:

```powershell
python -m pytest inDayTradeBook/tests
```

From this folder:

```powershell
python -m pytest tests
```
