ALTER TABLE raw_market_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, channel'
);

ALTER TABLE trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

ALTER TABLE orderbook_5 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('raw_market_events', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('trades', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('orderbook_5', INTERVAL '7 days', if_not_exists => TRUE);
